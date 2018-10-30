import argparse
import tensorflow as tf
import os
from dataset import TFRdataset
import argparse
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from utils.patches import reconstruct_from_patches, get_patch_from_3d_data,\
                          compute_patch_indices
from utils.sitk_utils import resample_to_spacing


def patch_wise_prediction(output, data, patch_shape,
                          overlap=0, batch_size=1):

    """
    :param batch_size:
    :param model:
    :param data:
    :param overlap:
    :return:
    """
    predictions = []
    indices = compute_patch_indices(data.shape[-3:],
                                    patch_size=patch_shape,
                                    overlap=overlap,
                                    start=0)
    batch = []
    i = 0
    pbar = tqdm(enumerate(indices),
                unit='patches',
                total=len(indices),
                desc='scan_patches')

    sess = tf.get_default_session()

    for j, _ in pbar:
        while len(batch) < batch_size:
            patch = get_patch_from_3d_data(data[0],
                                           patch_shape=patch_shape,
                                           patch_index=indices[i])
            batch.append(patch)
            i += 1
        prediction = sess.run(output,
                              feed_dict={'img:0': np.asarray(batch)})
        batch = []
        for predicted_patch in prediction:
            predictions.append(predicted_patch)
    output_shape = [int(output.shape[1])] + list(data.shape[-3:])
    return reconstruct_from_patches(predictions,
                                    patch_indices=indices,
                                    data_shape=output_shape)


def main(fpath, output_path):

    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=False)

    with tf.Session() as sess:
        # initiate dataloader
        dset = TFRdataset(fpath)

        # initiate model
        tf.saved_model.loader.load(sess,
                                   ['serve'],
                                   dset.dset['fpath_model'])
        output = sess.graph.get_tensor_by_name('Generator/Softmax_1:0')

        pbar = tqdm(enumerate(dset.dset['test']),
                    unit='scan',
                    total=len(dset.dset['test']),
                    desc='images to segment')

        for i, img in pbar:
            # read image
            sitk_img = sitk.ReadImage(fpath + img.split('./')[-1])

            # resample image to dset_spacing
            element_spacing = sitk_img.GetSpacing()
            sitk_img_resampled = resample_to_spacing(
                                            sitk_img,
                                            target_spacing=dset.spacing,
                                            interpolation='linear')
            resampled_array = sitk_img_resampled

            # add dims until rank fits with what is expected by model
            while len(resampled_array.shape) < 5:
                resampled_array = resampled_array[np.newaxis, :]

            predicted = patch_wise_prediction(
                                output=output,
                                data=resampled_array,
                                overlap=0,
                                batch_size=1,
                                patch_shape=dset.dset['patch_shape'])

            np_mask = np.argmax(predicted, 0).astype(np.uint8)

            sitk_mask = sitk.GetImageFromArray(np_mask)
            sitk_mask.SetSpacing(dset.spacing)

            # resample prediction to original image spacing
            if dset.num_modalities > 1:
                sitk_mask_resampled = resample_to_spacing(
                                            sitk_mask,
                                            target_spacing=element_spacing[:-1],
                                            interpolation='nearest')
            else:
                sitk_mask_resampled = resample_to_spacing(
                                            sitk_mask,
                                            target_spacing=element_spacing,
                                            interpolation='nearest')

            sitk_mask_resampled = sitk.GetImageFromArray(sitk_mask_resampled)

            # if required, either crop or pad sitk_mask_resampled
            if dset.num_modalities > 1:
                offset = (np.array(sitk_mask_resampled.GetSize())
                          - np.array(sitk_img.GetSize()[:-1]))/2
            else:
                offset = (np.array(sitk_mask_resampled.GetSize())
                          - np.array(sitk_img.GetSize()))/2

            cropping = np.zeros(shape=(len(offset), 2),
                                dtype=int)
            padding_required = False
            for index, value in enumerate(offset):
                if value > 0.0:
                    cropping[index][0] = int(np.floor(value))
                    cropping[index][1] = int(-np.ceil(value))

                elif value == 0.0:
                    cropping[index][0] = 0
                    cropping[index][1] = sitk_img.GetSize()[index]

                elif value < 0.0:
                    padding_required = True
                    cropping[index][1] = sitk_img.GetSize()[index]

            if padding_required:
                new_array = sitk_mask_resampled
                difference = np.array(sitk_img.GetSize()) \
                    - np.array(sitk_mask_resampled.GetSize())
                difference = np.clip(difference,
                                     0,
                                     10e6).astype(int)
                difference = np.roll(difference, 1)
                difference = [(diff, 0) for diff in difference]
                new_array = np.pad(new_array, difference, mode='constant')

                sitk_mask_resampled = sitk.GetImageFromArray(new_array)
                sitk_mask.SetSpacing(element_spacing)

            sitk_mask_cropped = sitk_mask_resampled[
                                        cropping[0][0]:cropping[0][1],
                                        cropping[1][0]:cropping[1][1],
                                        cropping[2][0]:cropping[2][1]]

            sitk.WriteImage(
                sitk_mask_cropped,
                os.path.join(output_path, img.split('/')[-1] + '.nii.gz'))

            tqdm.write("{} processed".format(img))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Predict on Test dataset of \
                                                MSD Challenge Task")

    parser.add_argument('-i', '--input', help='input (root) path for dataset',
                        required=True)
    parser.add_argument('-o', '--output', help='full path where to put images',
                        required=True)

    args = vars(parser.parse_args())
    fpath = args['input']
    output_path = args['output']
    output_path = os.path.join(output_path,
                               fpath.split('/')[-2])

    main(fpath, output_path)
