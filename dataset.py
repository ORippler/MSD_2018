import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import os
import json
import argparse
import logging
from utils.sitk_utils import resample_to_spacing
from utils.utils import start_logger
from batchgenerators.augmentations.spatial_transformations \
    import augment_spatial
from batchgenerators.augmentations.noise_augmentations \
    import augment_gaussian_noise


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_json(fpath):
    with open(fpath, 'r') as fp:
        return(json.load(fp))


def write_json(f, fpath):
    with open(fpath, 'w') as fp:
        return(json.dump(f, fp))


def transforms(volume, segmentation, patch_size_, fraction_, spacing_,
               variance_):
    # BEWARE SPACING: dset.spacing = (x,y,z) but volume is shaped (z,x,y))
    volume, augmentation, rval = augment_spatial(
                                    volume,
                                    segmentation,
                                    patch_size=patch_size_,
                                    patch_center_dist_from_border=np.array(
                                        patch_size_) // 2,
                                    border_mode_data='constant',
                                    border_mode_seg='constant',
                                    border_cval_data=np.min(volume),
                                    border_cval_seg=0,
                                    alpha=(0, 750),
                                    sigma=(10, 13),
                                    scale=(0.8, 1.2),
                                    do_elastic_deform=True,
                                    do_scale=True,
                                    do_rotation=True,
                                    angle_x=(0, 2*np.pi),
                                    angle_y=(0, 0),
                                    angle_z=(0, 0),
                                    fraction=fraction_,
                                    spacing=spacing_)
    if np.any(variance_ != 0):
        volume = augment_gaussian_noise(volume, noise_variance=variance_)
    return volume, augmentation, rval


class TFRdataset():
    def __init__(self, fpath):
        self.fpath = fpath
        self.dset = read_json(os.path.join(self.fpath, 'dataset.json'))
        self.num_classes = len(self.dset['labels'])
        self.num_modalities = len(self.dset['modality'])
        self.batch_size = None
        self.spacing = None
        self.shapes = None
        self.dset_path = os.path.join(self.fpath, 'dataset.tfrecord')
        print(self.dset.keys())

        try:
            self.spacing = self.dset['spacing']
        except:
            pass

        try:
            self.shapes = self.dset['shapes']
        except:
            pass

    def make_train_dataset(self, save_interpolation=None, compression=None,
                           output_path=None):

        if compression:
            options = tf.python_io.TFRecordOptions(
                            tf.python_io.TFRecordCompressionType.GZIP)
        else:
            options = tf.python_io.TFRecordOptions(
                            tf.python_io.TFRecordCompressionType.NONE)
        self.writer = tf.python_io.TFRecordWriter(
                            os.path.join(output_path, 'dataset.tfrecord'),
                            options=options)

        holder = {}
        spacings = {}
        logging.info('Constructing training dataset')
        for img_pair in (self.dset['training']):
            write = True
            holder['idx'] = img_pair['label'].split('./')[-1]
            for img_type, img_path in img_pair.items():
                sitk_img = sitk.ReadImage(
                                os.path.join(self.fpath,
                                             img_path.split('./')[-1]))

                origin_ = sitk_img.GetOrigin()
                orientation_ = sitk_img.GetDirection()
                spacings[img_type] = sitk_img.GetSpacing()
                if img_type != 'label':
                    img = self.resample_image_spacing(
                                    sitk_img, interpolation_='Spline')
                    holder[img_type] = img
                    holder['spacing_orig'] = sitk_img.GetSpacing()
                    holder['shape_orig'] = sitk_img.GetSize()
                elif img_type == 'label':
                    # modalities channel spacing
                    if len(self.spacing) - len(sitk_img.GetSpacing()) != 0:
                        sitk_img.SetSpacing(
                            spacings.pop('image')[:3 - len(self.spacing)])
                    else:
                        sitk_img.SetSpacing(
                            spacings.pop('image'))
                    img = self.resample_image_spacing(
                                sitk_img, interpolation_='nearest')
                    try:
                        holder[img_type] = self.get_multi_class_labels(img)
                    except AssertionError:
                        logging.warning('label conversion failed for {}, \
                                         excluding from training dataset'.format(holder['idx']))
                        write = False
                if save_interpolation:
                    sitk_img_resampled = sitk.GetImageFromArray(img)
                    sitk_img_resampled.SetSpacing(self.spacing)
                    sitk_img_resampled.SetOrigin(origin_)
                    sitk_img_resampled.SetDirection(orientation_)
                    sitk.WriteImage(sitk_img_resampled,
                                    os.path.join(
                                        output_path,
                                        value.split('./')[-1].split('.gz')[0]))
            if write:
                self.write_to_tfrecord(holder)
                logging.info(holder['idx'] + ' processed')
        self.writer.close()

    def scan_dataset_spacing_shapes(self):
        spacings = []
        shapes = []
        for img_pair in (self.dset['training']):
            itkimg = sitk.ReadImage(
                        os.path.join(self.fpath,
                                     img_pair['image'].split('./')[1]))
            spacings.append(itkimg.GetSpacing())
            shapes.append(itkimg.GetSize())
            logging.info(img_pair['image'] + ' scanned')
        self.spacing = np.median(spacings, axis=0)
        self.shapes = np.median(shapes, axis=0)
        # json compatibility
        self.dset['spacing'] = self.spacing.tolist()
        self.dset['shapes'] = self.shapes.tolist()
        logging.info(self.fpath + 'spacing set: {}'.format(self.spacing))
        logging.info(self.fpath + 'shapes set: {}'.format(self.shapes))
        write_json(self.dset, os.path.join(self.fpath, 'dataset.json'))

    def resample_image_spacing(self, itk_image, interpolation_=None):
        itk_image_rank = len(itk_image.GetSpacing())
        if itk_image_rank - len(self.spacing) != 0:
            resampled = resample_to_spacing(
                            itk_image,
                            target_spacing=self.spacing[:itk_image_rank
                                                        - len(self.spacing)],
                            interpolation=interpolation_)
        else:
            resampled = resample_to_spacing(
                            itk_image,
                            target_spacing=self.spacing,
                            interpolation=interpolation_)
        return resampled

    def get_multi_class_labels(self, mask):
        """
        Translates a label map into a set of binary labels.
        :param data: numpy array containing the label map with shape:
                     (n_samples, 1, ...).
        :param n_labels: number of labels.
        :param labels: integer values of the labels.
        :return: binary numpy array of shape: (n_samples, n_labels, ...)
        """
        new_shape = [self.num_classes] + list(mask.shape)
        y = np.zeros(new_shape, np.uint8)
        uniques = np.unique(mask)
        assert (self.num_classes >= len(uniques)), "expected {} number of \
                classes, instead got {} : {}".format(self.num_classes,
                                                     len(np.unique(mask)),
                                                     uniques)
        for label_index in range(self.num_classes):
            y[label_index][mask == label_index] = 1
        return y

    def write_to_tfrecord(self, img_pair):
        try:
            assert(img_pair['image'].dtype == np.float32)

        except AssertionError:
            logging.info(img_pair['idx'] + ': np.float32 expected, got {}, \
            converting to np.float32'.format(img_pair['image'].dtype))
            img = img_pair['image'].astype(np.float32)
            img_pair.pop('image')
            img_pair['image'] = img

        try:
            assert(img_pair['label'].dtype == np.uint8)
        except AssertionError:
            logging.info(img_pair['idx'] + ': np.uint8 expected, got {}, \
            converting to uint8'.format(img_pair['label'].dtype))
            label = img_pair['label'].astype(np.uint8)
            img_pair.pop('label')
            img_pair['label'] = label

        if len(img_pair['image'].shape) == 4:
            try:
                assert(img_pair['image'].shape[1:]
                       == img_pair['label'].shape[1:])
            except AssertionError:
                logging.warning(img_pair['idx'] + ':expected shapes to be \
                                equal, but image was {} and label was {}'.format(
                                    img_pair['image'].shape[1:],
                                    img_pair['label'].shape[1:]))
        else:
            try:
                assert(img_pair['image'].shape == img_pair['label'].shape[1:])
            except AssertionError:
                logging.warning(img_pair['idx'] + ':expected shapes to be \
                                equal, but image was {} and label was {}'.format(
                                    img_pair['image'].shape,
                                    img_pair['label'].shape[1:]))

        raw_image = img_pair['image'].tobytes()
        raw_label = img_pair['label'].tobytes()
        shape_i = np.array(img_pair['image'].shape, np.int32).tobytes()
        shape_l = np.array(img_pair['label'].shape, np.int32).tobytes()

        # 32bit constrained from google protobof
        Kint32Max = 2147483647.0

        if Kint32Max - len(raw_image) - len(raw_label) - len(shape_i) - len(shape_l) < 0:
            logging.warning(
                img_pair['idx'] + ': raw bytecounts > Kint32Max, \
                cropping image and label')

            label = img_pair.pop('label')

            z_min = np.sort(np.unique(np.where(label[1] != 0)[0]))[0]-64
            z_max = np.sort(np.unique(np.where(label[1] != 0)[0]))[-1]+64

            img_pair['label'] = label[:, z_min:z_max]

            img = img_pair.pop('image')

            if len(img.shape) > 3:
                img_pair['image'] = img[:, z_min:z_max]
            else:
                img_pair['image'] = img[z_min:z_max]

            raw_image = img_pair['image'].tobytes()
            raw_label = img_pair['label'].tobytes()

            shape_i = np.array(img_pair['image'].shape, np.int32).tobytes()
            shape_l = np.array(img_pair['label'].shape, np.int32).tobytes()

        shape_i_orig = np.array(img_pair['spacing_orig'], np.int32).tobytes()
        spacing_i_orig = np.array(img_pair['shape_orig'], np.int32).tobytes()

        entry = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image': _bytes_feature(raw_image),
                        'shape_i': _bytes_feature(shape_i),
                        'img_dtype': _bytes_feature(
                            tf.compat.as_bytes(str(img_pair['image'].dtype))),
                        'label': _bytes_feature(raw_label),
                        'shape_l': _bytes_feature(shape_l),
                        'label_dtype': _bytes_feature(
                            tf.compat.as_bytes(str(img_pair['label'].dtype))),
                        'idx': _bytes_feature(
                            tf.compat.as_bytes(img_pair['idx'])),
                        'shape_i_orig': _bytes_feature(shape_i_orig),
                        'spacing_i_orig': _bytes_feature(spacing_i_orig)
                    }))
        self.writer.write(entry.SerializeToString())

    def provide_tf_dataset(self, number_epochs, shape=None, batch_size=1,
                           num_parallel_calls=8, mode='train', fraction=0.0,
                           shuffle_size=100, variance=0.1, compression=None,
                           multi_GPU=False):

        self.mode = mode
        self.shape = shape
        self.batch_size = batch_size
        self.fraction = fraction
        self.variance = variance

        if compression:
            compression_type_ = 'GZIP'
        else:
            compression_type_ = ''
        dataset = tf.data.TFRecordDataset(
                    self.dset_path, compression_type=compression_type_)
        dataset = dataset.apply(
                    tf.contrib.data.shuffle_and_repeat(shuffle_size,
                                                       count=number_epochs))
        dataset = dataset.apply(
                    tf.contrib.data.map_and_batch(
                        map_func=self.parse_image,
                        num_parallel_batches=num_parallel_calls,
                        batch_size=batch_size))

        dataset = dataset.prefetch(buffer_size=None)

        if multi_GPU:
            return dataset

        else:
            iterator = dataset.make_one_shot_iterator()
            inputs, outputs = iterator.get_next()
            return inputs, outputs

    def parse_image(self, entry):
        parsed = tf.parse_single_example(entry, features={
                    'image': tf.FixedLenFeature([], tf.string),
                    'shape_i': tf.FixedLenFeature([], tf.string),
                    'img_dtype': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.string),
                    'shape_l': tf.FixedLenFeature([], tf.string),
                    'label_dtype': tf.FixedLenFeature([], tf.string),
                    'idx': tf.FixedLenFeature([], tf.string),
                    'shape_i_orig': tf.FixedLenFeature([], tf.string),
                    'spacing_i_orig': tf.FixedLenFeature([], tf.string)
                    })

        # TODO: make dtype flexible
        shape_i = tf.decode_raw(parsed['shape_i'], tf.int32)
        image = tf.decode_raw(parsed['image'], tf.float32)
        image = tf.reshape(image, shape_i)
        image = tf.cond(
                    tf.greater_equal(
                        tf.rank(image), 4),
                    lambda: image,
                    lambda: tf.expand_dims(image, 0))

        shape_l = tf.decode_raw(parsed['shape_l'], tf.int32)
        segmentation = tf.decode_raw(parsed['label'], tf.uint8)
        segmentation = tf.reshape(segmentation, shape_l)

        stacked = tf.concat([image, tf.cast(segmentation, tf.float32)], axis=0)

        paddings = tf.clip_by_value(
                    tf.subtract([self.num_classes]+self.shape, shape_l),
                    0, 100000)
        paddings = tf.ceil(paddings/2)
        paddings_reshaped = tf.cast(tf.stack([paddings, paddings]), tf.int32)
        paddings_reshaped = tf.transpose(paddings_reshaped, [1, 0])
        stacked = tf.cond(
                        tf.reduce_mean(paddings) > 0.0,
                        lambda: tf.pad(stacked, paddings=paddings_reshaped,
                                       mode="reflect"),
                        lambda: stacked)

        cropped_i = stacked[:self.num_modalities, :, :, :]
        cropped_s = stacked[self.num_modalities:, :, :, :]

        cropped_i, cropped_s, rval = tf.py_func(
                                        transforms,
                                        [cropped_i, cropped_s, self.shape, self.fraction, self.spacing, self.variance],
                                        Tout=[tf.float32, tf.float32, tf.float64])
        cropped_s = tf.cast(cropped_s, tf.uint8)

        # static output shape required by tf.dataset.batch() method!
        cropped_i.set_shape([self.num_modalities] + self.shape)
        cropped_s.set_shape([self.num_classes] + self.shape)
        print("input shape" + str(cropped_i.shape))
        print("segmentation shape" + str(cropped_s.shape))

        objects = tf.where(cropped_s[1:] > 0)
        objects_in_crop = tf.reduce_mean(objects, axis=0)
        background = tf.where(cropped_s[0] > 0)
        ratio = tf.shape(objects)[0] / tf.shape(background)[0]
        objects_in_crop = tf.cond(
                            tf.shape(objects)[0] > 0,
                            lambda: objects_in_crop,
                            lambda: tf.zeros_like(objects_in_crop))
        tf.Print([], [ratio], message="ratio of fore to background in patch: ")

        if self.mode == 'train':
            return({'img': cropped_i, 'object_in_crop': objects_in_crop},
                   cropped_s)
        elif self.mode == 'dataloader_test':
            tf.Print(['img shape'], [tf.shape(stacked[:self.num_modalities])])
            return({'img': cropped_i, 'rval': rval},
                   cropped_s)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='generate dataset for'
                                     'MSD_challenge')
    parser.add_argument('-i', '--input', help='input (root) path',
                        required=True)
    parser.add_argument('-o', '--output',
                        help='path to store generated TFRecord and json',
                        required=True)
    parser.add_argument('-s', '--save_interpolation',
                        help='save interpolated images',
                        required=False,
                        default=None)
    parser.add_argument('-c', '--compression',
                        help='compress generated TFRecord',
                        required=False,
                        default=None)

    args = vars(parser.parse_args())

    fpath = args['input']
    output_path = args['output']
    compression = args['compression']
    save_images = args['save_interpolation']

    start_logger(os.path.join(fpath + 'dataset_generation'))
    logger = logging.getLogger(__name__)
    dset = TFRdataset(fpath)
    if dset.spacing is None or dset.shapes is None:
        dset.scan_dataset_spacing_shapes()
    dset.make_train_dataset(output_path=output_path,
                            compression=compression,
                            save_interpolation=save_images)
