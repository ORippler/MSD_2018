import argparse
import tensorflow as tf
import os
from model import model
from dataset import TFRdataset
import SimpleITK as sitk
import numpy as np
from metrics import generalized_dice_loss
import json
from utils.utils import distribution_strategy


def main(*kwargs):

    fpath = args['input']
    mdir = args['modeldir']
    loss = generalized_dice_loss
    os.environ["CUDA_VISIBLE_DEVICES"] = args['device']
    weighting = args['weighting']
    fraction = args['fraction']
    variance_ = args['variance']
    num_gpu = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    multi_gpu = distribution_strategy(num_gpu)

    dset = TFRdataset(fpath=fpath)

    # ensure minimum of 8 voxels in every dim
    if dset.num_modalities > 1:
        n_possible_depths = (np.log2(dset.shapes[:-1]) - 3).astype(int)
    else:
        n_possible_depths = (np.log2(dset.shapes) - 3).astype(int)

    # convert to channels first
    n_possible_depths = np.roll(n_possible_depths, 1)

    # limit model_depth due to GPU_mem
    depths = [4 if depth > 4 else depth for depth in n_possible_depths]

    if dset.num_modalities > 1:
        input_shape = [int(dset.shapes[-1])] \
                    + [2**(depth+3) for depth in n_possible_depths]
    else:
        input_shape = [1] \
                    + [2**(depth+3) for depth in n_possible_depths]

    # limit input_shape due to GPU_mem
    input_shape = list(np.clip(input_shape, 0,
                               input_shape[:-3] + [128, 256, 256]))

    if dset.num_modalities > 1:
        dset.spacing = [dset.dset['spacing'][-2],
                        dset.dset['spacing'][0],
                        dset.dset['spacing'][1]]
    else:
        dset.spacing = [dset.dset['spacing'][-1],
                        dset.dset['spacing'][0],
                        dset.dset['spacing'][1]]

    print(dset.spacing)
    print(input_shape)
    print(depths)

    # number of necessary repeats to get 300 "epochs" of 100 iterations each
    num_iters = dset.dset['numTraining'] / (100 * num_gpu)
    num_epoch_repeats = int(300 / num_iters)

    runconfig = tf.estimator.RunConfig(save_summary_steps=500,
                                       save_checkpoints_steps=500,
                                       train_distribute=multi_gpu)

    estimator = tf.estimator.Estimator(
                    model_fn=model,
                    model_dir=mdir,
                    params={'input_shape': input_shape,
                            'n_base_filters': 6,
                            'depth': depths,
                            'dropout_rate': 0.3,
                            'n_segmentaton_levels': np.max(depths),
                            'n_labels': dset.num_classes,
                            'optimizer': 'Adam',
                            'initial_learning_rate': 5e-4,
                            'loss_function': loss,
                            'weighting': weighting},
                    config=runconfig)

    def train_file(estimator, dataset):
        return estimator.train(
                    input_fn=lambda: dataset.provide_tf_dataset(
                                            number_epochs=num_epoch_repeats,
                                            shape=input_shape[1:],
                                            batch_size=1,
                                            num_parallel_calls=os.cpu_count(),
                                            fraction=fraction,
                                            shuffle_size=100,
                                            variance=(0, variance_),
                                            mode='train',
                                            compression=None,
                                            multi_GPU=num_gpu))

    train_file(estimator, dataset=dset)

    # Export model. Static input shapes are required until tf issue #20527
    # is fixed 
    if dset.num_modalities > 1:
        input_shape = [dset.shapes[-1]] \
                    + [val - (val % 2**(depths[index])) for index, val in enumerate(reversed(dset.shapes[:-1]))]
    else:
        input_shape = [1] \
                    + [val - (val % 2**(depths[index])) for index, val in enumerate(reversed(dset.shapes))]

    input_shape = list(np.clip(input_shape, 0,
                               [input_shape[0]] + [128, 512, 512]))

    exported_input = {'img': tf.placeholder(tf.float32,
                                            shape=[None] + input_shape,
                                            name='img')}
    input_receiver_fn = tf.estimator.export.\
                        build_raw_serving_input_receiver_fn(exported_input)
    estimator.export_savedmodel(mdir, input_receiver_fn)

    dset.dset['patch_shape'] = [int(i) for i in input_shape[1:]]

    mdir_folders = os.listdir(mdir)
    for i in mdir_folders:
        if os.path.isdir(os.path.join(mdir, i)):
            dset.dset['fpath_model'] = os.path.join(mdir, i)
            break

    with open(os.path.join(fpath, 'dataset.json'), 'w') as fp:
        json.dump(dset.dset, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 3D Net for \
                                                MSD_challenge dataset')

    parser.add_argument('-i', '--input', help='input (root) path for dataset',
                        required=True)

    parser.add_argument('-m', '--modeldir', help='path to store trained model',
                        required=True)

    parser.add_argument('-d', '--device', help='set Variable CUDA_VISIBLE DEVICES',
                        required=False, default=os.environ["CUDA_VISIBLE_DEVICES"])

    parser.add_argument('-w', '--weighting', help='choose weighting for dice loss.\
                        Must be one of ["linear", "volume"]',
                        required=False, default="linear")

    parser.add_argument('-f', '--fraction', help='fraction of train images cropped \
                        around forgeround',
                        required=False, type=float, default=0.0)

    parser.add_argument('-v', '--variance', help='variance of applied gaussian noise, \
                        0 == No noise', required=False, type=float, default=0.01)

    args = vars(parser.parse_args())
    main(args)
