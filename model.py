import tensorflow as tf
import numpy as np
import os
from functools import partial

Adam = tf.keras.optimizers.Adam
K = tf.keras.backend
Add = tf.keras.layers.Add
UpSampling3D = tf.keras.layers.UpSampling3D
Activation = tf.keras.layers.Activation
SpatialDropout3D = tf.keras.layers.SpatialDropout3D
Input = tf.keras.layers.Input
MaxPooling3D = tf.keras.layers.MaxPooling3D
LeakyReLU = tf.keras.layers.LeakyReLU
Softmax = tf.keras.activations.softmax
Lambda = tf.keras.layers.Lambda
Conv3D = tf.keras.layers.Conv3D
Concatenate = tf.keras.layers.concatenate
BatchNormalization = tf.keras.layers.BatchNormalization
GAP3D = tf.keras.layers.GlobalAveragePooling3D

K.set_image_data_format("channels_first")


def create_convolution_block(input_layer, n_filters, batch_normalization=False,
                             kernel=(3, 3, 3), activation=None, padding='same',
                             strides=(1, 1, 1), instance_normalization=False,
                             fov_in=None, j_in=None):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel: Kernel size
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding: 'same' or 'valid' padding of conv layers
    :return:
    """
    layer = Conv3D(n_filters, kernel,
                   padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        tf_instance_norm = tf.contrib.layers.instance_norm
        layer = Lambda(lambda x: tf_instance_norm(
                                    inputs=x, data_format='NCHW'))(layer)
    if np.any(fov_in):
        fov_out = fov_in + np.array(j_in) * (np.array(kernel) - 1)
        j_out = j_in * np.array(strides)
        if activation is None:
            return Activation('relu')(layer), fov_out, j_out
        else:
            return activation()(layer), fov_out, j_out
    else:
        if activation is None:
            return Activation('relu')(layer)
        else:
            return activation()(layer)

create_convolution_block = partial(create_convolution_block,
                                   activation=LeakyReLU,
                                   instance_normalization=True)


def create_localization_module(input_layer, n_filters,
                               kernel_=(3, 3, 3)):

    convolution1 = create_convolution_block(input_layer, n_filters,
                                            kernel=kernel_)
    convolution2 = create_convolution_block(convolution1, n_filters,
                                            kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, kernel_=(3, 3, 3),
                              size=(2, 2, 2)):

    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters,
                                           kernel=kernel_)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3,
                          data_format="channels_first", kernel_=(3, 3, 3),
                          fov=None, j=None):

    convolution1, fov, j = create_convolution_block(input_layer=input_layer,
                                                    n_filters=n_level_filters,
                                                    kernel=kernel_,
                                                    fov_in=fov,
                                                    j_in=j)
    dropout = SpatialDropout3D(rate=dropout_rate,
                               data_format=data_format)(convolution1)
    convolution2, fov, j = create_convolution_block(input_layer=dropout,
                                                    n_filters=n_level_filters,
                                                    kernel=kernel_,
                                                    fov_in=fov,
                                                    j_in=j)
    return convolution2, fov, j


def create_segmenter(input_l, n_labels, depth, nbf, nsl, fs=3, s=2, dr=0.3):

    current_layer = None
    level_output_layers = list()
    level_filters = list()

    for level_number in range(max(depth) + 1):
            n_level_filters = (2**level_number) * nbf
            level_filters.append(n_level_filters)

            if current_layer is None:
                fov = [1] * len(depth)
                j = [1] * len(depth)
                kernel = [fs] * len(depth)
                in_conv, fov, j = create_convolution_block(
                                        input_l, n_level_filters,
                                        fov_in=fov, j_in=j)
            else:
                _strides = [s if d_i >= level_number else 1 for d_i in depth]
                kernel = [fs if d_i >= level_number else 1 for d_i in depth]
                in_conv, fov, j = create_convolution_block(current_layer,
                                                           n_level_filters,
                                                           strides=_strides,
                                                           kernel=kernel,
                                                           fov_in=fov,
                                                           j_in=j)

            context_output_layer, fov, j = create_context_module(
                                                in_conv,
                                                n_level_filters,
                                                dropout_rate=dr,
                                                kernel_=kernel,
                                                fov=fov,
                                                j=j)

            print("level_number {}".format(level_number))
            print('fov: {}'.format(fov))
            print('j: {}'.format(j))
            print('kernel: {}'.format(kernel))

            summation_layer = Add()([in_conv, context_output_layer])
            level_output_layers.append(summation_layer)
            current_layer = summation_layer

    segmentation_layers = list()
    for level_number in reversed(range(max(depth))):
        size_ = [s if d_i > level_number else 1 for d_i in depth]
        up_kernel = [fs if d_i > level_number else 1 for d_i in depth]
        up_sampling = create_up_sampling_module(current_layer,
                                                level_filters[level_number],
                                                size=size_, kernel_=up_kernel)

        concatenation_layer = Concatenate(
                                [level_output_layers[level_number], up_sampling],
                                axis=1)

        conv_kernel = [fs if d_i >= level_number else 1 for d_i in depth]
        localization_output = create_localization_module(
                                            concatenation_layer,
                                            level_filters[level_number],
                                            kernel_=conv_kernel)
        current_layer = localization_output

        if level_number < nsl:
            segmentation_layers.insert(0, Conv3D(filters=n_labels,
                                                 kernel_size=(1, 1, 1),
                                                 padding='same')(current_layer))

    output_layer = None
    for level_number in reversed(range(nsl)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            size_ = [s if d_i >= level_number else 1 for d_i in depth]
            output_layer = UpSampling3D(size=size_)(output_layer)

    predictions = tf.nn.softmax(output_layer, axis=1, name='Softmax')

    return predictions


def model(features, labels, mode, params):

    depth = params['depth']
    n_base_filters = params['n_base_filters']
    dropout_rate = params['dropout_rate']
    n_segmentation_levels = params['n_segmentaton_levels']
    n_labels = params['n_labels']
    optimizer = params['optimizer']
    initial_learning_rate = params['initial_learning_rate']
    loss_function = params['loss_function']
    weighting = params['weighting']

    image = features['img']

    with tf.variable_scope('Generator'):

        predictions = create_segmenter(input_l=image,
                                       n_labels=n_labels,
                                       depth=depth,
                                       nbf=n_base_filters,
                                       nsl=n_segmentation_levels,
                                       dr=dropout_rate)

    # Dice loss
    if (mode == tf.estimator.ModeKeys.EVAL 
            or mode == tf.estimator.ModeKeys.TRAIN):

            labels = tf.cast(labels, tf.float32)

            for i in range(n_labels):
                loss_label, sum_preds_label = loss_function(
                                                    labels[:, i, :, :, :],
                                                    predictions[:, i, :, :, :],
                                                    weighting=weighting,
                                                    log_y_pred_=True)
                tf.summary.scalar('dice_loss_label ' + str(i), loss_label)
                tf.summary.scalar('sum_preds_label ' + str(i), sum_preds_label)

            dice_loss_G = loss_function(labels,
                                        predictions,
                                        weighting=weighting)
            tf.summary.scalar('dice_loss_all', dice_loss_G)

            _ilr = initial_learning_rate

            lr = tf.train.exponential_decay(
                        _ilr, global_step=tf.train.get_global_step(),
                        decay_rate=0.985, decay_steps=100, staircase=True)

            tf.summary.scalar('learning_rate', lr)

            optimizer_G = tf.train.AdamOptimizer(learning_rate=lr).minimize(
                            loss=dice_loss_G,
                            var_list=tf.trainable_variables(scope='Generator'),
                            global_step=tf.train.get_global_step())
            train_op = tf.group([optimizer_G])

            # tensorboard logging
            with tf.name_scope('extracted_images'):

                max_energy = features['object_in_crop']

                max_energy.set_shape([None, len(image.shape)-1])

                # TODO: Refactor such that offsets are calculated dynamically
                image_slice = image[:, 0, max_energy[0, 1], :, :]
                image_slice_exp = tf.expand_dims(image_slice, -1)

                mask_gt = tf.cast(
                    tf.argmax(labels, axis=1)[:, max_energy[0, 1], :, :] * 255 / n_labels,
                    tf.uint8)
                mask_gt_exp = tf.expand_dims(mask_gt, -1)

                mask_pred = tf.cast(
                    tf.argmax(predictions, axis=1)[:, max_energy[0, 1], :, :] * 255 / n_labels,
                    tf.uint8)
                mask_pred_exp = tf.expand_dims(mask_pred, -1)

                tf.summary.image('volume slice', image_slice_exp)
                tf.summary.image('mask gt', mask_gt_exp)
                tf.summary.image('mask pred', mask_pred_exp)

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {'total_loss': dice_loss_G}
        return tf.estimator.EstimatorSpec(mode, loss=dice_loss_G,
                                          eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.PREDICT:
        export_output_ = {'output': tf.estimator.export.PredictOutput(
                                                            predictions)}
        predictions = {'output': predictions}
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions,
                                          export_outputs=export_output_)

    assert mode == tf.estimator.ModeKeys.TRAIN

    return tf.estimator.EstimatorSpec(mode,
                                      loss=dice_loss_G, train_op=train_op)

if __name__ == '__main__':
    from metrics import generalized_dice_loss
    model(
      {'img': tf.placeholder(tf.float32, shape=[None]+[1, 64, 256, 256]),
       'object_in_crop': tf.placeholder(tf.int32, shape=[None, 4])},
      tf.placeholder(tf.float32, shape=[None]+[3, 64, 256, 256]),
      mode=tf.estimator.ModeKeys.TRAIN,
      params={'n_base_filters': 6,
              'depth': [3, 4, 4],
              'dropout_rate': 0.3,
              'n_segmentaton_levels': 4,
              'n_labels': 3,
              'optimizer': 'Adam',
              'initial_learning_rate': 5e-4,
              'loss_function': generalized_dice_loss,
              'weighting': 'linear'})
