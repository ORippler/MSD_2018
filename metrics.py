import tensorflow as tf

K = tf.keras.backend


def generalized_dice(y_true, y_pred, axis=(-3, -2, -1),
                     smooth=0.00001, weighting=None, log_y_pred=False):
    """
    Weighted dice coefficient. Default axis assumes a "channels first"
    data structure

    :param smooth: Epsilon to ensure numerical stability and loss for y_true
                   = 0
    :param y_true: Groundtruth Segmentation Map
    :param y_pred: Predicted Segmentation Map
    :param axis: Axis to comnpute predictions over
    :return: generalized dice score + logged absolute probabilities
    """

    if weighting == "volume":
        w = 1. / K.sum(y_true + smooth, axis=axis)**2

    elif weighting == 'linear':
        w = 1.

    if log_y_pred:
        return (K.mean(2. * (w * K.sum(y_true * y_pred,
                             axis=axis) + smooth/2)/(K.sum(y_true,
                                                           axis=axis) + w * K.sum(y_pred,
                                                                                  axis=axis) + smooth)),
                K.mean(K.sum(y_pred, axis=axis) + smooth))
    else:
        return K.mean(2. * (w * K.sum(y_true * y_pred,
                            axis=axis) + smooth/2)/(K.sum(y_true,
                                                          axis=axis) + w * K.sum(y_pred,
                                                                                 axis=axis) + smooth))


def generalized_dice_loss(y_true, y_pred, weighting='linear',
                          log_y_pred_=False):
    if log_y_pred_:
        dice = generalized_dice(y_true, y_pred, weighting=weighting,
                                log_y_pred=log_y_pred_)
        return 1 - dice[0], dice[1]
    else:
        return 1 - generalized_dice(y_true, y_pred, weighting=weighting)
