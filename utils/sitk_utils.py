import SimpleITK as sitk
import numpy as np


def calculate_origin_offset(new_spacing, old_spacing):
    return np.subtract(new_spacing, old_spacing)/2


def sitk_resample_to_spacing(image, new_spacing=(1.0, 1.0, 1.0),
                             interpolator=sitk.sitkLinear, default_value=0.):
    zoom_factor = np.divide(image.GetSpacing(), new_spacing)
    new_size = np.asarray(
                    np.ceil(
                        np.round(
                            np.multiply(zoom_factor, image.GetSize()),
                            decimals=5)),
                    dtype=np.int16)
    offset = calculate_origin_offset(new_spacing, image.GetSpacing())
    reference_image = sitk_new_blank_image(size=new_size,
                                           spacing=new_spacing,
                                           direction=image.GetDirection(),
                                           origin=image.GetOrigin() + offset,
                                           default_value=default_value)
    return sitk_resample_to_image(
                image, reference_image, interpolator=interpolator,
                default_value=default_value)


def extract_multimodality(image):
    modalities = {}
    img_array = sitk.GetArrayFromImage(image)
    for i in range(img_array.shape[0]):
        itk_slice = sitk.GetImageFromArray(img_array[i, :, :, :])
        itk_slice.SetSpacing(image.GetSpacing()[:-1])
        direction = image.GetDirection()
        itk_slice.SetDirection(direction[0:3]+direction[4:7]+direction[13:])
        itk_slice.SetOrigin(image.GetOrigin()[:-1])
        modalities[str(i)] = itk_slice
    return modalities


def sitk_resample_to_image(image, reference_image, default_value=0.,
                           interpolator=sitk.sitkLinear, transform=None,
                           output_pixel_type=None):
    if len(image.GetSize()) > 3:
        resampled_image = sitk.GetArrayFromImage(reference_image)
        modalities = extract_multimodality(image)
        references = extract_multimodality(reference_image)
        for k in modalities:
            image = modalities[k]
            reference_image = references[k]
            if transform is None:
                transform = sitk.Transform()
                transform.SetIdentity()
            if output_pixel_type is None:
                output_pixel_type = image.GetPixelID()
            resample_filter = sitk.ResampleImageFilter()
            resample_filter.SetInterpolator(interpolator)
            resample_filter.SetTransform(transform)
            resample_filter.SetOutputPixelType(output_pixel_type)
            resample_filter.SetDefaultPixelValue(default_value)
            resample_filter.SetReferenceImage(reference_image)
            resampled_image[int(k)] = sitk.GetArrayFromImage(
                                        resample_filter.Execute(image))
    else:
        if transform is None:
            transform = sitk.Transform()
            transform.SetIdentity()
        if output_pixel_type is None:
            output_pixel_type = image.GetPixelID()
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetInterpolator(interpolator)
        resample_filter.SetTransform(transform)
        resample_filter.SetOutputPixelType(output_pixel_type)
        resample_filter.SetDefaultPixelValue(default_value)
        resample_filter.SetReferenceImage(reference_image)
        resampled_image = sitk.GetArrayFromImage(
                            resample_filter.Execute(image))
    return resampled_image


def sitk_new_blank_image(size, spacing, direction, origin, default_value=0.):
    image = sitk.Image(size.tolist(), sitk.sitkFloat32)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    image.SetOrigin(origin)
    return image


def sitk_histogram_matching_image(source, target):
    hist_filter = sitk.HistogramMatchingImageFilter()
    hist_filter.SetNumberOfHistogramLevels(256)
    hist_filter.SetNumberOfMatchPoints(1)
    hist_filter.ThresholdAtMeanIntensityOn()
    return hist_filter.Execute(source, target)


def resample_to_spacing(data, target_spacing, interpolation='',
                        default_value=0.):
    image = data
    if interpolation is "linear":
        interpolator = sitk.sitkLinear
    elif interpolation is "nearest":
        interpolator = sitk.sitkNearestNeighbor
    elif interpolation is 'Spline':
        interpolator = sitk.sitkBSpline
    else:
        raise ValueError("'interpolation' must be either 'linear' or 'nearest'.\
                         '{}' is not recognized".format(interpolation))
    resampled_image = sitk_resample_to_spacing(
                            image, new_spacing=target_spacing,
                            interpolator=interpolator,
                            default_value=default_value)
    return resampled_image


def data_to_sitk_image(data, spacing=(1., 1., 1.)):
    if len(data.shape) == 3:
        data = np.rot90(data, 1, axes=(0, 2))
    image = sitk.GetImageFromArray(data)
    image.SetSpacing(np.asarray(spacing, dtype=np.float))
    return image


def sitk_image_to_data(image):
    data = sitk.GetArrayFromImage(image)
    if len(data.shape) == 3:
        data = np.rot90(data, -1, axes=(0, 2))
    return data
