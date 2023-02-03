
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dropout, concatenate, UpSampling3D
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import scipy
import cv2
from glob import glob
import imageio

def Unet3D(pretrained_weights = None, patients = 30, slices = 64, width = 128, height = 128, channels = 1): # len(patient_images_resampled) = 3
    input_size = (slices, width, height, channels)
    inputs = Input(input_size)
    conv1 = Conv3D(16, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv3D(16, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = Conv3D(32, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv3D(32, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = Conv3D(64, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv3D(64, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = Conv3D(128, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv3D(128, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = Conv3D(256, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv3D(256, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv3D(128, kernel_size=(2, 2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2, 2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis = 4)
    conv6 = Conv3D(128, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv3D(128, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv3D(64, kernel_size=(2, 2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2, 2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis = 4)
    conv7 = Conv3D(64, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv3D(64, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv3D(32, kernel_size=(2, 2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2, 2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis = 4)
    conv8 = Conv3D(32, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv3D(32, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv3D(16, kernel_size=(2, 2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2, 2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis = 4)
    conv9 = Conv3D(16, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv3D(16, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv3D(2, kernel_size=(3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv3D(1, kernel_size=(1, 1, 1), activation = 'sigmoid')(conv9)

    model = Model(inputs, conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ["loss", "accuracy"])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def load_scans(paths): # paths = (patient_id, paths_dicom_files) -> slices = (patient, dicom_files)
    '''
        Считывает все dicom файлы для всех пациентов \n
        paths = (patient, paths_to_dicom_files) \n
        return (patient, dicom_files)
    '''

    slices = [[0]]*len(paths)
    for i in range(len(paths)): # i - patient
        slices[i] = [pydicom.dcmread(path) for path in paths[i]]
        slices[i].sort(key = lambda x: int(x.InstanceNumber), reverse = True)

        try:
            slice_thickness = np.abs(slices[i][0].ImagePositionPatient[2] - slices[i][1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[i][0].SliceLocation - slices[i][1].SliceLocation)

        for slice in slices[i]:
            slice.SliceThickness = slice_thickness

    return slices

def get_pixels_hu(scans): # scans = (patients_id, dicom_files) -> (patient, hu_scale_images)

    """
    Преобразует все сканы из dicom_files в pixel_data согласно шкале HU \n
    scans = (patient, dicom_files) \n
    return (patient, hu_scale_images)
    """

    images_hu = []
    for scan in scans:
        image = np.stack([s.pixel_array for s in scan])
        image = image.astype(np.int16)
        image[image == -2000] = 0
        intercept = scan[0].RescaleIntercept
        slope = scan[0].RescaleSlope

        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)

        image += np.int16(intercept)
        images_hu.append(image)

    return images_hu
    # return np.array(images_hu, dtype=np.int16)

def hu_hist(patient): # patient = (hu_scale_images)

    """
    Диаграмма распределения значений HU \n
    patient = (hu_scale_images)
    """

    plt.hist(patient.flatten(), bins=100, color='c')
    plt.xlabel("Honsfeild Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

def resample(images, file_info, new_spacing=[1,1,1]): # изменение размеров, чтобы воксель был размера 1

    """
    Интерполяция с использованием пути к файлу \n
    images = (depth, width, height)
    file_info - any_dicom_file
    """

    spacing = np.array([float(file_info.SliceThickness),
                        float(file_info.PixelSpacing[0]),
                        float(file_info.PixelSpacing[1])])

    resize_factor = spacing / new_spacing
    new_real_shape = images.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / images.shape
    new_spacing = spacing / real_resize_factor

    # images = scipy.ndimage.zoom(images, real_resize_factor)
    images = scipy.ndimage.zoom(images, real_resize_factor, order=1)

    return images, new_spacing

def resample_path(images, file_path, new_spacing=[1,1,1]):

    """
    Интерполяция с использованием пути к файлу \n
    images = (depth, width, height)
    file_path - path_to_any_dicom_file
    """

    file_info = pydicom.read_file(file_path)
    spacing = np.array([float(file_info.SliceThickness),
                        float(file_info.PixelSpacing[0]),
                        float(file_info.PixelSpacing[1])])

    resize_factor = spacing / new_spacing
    new_real_shape = images.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / images.shape
    new_spacing = spacing / real_resize_factor

    # images = scipy.ndimage.zoom(images, real_resize_factor)
    images = scipy.ndimage.zoom(images, real_resize_factor, order=1)

    return images, new_spacing

def resize(images, target_size): # простое изменение размеров изображений
    """
    Изменение размеров изображений \n
    images = (depth, width, height) \n
    target_size - any_digit \n
    return new_images = (depth, width_scaled, height_scaled)
    """
    new_images = []
    for image in images:
        new_image = cv2.resize(image,(target_size,target_size))
        new_images.append(new_image)
    return new_images

def resize_data_volume(data, target_scale):
    """
    Resize the data to the dim size \n
    data = (depth, width, height) \n
    target_scale = [depth_scale, width_scale, height_scale]
    """
    depth, height, width = data.shape
    print("resize factor is", target_scale / data.shape)
    scale = [target_scale[0] * 1.0 / depth, target_scale[1] * 1.0 / height, target_scale[2] * 1.0 / width]
    return scipy.ndimage.interpolation.zoom(data, scale, order=1)

def resize_data_volume_only_depth(data, target_scale):
    """
    Resize the data to the dim size \n
    data = (depth, width, height) \n
    target_scale - any_digit
    """
    depth, _, _ = data.shape
    print("resize factor is", target_scale * 1.0 / depth)
    scale = [target_scale * 1.0 / depth, 1, 1]
    return scipy.ndimage.interpolation.zoom(data, scale, order=1)

def sample_stack(stack, rows=6, cols=6, start_width=10, show_every=2):

    """
    Показывает несколько картинок сразу \n
    stack = (depth, width, height)
    """

    fig, ax = plt.subplots(rows, cols, figsize=[18, 20])
    for i in range (rows*cols):
        ind = start_width + i*show_every
        ax[int(i/rows), int(i % rows)].set_title(f'slice {ind}')
        ax[int(i/rows), int(i % rows)].imshow(stack[ind], cmap='gray')
        ax[int(i/rows), int(i % rows)].axis('off')
    plt.show()

def load_images (paths): # paths = (patient, paths_png_files) -> slices = (patient, depth, width, height)

    """
    Считывание label images \n
    paths = (patient, paths_to_png_files) \n
    return slices = (patient, depth, width, height)
    """

    slices = [[0]] * len(paths)
    for i in range(len(paths)): # i - patient's id
        slices[i] = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in paths[i]]
        # plt.imshow(slices[i][0], cmap=plt.cm.bone)
        # plt.show()
    return slices

def create_images_gifs(patients_id, patients_images, path_to_save="output.gif"):

    """
    Создание gif \n
    patients_id = [any array] \n
    patients_images = (patient, depth, width, height) \n
    """

    final_gif = imageio.get_writer(path_to_save)
    temp_gifs = []
    for i in range(len(patients_id)):
        patients_images[i] = patients_images[i].astype(np.uint8)
        imageio.mimsave(f"./{patients_id[i]}.gif", patients_images[i], duration=0.1)
        temp_gifs.append(imageio.get_reader(f"./{patients_id[i]}.gif"))
    number_of_frames = min([value.get_length() for value in temp_gifs])
    for _ in range(number_of_frames):
        next_img = [temp_gifs[i].get_next_data() for i in range(len(patients_id))]
        new_image = np.hstack(next_img) # vstack делает по вертикали
        final_gif.append_data(new_image)
    final_gif.close()

def show_shape(array, index=0):

    """
    Показывает shape. Причина создания: numpy.size возвращает не то значение, которое ожидается \n
    array = (patient, depth, width, height)
    """

    return (len(array), len(array[index]), len(array[index][index]), len(array[index][index][index]))


# tensorflow.test.is_gpu_available()
# model = Unet3D()
# model.summary()
