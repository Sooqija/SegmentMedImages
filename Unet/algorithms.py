import random
import time
import os
from pydicom import dcmread
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt

def GenerateUniqueRandomIntegers(count, limit):
    if count > limit:
        print("Count > Limit")
        return []
    random.seed(time.time())
    result = []
    digit = random.randint(0, limit)
    i = 0
    while count - i != 0:
        if digit not in result:
            result.append(digit)
            i += 1
            digit = random.randint(0, limit)
        else:
            digit = random.randint(0, limit)
    return result

def GenerateTrainData(save_image_path, save_label_path, save_test_path, dir_used):
    info_file = open("./info.txt", "w")
    info_file.write("Paths data has been used\n")
    train_counter = 0
    test_counter = 0
    for i in dir_used:
        rand_integers = GenerateUniqueRandomIntegers(20, 85) # works only for len(dir_used == 3)
        info_file.write("\nFor Train\n")
        for j in rand_integers[:10]:
            if j < 10:
                print("./CT_data_batch1/{}/Ground/liver_GT_00{}".format(i, j))
                ds = dcmread("./CT_data_batch1/{}/DICOM_anon/i000{},0000b.dcm".format(i, j))
                label = io.imread("./CT_data_batch1/{}/Ground/liver_GT_00{}.png".format(i, j))
                info_file.write("./CT_data_batch1/{}/DICOM_anon/i000{},0000b.dcm\n".format(i, j))
            else:
                print("./CT_data_batch1/{}/Ground/liver_GT_000{}".format(i, j))
                ds = dcmread("./CT_data_batch1/{}/DICOM_anon/i00{},0000b.dcm".format(i, j))
                label = io.imread("./CT_data_batch1/{}/Ground/liver_GT_0{}.png".format(i, j))
                info_file.write("./CT_data_batch1/{}/DICOM_anon/i00{},0000b.dcm\n".format(i, j))
            temp = ds.pixel_array
            temp = temp / 255
            io.imsave(os.path.join(save_image_path, "{}.png".format(train_counter)), temp)
            io.imsave(os.path.join(save_label_path, "{}.png".format(train_counter)), label)
            train_counter += 1
        info_file.write("\nFor Test\n")
        for j in rand_integers[10:]:
            if j < 10:
                print("./CT_data_batch1/{}/Ground/liver_GT_00{}".format(i, j))
                ds = dcmread("./CT_data_batch1/{}/DICOM_anon/i000{},0000b.dcm".format(i, j))
                info_file.write("./CT_data_batch1/{}/DICOM_anon/i000{},0000b.dcm\n".format(i, j))
            else:
                print("./CT_data_batch1/{}/Ground/liver_GT_000{}".format(i, j))
                ds = dcmread("./CT_data_batch1/{}/DICOM_anon/i00{},0000b.dcm".format(i, j))
                info_file.write("./CT_data_batch1/{}/DICOM_anon/i00{},0000b.dcm\n".format(i, j))
            temp = ds.pixel_array
            temp = temp / 255
            io.imsave(os.path.join(save_test_path, "{}.png".format(test_counter)), temp)
            test_counter += 1
    info_file.close()

def CreateChartStatisticOfLearning(n_epoch = 0, loss = [], acc = []):
    epochs = np.linspace(1, n_epoch, n_epoch)
    print(epochs)
    if loss and acc and n_epoch:
        plt.figure(figsize=(10.,5.))
        plt.title("Training Loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.plot(epochs, loss, "orange")
        plt.grid()
        plt.show()

        plt.figure(figsize=(8.,5.))
        plt.title("Training Accuracy")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.plot(epochs, acc, "blue")
        plt.grid()
        plt.show()

