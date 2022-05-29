import random
import time
from pydicom import dcmread
import skimage.io as io

def GenerateUniqueRandomIntegers(count, limit):
    if count < limit:
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
    for i in data_used:
        rand_integers = GenerateUniqueRandomIntegers(20, 85) # works only for len(dir_used == 3)
        k = 0
        info_file.write("\nFor Train\n")
        for j in rand_integers[:10]:
            if j < 10:
                print("./CT_data_batch1/{}/Ground/liver_GT_00{}".format(i, j))
                ds = dcmread("./CT_data_batch1/{}/DICOM_anon/i000{},0000b.dcm".format(i, j))
                label = io.imread("./CT_data_batch1/{}/Ground/liver_GT_00{}.png".format(i, j))
                f.write("./CT_data_batch1/{}/DICOM_anon/i000{},0000b.dcm\n".format(i, j))
            else:
                print("./CT_data_batch1/{}/Ground/liver_GT_000{}".format(i, j))
                ds = dcmread("./CT_data_batch1/{}/DICOM_anon/i00{},0000b.dcm".format(i, j))
                label = io.imread("./CT_data_batch1/{}/Ground/liver_GT_0{}.png".format(i, j))
                f.write("./CT_data_batch1/{}/DICOM_anon/i00{},0000b.dcm\n".format(i, j))
            temp = ds.pixel_array
            temp = temp / 255
            io.imsave(os.path.join(save_image_path, "{}.png".fomat(k)), temp)
            io.imsave(os.path.join(save_label_path, "{}.png".fomat(k)), label)
            k += 1
        k = 0
        for j in rand_integers[10:]:
            if j < 10:
                print("./CT_data_batch1/{}/Ground/liver_GT_00{}".format(i, j))
                ds = dcmread("./CT_data_batch1/{}/DICOM_anon/i000{},0000b.dcm".format(i, j))
                label = io.imread("./CT_data_batch1/{}/Ground/liver_GT_00{}.png".format(i, j))
                f.write("./CT_data_batch1/{}/DICOM_anon/i000{},0000b.dcm\n".format(i, j))
            else:
                print("./CT_data_batch1/{}/Ground/liver_GT_000{}".format(i, j))
                ds = dcmread("./CT_data_batch1/{}/DICOM_anon/i00{},0000b.dcm".format(i, j))
                label = io.imread("./CT_data_batch1/{}/Ground/liver_GT_0{}.png".format(i, j))
                f.write("./CT_data_batch1/{}/DICOM_anon/i00{},0000b.dcm\n".format(i, j))

            k += 1
    f.close()