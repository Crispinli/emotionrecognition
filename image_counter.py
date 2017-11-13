import os
import numpy as np

root = r"D:\mec_data2\test_set"
npy_path = r"D:\mec_data2" + os.sep + "test_image_counter_npy.npy"
text_path = r"D:\mec_data2" + os.sep + "test_image_counter_text.txt"

if os.path.exists(npy_path) == False:
    image_counter = {}
    with open(text_path, 'a') as file:
        floders = os.listdir(root)
        for floder in floders:
            sub_root = root + os.sep + floder
            images = os.listdir(sub_root)
            counts = len(images)
            image_counter[floder] = counts
            file.write(floder + " " + str(counts) + "\n")
            print(floder, counts)

    np.save(npy_path, image_counter)
    print("completed!")
else:
    image_counter = np.load(npy_path).tolist()
    image_counter_lower_50 = {}
    image_counter_higher_50 = {}
    image_counter_0 = {}
    for key in image_counter:
        if image_counter[key] >= 50:
            image_counter_higher_50[key] = image_counter[key]
        elif image_counter[key] > 0 and image_counter[key] < 50:
            image_counter_lower_50[key] = image_counter[key]
        else:
            image_counter_0[key] = image_counter[key]

    print(len(image_counter_lower_50), len(image_counter_higher_50), len(image_counter_0))
    np.save(r"D:\mec_data2" + os.sep + "test_image_counter_lower_50_npy.npy", image_counter_lower_50)
    np.save(r"D:\mec_data2" + os.sep + "test_image_counter_higher_50_npy.npy", image_counter_higher_50)
    np.save(r"D:\mec_data2" + os.sep + "test_image_counter_0_npy.npy", image_counter_0)
