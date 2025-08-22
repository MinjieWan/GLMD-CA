import cv2
import os
from glob import glob
from tqdm import tqdm


def main():
    for dataType in [("train2017", "train"), ("val2017", "val")]:
        color_data_folder = "./{}/".format(dataType[0])
        gray_save_folder = "./{}/".format(dataType[1])
        if not os.path.exists(gray_save_folder):
            os.mkdir(gray_save_folder)
        color_img_list = glob(color_data_folder + "*.jpg")

        for img_path in tqdm(color_img_list, total=len(color_img_list)):
            img_name = img_path.split("/")[-1]
            color_img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            img_save_path = os.path.join(gray_save_folder, img_name)
            cv2.imwrite(img_save_path, gray_img)


if __name__ == "__main__":
    main()
