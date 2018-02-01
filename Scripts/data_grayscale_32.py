import numpy as np
import pandas as pd
import cv2
import glob

img_list = glob.glob("../Data/lfw/**/*.jpg")
img_data = []

for img_file in img_list:
    img = cv2.imread(img_file)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    for i in range(3):
        gray_img = cv2.pyrDown(gray_img)
    gray_img_list = [img_file]+np.ravel(gray_img).tolist()
    img_data.append(gray_img_list)

img_df = pd.DataFrame(img_data)
img_df.to_csv("../Data/lfw_grayscale_32.csv",header=False,index=False)