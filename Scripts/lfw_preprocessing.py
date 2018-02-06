import numpy as np
import pandas as pd
import cv2
import glob

img_list = glob.glob("../Data/lfw/**/*.jpg")
img_data = []

for i in range(len(img_list)):
    img_file = img_list[i]
    img = cv2.imread(img_file)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (64,64), interpolation=cv2.INTER_AREA)
    gray_img_list = [img_file]+np.ravel(gray_img).tolist()
    img_data.append(gray_img_list)
    if (i%100)==0:
        print("Converted {}/{}".format(i+1,len(img_list)))

img_df = pd.DataFrame(img_data)
img_df.to_csv("../Data/lfw_grayscale_64.csv",header=False,index=False)