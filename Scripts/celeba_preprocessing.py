import numpy as np
import pandas as pd
import cv2
import glob

img_list = glob.glob("../Data/img_align_celeba/*.jpg")
img_data = []

for i in range(len(img_list)):
    img_file = img_list[i]
    img = cv2.imread(img_file)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_img = gray_img[20:198,:]
    gray_img = cv2.resize(gray_img, (32,32), interpolation=cv2.INTER_AREA)
    gray_img_list = np.ravel(gray_img).tolist()
    img_data.append(gray_img_list)
    if (i%100)==0:
        print("Converted {}/{}".format(i+1,len(img_list)))
    if ((i+1)%16277)==0:
        img_df = pd.DataFrame(img_data)
        img_df.to_csv("../Data/celeba_grayscale_32/train_{:02d}.csv".format((i+1)//16277),header=False,index=False)
        img_data = []
        if((i+1)//16277)==10:
            break