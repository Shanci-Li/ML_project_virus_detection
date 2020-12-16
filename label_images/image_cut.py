from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import sys

def segment_and_transform_image(image_path, output_path ,size):
    files = os.listdir(image_path)
    files = np.sort(files)
    print("started segment the image")
    for f in files:
        imgpath = image_path + "/" +  f
        img = Image.open(imgpath)
        img_size = img.size
        m = img_size[0]   
        n = img_size[1]     
        w,h = size
        nn = 0  
        # segment a image into 81 sub-images               
        for i in range(9):
            x = 192*i      
            for j in range(9):
                y = 192* j
                # the segmented region
                region = img.crop((x, y, x+w, y+h))     
                nn = nn +1
                # transform the depth of image from 8 to 24
                img_to_depth_24 = region.convert('RGB')
                img_array = np.array(img_to_depth_24) 
                # save the image
                dirpath = output_path
                file_name, file_extend = os.path.splitext(f)
                dst = os.path.join(os.path.abspath(dirpath), file_name + '_' +str(nn) + '.png')
                cv2.imwrite(dst, img_array)
    print("finished")

# the path of original image and modified image
image_path = sys.argv[1]  
output_path = sys.argv[2]

segment_and_transform_image(image_path, output_path ,(512,512))

