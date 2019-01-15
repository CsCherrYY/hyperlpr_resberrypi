import os
import shutil
from PIL import Image
import numpy as np
import cv2
from hyperlpr import *
from hyperlpr import pipline as pp
path="data_plate/"

for root,dirs,files in os.walk(path):
    for file in files:
        if file[1]=='0':
            os.rename(path+file,path+file[0]+'O'+file[2:len(file)])
