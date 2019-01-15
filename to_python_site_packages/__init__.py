name = "hyperlpr_python_pkg"
import sys
from .hyperlpr import LPR
import os



PR = LPR(os.path.join(os.path.split(os.path.realpath(__file__))[0],"models"))



def HyperLPR_PlateRecogntion(Input_BGR,minSize=30,charSelectionDeskew=True):
    return PR.plateRecognition(Input_BGR,minSize) ###force to recognize if set charSelectionDeskew=False


# image  = cv2.imread("/Users/yujinke/车牌图片/车牌图片-夜晚白天/20170117081847-蓝-粤B0M855-0.jpg")
# print(HyperLPR_PlateRecogntion(image))
