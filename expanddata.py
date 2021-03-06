#!/usr/bin/env python
# coding=utf-8
#扩展数据集至原来20倍
import os
import cv2
import numpy as np
from math import *
import random

index = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11, "皖": 12,
         "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24,
         "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36,
         "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48,
         "J": 49, "K": 50, "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
         "W": 61, "X": 62, "Y": 63, "Z": 64};


def GetFileList(dir, fileList):
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # 如果需要忽略某些文件夹，使用以下代码
            # if s == "xxx":
            # continue
            newDir = os.path.join(dir, s)
            GetFileList(newDir, fileList)
    return fileList


# create random value between 0 and val-1
def r(val):
    return int(np.random.random() * val)


def AddGauss(img, level):
    return cv2.blur(img, (level * 2 + 1, level * 2 + 1));


def rot(img, angel, shape, max_angel):
    size_o = [shape[1], shape[0]]

    size = (shape[1] + int(shape[0] * cos((float(max_angel) / 180) * 3.14)), shape[0])

    interval = abs(int(sin((float(angel) / 180) * 3.14) * shape[0]));

    pts1 = np.float32([[0, 0], [0, size_o[1]], [size_o[0], 0], [size_o[0], size_o[1]]])
    if (angel > 0):

        pts2 = np.float32([[interval, 0], [0, size[1]], [size[0], 0], [size[0] - interval, size_o[1]]])
    else:
        pts2 = np.float32([[0, 0], [interval, size[1]], [size[0] - interval, 0], [size[0], size_o[1]]])

    M = cv2.getPerspectiveTransform(pts1, pts2);
    dst = cv2.warpPerspective(img, M, size);

    return dst;


def rotRandrom(img, factor, size):
    shape = size;
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)], [r(factor), shape[0] - r(factor)], [shape[1] - r(factor), r(factor)],
                       [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2);
    dst = cv2.warpPerspective(img, M, size);
    return dst;


def cropFill(img, bot):
    leftIdx = 0
    rightIdx = 0
    for col in range(img.shape[1]):
        if sum(sum(img[0:, col])) != 0:
            leftIdx = col
            break
    for col in range(img.shape[1]):
        if sum(sum(img[0:, img.shape[1] - col - 1])) != 0:
            rightIdx = img.shape[1] - col
            break
    imgRoi = img[0:, leftIdx: rightIdx]
    tmp=0
    if tmp==0:
        envPath = './envs/' + str(tmp) + '.bmp'  # env文件夹下保存了28张背景图片，从0.png到27.png
    else:
        envPath = './envs/' + str(tmp) + '.jpg'
    env = cv2.imread(envPath)
    env = cv2.resize(env, (imgRoi.shape[1], imgRoi.shape[0]))

    img2gray = cv2.cvtColor(imgRoi, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    bak = (imgRoi == 0);
    bak = bak.astype(np.uint8) * 255;
    inv = cv2.bitwise_and(bak, env)
    img_temp = cv2.bitwise_or(inv, imgRoi, mask=mask_inv)
    imgRoi = cv2.bitwise_or(imgRoi, img_temp)

    return imgRoi


def tfactor(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV);
    hsv[:, :, 1] = hsv[:, :, 1] * (0.7 + np.random.random() * 0.3);
    hsv[:, :, 1] = hsv[:, :, 1] * (0.4 + np.random.random() * 0.6);
    hsv[:, :, 2] = hsv[:, :, 2] * (0.4 + np.random.random() * 0.6);

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR);
    return img


if __name__ == '__main__':
    lists = GetFileList('./single_plate/', [])
    path = "./single_plate/"
    path_out = "./single_plate_expand/"
    for root,dirs,files in os.walk(path):
        for file in files:# 遍历所有图片
            img = cv2.imdecode(np.fromfile(path+file, dtype=np.uint8),-1)
            img = cv2.resize(img, (272, 72))
            #cv2.imshow('src', img)
            #cv2.waitKey(0)

            for times in range(20):  # 20次变换可得到20张增强的图片
                src = img
                dst = AddGauss(src, r(3))
                dst = rot(dst, r(60) - 30, dst.shape, 20);
                dst = rotRandrom(dst, 5, (dst.shape[1], dst.shape[0]));
                dst = cropFill(dst, 3)
                dst = tfactor(dst)
                #cv2.imshow('dst', dst)
                cv2.imencode('.jpg', dst)[1].tofile(path_out+file[0:7]+"_"+str(times)+'.jpg')
                #cv2.waitKey(0)