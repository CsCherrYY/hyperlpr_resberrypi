from hyperlpr import *
import cv2
import numpy as np
import time
import shutil
#image = cv2.imread("demo.jpg")
#print(HyperLPR_PlateRecogntion(image))
def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i) # 取文件绝对路径
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)


test=1   #0:full 1:plate
if test==1:
    path = "./single_plate/"
else:
    path = "./data_plate/"
out_path="./plate_test/"
i=0
non_recognize=0
wa=0
acc=0
total_time=0
#del_file(out_path + "non_recognize/")
#del_file(out_path + "acc/")
#del_file(out_path + "wa/")
fp=open(out_path + "wa/" + "wa.txt" ,"w")
acc_h,acc_w,wa_h,wa_w,img_h,img_w=0,0,0,0,0,0
for root,dirs,files in os.walk(path):
    for file in files:
        image = cv2.imdecode(np.fromfile(path + file, dtype=np.uint8),cv2.IMREAD_COLOR)
        i+=1
        start_time=time.time()

        if test==1 or test==0:
            image = cv2.copyMakeBorder(image,int(image.shape[0]*0.1),int(image.shape[0]*0.1),int(image.shape[1]*0.225),
                                   int(image.shape[1]*0.225), cv2.BORDER_CONSTANT, value=[255, 255, 255])
        '''
        arg1: src - 输入图像
        arg2: top - 顶部边缘的宽度
        arg3: bottom - 底部边缘的宽度
        arg4: left - 左边边缘的宽度
        arg5: right - 右边边缘的宽度
        arg6: borderType - 填充边缘的type，如：cv2.BORDER_CONSTANT，cv2.BORDER_REFLECT，cv2.BORDER_REFLECT_101
        arg7: value - 填充的颜色
        '''
        sp=image.shape
        size_h=sp[0]
        size_w=sp[1]
        #img_h += size_h
        #img_w += size_w
        res = HyperLPR_PlateRecogntion(image)
        #cv2.imshow('src1', image)
        end_time=time.time()
        total_time=total_time+end_time-start_time
        if len(res)==0:
            non_recognize+=1
            shutil.copy(path + file, out_path + "non_recognize/" + file)
            print("non_recognize:"+file)
        elif res[0][0][0:7]==file[0:7] or (res[0][0][0]==file[0] and res[0][0][2:7]==file[2:7] and file[1]=='O'):
            acc+=1
            #acc_h += size_h
            #acc_w += size_w
            #shutil.copy(path + file, out_path + "acc/" + file)
            print("acc:"+file)
        else:
            wa+=1
            #wa_h += size_h
            #wa_w += size_w
            print("wa:"+file)
            shutil.copy(path + file, out_path + "wa/" + file[0:7] + "wr" + res[0][0][0:7] +".jpg")
            fp.write(file+"wr"+res[0][0]+"\n")
fp.close()

total_time=total_time/i
print("平均时间=%f"%total_time)
#print("img_h_avg=%.2f,img_w_avg=%.2f,acc_h_avg=%.2f,acc_w_avg=%.2f,wa_h_avg=%.2f,wa_w_avg=%.2f"%(img_h/i,img_w/i,acc_h/acc,acc_w/acc,wa_h/wa,wa_w/wa))
print("总样本数=%d,正确数=%d,未检出数=%d,错误数=%d"%(i,acc,non_recognize,wa))
print("正确率=%f,未检出率=%f,错误率=%f"%((acc/i),(non_recognize/i),(wa/i)))
print("检出样本中正确率=%f"%(acc/(acc+wa)))
print("检出率=%f"%(1-(non_recognize/i)))