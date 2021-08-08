'''
@Time : 2020/1/11 下午2:53 

# @Author : Xuebing
# @File : hand_test.py
# @Software: CLion
'''

from __future__ import print_function
import os
import cv2
import numpy as np
from config.config import Config
from torch.nn import DataParallel
from models.resnet import *
from models.focal_loss import *
import xlrd

hand_Cascade = cv2.CascadeClassifier("xml/cascade.xml")
hand_Cascade.load('xml/cascade.xml')

def detect_image(image_file):
    image=cv2.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rect = hand_Cascade.detectMultiScale(  # 主要修改以下参数
        gray,
        scaleFactor=6,
        minNeighbors=100,
        minSize=(200, 200)
    )
    for (x, y, w, h) in rect:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255))
        cv2.imshow('hand_location', image)
        cropped = image[y:y+h, x:x+w]  # 裁剪坐标为[y0:y1, x0:x1]
        cropped = cv2.resize(cropped,(128,128),cv2.INTER_LINEAR)
    return cropped

def load_image(cropped):
    image = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
    if image is None:
        return None
    image = np.dstack((image, np.fliplr(image)))  # [128,128]
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image[:,:,1:]

def get_feature(model,path):
    features = None
    images=None
    image=load_image(path)
    data = torch.from_numpy(image)
    data = data.to(torch.device("cuda"))
    output = model(data)
    output = output.data.cpu().numpy()
    fe_1 = output[:1,:]
    fe_2 = output[1:,:]
    feature = np.hstack((fe_1, fe_2))
    return feature

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def cal_verification(sims):
    sims = np.asarray(sims)
    flag='0'
    for i in sims:
        if i>0.55:
            flag='1'
            return flag
    return '0'

def identity_verification(sims):
    max_index=sims.index(max(sims))
    wb=xlrd.open_workbook('datasets/id_feature.xls')
    worksheet=wb.sheet_by_index(0)
    return worksheet.cell_value(max_index,1)

def test_performance(feature, features):
    sims = []
    for fea in features:
        sim = cosin_metric(feature, fea) #np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        sims.append(sim)
    return sims

def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    flag_list = []
    for pair in pairs:
        splits = pair.split()
        if splits[0] not in data_list:
            data_list.append(splits[0])
            flag_list.append(splits[1])
    return data_list,flag_list

def acc_identity(dir_1,dir_2):
    list_1=[]
    list_2=[]
    list_2_imgs=[]
    index=[]
    error_imgs=[]
    with open(dir_1,'r') as f1:
        paris = f1.readlines()
        for pair in paris:
            list_1.append(pair.split('/')[0][:-1])
    with open(dir_2,'r') as f2:
        pairs = f2.readlines()
        for pair in pairs:
            list_2.append(pair.split('/')[0])
            list_2_imgs.append(pair.split(' ')[0])
    for i in range(len(list_1)):
        if list_1[i] =='error':
            index.append(i)
    # print(index)

    for i in index:
        error_imgs.append(list_2_imgs[i])

    print(error_imgs)
    for i in index[::-1]:
        list_1.pop(i)
        list_2.pop(i)

    print(len(list_1),list_1)
    print(len(list_2),list_2)

    b=len(list_1)
    count=0
    for i in range(b):
        if list_1[i]==list_2[i]:
            count+=1
    print('acc={}'.format(count/b))




if __name__ == '__main__':
    features=np.load('datasets/features.npy')
    flags=[]
    opt = Config()
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    model.eval()
    model = DataParallel(model)
    model.load_state_dict(torch.load(opt.test_model_path),strict=True)
    model.to(torch.device("cuda"))
    img=('datasets/hand/test/test_1-6123-R/209.jpg')
    cropped=detect_image(img)
    feature=get_feature(model,cropped)
    sims = test_performance(feature, features)
    flag=cal_verification(sims)
    if flag=='1':
        print('Verify success')
        identity = identity_verification(sims)
        feature_map=cv2.imread('datasets/hand/test_feature/'+identity+'/'+'201.jpg')
        cv2.imshow('feature_map',feature_map)
        print('identity is {}'.format(identity))
        cv2.waitKey(0)
    else:
        print('Verify error')
    # identity_list,flag_list = get_lfw_list(opt.lfw_test_list)  # list of image for test
    # img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]
    #
    # h = open('result.txt','w')
    #
    # for img in img_paths:
    #     feature=get_feature(model,img)
    #     sims=test_performance(feature,features)
    #     flag=cal_verification(sims)
    #     flags.append(flag)
    #     if flag=='1':
    #         # print('Verify success')
    #         identity=identity_verification(sims)
    #         # print(identity)
    #         h.write(identity+'\n')
    #     else:
    #         # print('Verify error')
    #         h.write('error'+'\n')
    # h.close()
    # print(flags)
    # print(flag_list)
    # flags=np.asarray(flags)
    # flag_list=np.asarray(flag_list)
    # acc = np.mean((flag_list == flags).astype(int))
    # print(acc)
    #
    # acc_identity('result.txt','datasets/hand_img_test.txt')
    #

    # if flag=='1':
    #     print('Verify success')
    #     identity=identity_verification(sims)
    #     print(identity)
    # else:
    #     print('Verify error')

