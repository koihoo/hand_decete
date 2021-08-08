'''
@Time : 2020/1/11 下午2:15 

# @Author : Xuebing

# @File : create_feature.py 

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
import xlrd,xlwt

def test_txt():
    f = open(opt.hand_feature_file,'w')
    path = opt.hand_feature_root
    img_file = os.listdir(path)
    print("len", len(img_file))
    for file in img_file:
        img_path = os.listdir(os.path.join(path, file))
        new_context = os.path.join(file, img_path[0])+'\n'
        f.write(new_context)
    f.close()

    return f.name

def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()
        if splits[0] not in data_list:
            data_list.append(splits[0])
    return data_list

def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    image = np.dstack((image, np.fliplr(image)))  # [128,128]
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image

def get_featurs(model, test_list, batch_size):
    images = None
    features = None
    name_fea={}
    for i, img_path in enumerate(test_list):  # 181
        image = load_image(img_path)
        name_fea[str(i)]=img_path.split('/')[-2]
        if image is None:
            print('read {} error'.format(img_path))
        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)
        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:

            data = torch.from_numpy(images)
            data = data.to(torch.device("cuda"))  # torch.Size([60, 1, 128, 128])
            output = model(data)
            output = output.data.cpu().numpy()  # (60, 512)
            fe_1 = output[::2]
            fe_2 = output[1::2]
            feature = np.hstack((fe_1, fe_2))  # (30, 1024)
            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))
            images = None
    return name_fea,features

if __name__ == '__main__':
    opt = Config()
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    model.eval()
    model = DataParallel(model)
    # load_model(model, opt.test_model_path)
    model.load_state_dict(torch.load(opt.test_model_path))
    model.to(torch.device("cuda"))
    txt_name=test_txt()
    identity_list = get_lfw_list(opt.hand_feature_file)  # list of image for test
    img_paths = [os.path.join(opt.hand_feature_root, each) for each in identity_list]
    name_fea,features = get_featurs(model, img_paths, batch_size=opt.test_batch_size)
    wb = xlwt.Workbook()#创建一个excel文件
    sh = wb.add_sheet('Sheet1')
    for key,value in name_fea.items():
        sh.write(int(key),0,key)
        sh.write(int(key),1,value)

    wb.save('datasets/id_feature.xls')
    np.save(opt.feature_file,features)

