#!/usr/bin/env python
#手掌识别
import cv2
import os

hand_Cascade = cv2.CascadeClassifier("../xml/cascade.xml")
hand_Cascade.load('../xml/cascade.xml')

data_dir='G:/img/hand_detect/video/test'
video_dirs=os.listdir(data_dir)
video_dirs.sort()
# print(video_dirs)
#
save_dir='../datasets/hand/test'
image=10 #提取数量

for video_dir in video_dirs:
    video_list=os.listdir(os.path.join(data_dir,video_dir))
    for video in video_list:
        os.mkdir(os.path.join(save_dir,video_dir+'-'+video[:-4]))
        cap = cv2.VideoCapture(os.path.join(data_dir,video_dir,video))
        i=1
        while True:
            if i<200:
                i+=1
                continue
            ret, frame = cap.read()
            if ret == False:
                break
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # rect = hand_Cascade.detectMultiScale(        #主要修改以下参数
            #     gray,
            #     scaleFactor=6,
            #     minNeighbors=100,
            #     minSize=(250,250)
            #     )
            # for (x, y, w, h) in rect:
            #     out=frame[y:y+h,x:x+w]
            #     out=cv2.resize(out,(128,128))
            cv2.imwrite(os.path.join(save_dir,video_dir+'-'+video[:-4])+'/'+str(i)+'.jpg',frame)
            i+=1
            if i-200>image:
                break

        cap.release()

