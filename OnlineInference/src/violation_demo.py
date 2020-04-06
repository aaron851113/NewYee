import cv2
import os
import torch
import time
import numpy as np


def save_violation(violation_frame,img_stack,img_result,frame_num,check_frame,vio):
    if len(img_stack) >= frame_num *2:
        img_stack = img_stack[1:]
        img_stack.append(img_result) #只保留最新frame_num筆img_result
    else:   
        img_stack.append(img_result)
    if violation_frame != -1 :
        f = frame_num
        index = [0,f//2,f,int(f*1.5),f*2-1]

        if len(img_stack) == f*2 and (check_frame%frame_num==0):
            tmp_path = './violation/'+str(violation_frame)+'/'
            for i in index :
                save_index = violation_frame-f+i
                img_path = tmp_path + str(save_index)+'.png'
                im_name = str(violation_frame)
                cv2.imwrite(img_path,img_stack[i])
            
            violation_frame = -1

        check_frame+=1
        vio = True
        if check_frame>frame_num-1:
            check_frame=0
        return violation_frame,img_stack,check_frame,vio
    
    if vio :
        check_frame+=1
    if check_frame>frame_num-1:
        check_frame=0
    return violation_frame,img_stack,check_frame,vio
    
    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('==============>>>>>>  Scuessfully Make folder :' + directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)











