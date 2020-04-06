import cv2
import os
import torch
import time
import numpy as np

def save_violation(violation_frame,decision_box,img_stack,bbox_stack,img_result,frame_num):
    if violation_frame != -1 :
        global im_name
        global plate_word
        img_stack.append(img_result)
        bbox_stack.append(decision_box)
        f = frame_num
        index = [0,f//2,f,int(f*1.5),f*2]
        # 且湊滿frame_num*2張frame，更改旗標，清空
        if len(img_stack) == f*2+1 :
            tmp_path = './violation/'+str(violation_frame)+'/'
            for i in index :
                save_index = violation_frame-f+i
                img_path = tmp_path + str(save_index)+'.png'
                im_name = str(violation_frame)
                cv2.imwrite(img_path,img_stack[i])
                ############################車牌辨識#####################################
                #裁切圖片再丟進去辨識
                xmin = 0
                ymin = 0
                xmax = 0
                ymax = 0
                for num in range(len(bbox_stack[i])):
                    if bbox_stack[i][num]['decision'] != 'pass' :
                        xmin = int(bbox_stack[i][num]['bbox']['points'][0])
                        ymin = int(bbox_stack[i][num]['bbox']['points'][1])
                        xmax = int(bbox_stack[i][num]['bbox']['points'][2])
                        ymax = int(bbox_stack[i][num]['bbox']['points'][3])
                print(i)
                print("Place:",xmin,ymin,xmax,ymax)
                #xmin = int(decision_boxes[i]['bbox']['points'][0])
                #裁切圖片再丟進去辨識
                if xmin > 0 :
                    im_result = img_stack[i][ymin:ymax,xmin:xmax]
                    img_path2 = tmp_path + '_' +str(i)+'.png'
                    cv2.imwrite(img_path2,im_result)
                
            print('==============>>>>>>  Sucess save Violation Image folder : ',violation_frame,"in",im_name)
            #print("license number ----------------------",plate_word)
            f = frame_num*-1
            img_stack = img_stack[f:]
            bbox_stack = bbox_stack[f:]
            violation_frame = -1
        """
        #########change dir name
        for folder_name in os.listdir('./violation'):
            if folder_name == im_name and len(plate_word) > 0:
                os.rename('./violation/{}'.format(im_name),'./violation/{}'.format(plate_word))
        """
    # 如果沒有違規        
    else : 
        if len(img_stack) < frame_num :
            img_stack.append(img_result)
            bbox_stack.append(decision_box)
        else:
            f = (frame_num*-1)+1
            img_stack = img_stack[f:]
            bbox_stack = bbox_stack[f:]
            img_stack.append(img_result) #只保留最新frame_num筆img_result
            bbox_stack.append(decision_box)
    return violation_frame,img_stack,bbox_stack
    
    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('==============>>>>>>  Scuessfully Make folder :' + directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)











