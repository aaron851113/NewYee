import cv2
import os
import torch
import time
import numpy as np

def save_violation(bbox_stack, match_bbox, violation_frame, violation_id, img_stack, img_result, frame_num, violation_id_list):
    if violation_frame != -1 :
        global im_name
        global plate_word
        img_stack.append(img_result)
        bbox_stack.append(match_bbox)
        f = frame_num
        index = [0,f//2,f,int(f*1.5),f*2] #frame_num=20
        # 且湊滿frame_num*2張frame，更改旗標，清空
        if len(img_stack) == f*2+1 :
            tmp_path = './violation/'+str(violation_frame)+'/'
            for i in index :
                save_index = violation_frame-f+i
                img_path = tmp_path + str(save_index)+'.png'
                im_name = str(violation_frame)
                cv2.imwrite(img_path,img_stack[i])

            f = frame_num*-1
            img_stack = img_stack[f:]
            bbox_stack = bbox_stack[f:]
            violation_frame = -1
            if violation_id not in violation_id_list:
                violation_id_list.append(violation_id)
            violation_id = 0
    # 如果沒有違規        
    else : 
        if len(img_stack) < frame_num :
            img_stack.append(img_result)
            bbox_stack.append(match_bbox)
        else:
            f = (frame_num*-1)+1
            img_stack = img_stack[f:]
            bbox_stack = bbox_stack[f:]
            img_stack.append(img_result) #只保留最新frame_num筆img_result
            bbox_stack.append(match_bbox)
            
    return violation_frame, violation_id, img_stack, bbox_stack, violation_id_list
    
    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('==============>>>>>>  Scuessfully Make folder :' + directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)





def match_bbox_tracker(bbox,tracker):
    match = []        
    if len(tracker) == 0:
        for i in range(0,len(bbox)):
            tmp_id = -1 
            tmp_cls = -1
            match.append([bbox[i]['decision'],
                          int(bbox[i]['bbox']['points'][0]),int(bbox[i]['bbox']['points'][1]),
                          int(bbox[i]['bbox']['points'][2]),int(bbox[i]['bbox']['points'][3]),
                          tmp_id,tmp_cls])
        return match
    else : 
        origin_tracker = tracker.copy()
        origin_tracker.sort(axis=0)
        """
        for a in range(0,len(bbox)):
            print(bbox[a])
        for b in range(0,len(origin_tracker)):
            print(origin_tracker[b])
        """
        for i in range(0,len(bbox)):
            tmp = -1
            for j in range(0,len(origin_tracker)):
                x1_ = abs(bbox[i]['bbox']['points'][0]-origin_tracker[j][0])
                y1_ = abs(bbox[i]['bbox']['points'][1]-origin_tracker[j][1])
                x2_ = abs(bbox[i]['bbox']['points'][2]-origin_tracker[j][2])
                y2_ = abs(bbox[i]['bbox']['points'][3]-origin_tracker[j][3])
                if (x1_ + y1_) < 200 and (x2_ + y2_ < 200) :
                    match.append([bbox[i]['decision'],
                                  int(bbox[i]['bbox']['points'][0]),int(bbox[i]['bbox']['points'][1]),
                                  int(bbox[i]['bbox']['points'][2]),int(bbox[i]['bbox']['points'][3]),
                                  int(origin_tracker[j][4]),int(origin_tracker[j][5])])
                    tmp = 1
                    break
            if tmp == -1 :
                tmp_id = -1 
                tmp_cls = -1
                match.append([bbox[i]['decision'],
                              int(bbox[i]['bbox']['points'][0]),int(bbox[i]['bbox']['points'][1]),
                              int(bbox[i]['bbox']['points'][2]),int(bbox[i]['bbox']['points'][3]),
                              tmp_id,tmp_cls])
        return match






