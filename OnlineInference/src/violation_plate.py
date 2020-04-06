import cv2
import os
import torch
import time
import numpy as np
from charnet.config import cfg

plate_word = ''
im_name = ''
def save_violation(bbox_stack,match_bbox,cfg,charnet,violation_frame,violation_id,violation_id_satck,img_stack,img_result,frame_num):
    if len(img_stack) >= frame_num *2:
        img_stack = img_stack[1:]
        img_stack.append(img_result) #只保留最新frame_num筆img_result
        bbox_stack = bbox_stack[1:] 
        bbox_stack.append(match_bbox)
    else:   
        img_stack.append(img_result)
        bbox_stack.append(match_bbox)
    
    if violation_frame != -1 :
        global im_name
        global plate_word
        f = frame_num
        index = [0,f//2,f,int(f*1.5),f*2-1]
        vio_objid = []
        if len(img_stack) == f*2 : 
            tmp_path = './violation/'+str(violation_frame)+'/'
            for i in index :
                save_index = violation_frame-f+i
                img_path = tmp_path + str(save_index)+'.png'
                im_name = str(violation_frame)
                cv2.imwrite(img_path,img_stack[i])
            
            for i in range(len(bbox_stack)):
                #車牌辨識##############charnet##########
                charnet.cuda()
                xmin = 0
                ymin = 0
                xmax = 0
                ymax = 0
                for num in range(len(bbox_stack[i])):
                    if bbox_stack[i][num][5] == violation_id :
                        xmin = int(bbox_stack[i][num][1])
                        ymin = int(bbox_stack[i][num][2])
                        xmax = int(bbox_stack[i][num][3])
                        ymax = int(bbox_stack[i][num][4])
                       
                #裁切圖片再丟進去辨識
                if xmin > 0 :
                    print('find_plate_frame :',i,"  Place:",xmin,ymin,xmax,ymax)
                    im_result = img_stack[i][ymin:ymax,xmin:xmax]
                    img_path2 = tmp_path + '_' + str(i)+'.png'
                    cv2.imwrite(img_path2,im_result)
                    im, scale_w, scale_h, original_w, original_h = resize(im_result, size=cfg.INPUT_SIZE)
                    start_time = time.time()
                    with torch.no_grad():
                        char_bboxes, char_scores, word_instances = charnet(im, scale_w, scale_h, original_w, original_h)
                        plate_word = save_word_recognition(
                                word_instances, im_name,
                                "./charnet_result", cfg.RESULTS_SEPARATOR
                            )
                    #draw the box and result
                    #draw_box_word(im_name,im_result,"./charnet_result")  
                    print("--- %s seconds ---" % (time.time() - start_time))

                #########################
            print('==>>>>>>  Sucess save Violation Image folder : ',violation_frame,"in",im_name)
            violation_frame = -1
            violation_id_satck.append(violation_id)
            violation_id = 0
            print("license number ------------>",plate_word)
        #########change dir name
        for folder_name in os.listdir('./violation'):
            if folder_name == im_name and len(plate_word) > 0:
                if not os.path.isdir('./violation/{}'.format(plate_word)):
                    os.rename('./violation/{}'.format(im_name),'./violation/{}'.format(plate_word))
        
    return violation_frame,violation_id,violation_id_satck,img_stack,bbox_stack
    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('==============>>>>>>  Scuessfully Make folder :' + directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


####charnet車牌辨識
def save_word_recognition(word_instances, image_id, save_root, separator=chr(31)):
    global plate_word
    plate_word = ''
    with open('{}/{}.txt'.format(save_root, image_id), 'wt') as fw:
        for word_ins in word_instances:
            if len(word_ins.text) > 0:
                if len(word_ins.text) > len(plate_word):
                    plate_word = word_ins.text

                fw.write(separator.join([str(_) for _ in word_ins.word_bbox.astype(np.int32).flat]))
                fw.write(separator)
                fw.write(word_ins.text)
                fw.write('\n')
                print("detect word==============",plate_word)
    return plate_word

def resize(im, size):
    h, w, _ = im.shape
    scale = max(h, w) / float(size)
    image_resize_height = int(round(h / scale / cfg.SIZE_DIVISIBILITY) * cfg.SIZE_DIVISIBILITY)
    image_resize_width = int(round(w / scale / cfg.SIZE_DIVISIBILITY) * cfg.SIZE_DIVISIBILITY)
    scale_h = float(h) / image_resize_height
    scale_w = float(w) / image_resize_width
    im = cv2.resize(im, (image_resize_width, image_resize_height), interpolation=cv2.INTER_LINEAR)
    return im, scale_w, scale_h, w, h


def vis(img, word_instances):
    img_word_ins = img.copy()
    for word_ins in word_instances:
        word_bbox = word_ins.word_bbox
        cv2.polylines(img_word_ins, [word_bbox[:8].reshape((-1, 2)).astype(np.int32)],
                      True, (0, 255, 0), 2)
        cv2.putText(
            img_word_ins,
            '{}'.format(word_ins.text),
            (word_bbox[0], word_bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
    return img_word_ins

###########


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
        
        for a in range(0,len(bbox)):
            print(bbox[a])
        for b in range(0,len(origin_tracker)):
            print(origin_tracker[b])
        
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
    
    


