import cv2
import os
#from charnet.modeling.model import CharNet
import torch
from charnet.config import cfg
import time
import numpy as np
plate_word = ''
im_name = ''
def save_violation(bbox_stack,decision_boxes,cfg,charnet,violation_frame,img_stack,img_result,frame_num,check_frame,vio):
    if len(img_stack) >= frame_num *2:
        img_stack = img_stack[1:]
        img_stack.append(img_result) #只保留最新frame_num筆img_result
        bbox_stack = bbox_stack[1:] 
        bbox_stack.append(decision_boxes)
    else:   
        img_stack.append(img_result)
        bbox_stack.append(decision_boxes)
    if violation_frame != -1 :
        global im_name
        global plate_word
        #img_stack.append(img_result)
        #bbox_stack.append(decision_boxes)
        f = frame_num
        #index = [0,f//2,f,int(f*1.5),f*2]
        index = [0,f//2,f,int(f*1.5),f*2-1]
        vio_objid = []
        #for i in range(len(bbox_stack[f])):
        #    if bbox_stack[f][i]['decision'] != 'pass' :
        #        vio_objid.append(int(bbox_stack[f][i]['obj_id']))
        #print("violation object ID: ",vio_objid)
        # 且湊滿frame_num*2張frame，更改旗標，清空
        #if len(img_stack) == f*2+1 : 
        if len(img_stack) == f*2 and (check_frame%20==0):
            tmp_path = './violation/'+str(violation_frame)+'/'
            for i in index :
                save_index = violation_frame-f+i
                img_path = tmp_path + str(save_index)+'.png'
                im_name = str(violation_frame)
                cv2.imwrite(img_path,img_stack[i])
                
                #車牌辨識##############charnet##########
                #charnet = CharNet()
                #charnet.load_state_dict(torch.load(cfg.WEIGHT))
                #charnet.eval()
                #charnet.cuda()
                #print(decision_boxes)
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
                print("find")
                if xmin > 0 :
                    im_result = img_stack[i][ymin:ymax,xmin:xmax]
                    img_path2 = tmp_path + str(save_index)+ str(i)+'.png'
                    cv2.imwrite(img_path2,im_result)
                #im_result = img_stack[i][360:1080, 640:1920]
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
            print('==============>>>>>>  Sucess save Violation Image folder : ',violation_frame,"in",im_name)
            #f = frame_num*-1
            #img_stack = img_stack[f:]
            violation_frame = -1
            print("license number ----------------------",plate_word)
        #########change dir name
        for folder_name in os.listdir('./violation'):
            if folder_name == im_name and len(plate_word) > 0:
                if not os.path.isdir('./violation/{}'.format(plate_word)):
                    os.rename('./violation/{}'.format(im_name),'./violation/{}'.format(plate_word))
        check_frame+=1
        vio = True
        if check_frame>19:
            check_frame=0
        return violation_frame,img_stack,bbox_stack,check_frame,vio
    '''
    # 如果沒有違規        
    else : 
        if len(img_stack) < frame_num :
            img_stack.append(img_result)
            bbox_stack.append(decision_boxes)
        else:
            f = (frame_num*-1)+1
            img_stack = img_stack[f:] 
            img_stack.append(img_result) #只保留最新frame_num筆img_result
            bbox_stack = bbox_stack[f:] 
            bbox_stack.append(decision_boxes)
    '''  
    if vio :
        check_frame+=1
    if check_frame>19:
        check_frame=0
    return violation_frame,img_stack,bbox_stack,check_frame,vio
    
    
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









