

import cv2

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize

import torch
import torch.nn as nn
from tempfile import TemporaryFile


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/coco20181204T1440/mask_rcnn_coco_0020.pth")
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

# /home/steward/pytorch-mask-rcnn/logs/coco20181202T1943/rcnn/logs/rcnn/logs/coco20181203T1615//
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
OUTPUT_FILE_DIR = os.path.join(ROOT_DIR, 'output_file')

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 2
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object.
model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
if config.GPU_COUNT:
    model = model.cuda()

# Load weights trained on MS-COCO
model.load_state_dict(torch.load(COCO_MODEL_PATH))

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
# class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                'bus', 'train', 'truck', 'boat', 'traffic light',
#                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                'teddy bear', 'hair drier', 'toothbrush']
# [62,63,65,67,72,73,76,84,91,92]
class_names = ['BG', 'chair', 'couch', 'bed', 'dining table', 'tv', 'laptop', 'keyboard', 'book', 'door', 'window']

def write_output_file(file_name, boxes, class_ids, scores, class_names):
  # print(file_name)
  with open(file_name, 'w+') as output_file:
    N = boxes.shape[0]
    for i in range(N):
      if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
      y1, x1, y2, x2 = boxes[i]

      # Label
      class_id = class_ids[i]
      score = scores[i] if scores is not None else None
      label = class_names[class_id]

      output_file.write('%d %d %d %d %.4f %s\n' %(y1, x1, y2, x2, score, label))

test_names = next(os.walk('images'))[2]
cos_simi = nn.CosineSimilarity(0)

#file_names = next(os.walk('table_npy'))[2]

for k, (test_name) in enumerate(test_names):
    #loadfile = 'mynpy/11.npy'
    #a1 = torch.from_numpy(np.load(loadfile)).cuda()
    #img_path = 'images/11.jpg'
    #img_path = 'test_table/%s'%test_name #'images/20181206_140950.jpg'
    #image = skimage.io.imread(img_path)
    
    
    img_path = 'images/%s'%test_name #'images/20181206_140950.jpg'
    image = skimage.io.imread(img_path)
    # Run detection
    results, my_roi, index = model.detect([image])
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])
    plt.show()

    #import pdb; pdb.set_trace()

    N = r['rois'].shape[0]
    for xi in range(N):
        y1, x1, y2, x2 = r['rois'][xi]
        image_bd = image[y1:y2, x1:x2]
        image_bd_mask = results[0]['masks'][:,:,xi][y1:y2, x1:x2]



        max_rate = 300/max(image_bd.shape[0], image_bd.shape[1])
        #import pdb; pdb.set_trace()
        mask = cv2.resize(image_bd_mask, (int(image_bd_mask.shape[1]*max_rate), int(image_bd_mask.shape[0]*max_rate)), interpolation=cv2.INTER_CUBIC)
        re_image = cv2.resize(image_bd, (int(image_bd.shape[1]*max_rate), int(image_bd.shape[0]*max_rate)), interpolation=cv2.INTER_CUBIC)

        #import pdb; pdb.set_trace()
        back = cv2.imread('background.png')
        # C = np.where(mask==1)
        # back[C[0], C[1],:] = re_image[C[0],C[1],:]
        back[10:re_image.shape[0]+10, 10:re_image.shape[1]+10] = re_image
        image_new = back
    
    
        # Run detection
        my_results, my_roi, index = model.detect([image_new])
        a1 = my_roi[index[0]].view(-1).data

        my_r = my_results[0]
        visualize.display_instances(image_new, my_r['rois'], my_r['masks'], my_r['class_ids'],class_names, my_r['scores'])
        plt.show()

        filename = []
        similarity = []

        label =  class_names[r['class_ids'][xi]]
        
        if label=='dining table':
            file_names = next(os.walk('table_npy'))[2]

            for i, (file_name) in enumerate(file_names):

                if(file_name[:8]!='table001'and file_name[:8]!='table003'):

                    #r1 = torch.from_numpy(np.load(os.path.join('table_npy', file_name))).cuda()
                    r1 = torch.from_numpy(np.load(os.path.join('table_npy', file_name))).cuda()

                    similarity_2 = cos_simi(a1,r1) #cos_simi(a1.unsqueeze(0),r1.unsqueeze(0))
                    
                    filename.append(file_name)
                    similarity.append(similarity_2)
                    print(file_name,similarity_2)

            print('-----------------------')
            print(img_path, filename[similarity.index(max(similarity))], max(similarity))
            # readref = 'easy_table/%s/renders/%s.png'%(filename[similarity.index(max(similarity))].split('.')[0].split('_')[0],filename[similarity.index(max(similarity))].split('.')[0][9:])

            # loadimg = image_new
            # refimg = cv2.imread(readref)

            # result = np.zeros((max(refimg.shape[0], loadimg.shape[0]), refimg.shape[1]+loadimg.shape[1], refimg.shape[2]))
            # result[0:loadimg.shape[0],0:loadimg.shape[1],:] = loadimg
            # result[0:refimg.shape[0],loadimg.shape[1]:,:] = refimg

            # cv2.imwrite('results/easy_table_%s_%d.png'%(img_path.split('test_img/')[1].split('.')[0],xi), result)
            # print('=======================')

        elif label=='chair':

            file_names = next(os.walk('chair_npy'))[2]
            
            for i, (file_name) in enumerate(file_names):

                r1 = torch.from_numpy(np.load(os.path.join('chair_npy', file_name))).cuda()

                similarity_2 = cos_simi(a1,r1) 
                
                filename.append(file_name)
                similarity.append(similarity_2)
                print(file_name,similarity_2)

            print('-----------------------')
            
            print(img_path, filename[similarity.index(max(similarity))], max(similarity))

            # readref = 'rendered_chairs/%s/renders/%s.png'%(filename[similarity.index(max(similarity))].split('.')[0].split('_')[0],filename[similarity.index(max(similarity))].split('.')[0][9:])

            # loadimg = image_new
            # refimg = cv2.imread(readref)

            # result = np.zeros((max(refimg.shape[0], loadimg.shape[0]), refimg.shape[1]+loadimg.shape[1], refimg.shape[2]))
            # result[0:loadimg.shape[0],0:loadimg.shape[1],:] = loadimg
            # result[0:refimg.shape[0],loadimg.shape[1]:,:] = refimg

            # cv2.imwrite('results/easy_chair_%s_%d.png'%(img_path.split('test_img/')[1].split('.')[0],xi), result)