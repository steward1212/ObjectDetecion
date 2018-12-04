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


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/coco20181203T1615/mask_rcnn_coco_0020.pth")
# /home/steward/pytorch-mask-rcnn/logs/coco20181202T1943/rcnn/logs/rcnn/logs/coco20181203T1615//
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
OUTPUT_FILE_DIR = os.path.join(ROOT_DIR, 'output_file')

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 0
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
#                'teddy bear', 'hair drier', 'toothbrush', 'door', 'window']
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

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]

# figsize=(16, 16)
# _, ax = plt.subplots(1, figsize=figsize)
for i, (file_name) in enumerate(file_names):
  print(file_name)
  image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))

  # Run detection
  results = model.detect([image])

  # Visualize results
  r = results[0]
  visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                              class_names, r['scores'])
  plt.show()

  # output_file_name = os.path.join(OUTPUT_FILE_DIR, '%s.txt' % (os.path.splitext(file_name)[0]))
  # write_output_file(output_file_name, r['rois'], r['class_ids'], r['scores'], class_names)

  # output_name = os.path.join(OUTPUT_DIR, file_name)
  # plt.savefig(output_name)
  print('%d/%d' % (i, len(file_names)))
  # plt.cla()
  # break



