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
from skimage.transform import rescale, resize, downscale_local_mean


# Root directory of the project
ROOT_DIR = os.getcwd()
background_image = skimage.io.imread(os.path.join(ROOT_DIR, 'background.png'))

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
# rcnn/logs/coco20181202T1943/rcnn/logs/coco20181204T1440/
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/coco20181204T1440/mask_rcnn_coco_0020.pth")
# /home/steward/pytorch-mask-rcnn/logs/coco20181202T1943/rcnn/logs/rcnn/logs/coco20181203T1615//
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
OUTPUT_FILE_DIR = os.path.join(ROOT_DIR, 'output_file')
OUTPUT_BOUNDING_BOX = os.path.join(IMAGE_DIR, 'output_bounding_box')

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 1
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

def apply_mask(image, mask, background_image):
    """Apply the given mask to the image.
    """
    # x, y, z = image.shape

    # print(image.shape)
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c], background_image[:, :, c])


    # for c in range(3):
    #     image[:, :, c] = np.where(mask == 0,
    #                               image[:, :, c], 255)

    return image

def crop_bounding_box(image, boxes, class_ids, scores, class_names, masks, background_image):
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
      # box_image = image[y1:y2, x1:x2]
      # print(len(mask))
      masked_image = image.astype(np.uint32).copy()
      mask = masks[:, :, i]
      masked_image = apply_mask(masked_image, mask, background_image)
      # width, height = masked_image.size
      # print(width, height)
      # break
      skimage.io.imsave(os.path.join(OUTPUT_BOUNDING_BOX, '%s%d.jpg' % (label, i)), masked_image)





# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]

# figsize=(16, 16)
# _, ax = plt.subplots(1, figsize=figsize)
for i, (file_name) in enumerate(file_names):
  # print(file_name)
  image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))
  # x, y, z = image.shape

  # background_image = resize(background_image, (x, y), anti_aliasing=True)

  # Run detection
  results = model.detect([image])

  # Visualize results
  r = results[0]
  visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                              class_names, r['scores'])
  plt.show()

  # output_file_name = os.path.join(OUTPUT_FILE_DIR, '%s.txt' % (os.path.splitext(file_name)[0]))
  # crop_bounding_box(image, r['rois'], r['class_ids'], r['scores'], class_names, r['masks'], background_image)

  # output_name = os.path.join(OUTPUT_DIR, file_name)
  # plt.savefig(output_name)
  print('%d/%d' % (i, len(file_names)))
  # plt.cla()
  # break



