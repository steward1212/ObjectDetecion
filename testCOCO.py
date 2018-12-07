# %matplotlib inline
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# initialize COCO api for instance annotations
coco=COCO('mask_data/mycoco.json')

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['chair'])
print(catIds)
imgIds = coco.getImgIds(catIds=catIds)
print(imgIds)
# imgIds = coco.getImgIds(imgIds = [20180000133])
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

# load and display image
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
file_path = 'mask_data/%s/renders/%s_%s' % (img['file_name'].split('_')[0], img['file_name'].split('_')[1], img['file_name'].split('_')[2])
# print(file_path)
I = io.imread(file_path)
# plt.axis('off')
# plt.imshow(I)
# plt.show()


# load and display instance annotations
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
print(anns)
coco.showAnns(anns)
plt.show()
