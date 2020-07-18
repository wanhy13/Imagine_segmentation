'''
A lot of this code is inspired by and or copied from:
https://www.kaggle.com/counter/image-segmentation-for-self-driving-cars
https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/unet.py

Thanks to those developers for their work.
'''

import cv2
import os
import glob
import itertools
from tqdm import tqdm
import numpy as np

VOC_DATA_PATH = "/content/VOCdevkit/VOC2012"
VOC_TRAIN_TXT = "/content/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
VOC_VAL_TXT = "/content/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"

def convert_VOC(img):
    img[(img[:,:,0]==0) * (img[:, :, 1]==0) * (img[:,:,2]==0)] = 0
    img[(img[:,:,0]==0) * (img[:, :, 1]==0) * (img[:,:,2]==128)] = 1
    img[(img[:,:,0]==0) * (img[:, :, 1]==128) * (img[:,:,2]==0)] = 2
    img[(img[:,:,0]==0) * (img[:, :, 1]==128) * (img[:,:,2]==128)] = 3
    img[(img[:,:,0]==128) * (img[:, :, 1]==0) * (img[:,:,2]==0)] = 4
    img[(img[:,:,0]==128) * (img[:, :, 1]==0) * (img[:,:,2]==128)] = 5
    img[(img[:,:,0]==128) * (img[:, :, 1]==128) * (img[:,:,2]==0)] = 6
    img[(img[:,:,0]==128) * (img[:, :, 1]==128) * (img[:,:,2]==128)] = 7
    img[(img[:,:,0]==0) * (img[:, :, 1]==0) * (img[:,:,2]==64)] = 8
    img[(img[:,:,0]==0) * (img[:, :, 1]==0) * (img[:,:,2]==192)] = 9
    img[(img[:,:,0]==0) * (img[:, :, 1]==128) * (img[:,:,2]==64)] = 10
    img[(img[:,:,0]==0) * (img[:, :, 1]==128) * (img[:,:,2]==192)] = 11
    img[(img[:,:,0]==128) * (img[:, :, 1]==0) * (img[:,:,2]==64)] = 12
    img[(img[:,:,0]==128) * (img[:, :, 1]==0) * (img[:,:,2]==192)] = 13
    img[(img[:,:,0]==128) * (img[:, :, 1]==128) * (img[:,:,2]==64)] = 14
    img[(img[:,:,0]==128) * (img[:, :, 1]==128) * (img[:,:,2]==192)] = 15
    img[(img[:,:,0]==0) * (img[:, :, 1]==64) * (img[:,:,2]==0)] = 16
    img[(img[:,:,0]==0) * (img[:, :, 1]==64) * (img[:,:,2]==128)] = 17
    img[(img[:,:,0]==0) * (img[:, :, 1]==192) * (img[:,:,2]==0)] = 18
    img[(img[:,:,0]==0) * (img[:, :, 1]==192) * (img[:,:,2]==128)] = 19
    img[(img[:,:,0]==128) * (img[:, :, 1]==64) * (img[:,:,2]==0)] = 20

    img[img[:, :, 0] > 20] = 0 # Edge as Background
    return img


def convert_segmmentations_VOC():
    if not os.path.exists(os.path.join(VOC_DATA_PATH, "converted")):
        os.makedirs(os.path.join(VOC_DATA_PATH, "converted"))
    for image_dir in os.listdir(os.path.join(VOC_DATA_PATH, "SegmentationClass")):
        img = cv2.imread(os.path.join(VOC_DATA_PATH, "SegmentationClass", image_dir), 1)
        img = convert_VOC(img)
        cv2.imwrite(os.path.join(VOC_DATA_PATH, "converted", image_dir),img)


def make_dataset_VOC():
    '''
    Make necessary dirs
    '''

    print('Making your VOC dataset')

    convert_segmmentations_VOC()

    if not os.path.exists(os.path.join(VOC_DATA_PATH, "train")):
        os.makedirs(os.path.join(VOC_DATA_PATH, "train"))
    if not os.path.exists(os.path.join(VOC_DATA_PATH, "val")):
        os.makedirs(os.path.join(VOC_DATA_PATH, "val"))
    if not os.path.exists(os.path.join(VOC_DATA_PATH, "train", "imgs")):
        os.makedirs(os.path.join(VOC_DATA_PATH, "train", "imgs"))
    if not os.path.exists(os.path.join(VOC_DATA_PATH, "train", "segs")):
        os.makedirs(os.path.join(VOC_DATA_PATH, "train", "segs"))
    if not os.path.exists(os.path.join(VOC_DATA_PATH, "val", "imgs")):
        os.makedirs(os.path.join(VOC_DATA_PATH, "val", "imgs"))
    if not os.path.exists(os.path.join(VOC_DATA_PATH, "val", "segs")):
        os.makedirs(os.path.join(VOC_DATA_PATH, "val", "segs"))
    ''' End '''
    train_list = open(VOC_TRAIN_TXT, "r").readlines()
    val_list = open(VOC_VAL_TXT, "r").readlines() 
    for image_name in train_list:
        image_name = image_name.strip()
        # Save images
        img = cv2.imread(os.path.join(VOC_DATA_PATH, "JPEGImages", image_name+".jpg"),1)
        cv2.imwrite(os.path.join(VOC_DATA_PATH, "train", "imgs", image_name+".jpg"), img)
        # Save segmentations
        img = cv2.imread(os.path.join(VOC_DATA_PATH, "converted", image_name+".png"))
        cv2.imwrite(os.path.join(VOC_DATA_PATH, "train", "segs", image_name+".png"), img)
    for image_name in val_list:
        image_name = image_name.strip()
        # Save images
        img = cv2.imread(os.path.join(VOC_DATA_PATH, "JPEGImages", image_name+".jpg"),1)
        cv2.imwrite(os.path.join(VOC_DATA_PATH, "val", "imgs", image_name+".jpg"), img)
        # Save segmentations
        img = cv2.imread(os.path.join(VOC_DATA_PATH, "converted", image_name+".png"))
        cv2.imwrite(os.path.join(VOC_DATA_PATH, "val", "segs", image_name+".png"), img)

def verify_segmentation_dataset( images_path , segs_path , n_classes ):
	
	img_seg_pairs = get_pairs_from_paths( images_path , segs_path )

	assert len(img_seg_pairs)>0 , "Dataset looks empty or path is wrong "
	
	for im_fn , seg_fn in tqdm(img_seg_pairs) :
		img = cv2.imread( im_fn )
		seg = cv2.imread( seg_fn )

		assert ( img.shape[0]==seg.shape[0] and img.shape[1]==seg.shape[1] ) , "The size of image and the annotation does not match or they are corrupt "+ im_fn + " " + seg_fn
		assert ( np.max(seg[:,:,0]) < n_classes) , "The pixel values of seg image should be from 0 to "+str(n_classes-1) + " . Found pixel value "+str(np.max(seg[:,:,0]))

	print("Dataset verified! ")

def get_pairs_from_paths( images_path , segs_path, recursive = False ):
	if recursive == True:
		images = glob.glob( 
			os.path.join(images_path,"**/*.jpg")) + \
			glob.glob( os.path.join(images_path,"**/*.png")  ) +  \
			glob.glob( os.path.join(images_path,"**/*.jpeg")  )
		segmentations  =  glob.glob( os.path.join(segs_path,"**/*.png")  ) 

		base_name = [os.path.basename(i) for i in segmentations]
		ret = []

		for im in images:
			seg_bnme = os.path.basename(im).replace(".jpg" , ".png").replace(".jpeg" , ".png")
			assert ( seg_bnme in base_name ),  (im + " is present in "+images_path +" but "+seg_bnme+" is not found in "+segs_path + " . Make sure annotation image are in .png"  )
			ret.append((im , segmentations[base_name.index(seg_bnme)]) )
		return ret
	else:
		images = glob.glob( os.path.join(images_path,"*.jpg")  ) + glob.glob( os.path.join(images_path,"*.png")  ) +  glob.glob( os.path.join(images_path,"*.jpeg")  )
		segmentations  =  glob.glob(os.path.join(segs_path,"*.png")) 

		segmentations_d = dict(zip(segmentations,segmentations ))

		ret = []

		for im in images:
			seg_bnme = os.path.basename(im).replace(".jpg" , ".png").replace(".jpeg" , ".png")
			seg = os.path.join( segs_path , seg_bnme  )
			assert ( seg in segmentations_d ),  (im + " is present in "+images_path +" but "+seg_bnme+" is not found in "+segs_path + " . Make sure annotation image are in .png"  )
			ret.append((im , seg) )

		return ret
