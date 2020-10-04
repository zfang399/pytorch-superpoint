import numpy as np
import glob
import os
import cv2

task = "val" # train / val

# Npz names
npz_folder = "logs/magicpoint_synth_homoAdapt_aidtr_from_coco/predictions/"
npz_path = npz_folder+task+'/'
npzs = [f.split('.')[0] for f in map(os.path.basename, glob.glob(npz_path+"*.npz"))]

# Image path
img_folder = "datasets/Aidtr/"
img_path = img_folder+task+'/'

# percent threshold of gray points above which to discard the point
thresh = 0.25

for i in range(len(npzs)):
    # Loop through all images
    print("{0}/{1}".format(i, len(npzs)-1))
    npz_fullpath = npz_path + npzs[i] + '.npz'
    img_fullpath = img_path + npzs[i] + '.png'

    pts = np.load(npz_fullpath)['pts']
    image = cv2.resize(cv2.imread(img_fullpath), (320, 240))

    # the points to keep
    pts_keep = []

    for j in range(pts.shape[0]):
        # get the point of interest
        p = pts[j]

        # get the box around that point
        x_min, y_min, x_max, y_max = max(int(p[0]-5),0), max(int(p[1]-5),0), min(int(p[0]+5), 319), min(int(p[1]+5), 239)

        # crop and count gray points
        crop = image[y_min:y_max, x_min:x_max].reshape(-1,3)
        gray_vec = np.array([128,128,128])
        comp = ((crop==gray_vec[None,:]).sum(1)==3).sum()
        if comp < thresh * np.prod(crop.shape[:1]):
            pts_keep.append(p)

    pts_keep = np.array(pts_keep)

    print("{0} --> {1}".format(pts.shape[0], pts_keep.shape[0]))

    np.savez(npz_fullpath, pts=pts_keep)

