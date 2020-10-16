import utils
import numpy as np
import os
import imageio
import random
import cv2
import sys
import torch
import yaml

from utils import aidtr_utils

# Dataset save location
sys.path.append("/home/zhaoyuaf/research/pytorch-superpoint")
mod = 'ac'

# Superpoint settings
# Configuration
with open("configs/superpoint_repeatability_heatmap.yaml", "r") as f:
    config = yaml.load(f)

# Load checkpoint 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from utils.loader import get_module
Val_model_heatmap = get_module("", config["front_end_model"])
## load pretrained
val_agent = Val_model_heatmap(config["model"], device=device)
val_agent.loadModel()
patch_size = config["model"]["subpixel"]["patch_size"]

def img_preprocess(image_name):
    image = cv2.imread(image_name)
    sizer = np.array([240, 320])
    s = max(sizer /image.shape[:2])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image[:int(sizer[0]/s),:int(sizer[1]/s)]
    image = cv2.resize(image, (sizer[1], sizer[0]),
                             interpolation=cv2.INTER_AREA)
    image = image.astype('float32') / 255.0
    if image.ndim == 2:
        image = image[:,:, np.newaxis]
    return image

# Forward function
def get_pts_des_from_agent(val_agent, img_name, device="cpu"):
    """
    pts: numpy (N, 3)
    desc: numpy (N, 256)
    """
    img = img_preprocess(img_name)
    heatmap_batch = val_agent.run(torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device))
    pts = val_agent.heatmap_to_pts()
    pts = val_agent.soft_argmax_points(pts, patch_size=patch_size)
    desc_sparse = val_agent.desc_to_sparseDesc()
    return pts[0].transpose(), desc_sparse[0].transpose()

# NGRANSAC settings
dump_folder = '/projects/katefgroup/datasets/aidtr/processed/ngransac_%s' % mod
dataset_name = "ngransac_%s" % mod
variant = "test"
nfeatures = -1 # does not restrict feature count
multiview_pair_per_timestep = 3
sequence_pair_per_timestep = 3

if not os.path.exists(dump_folder):
    os.makedirs(dump_folder)
out_dir = os.path.join(dump_folder, variant+"_data")
out_dir += "_rs" # use rs as postfix
out_dir += '/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

MODULE_LIST = (0, 1, 2, 3, 4) # module0 - module 4
out_image_size = (320, 240)
RGB_INDEX = 2

raw_data_dir = '/projects/katefgroup/datasets/aidtr/hideandseek/'
subfolders = [f.path for f in os.scandir(raw_data_dir) if f.is_dir()]

# One folder for val, Six for train
dir_to_process = []
if variant == "train":
    for folder in subfolders:
        if "Thorday-20200416T210652" not in folder:
            dir_to_process.append(folder)
else:
    for folder in subfolders:
        if "Thorday-20200416T210652" in folder:
            dir_to_process.append(folder)

print('total: {} folders to process'.format(len(dir_to_process)))
print(dir_to_process)

S = len(MODULE_LIST)

# Sequence data
def get_intervals(range_max, S):
    res = []
    for s in range(S):
        res.append(range(range_max)[s::S])

    return list(zip(*res))

ngransac_sample_count = 0

for root_dir in dir_to_process:
    util = aidtr_utils.AIDTR_UTILS(root_dir, dump_folder, out_image_size=out_image_size, id=root_dir.split('/')[-1])
    all_timesteps = util.get_all_timesteps()

    for timestep in all_timesteps:
        rgb_imgname_list = []
        pix_T_cam_list = []
        cam_T_world_list = []

        module_cnt = 0

        for module in MODULE_LIST:
            res, image_name = util.get_image_name(timestep, module, RGB_INDEX)
            if not res:
                break

            pix_T_cam = util.get_pix_T_cam(module, RGB_INDEX)
            cam_T_world = util.get_cam_T_world(timestep, module, RGB_INDEX)

            rgb_imgname_list.append(image_name)
            pix_T_cam_list.append(pix_T_cam)
            cam_T_world_list.append(cam_T_world)

            module_cnt += 1

        if module_cnt < 2:
            print("not enough images to have pairs")
            continue

        bf = cv2.BFMatcher()
        for i in range(module_cnt-1):
            kp1, desc1 = get_pts_des_from_agent(val_agent, rgb_imgname_list[i], device=device)
            kp2, desc2 = get_pts_des_from_agent(val_agent, rgb_imgname_list[i+1], device=device)

            pts1 = []
            pts2 = []
            ratios = []

            matches = bf.knnMatch(desc1, desc2, k = 2)
            for (m,n) in matches:
                pts1.append(kp1[m.queryIdx][:2]) # Take the point location, not the score
                pts2.append(kp2[m.trainIdx][:2])
                ratios.append(m.distance / n.distance)

            pts1 = np.array([pts1])
            pts2 = np.array([pts2])
            ratios = np.array([ratios])
            ratios = np.expand_dims(ratios, 2)

            calibration_1 = pix_T_cam_list[i]
            K1 = calibration_1[:3,:3]
            calibration_2 = pix_T_cam_list[i+1]
            K2 = calibration_2[:3,:3]

            cam_T_world_1 = cam_T_world_list[i]
            GT_R1 = cam_T_world_1[:3,:3]
            GT_t1 = cam_T_world_1[:3,-1].reshape(1,3)
            cam_T_world_2 = cam_T_world_list[i+1]
            GT_R2 = cam_T_world_2[:3,:3]
            GT_t2 = cam_T_world_2[:3,-1].reshape(1,3)

            GT_R_Rel = np.matmul(GT_R2, np.transpose(GT_R1))
            GT_t_Rel = GT_t2.T - np.matmul(GT_R_Rel, GT_t1.T)

            # save data tensor and ground truth transformation
            np.save(out_dir + 'pair_multiview_{0}.npy'.format(ngransac_sample_count), [
                pts1.astype(np.float32), 
                pts2.astype(np.float32), 
                ratios.astype(np.float32), 
                np.array([240,320]), 
                np.array([240,320]), 
                K1.astype(np.float32), 
                K2.astype(np.float32), 
                GT_R_Rel.astype(np.float32), 
                GT_t_Rel.astype(np.float32)
                ])

            ngransac_sample_count += 1
            print("Collected multiview pairs {}".format(ngransac_sample_count))
