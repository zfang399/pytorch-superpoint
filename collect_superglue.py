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
sys.path.append("/home/akashsharma/projects/pytorch-superpoint")
mod = 'ab'

# Superpoint settings
# Configuration
with open("configs/superpoint_repeatability_heatmap.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Load checkpoint
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from utils.loader import get_module
Val_model_heatmap = get_module("", config["front_end_model"])
## load pretrained
val_agent = Val_model_heatmap(config["model"], device=device)
val_agent.loadModel()
patch_size = config["model"]["subpixel"]["patch_size"]

def resize_image(image, size):
    s = max(size /image.shape[:2])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image[:int(size[0]/s),:int(size[1]/s)]
    image = cv2.resize(image, (size[1], size[0]),
                             interpolation=cv2.INTER_AREA)
    return image

# Forward function
def get_pts_des_from_agent(val_agent, image, device="cpu"):
    """
    pts: numpy (N, 3)
    desc: numpy (N, 256)
    """
    image = image.astype('float32') / 255.0
    if image.ndim == 2:
        image = image[:,:, np.newaxis]

    heatmap_batch = val_agent.run(torch.from_numpy(image).permute(2,0,1).unsqueeze(0).to(device))
    pts = val_agent.heatmap_to_pts()
    pts = val_agent.soft_argmax_points(pts, patch_size=patch_size)
    desc_sparse = val_agent.desc_to_sparseDesc()
    assert(pts[0].shape[1] == desc_sparse[0].shape[1])
    return pts[0].transpose(), desc_sparse[0].transpose()

# NGRANSAC settings
dump_folder = '/projects/katefgroup/datasets/aidtr/processed/superglue_%s' % mod
dataset_name = "superglue_%s" % mod
variant = "train"
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

superglue_sample_count = 0
for root_dir in dir_to_process:
    util = aidtr_utils.AIDTR_UTILS(root_dir, dump_folder, out_image_size=out_image_size, id=root_dir.split('/')[-1])
    all_timesteps = util.get_all_timesteps()

    timestep_intervals = get_intervals(len(all_timesteps), S)

    for interval in timestep_intervals:
        for module in MODULE_LIST:
            rgb_cam_list = []
            rgb_imgname_list = []
            pix_T_cam_list = []
            cam_T_world_list = []

            timestep_cnt = 0

            for timestep_id in interval:
                timestep = all_timesteps[timestep_id]
                res, image_name = util.get_image_name(timestep, module, RGB_INDEX)
                if not res:
                    continue

                rgb_image = util.load_image(image_name)
                pix_T_cam = util.get_pix_T_cam(module, RGB_INDEX)
                cam_T_world = util.get_cam_T_world(timestep, module, RGB_INDEX)

                rgb_cam_list.append(rgb_image)
                rgb_imgname_list.append(image_name)
                pix_T_cam_list.append(pix_T_cam)
                cam_T_world_list.append(cam_T_world)

                timestep_cnt += 1

            if timestep_cnt < 2:
                print("not enough sequential images to have pairs")
                continue

            bf = cv2.BFMatcher()
            for i in range(timestep_cnt):
                image = cv2.imread(rgb_imgname_list[i])
                image = resize_image(image, np.array([480, 640]))

                width, height = image.shape[0], image.shape[1]

                corners = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)
                warp = np.random.randint(-224, 224, size=(4, 2)).astype(np.float32)

                M = cv2.getPerspectiveTransform(corners, corners + warp)
                warped_image = cv2.warpPerspective(src=image, M=M, dsize=(image.shape[1], image.shape[0])) # return an image type

                kp1, desc1 = get_pts_des_from_agent(val_agent, image, device=device)
                kp2, desc2 = get_pts_des_from_agent(val_agent, warped_image, device=device)

                kp1 = np.array([kp1])
                kp2 = np.array([kp2])
                desc1 = np.array([desc1])
                desc2 = np.array([desc2])

                assert(kp1.shape[1] == desc1.shape[1])
                assert(kp2.shape[1] == desc2.shape[1]) # line fails
                assert(image.shape == (480, 640))
                assert(warped_image.shape == (480, 640))

                #save data tensor and ground truth transformation
                np.save(out_dir + 'pair_superglue_{0}.npy'.format(superglue_sample_count), [
                    kp1.astype(np.float32),
                    kp2.astype(np.float32),
                    desc1.astype(np.float32),
                    desc2.astype(np.float32),
                    image,
                    warped_image,
                    M
                    ])

                superglue_sample_count += 1
                print(f"Collected image {superglue_sample_count}")



