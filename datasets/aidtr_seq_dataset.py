"""

"""

import numpy as n
import os
import cv2
from pathlib import Path

import torch
import torch.utils.data as data

# from .base_dataset import BaseDataset
# from .utils import pipeline
from utils.tools import dict_update

from models.homographies import sample_homography
from settings import DATA_PATH

from utils import aidtr_utils

from imageio import imread
import numpy as np

def load_as_float(path):
    return imread(path).astype(np.float32)/255

class AidtrSeqDataset(data.Dataset):
    default_config = {
        'dataset': 'aidtr_seq',  # or 'coco'
        'cache_in_memory': False,
        'preprocessing': {
            'resize': [240,320]
        }
    }

    def __init__(self, transform=None, **config):
        self.config = self.default_config
        self.config = dict_update(self.config, config)
        self.files = self._init_dataset(**self.config)
        sequence_set = []
        for (img1, img2, rel_pose) in zip(self.files['image1_paths'], self.files['image2_paths'], self.files['relative_poses']):
            sample = {'image1': img1, 'image2': img2, 'rel_pose': rel_pose}
            sequence_set.append(sample)
        self.samples = sequence_set
        self.transform = transform
        if config['preprocessing']['resize']:
            self.sizer = np.array(config['preprocessing']['resize'])
        pass

    def __getitem__(self, index):
        """

        :param index:
        :return:
            image:
                tensor (1,H,W)
            warped_image:
                tensor (1,H,W)
        """
        def _read_image(path):
            input_image = cv2.imread(path)
            return input_image

        def _preprocess(image):
            s = max(self.sizer /image.shape[:2])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = image[:int(self.sizer[0]/s),:int(self.sizer[1]/s)]
            image = cv2.resize(image, (self.sizer[1], self.sizer[0]),
                                     interpolation=cv2.INTER_AREA)
            image = image.astype('float32') / 255.0
            if image.ndim == 2:
                image = image[:,:, np.newaxis]
            if self.transform is not None:
                image = self.transform(image)
            return image

        sample = self.samples[index]
        image1 = _preprocess(_read_image(sample['image1']))
        image2 = _preprocess(_read_image(sample['image2']))
        rel_pose = sample['rel_pose']
        sample = {'image1': image1, 
                  'image2': image2,
                  'rel_pose': rel_pose}
        return sample

    def __len__(self):
        return len(self.samples)

    def _init_dataset(self, **config):
        def _get_intervals(range_max, S):
            res = []
            for s in range(S):
                res.append(range(range_max)[s::S])

            return list(zip(*res))

        # returns
        image1_paths = []
        image2_paths = []
        relative_poses = []

        # settings
        raw_data_dir = '/projects/katefgroup/datasets/aidtr/hideandseek/'
        dump_folder = '/projects/katefgroup/datasets/aidtr/processed/ngransac_aa'
        out_image_size = (512, 384)
        MODULE_LIST = (0, 1, 2, 3, 4) # module0 - module 4
        S = len(MODULE_LIST)
        RGB_INDEX = 2

        dir_to_process = [f.path for f in os.scandir(raw_data_dir) if f.is_dir()]

        for root_dir in dir_to_process:
            util = aidtr_utils.AIDTR_UTILS(root_dir, dump_folder, out_image_size=out_image_size, id=root_dir.split('/')[-1])
            all_timesteps = util.get_all_timesteps()

            timestep_intervals = _get_intervals(len(all_timesteps), S)

            for interval in timestep_intervals:
                for module in MODULE_LIST:
                    rgb_imgname_list = []
                    cam_T_world_list = []

                    timestep_cnt = 0

                    # get timesteps
                    for timestep_id in interval:
                        timestep = all_timesteps[timestep_id]
                        res, image_name = util.get_image_name(timestep, module, RGB_INDEX)
                        if not res:
                            continue

                        pix_T_cam = util.get_pix_T_cam(module, RGB_INDEX)
                        cam_T_world = util.get_cam_T_world(timestep, module, RGB_INDEX)

                        rgb_imgname_list.append(image_name)
                        cam_T_world_list.append(pix_T_cam)

                        timestep_cnt += 1 

                    if timestep_cnt < 2:
                        # not enough sequential images to have pairs
                        continue

                    # loop through timesteps to get pairs
                    for i in range(timestep_cnt - 1):
                        image1_paths.append(rgb_imgname_list[i])
                        image2_paths.append(rgb_imgname_list[i+1])
                        pose1 = cam_T_world_list[i]
                        pose2 = cam_T_world_list[i+1]
                        rel_pose = np.matmul(np.linalg.inv(pose1), pose2)
                        relative_poses.append(rel_pose)

        files = {'image1_paths': image1_paths,
                 'image2_paths': image2_paths,
                 'relative_poses': relative_poses}
        return files


# # Check the dataset
# import yaml
# with open("configs/superpoint_repeatability_heatmap.yaml", "r") as f:
#     config = yaml.load(f)
# test_set = AidtrSeqDataset(
#             transform=None,
#             **config['data'],
#         )
