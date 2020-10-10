import csv
import os
import re
from os import listdir
from os.path import isfile, join

import imageio
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

import cv2


class AIDTR_UTILS(object):
    def __init__(self, root_dir, dump_folder, out_image_size, id='0000'):
        self.root_dir = root_dir
        self.out_image_size = out_image_size
        self.id = id

        # self.raw_img_size = (1024.0, 750.0)
        # self.scale_factor_x = self.out_image_size[0] / self.raw_img_size[0]
        # self.scale_factor_y = self.out_image_size[1] / self.raw_img_size[1]

        self.EXTRINSIC_FILE_NAME = 'camera_to_local_poses.csv'
        self.IMAGE_FOLDER = 'rectified'
        self.INTRINSICS_FOLDER = 'intrinsics'

        self.dump_folder = dump_folder
        if not os.path.exists(dump_folder):
            os.makedirs(dump_folder)

        self.extrinsics_raw_data = self.load_csv(join(root_dir, self.EXTRINSIC_FILE_NAME)) # a numpy array with string, 120 x 9
        self.image_list = self.list_images(join(self.root_dir, self.IMAGE_FOLDER))


    ### io related

    def load_image(self, image_file_name):
        image = imageio.imread(image_file_name)
        image = cv2.resize(image, dsize=self.out_image_size)

        return image

    def load_csv(self, csv_file_name):
        with open(csv_file_name, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
            data = np.array(data)

        return data[1:] # first column is column name

    def get_all_timesteps(self):
        return np.unique(self.extrinsics_raw_data[:, 0])

    def get_cam_name(self, m, l):
        return "camera_" + "module" + str(m) + "_lens" + str(l) + "_rect"

    def get_intrinsics_file_name(self, m, l):
        return join(self.root_dir, self.INTRINSICS_FOLDER, "camera_" + "module" + str(m) + "_lens" + str(l) + ".yaml")

    def get_pix_T_cam(self, m, l):
        intrinsics_file_name = self.get_intrinsics_file_name(m, l)
        with open(intrinsics_file_name) as f:
            raw_data = yaml.load(f, Loader=yaml.FullLoader)
            intrinsics = np.reshape(raw_data['projection_matrix']['data'], (3, 4))
            image_width = raw_data['image_width'] # 1024
            image_height = raw_data['image_height'] # 750

        pix_T_cam = intrinsics[:, 0:3]

        scale_factor_x = self.out_image_size[0] / image_width
        scale_factor_y = self.out_image_size[1] / image_height

        # rescale here
        pix_T_cam[0, :] = scale_factor_x * pix_T_cam[0, :]
        pix_T_cam[1, :] = scale_factor_y * pix_T_cam[1, :]

        pix_T_cam_4x4 = np.eye(4)
        pix_T_cam_4x4[0:3, 0:3] = pix_T_cam

        return pix_T_cam_4x4

    def get_rect_T_cam(self, m, l):
        intrinsics_file_name = self.get_intrinsics_file_name(m, l)
        with open(intrinsics_file_name) as f:
            raw_data = yaml.load(f, Loader=yaml.FullLoader)
            rectification_matrix = np.reshape(raw_data['rectification_matrix']['data'], (3, 3))

        rectification_matrix_4x4 = np.eye(4)
        rectification_matrix_4x4[0:3, 0:3] = rectification_matrix

        return rectification_matrix_4x4

    def list_images(self, image_dir):
        image_files_list = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]

        return image_files_list

    def get_image_name(self, timestep, m, l):
        name_to_comp = 'p0' + 'm' + str(m) + 'l' + str(l) + '.*' + timestep + '.png'
        for name in self.image_list:
            res = re.match(name_to_comp, name)
            if res:
                return True, os.path.join(self.root_dir, self.IMAGE_FOLDER, name)

        print('warning: {} not found'.format(name_to_comp))
        return False, None

        # assert False, (name_to_comp, 'no such a file')

    def get_extrinsics_raw(self, timestep, m, l):
        cam_name = self.get_cam_name(m, l)
        for rows in self.extrinsics_raw_data:
            if rows[0] == timestep and rows[1] == cam_name:
                return rows

        assert False, 'no such a file'

    def get_cam_T_world(self, timestep, m, l):
        world_T_cam = np.eye(4)
        raw_data = self.get_extrinsics_raw(timestep, m, l)
        r, t = self.get_rt_from_raw(raw_data)

        world_T_cam[0:3, 0:3] = r
        world_T_cam[0:3, 3] = t

        cam_T_world = np.linalg.inv(world_T_cam)

        return cam_T_world

    def generate_dump_file_name(self, timestep, m, l0, l1):
        return join(self.dump_folder, self.id + '_' + timestep + '_module' + str(m) + '_l_' + str(l0) + '_' + str(l1) + '_'
            + str(self.out_image_size[0]) + 'x' + str(self.out_image_size[1]) + '.npz')

    def dump_dict(self, dict_to_save, timestep, m, l0, l1, verbose=True):
        np.savez(self.generate_dump_file_name(timestep, m, l0, l1), **dict_to_save)
        if verbose:
            print('dumped {}'.format(self.generate_dump_file_name(timestep, m, l0, l1)))
        return True

    def generate_seq_dump_file_name(self, S, m, l0, l1, timestep):
        return join(self.dump_folder, self.id + '_S' + str(S) + '_module' + str(m) + '_' + str(timestep) + '_l_' + str(l0) + '_' + str(l1) + '_'
            + str(self.out_image_size[0]) + 'x' + str(self.out_image_size[1]) + '.npz')

    def dump_seq_dict(self, dict_to_save, S, m, l0, l1, timestep, verbose=True):
        np.savez(self.generate_seq_dump_file_name(S, m, l0, l1, timestep), **dict_to_save)
        if verbose:
            print('dumped {}'.format(self.generate_seq_dump_file_name(S, m, l0, l1, timestep)))
        return True

    def generate_multiview_dump_file_name(self, S, l0, l1, timestep):
        return join(self.dump_folder, self.id + '_S' + str(S) + '_' + str(timestep) + '_l_' + str(l0) + '_' + str(l1) + '_'
            + str(self.out_image_size[0]) + 'x' + str(self.out_image_size[1]) + '.npz')

    def dump_multiview_dict(self, dict_to_save, S, l0, l1, timestep, verbose=True):
        np.savez(self.generate_multiview_dump_file_name(S, l0, l1, timestep), **dict_to_save)
        if verbose:
            print('dumped {}'.format(self.generate_multiview_dump_file_name(S, l0, l1, timestep)))
        return True

    ### io related end ###

    ### geometry related ###

    def get_rt_from_raw(self, raw):
        assert len(raw) == 9
        t = np.array(raw[2:5]).astype(float)
        quat = np.array(raw[5:]).astype(float)
        r = R.from_quat(quat).as_matrix()

        return r, t

    def get_axis_from_rt(self, r, t):
        x_axis = np.array([t, t+r[:,0]]).transpose() # each 2 x 3
        y_axis = np.array([t, t+r[:,1]]).transpose()
        z_axis = np.array([t, t+r[:,2]]).transpose()

        return x_axis, y_axis, z_axis

    ### geometry related end ###

    ### visualization related ###





# images = []
# for filename in filenames:
#     images.append(imageio.imread(filename))
# imageio.mimsave('/path/to/movie.gif', images)
