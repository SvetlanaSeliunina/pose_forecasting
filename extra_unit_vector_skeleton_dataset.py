import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset

KPS = [8, 9, 10, 11, 12, 13, 14, 20, 1, 0, 19, 5, 6, 7, 2, 3, 4]
NUM_KPS_USED = 17

joint_name_used = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "LeftUpLeg", "LeftLeg",
                   "LeftFoot", "Spine", "Spine1", "Neck", "Head", "LeftShoulder", "LeftArm",
                   "LeftForeArm", "RightShoulder", "RightArm",
                   "RightForeArm"]


class extra_Unit_Vector_Skeleton_Dataset(Dataset):

    def __init__(self, data_dir, input_n, output_n, skip_rate, transform=None):
        """
        :param data_dir:
        :param input_n:
        :param output_n:
        :param skip_rate:
        """
        self.path_to_data = os.path.join(data_dir, 'VisionLabSS23_3DPoses')
        self.in_n = input_n
        self.out_n = output_n
        self.sample_rate = 2
        self.transform = transform
        self.p3d = {}
        self.data_idx = []
        seq_len = self.in_n + self.out_n
        # 32 human3.6 joint name:
        joint_name = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "Site", "LeftUpLeg", "LeftLeg",
                      "LeftFoot",
                      "LeftToeBase", "Site", "Spine", "Spine1", "Neck", "Head", "Site", "LeftShoulder", "LeftArm",
                      "LeftForeArm",
                      "LeftHand", "LeftHandThumb", "Site", "L_Wrist_End", "Site", "RightShoulder", "RightArm",
                      "RightForeArm",
                      "RightHand", "RightHandThumb", "Site", "R_Wrist_End", "Site"]

        parents = [None, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

        key = 0

        directory = self.path_to_data

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".json"):
                name = os.path.join(directory, filename)
                print("Reading ", name)
                with open(name, 'r') as f:
                    pose_data = json.load(f)

                all_xs = []
                all_ys = []
                all_zs = []

                for frame in pose_data:
                    kps = frame['person']['keypoints']

                    for kp_idx in range(NUM_KPS_USED):
                        if KPS[kp_idx] == 19:
                            all_xs.append((kps[17]['pos'][0] + kps[18]['pos'][0]) / 2 - kps[8]['pos'][0])
                            all_ys.append((kps[17]['pos'][1] + kps[18]['pos'][1]) / 2 - kps[8]['pos'][1])
                            all_zs.append((kps[17]['pos'][2] + kps[18]['pos'][2]) / 2 - kps[8]['pos'][2])
                        elif KPS[kp_idx] == 20:
                            all_xs.append((kps[8]['pos'][0] + kps[1]['pos'][0]) / 2 - kps[8]['pos'][0])
                            all_ys.append((kps[8]['pos'][1] + kps[1]['pos'][1]) / 2 - kps[8]['pos'][1])
                            all_zs.append((kps[8]['pos'][2] + kps[1]['pos'][2]) / 2 - kps[8]['pos'][2])
                        else:
                            kp = kps[KPS[kp_idx]]
                            all_xs.append(kp['pos'][0] - kps[8]['pos'][0])
                            all_ys.append(kp['pos'][1] - kps[8]['pos'][1])
                            all_zs.append(kp['pos'][2] - kps[8]['pos'][2])

                all = np.array((all_xs, all_zs, all_ys), np.float32).T
                all = all.reshape(-1, 17 * 3)
                n, d = all.shape
                even_list = range(0, n, self.sample_rate)
                num_frames = len(even_list)
                all = np.array(all[even_list, :])

                all = all.reshape(-1, 17, 3)
                norm_sequence = []

                for frame in all:

                    xs = frame[:, 0]
                    ys = frame[:, 1]
                    zs = frame[:, 2]

                    joints = list(enumerate(zip(xs, ys, zs, parents)))

                    norm_coordinates = []

                    for joint in joints:

                        id, (x, y, z, parent) = joint

                        if parent is None:
                            norm_coordinates.extend([0, 0, 0])
                        else:
                            p_id, (p_x, p_y, p_z, _) = joints[parent]

                            delta_x = (x - p_x)
                            delta_y = (y - p_y)
                            delta_z = (z - p_z)

                            length = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)

                            norm_delta_x = delta_x / length
                            norm_delta_y = delta_y / length
                            norm_delta_z = delta_z / length

                            norm_coordinates.append(norm_coordinates[parent * all.shape[2] + 0] + norm_delta_x / 2)
                            norm_coordinates.append(norm_coordinates[parent * all.shape[2] + 1] + norm_delta_y / 2)
                            norm_coordinates.append(norm_coordinates[parent * all.shape[2] + 2] + norm_delta_z / 2)

                    norm_sequence.append(norm_coordinates)

                all = torch.Tensor(norm_sequence).numpy()

                self.p3d[key] = all
                valid_frames = np.arange(0, num_frames - seq_len + 1, skip_rate)
                tmp_data_idx_1 = [key] * len(valid_frames)
                tmp_data_idx_2 = list(valid_frames)
                self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                key += 1

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        # print (self.p3d[key][fs][:,self.dimensions_to_use].shape)
        sample = self.p3d[key][fs]
        if self.transform:
            sample = self.transform(sample)
        return sample
