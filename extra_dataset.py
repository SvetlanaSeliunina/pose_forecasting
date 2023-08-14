from torch.utils.data import Dataset
import numpy as np
import data_utils
import torch
import os
import json

#Nose = 0,
#Neck = 1,
#RShoulder = 2,
#RElbow = 3,
#RWrist = 4,
#LShoulder = 5,
#LElbow = 6,
#LWrist = 7,
#MidHip = 8,
#RHip = 9,
#RKnee = 10,
#RAnkle = 11,
#LHip = 12,
#LKnee = 13,
#LAnkle = 14,
#REye = 15,
#LEye = 16,
#REar = 17,
#LEar = 18,
##Head = 19,  #unused
##Belly = 20, #unused
##LBToe = 21, #unused
##LSToe = 22, #unused
##LHeel = 23, #unused
##RBToe = 24, #unused
##RSToe = 25, #unused
##RHeel = 26, #unused
##NUM_KEYPOINTS = 27;

KPS = [8, 9, 10, 11, 12, 13, 14, 20, 1, 0, 19, 5, 6, 7, 2, 3, 4]
NUM_KPS_USED = 17

joint_name_used = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "LeftUpLeg", "LeftLeg",
                      "LeftFoot", "Spine", "Spine1", "Neck", "Head", "LeftShoulder", "LeftArm",
                      "LeftForeArm", "RightShoulder", "RightArm",
                      "RightForeArm"]

class extra_Dataset(Dataset):

    def __init__(self,data_dir,input_n,output_n,skip_rate):
        """
        :param path_to_data:
        :param input_n:
        :param output_n:
        :param sample_rate:
        """
        self.path_to_data = os.path.join(data_dir,'VisionLabSS23_3DPoses')
        self.in_n = input_n
        self.out_n = output_n
        self.sample_rate = 2
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
                        if (KPS[kp_idx] == 19):
                            all_xs.append((kps[17]['pos'][0] + kps[18]['pos'][0])/2)
                            all_ys.append((kps[17]['pos'][1] + kps[18]['pos'][1])/2)
                            all_zs.append((kps[17]['pos'][2] + kps[18]['pos'][2])/2)
                        elif (KPS[kp_idx] == 20):
                            all_xs.append((kps[8]['pos'][0] + kps[1]['pos'][0]) / 2)
                            all_ys.append((kps[8]['pos'][1] + kps[1]['pos'][1]) / 2)
                            all_zs.append((kps[8]['pos'][2] + kps[1]['pos'][2]) / 2)
                        else:
                            kp = kps[KPS[kp_idx]]
                            all_xs.append(kp['pos'][0])
                            all_ys.append(kp['pos'][1])
                            all_zs.append(kp['pos'][2])

                xs = np.array(all_xs) / np.linalg.norm(np.array(all_xs))
                ys = np.array(all_ys) / np.linalg.norm(np.array(all_ys))
                zs = np.array(all_zs) / np.linalg.norm(np.array(all_zs))
                all = np.array((xs, ys, zs)).T
                all = all.reshape(-1,17*3)
                n, d = all.shape
                even_list = range(0, n, self.sample_rate)
                num_frames = len(even_list)
                all = np.array(all[even_list, :])
                self.p3d[key] = all;
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
        #print (self.p3d[key][fs][:,self.dimensions_to_use].shape)
        return self.p3d[key][fs]