import pandas as pd
import numpy as np
import data_utils
import torch
import os

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def logic(index, sample_rate):
    if index % sample_rate == 0:
       return True
    return False

class H36M_Unit_Vector_Skeleton_Dataset(Dataset):

    def __init__(self,data_dir,input_n,output_n,skip_rate, actions=None, split=0, transform=None):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing
        :param sample_rate:
        """
        self.path_to_data = os.path.join(data_dir,'h3.6m/dataset')
        self.split = split
        self.transform = transform
        self.in_n = input_n
        self.out_n = output_n
        self.sample_rate = 2
        self.p3d = {}
        self.data_idx = []
        seq_len = self.in_n + self.out_n
        subs = [[1, 6, 7, 8, 9], [11, 5], [5]]

        if actions is None:
            acts = ["walking", "eating", "smoking", "discussion", "directions",
                    "greeting", "phoning", "posing", "purchases", "sitting",
                    "sittingdown", "takingphoto", "waiting", "walkingdog",
                    "walkingtogether"]
        else:
            acts = actions
            
        # 32 human3.6 joint name:
        joint_name = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "Site", "LeftUpLeg", "LeftLeg",
                      "LeftFoot",
                      "LeftToeBase", "Site", "Spine", "Spine1", "Neck", "Head", "Site", "LeftShoulder", "LeftArm",
                      "LeftForeArm",
                      "LeftHand", "LeftHandThumb", "Site", "L_Wrist_End", "Site", "RightShoulder", "RightArm",
                      "RightForeArm",
                      "RightHand", "RightHandThumb", "Site", "R_Wrist_End", "Site"]

        joints_to_ignore = np.array([4,5,9,10,11,16,20,21,22,23,24,28,29,30,31])   
        joints_to_use = np.setdiff1d(np.arange(32), joints_to_ignore)
        parents = [None, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

        subs = subs[split]
        key = 0
        for subj in subs:
            for action_idx in np.arange(len(acts)):
                action = acts[action_idx]
                for subact in [1, 2]:  # subactions
                    
                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, subact)
                    
                    df = pd.read_csv(filename, header=None, 
                     skiprows=lambda x: logic(x, self.sample_rate))
    
                    df.iloc[:, 0:6] = 0
                
                    tensor = torch.tensor(df.values).float().cuda()
                    tensor = data_utils.expmap2xyz_torch(tensor)
                
                    tensor = tensor[:, joints_to_use, :]
                
                    norm_sequence = []
                
                    for frame in tensor:
                
                        xs = frame[:, 0]
                        ys = frame[:, 1]
                        zs = frame[:, 2]
                
                        joints = list(enumerate(zip(xs, ys, zs, parents)))
                
                        norm_coordinates = []
                        
                        for joint in joints:
                            
                            id, (x, y, z, parent) = joint
                        
                            if parent == None:
                                norm_coordinates.extend([0, 0, 0])
                            else:            
                                p_id, (p_x, p_y, p_z, _) = joints[parent]
                
                                delta_x = (x - p_x)
                                delta_y = (y - p_y)
                                delta_z = (z - p_z)
                
                                length = torch.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
                                
                                norm_delta_x = delta_x/length
                                norm_delta_y = delta_y/length
                                norm_delta_z = delta_z/length
                
                                norm_coordinates.append(norm_coordinates[parent*tensor.shape[2] + 0] + norm_delta_x/2)
                                norm_coordinates.append(norm_coordinates[parent*tensor.shape[2] + 1] + norm_delta_y/2)
                                norm_coordinates.append(norm_coordinates[parent*tensor.shape[2] + 2] + norm_delta_z/2)
                                                        
                        norm_sequence.append(norm_coordinates)
                
                    tensor = torch.Tensor(norm_sequence)
                
                    self.p3d[key] = tensor.view(df.shape[0], -1).cpu().numpy()
                
                    valid_frames = np.arange(0, df.shape[0] - seq_len + 1, skip_rate)
                
                    tmp_data_idx_1 = [key] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    key += 1

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        sample = self.p3d[key][fs]
        if self.transform:
            sample = self.transform(sample)
        return sample