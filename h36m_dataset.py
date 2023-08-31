from torch.utils.data import Dataset
import numpy as np
import data_utils
import torch
import os


class H36M_Dataset(Dataset):

    def __init__(self, data_dir, input_n, output_n, skip_rate, actions=None, split=0):
        """
        :param data_dir:
        :param actions:
        :param input_n:
        :param output_n:
        :param split: 0 train, 1 testing
        :param skip_rate:
        """
        self.path_to_data = os.path.join(data_dir, 'h3.6m/dataset')
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        self.sample_rate = 2
        self.p3d = {}
        self.data_idx = []
        seq_len = self.in_n + self.out_n
        subs = [[1, 6, 7, 8, 9], [11, 5]]
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

        subs = subs[split]
        key = 0
        for subj in subs:
            for action_idx in np.arange(len(acts)):
                action = acts[action_idx]
                for subact in [1, 2]:  # subactions
                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, subact)
                    the_sequence = data_utils.readCSVasFloat(filename)
                    n, d = the_sequence.shape
                    even_list = range(0, n, self.sample_rate)
                    num_frames = len(even_list)
                    the_sequence = np.array(the_sequence[even_list, :])
                    the_sequence = torch.from_numpy(the_sequence).float().cuda()
                    # remove global rotation and translation
                    the_sequence[:, 0:6] = 0
                    p3d = data_utils.expmap2xyz_torch(the_sequence)
                    # p3d = torch.nn.functional.normalize(p3d, p=2.0, dim=2)
                    self.p3d[key] = p3d.view(num_frames, -1).cpu().data.numpy()

                    valid_frames = np.arange(0, num_frames - seq_len + 1, skip_rate)

                    tmp_data_idx_1 = [key] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    key += 1

        joint_to_ignore = np.array([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])

        dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        self.dimensions_to_use = np.setdiff1d(np.arange(96), dimensions_to_ignore)
        joint_name_used = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "LeftUpLeg", "LeftLeg",
                           "LeftFoot", "Spine", "Spine1", "Neck", "Head", "LeftShoulder", "LeftArm",
                           "LeftForeArm", "RightShoulder", "RightArm",
                           "RightForeArm"]

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        return self.p3d[key][fs][:, self.dimensions_to_use]
