import numpy as np
import torch
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt

def _some_variables():
    """
    borrowed from
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/forward_kinematics.py#L100

    We define some variables that are useful to run the kinematic tree

    Args
      None
    Returns
      parent: 32-long vector with parent-child relationships in the kinematic tree
      offset: 96-long vector with bone lenghts
      rotInd: 32-long list with indices into angles
      expmapInd: 32-long list with indices into expmap angles
    """

    parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
                       17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31]) - 1

    offset = np.array(
        [0.000000, 0.000000, 0.000000, -132.948591, 0.000000, 0.000000, 0.000000, -442.894612, 0.000000, 0.000000,
         -454.206447, 0.000000, 0.000000, 0.000000, 162.767078, 0.000000, 0.000000, 74.999437, 132.948826, 0.000000,
         0.000000, 0.000000, -442.894413, 0.000000, 0.000000, -454.206590, 0.000000, 0.000000, 0.000000, 162.767426,
         0.000000, 0.000000, 74.999948, 0.000000, 0.100000, 0.000000, 0.000000, 233.383263, 0.000000, 0.000000,
         257.077681, 0.000000, 0.000000, 121.134938, 0.000000, 0.000000, 115.002227, 0.000000, 0.000000, 257.077681,
         0.000000, 0.000000, 151.034226, 0.000000, 0.000000, 278.882773, 0.000000, 0.000000, 251.733451, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999627, 0.000000, 100.000188, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 257.077681, 0.000000, 0.000000, 151.031437, 0.000000, 0.000000, 278.892924,
         0.000000, 0.000000, 251.728680, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999888,
         0.000000, 137.499922, 0.000000, 0.000000, 0.000000, 0.000000])
    offset = offset.reshape(-1, 3)

    rotInd = [[5, 6, 4],
              [8, 9, 7],
              [11, 12, 10],
              [14, 15, 13],
              [17, 18, 16],
              [],
              [20, 21, 19],
              [23, 24, 22],
              [26, 27, 25],
              [29, 30, 28],
              [],
              [32, 33, 31],
              [35, 36, 34],
              [38, 39, 37],
              [41, 42, 40],
              [],
              [44, 45, 43],
              [47, 48, 46],
              [50, 51, 49],
              [53, 54, 52],
              [56, 57, 55],
              [],
              [59, 60, 58],
              [],
              [62, 63, 61],
              [65, 66, 64],
              [68, 69, 67],
              [71, 72, 70],
              [74, 75, 73],
              [],
              [77, 78, 76],
              []]

    expmapInd = np.split(np.arange(4, 100) - 1, 32)

    return parent, offset, rotInd, expmapInd

def expmap2rotmat_torch(r):
    """
    Converts expmap matrix to rotation
    batch pytorch version ported from the corresponding method above
    :param r: N*3
    :return: N*3*3
    """
    theta = torch.norm(r, 2, 1)
    r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
    r1 = torch.zeros_like(r0).repeat(1, 3)
    r1[:, 1] = -r0[:, 2]
    r1[:, 2] = r0[:, 1]
    r1[:, 5] = -r0[:, 0]
    r1 = r1.view(-1, 3, 3)
    r1 = r1 - r1.transpose(1, 2)
    n = r1.data.shape[0]
    R = torch.eye(3, 3).repeat(n, 1, 1).float().cuda() + torch.mul(
        torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1) + torch.mul(
        (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)), torch.matmul(r1, r1))
    return R


def fkl_torch(angles, parent, offset, rotInd, expmapInd):
    """
    pytorch version of fkl.

    convert joint angles to joint locations
    batch pytorch version of the fkl() method above
    :param angles: N*99
    :param parent:
    :param offset:
    :param rotInd:
    :param expmapInd:
    :return: N*joint_n*3
    """
    n = angles.data.shape[0]
    j_n = offset.shape[0]
    p3d = Variable(torch.from_numpy(offset)).float().cuda().unsqueeze(0).repeat(n, 1, 1)
    angles = angles[:, 3:].contiguous().view(-1, 3)
    R = expmap2rotmat_torch(angles).view(n, j_n, 3, 3)
    for i in np.arange(1, j_n):
        if parent[i] > 0:
            R[:, i, :, :] = torch.matmul(R[:, i, :, :], R[:, parent[i], :, :]).clone()
            p3d[:, i, :] = torch.matmul(p3d[0, i, :], R[:, parent[i], :, :]) + p3d[:, parent[i], :]
    return p3d

class Ax3DPose(object):
  def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Create a 3d pose visualizer that can be updated with new poses.

    Args
      ax: 3d axis to plot the 3d pose on
      lcolor: String. Colour for the left part of the body
      rcolor: String. Colour for the right part of the body
    """

    # Start and endpoints of our representation
    self.I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1
    self.J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1
    # Left / right indicator
    self.LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    self.ax = ax

    vals = np.zeros((32, 3))

    # Make connection matrix
    self.plots = []
    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots.append(self.ax.plot(x, y, z, lw=2, c=lcolor if self.LR[i] else rcolor))

    self.ax.set_xlabel("x")
    self.ax.set_ylabel("y")
    self.ax.set_zlabel("z")

  def update(self, channels, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Update the plotted 3d pose.

    Args
      channels: 96-dim long np array. The pose to plot.
      lcolor: String. Colour for the left part of the body.
      rcolor: String. Colour for the right part of the body.
    Returns
      Nothing. Simply updates the axis with the new pose.
    """

    assert channels.size == 96, "channels should have 96 entries, it has %d instead" % channels.size
    vals = np.reshape( channels, (32, -1) )

    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots[i][0].set_xdata(x)
      self.plots[i][0].set_ydata(y)
      self.plots[i][0].set_3d_properties(z)
      self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)

    r = 750;
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    self.ax.set_xlim3d([-r+xroot, r+xroot])
    self.ax.set_zlim3d([-r+zroot, r+zroot])
    self.ax.set_ylim3d([-r+yroot, r+yroot])

    self.ax.set_aspect('equal')

def main():
    # Load all the data
    parent, offset, rotInd, expmapInd = _some_variables()

    # numpy implementation
    # with h5py.File('samples.h5', 'r') as h5f:
    #     expmap_gt = h5f['expmap/gt/walking_0'][:]
    #     expmap_pred = h5f['expmap/preds/walking_0'][:]
    expmap_pred = np.array(
        [0.0000000, 0.0000000, 0.0000000, -0.0000001, -0.0000000, -0.0000002, 0.3978439, -0.4166636, 0.1027215,
         -0.7767256, -0.0000000, -0.0000000, 0.1704115, 0.3078358, -0.1861640, 0.3330379, -0.0000000, -0.0000000,
         -0.0000000, -0.0000000, -0.0000000, 0.0679339, 0.2255526, 0.2394881, -0.0989492, -0.0000000, -0.0000000,
         0.0677801, -0.3607298, 0.0503249, 0.1819232, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000,
         0.3236777, -0.0476493, -0.0651256, -0.3150051, -0.0665669, 0.3188994, -0.5980227, -0.1190833, -0.3017127,
         1.2270271, -0.1010960, 0.2072986, -0.0000000, -0.0000000, -0.0000000, -0.2578378, -0.0125206, 2.0266378,
         -0.3701521, 0.0199115, 0.5594162, -0.4625384, -0.0000000, -0.0000000, 0.1653314, -0.3952765, -0.1731570,
         -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000,
         -0.0000000, -0.0000000, -0.0000000, 2.7825687, -1.4196042, -0.0936858, -1.0348599, -2.7419815, 0.4518218,
         -0.3902033, -0.0000000, -0.0000000, 0.0597317, 0.0547002, 0.0445105, -0.0000000, -0.0000000, -0.0000000,
         -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000
         ])
    expmap_gt = np.array(
        [0.2240568, -0.0276901, -0.7433901, 0.0004407, -0.0020624, 0.0002131, 0.3974636, -0.4157083, 0.1030248,
         -0.7762963, -0.0000000, -0.0000000, 0.1697988, 0.3087364, -0.1863863, 0.3327336, -0.0000000, -0.0000000,
         -0.0000000, -0.0000000, -0.0000000, 0.0689423, 0.2282812, 0.2395958, -0.0998311, -0.0000000, -0.0000000,
         0.0672752, -0.3615943, 0.0505299, 0.1816492, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000,
         0.3223563, -0.0481131, -0.0659720, -0.3145134, -0.0656419, 0.3206626, -0.5979006, -0.1181534, -0.3033383,
         1.2269648, -0.1011873, 0.2057794, -0.0000000, -0.0000000, -0.0000000, -0.2590978, -0.0141497, 2.0271597,
         -0.3699318, 0.0128547, 0.5556172, -0.4714990, -0.0000000, -0.0000000, 0.1603251, -0.4157299, -0.1667608,
         -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000,
         -0.0000000, -0.0000000, -0.0000000, 2.7811005, -1.4192915, -0.0932141, -1.0294687, -2.7323222, 0.4542309,
         -0.4048152, -0.0000000, -0.0000000, 0.0568960, 0.0525994, 0.0493068, -0.0000000, -0.0000000, -0.0000000,
         -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000
         ])

    exp1 = Variable(torch.from_numpy(np.vstack((expmap_pred, expmap_gt))).float()).cuda()
    xyz = fkl_torch(exp1, parent, offset, rotInd, expmapInd)
    xyz = xyz.cpu().data.numpy()
    # === Plot and animate ===
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ob = Ax3DPose(ax)
    # Plot the conditioning ground truth
    ob.update(xyz[0, :])
    plt.show(block=False)
    fig.canvas.draw()
    plt.pause(10)

    # Plot the prediction
    ob.update(xyz[1, :], lcolor="#9b59b6", rcolor="#2ecc71")
    plt.show(block=False)
    fig.canvas.draw()
    plt.pause(10)


if __name__ == '__main__':
    main()
