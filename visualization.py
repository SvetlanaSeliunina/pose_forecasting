%matplotlib notebook
#%matplotlib widget

%load_ext autoreload
%autoreload 2

import pandas as pd # For Variable inspector

import torch
import os
from h36m_dataset import H36M_Dataset
from extra_dataset import extra_Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import matplotlib.animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from IPython.display import HTML

# This function works with %matplotlib widget, but not with %matplotlib notebook
def FrameVisualization(frame):
    
    frame = frame.numpy()
    
    fig = plt.figure(figsize = (10, 10))
    ax = plt.axes(projection ="3d")
    ax.set_aspect('equal', adjustable='box')
    
    plt.xlabel("x")
    plt.ylabel("z")
    plt.xlabel("y")

    coordinates = frame.reshape(17, 3)
    
    xs = coordinates[:, 0]
    ys = coordinates[:, 1]
    zs = coordinates[:, 2]

    ###############################################
    # child id 0 - hip <==> no parent
    
    # child id 1 - rhip <==> parent id = 0 - hip
    # child id 2 - rknee <==> parent id = 1 - rhip
    # child id 3 - rfoot <==> parent id = 2 - rknee

    # child id 4 - lhip <==> parent id = 0 - hip
    # child id 5 - lknee <==> parent id = 4 - lhip
    # child id 6 - lhip <==> parent id = 5 - lfoot

    # child id 7 - spine <==> parent id = 0 - hip
    # child id 8 - thorax <==> parent id = 7 - spine
    # child id 9 - neck <==> parent id = 8 - thorax
    # child id 10 - head <==> parent id = 9 - neck

    # child id 11 - lshoulder <==> parent id = 8 - thorax
    # child id 12 - lelbow <==> parent id = 11 - lshoulder
    # child id 13 - lwrist <==> parent id = 12 - lelbow

    # child id 14 - rshoulder <==> parent id = 8 - thorax
    # child id 15 - relbow <==> parent id = 14 - rsgoulder
    # child id 16 - rwrist <==> parent id = 15 - rwrist
    ################################################

    parents = [None, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

    points = list(enumerate(zip(xs, ys, zs, parents)))

    #ax = plt.gca()
    #ax.cla()

    ax.scatter(xs, zs, ys, color = "green")
    #markers = ["${}$".format(id) for id in range(len(points))]
    #ax.plot(xs, zs, ys, linestyle="", marker=markers, color="green")
    
    for point in points:
        
        id, (x, y, z, p) = point
        
        # This was done for debug purposes, but is drawn slowly
        #ax.scatter(x, z, y, color = "green", marker=r"$ {} $".format(id), s=150)
        #ax.plot(x, z, y, linestyle="", marker="${}$".format(id), color="green")

        if p == None:
            continue
        
        p_id, (p_x, p_y, p_z, _) = points[p]

        link_xs = [x, p_x]
        link_ys = [y, p_y]
        link_zs = [z, p_z]

        if id in [1, 2, 3, 14, 15, 16]:
            color = "red"
        elif id in [4, 5, 6, 11, 12, 13]:
            color = "blue"
        else:
            color = "black"
        
        ax.plot(link_xs, link_zs, link_ys, color = color)

        def init():
    
    global frame_num
    frame_num = 0
    
# This is the function that is used for animation using matplotlib.animation
def FrameUpdate(frame, lines):

    global frame_num
    
    xs = frame[:, 0]
    ys = frame[:, 1]
    zs = frame[:, 2]

    joints = list(enumerate(zip(xs, ys, zs, parents)))

    axd = plt.gcf().axes
    
    for i, ax in enumerate(axd):
        for joint in joints:
            
            id, (x, y, z, parent) = joint
        
            if parent == None:
                #lines[id].set_data([x], [z])
                #lines[id].set_3d_properties([y])
                continue
            
            p_id, (p_x, p_y, p_z, _) = joints[parent]
            #print(id, p_id)
            link_xs = [x, p_x]
            link_ys = [y, p_y]
            link_zs = [z, p_z]
        
            lines[i*(len(joints)+1) + id].set_data(link_xs, link_zs)
            lines[i*(len(joints)+1) + id].set_3d_properties(link_ys)
        
        lines[i*(len(joints)+1) + 17].set_data(xs, zs)
        lines[i*(len(joints)+1) + 17].set_3d_properties(ys)
        if frame_num >= 10:
            #lines[-1].set(color = "orange")
            lines[i*(len(joints)+1) + 17].set_fillstyle('none')

    frame_num += 1
        
    return lines,

def SequenceAnimation(sequence):
    # We receive a tensor of size (20, 51)
    # We convert the line of size 51 to an array of 17 joints with 3 coordinates for each of 20 frames
    sequence = sequence.numpy().reshape(20, 17, 3)

    ##############################################
    
    def annotate_axes(ax, text, fontsize=18):
        ax.text(x=0.5, y=0.5, z=0.5, s=text,
                va="center", ha="center", fontsize=fontsize, color="black")

    # (plane, (elev, azim, roll))
    views = [('XY',   (90, -90, 0)),
             ('ISO',  (30, 30, 0)),
             ('XZ',    (0, -90, 0)),
             ('YZ',    (0,   0, 0))]
    
    layout = [['XY', 'ISO'],
              ['XZ', 'YZ']]
    fig, axd = plt.subplot_mosaic(layout, subplot_kw={'projection': '3d'},
                                  figsize=(12, 12))
    fig.tight_layout()
    
    for plane, angles in views:
       
        axd[plane].set(xlim3d=(-1, 1), xlabel='X')
        axd[plane].set(ylim3d=(-1, 1), ylabel='Y')
        axd[plane].set(zlim3d=(-1, 1), zlabel='Z')
    
        axd[plane].set_proj_type('ortho')
        axd[plane].view_init(elev=angles[0], azim=angles[1], roll=angles[2])
        axd[plane].set_box_aspect(None, zoom=2)
        axd[plane].set_aspect('equal', adjustable='box')
    
        label = f'{plane}\n{angles}'
        annotate_axes(axd[plane], label, fontsize=14)
    
    axd['XY'].set_zticklabels([])
    axd['XY'].set_zlabel('')

    axd['XZ'].set_yticklabels([])
    axd['XZ'].set_ylabel('')

    axd['YZ'].set_xticklabels([])
    axd['YZ'].set_xlabel('')
    
    #label = 'mplot3d primary view planes\n' + 'ax.view_init(elev, azim, roll)'
    #annotate_axes(axd['XYZ'], label, fontsize=18)
    #axd['ISO'].set_axis_off()
    axd['ISO'].set_proj_type('persp', focal_length=1)
    
    ##############################################

    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")

    # ax.set_aspect('equal', adjustable='box')
    # plt.title("Sequence Visualization mpl.animation")
    
    # ax.set(xlim3d=(-1, 1), xlabel='X')
    # ax.set(ylim3d=(-1, 1), ylabel='Y')
    # ax.set(zlim3d=(-1, 1), zlabel='Z')

    ###############################################
    # child id 0 - hip <==> no parent
    
    # child id 1 - rhip <==> parent id = 0 - hip
    # child id 2 - rknee <==> parent id = 1 - rhip
    # child id 3 - rfoot <==> parent id = 2 - rknee

    # child id 4 - lhip <==> parent id = 0 - hip
    # child id 5 - lknee <==> parent id = 4 - lhip
    # child id 6 - lhip <==> parent id = 5 - lfoot

    # child id 7 - spine <==> parent id = 0 - hip
    # child id 8 - thorax <==> parent id = 7 - spine
    # child id 9 - neck <==> parent id = 8 - thorax
    # child id 10 - head <==> parent id = 9 - neck

    # child id 11 - lshoulder <==> parent id = 8 - thorax
    # child id 12 - lelbow <==> parent id = 11 - lshoulder
    # child id 13 - lwrist <==> parent id = 12 - lelbow

    # child id 14 - rshoulder <==> parent id = 8 - thorax
    # child id 15 - relbow <==> parent id = 14 - rsgoulder
    # child id 16 - rwrist <==> parent id = 15 - rwrist
    
    global parents
    parents = [None, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    
    ################################################

    # Here we define the list of lines for the visualization
    lines = []
    # First 17 lines are links of the skeleton.
    # Here we define the colors for right, left and hips-head links.
    for plane, angles in views:
        for id in range(sequence.shape[1]):
           if id in [1, 2, 3, 14, 15, 16]:
               color = "red"
           elif id in [4, 5, 6, 11, 12, 13]:
               color = "blue"
           else:
               color = "black"
           lines.append(axd[plane].plot([], [], [], color = color)[0])

        # The last line is saved for joint visualization
        lines.append(axd[plane].plot([], [], [], linestyle="", marker="o", color="green")[0])

    animation = FuncAnimation(fig, FrameUpdate, frames=sequence, fargs=([lines]), init_func=init, blit=True)
    display(HTML(animation.to_jshtml()))