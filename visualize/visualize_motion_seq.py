# reference:
# https://stackoverflow.com/questions/8248467/matplotlib-tight-layout-doesnt-take-into-account-figure-suptitle

import argparse
import numpy as np
from processor.data_tools import _some_variables, fkl
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_poses(input_filepath, target_filepath, output_filepath):
    # load motion sequence tensors
    input_poses_4d = np.load(input_filepath)
    target_poses_4d = np.load(target_filepath)
    output_poses_4d = np.load(output_filepath)

    # define pose stats
    pose_stats = {}
    num_samples, input_seq_length, num_joints, _ = input_poses_4d.shape
    num_joints -= 1
    # define num_samples, num_joints
    pose_stats["num_samples"] = num_samples
    pose_stats["num_joints"] = num_joints
    # define input poses
    pose_stats["input_poses_4d"] = input_poses_4d
    pose_stats["input_seq_length"] = input_seq_length
    # define target poses
    _, target_seq_length, _, _ = target_poses_4d.shape
    pose_stats["target_poses_4d"] = target_poses_4d
    pose_stats["target_seq_length"] = target_seq_length
    # define output poses
    _, output_seq_length, _, _ = output_poses_4d.shape
    pose_stats["output_poses_4d"] = output_poses_4d
    pose_stats["output_seq_length"] = output_seq_length
    return pose_stats

def visualize_skeleton_per_frame(
    ax, pose_3d, parent, offset, posInd, expmapInd, bone_color="red", label_joints=True):
    xyz = fkl(pose_3d, parent, offset, posInd, expmapInd)
    xyz_2d = xyz.reshape(-1, 3)
    #plot joints
    ax.scatter(
        xyz_2d[:, 0],
        xyz_2d[:, 1],
        xyz_2d[:, 2])
    #plot bones
    bones = []
    for i in range(parent.shape[0]):
        if parent[i] != -1:
            bones.append((i, parent[i]))
    neighbor_link_partition = {
        "all": bones
    }
    for _, part_links in neighbor_link_partition.items():
        for bone in part_links:
            ax.plot(
                xyz_2d[bone, 0],
                xyz_2d[bone, 1],
                xyz_2d[bone, 2],
                color = bone_color,
                linewidth = 2)
    #label joints
    if label_joints:
        for joint_index in range(xyz_2d.shape[0]):
            ax.text(
                xyz_2d[joint_index, 0],
                xyz_2d[joint_index, 1],
                xyz_2d[joint_index, 2],
                f"{joint_index}",
                fontsize = 12,
                color="blue")

def visualize_input(poses_4d, frames_to_vis, img_dir):
    num_samples, sequence_length, num_joints, _ = poses_4d.shape
    num_joints -= 1
    num_frames = len(frames_to_vis)
    poses_3d = poses_4d.reshape((num_samples, sequence_length, -1))
    parent, offset, posInd, expmapInd = _some_variables()

    for sample_id in range(num_samples):
        # for each sample, create a directory and save the sequence
        img_dir_per_sample = f"{img_dir}/sample_{sample_id:02d}/"
        if not os.path.exists(img_dir_per_sample):
            os.makedirs(img_dir_per_sample)
        
        fig_w = 8
        fig = plt.figure(figsize=(fig_w*1*num_frames, fig_w))
        #fig.suptitle(f"Input Motion: {num_frames} Frames")
        # define frame counter for generating plots
        frame_counter = 1
        for frame_id in frames_to_vis:
            # build axis
            ax = fig.add_subplot(1, num_frames, frame_counter, projection='3d')
            frame_counter += 1

            # visualize each selected frame
            visualize_skeleton_per_frame(
                ax,
                poses_3d[sample_id, frame_id, :],
                parent,
                offset,
                posInd,
                expmapInd)

            # set up labels, view and scale axes
            #ax.set_xlabel("x")
            #ax.set_ylabel("y")
            #ax.set_zlabel("z")
            rel_frame_id = -1*(sequence_length-1-frame_id)
            ax.set_title(f"Frame {rel_frame_id}, {40*rel_frame_id} ms")
            ax.view_init(elev=10, azim=45)
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            z_min, z_max = ax.get_zlim()
            ax.set_box_aspect(
                (abs(x_max-x_min),
                abs(y_max-y_min),
                abs(z_max-z_min)))
            ax.set_axis_off()

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        img_filename = f"{img_dir_per_sample}/input_poses.png"
        plt.savefig(img_filename)
        plt.close(fig)
        print(f"sample {sample_id}'s poses saved to {img_filename}")

def visualize_target_and_output(target_poses_4d, output_poses_4d, frames_to_vis, img_dir):
    num_samples, sequence_length, num_joints, _ = target_poses_4d.shape
    num_joints -= 1
    num_frames = len(frames_to_vis)
    target_poses_3d = target_poses_4d.reshape((num_samples, sequence_length, -1))
    output_poses_3d = output_poses_4d.reshape((num_samples, sequence_length, -1))
    parent, offset, posInd, expmapInd = _some_variables()

    for sample_id in range(num_samples):
        # for each sample, create a directory and save the sequence
        img_dir_per_sample = f"{img_dir}/sample_{sample_id:02d}/"
        if not os.path.exists(img_dir_per_sample):
            os.makedirs(img_dir_per_sample)
        
        fig_w = 8
        fig = plt.figure(figsize=(fig_w*1*num_frames, fig_w))
        #fig.suptitle(f"Output and Target Motion: {num_frames} Frames")
        # define frame counter for generating plots
        frame_counter = 1
        for frame_id in frames_to_vis:
            # build axis
            ax = fig.add_subplot(1, num_frames, frame_counter, projection='3d')
            frame_counter += 1

            # visualize each selected frame
            visualize_skeleton_per_frame(
                ax,
                output_poses_3d[sample_id, frame_id, :],
                parent,
                offset,
                posInd,
                expmapInd,
                bone_color="red",
                label_joints=True)
            visualize_skeleton_per_frame(
                ax,
                target_poses_3d[sample_id, frame_id, :],
                parent,
                offset,
                posInd,
                expmapInd,
                bone_color="lightsalmon",
                label_joints=False)

            # set up labels, view and scale axes
            #ax.set_xlabel("x")
            #ax.set_ylabel("y")
            #ax.set_zlabel("z")
            rel_frame_id = 1+frame_id
            ax.set_title(f"Frame {rel_frame_id}, {40*rel_frame_id} ms")
            ax.view_init(elev=10, azim=45)
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            z_min, z_max = ax.get_zlim()
            ax.set_box_aspect(
                (abs(x_max-x_min),
                abs(y_max-y_min),
                abs(z_max-z_min)))
            ax.set_axis_off()

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        img_filename = f"{img_dir_per_sample}/output-and-target_poses.png"
        plt.savefig(img_filename)
        plt.close(fig)
        print(f"sample {sample_id}'s poses saved to {img_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualizer for DMGNN')
    parser.add_argument(
        '--npy_input_path', type=str, default='', help='path to .npy file containing encoder inputs')
    parser.add_argument(
        '--npy_target_path', type=str, default='', help='path to .npy file containing encoder targets')
    parser.add_argument(
        '--npy_output_path', type=str, default='', help='path to .npy file containing encoder outputs')
    parser.add_argument(
        '--img_dir', type=str, default='motion/', help='directory to save skeletal images')
    args = parser.parse_args()

    # load poses and stats
    pose_stats = load_poses(args.npy_input_path, args.npy_target_path, args.npy_output_path)
    num_samples = pose_stats["num_samples"]
    num_joints = pose_stats["num_joints"]
    input_poses_4d = pose_stats["input_poses_4d"]
    input_seq_length = pose_stats["input_seq_length"]
    target_poses_4d = pose_stats["target_poses_4d"]
    target_seq_length = pose_stats["target_seq_length"]
    output_poses_4d = pose_stats["output_poses_4d"]
    output_seq_length = pose_stats["output_seq_length"]
    
    # visualize input
    intput_frames_to_vis = [input_seq_length-i for i in range(5, 0, -2)]
    visualize_input(input_poses_4d, intput_frames_to_vis, args.img_dir)

    # visualize output vs target
    output_frames_to_vis = [0, 5, 9]
    visualize_target_and_output(target_poses_4d, output_poses_4d, output_frames_to_vis, args.img_dir)