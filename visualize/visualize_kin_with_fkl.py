# script to visualize skeletal sequences

from processor.data_tools import _some_variables, fkl
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualizer for DMGNN')
    parser.add_argument('--npy_path', type=str, default='', help='path to .npy file containing skeletal poses')
    parser.add_argument('--img_dir', type=str, default='motion/', help='directory to save skeletal images')
    args = parser.parse_args()

    poses_4d = np.load(args.npy_path)
    num_samples, sequence_length, num_joints, _ = poses_4d.shape
    num_joints -= 1
    poses_3d = poses_4d.reshape((num_samples, sequence_length, -1))

    print(f"Pose file: {args.npy_path}")
    print(f"Plotting pose sequences of {num_samples} samples")
    print(f"Sequence length: {sequence_length}, number of joints: {num_joints}")
    print(f"Saving visualizations to {args.img_dir}/sample_<sample id>")

    parent, offset, posInd, expmapInd = _some_variables()
    for sample_id in range(num_samples):
        img_dir_per_sample = f"{args.img_dir}/sample_{sample_id:02d}/"
        if not os.path.exists(img_dir_per_sample):
            os.makedirs(img_dir_per_sample)

        for time_id in range(sequence_length):
            xyz = fkl(poses_3d[sample_id, time_id, :], parent, offset, posInd, expmapInd)
            xyz_2d = xyz.reshape(-1, 3)
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(projection='3d')
            #plot joints
            ax.scatter(
                xyz_2d[:, 2],
                xyz_2d[:, 1],
                xyz_2d[:, 0])
            #plot bones
            # neighbor_link_partition = {
            #     "all": [
            #         (23, 22), (22, 21), (21, 17), (21, 2), (2, 3), (3, 4), (4, 5), (2, 8),
            #         (8, 30), (8, 9), (9, 10), (10, 11), (17, 30), (30, 31), (31, 37)
            #     ],
            # }
            bones = []
            for i in range(parent.shape[0]):
                if parent[i] != -1:
                    bones.append((i, parent[i]))
            neighbor_link_partition = {
                "all": bones
            }
            for part_name, part_links in neighbor_link_partition.items():
                for bone in part_links:
                    ax.plot(xyz_2d[bone, 2],
                        xyz_2d[bone, 1],
                        xyz_2d[bone, 0],
                        color = "red",
                        linewidth = 2)
            #label joints
            for joint_index in range(xyz_2d.shape[0]):
                ax.text(xyz_2d[joint_index, 2],
                    xyz_2d[joint_index, 1],
                    xyz_2d[joint_index, 0],
                    f"{joint_index}",
                    fontsize = 12,
                    color="blue")
            plt.savefig(f"{img_dir_per_sample}/pose_{time_id:02d}.png")
            plt.close(fig)
        print(f"sample {sample_id}'s poses saved to {img_dir_per_sample}")