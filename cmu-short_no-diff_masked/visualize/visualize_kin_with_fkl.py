from processor.data_tools import _some_variables, fkl
import numpy as np
import matplotlib.pyplot as plt

parent, offset, posInd, expmapInd = _some_variables()

# for i in range(len(posInd)):
#     print(expmapInd[i])

# print(len(expmapInd))
# for i in range(len(expmapInd)):
#     print(expmapInd[i])

encoder_inputs_4d = np.load(
    "/home/eric/eece571f/DMGNN/cmu-short_no-diff_masked/visualize/encoder_inputs_38_joints_walking.npy",
)
encoder_inputs_3d = encoder_inputs_4d.reshape(
    encoder_inputs_4d.shape[0],
    encoder_inputs_4d.shape[1],
    -1
)
print(f"encoder_inputs_3d has shape {encoder_inputs_3d.shape}")

sample_id = 0
for time_id in range(encoder_inputs_3d.shape[1]):
    xyz = fkl(encoder_inputs_3d[sample_id, time_id, :], parent, offset, posInd, expmapInd)
    # print(xyz.shape)
    xyz_2d = xyz.reshape(-1, 3)
    # for i in range(xyz_2d.shape[0]):
    #     print(xyz_2d[i])
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    #print(encoder_inputs_sample_x_time_t.shape)
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
    plt.savefig(f"skeletons_38_joints_walking_fkl/skeleton_<{time_id}>.png")
    plt.close(fig)
print(f"plotted {encoder_inputs_3d.shape[1]} skeletons")