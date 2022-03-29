import numpy as np
from processor.data_tools import expmap2rotmat
import matplotlib.pyplot as plt


def compute_affine_xform_mat(sample_index, time_index, joint_index, joint_rotmats, parents, bone_length):
    r"""
    Compute the 4D affine transformation matrix from joint's local coordinates
    to world coordinates.

    :return: 4, 4
    """
    # compute transform matrix from this joint to its parent
    # (counting only rotation here)
    xform_mat_4d = np.zeros([4, 4])
    xform_mat_4d[0:3, 0:3] = joint_rotmats[
        sample_index, time_index, joint_index, :, :
    ]
    xform_mat_4d[3, 3] = 1.0
    #trans_4d = np.eye(4)
    #trans_4d[0:3, 3] = np.array([0.0, 0.0, bone_length])
    #rot_4d = np.eye(4)
    #rot_4d[0:3, 0:3] = joint_rotmats[
    #    sample_index, time_index, joint_index, :, :
    #]
    #xform_mat_4d = np.matmul(
    #    rot_4d, trans_4d
    #)

    # compute transform matrix from parent to world
    if parents[joint_index] == -1:
        # base case: root joint to world
        pass
    else:
        # recursive case
        xform_mat_4d_last = compute_affine_xform_mat(
            sample_index,
            time_index,
            parents[joint_index],
            joint_rotmats,
            parents,
            bone_length
        )
        # include translation
        xform_mat_4d[0:3, 3] = np.array([0.0, 0.0, bone_length])
        # build hierarchy
        xform_mat_4d = np.matmul(xform_mat_4d_last, xform_mat_4d)
    return xform_mat_4d
        
def transform_along_kin_chain(sample_index, time_index, joint_index, joint_rotmats, parents, bone_length):
    r"""
    Compute one joint's position given joint rotation matrices,
    the kinematic tree and bone length.
    :param joint_rotmats: batch_size, T, J, 3, 3
    :param parents: J,
    :param bone_length: scalar

    :return: 3,
    """
    xform_mat_4d = compute_affine_xform_mat(
        sample_index,
        time_index,
        joint_index,
        joint_rotmats,
        parents,
        bone_length
    )
    joint_pos = np.matmul(xform_mat_4d, np.array([0.0, 0.0, 0.0, 1.0]))[0:3]
    return joint_pos

# exp map to joint pos
def compute_joint_pos(joint_exps, parents, bone_length):
    r"""
    Compute joint positions from joint rotations (in exponential maps).
    :param joint_exps: batch_size, T, J, 3
    :param parents: J,
    :param bone_length: scalar

    :return: batch_size, T, J, 3
    """
    batch_size, T, J, _ = joint_exps.shape

    # exp maps to rot matrices
    joint_rotmats = np.zeros([batch_size, T, J, 3, 3])
    for sample_index in range(batch_size):
        for time_index in range(T):
            for joint_index in range(J):
                joint_rotmats[sample_index, time_index, joint_index, :, :] = expmap2rotmat(
                    joint_exps[sample_index, time_index, joint_index, :]
                )

    # for each joint, compute its position by recursively transforming
    # along the kinematic chain
    joint_pos = np.zeros([batch_size, T, J, 3])
    for sample_index in range(batch_size):
        for time_index in range(T):
            for joint_index in range(J):
                joint_pos[sample_index, time_index, joint_index, :] = transform_along_kin_chain(
                    sample_index,
                    time_index,
                    joint_index,
                    joint_rotmats,
                    parents,
                    bone_length
                )
    return joint_pos


if __name__ == '__main__':
    # define the kinematic tree on the CMU dataset
    #cmu_parents = np.array([
    #    8, 0, 1, 2, 8, 4, 5, 6, 9, 10, 11, -1, 11, 12, 10, 14, 15, 16, 17, 16, 11, 20, 21, 22, 23, 22
    #    8, 0, 1, 2, 8, 4, 5, 6, -1, 8, 9, 10, 11, 12, 10, 14, 15, 16, 17, 16, 11, 20, 21, 22, 23, 22
    #    -1, 0, 1, 2, 8, 4, 5, 6, 0, 8, 9, 10, 11, 12, 10, 14, 15, 16, 17, 16, 11, 20, 21, 22, 23, 22
    #])
    cmu_parents = np.array([
        -1, 0,  1,  2,  3,  4,  5,  0,  7,  8,  9, 10, 11, 0,  13, 14, 15, 16, 17,
        18, 15, 20, 21, 22, 23, 24, 25, 23, 27, 15, 29, 30, 31, 32, 33, 34, 32, 36])

    # define bones
    # neighbor_link_partition = {
    #     "left_arm": [
    #         (10, 14), (14, 15), (15, 16), (16, 19), (16, 17), (17, 18)
    #     ],
    #     "right_arm": [
    #         (11, 20), (20, 21), (21, 22), (22, 23), (23, 24), (23, 25)
    #     ],
    #     "torso": [
    #         (8, 9), (9, 10), (10, 11)
    #     ],
    #     "head": [
    #         (11, 12), (12, 13)
    #     ],
    #     "left_leg": [
    #         (0, 8), (0, 1), (1, 2), (2, 3)
    #     ],
    #     "right_leg": [
    #         (4, 8), (4, 5), (5, 6), (6, 7)
    #     ]
    # }
    # neighbor_link_partition = {
    #     "all": [
    #         (1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 0), (8, 7), (9, 8), (10, 9),
    #         (11, 10), (12, 11), (13, 0), (14, 13), (15, 14), (16, 15), (17, 16), (18, 17),
    #         (19, 18), (20, 15), (21, 20), (22, 21), (23, 22), (24, 23), (25, 24), (26, 25),
    #         (27, 23), (28, 27), (29, 15), (30, 29), (31, 30), (32, 31), (33, 32), (34, 33),
    #         (35, 34), (36, 32), (37, 36)
    #     ],
    # }
    neighbor_link_partition = {
        "all": [
            (23, 22), (22, 21), (21, 17), (21, 2), (2, 3), (3, 4), (4, 5), (2, 8),
            (8, 30), (8, 9), (9, 10), (10, 11), (17, 30), (30, 31), (31, 37)
        ],
    }

    # define bone length for visualization
    bone_length = 2.0

    # convert joint angles to joint positions in
    # the world frame
    encoder_inputs_4d = np.load(
        "/home/eric/eece571f/DMGNN/cmu-short_no-diff_masked/visualize/encoder_inputs_38_joints_walking.npy",
    )
    encoder_inputs_cartesian = compute_joint_pos(
        encoder_inputs_4d,
        cmu_parents,
        bone_length
    )
    print(encoder_inputs_cartesian.shape)

    #visualize skeletons
    for time_id in range(encoder_inputs_cartesian.shape[1]):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        encoder_inputs_sample_x_time_t = encoder_inputs_cartesian[4, time_id, :].reshape(-1, 3) # J, 3
        #print(encoder_inputs_sample_x_time_t.shape)
        #plot joints
        ax.scatter(
            encoder_inputs_sample_x_time_t[:, 0],
            encoder_inputs_sample_x_time_t[:, 1],
            encoder_inputs_sample_x_time_t[:, 2])
        #plot bones
        # for part_name, part_links in neighbor_link_partition.items():
        #     if part_name == "left_arm" or part_name == "right_arm":
        #         link_color = "red"
        #         link_width = 2
        #     elif part_name == "torso":
        #         link_color = "cyan"
        #         link_width = 4
        #     elif part_name == "head":
        #         link_color = "purple"
        #         link_width = 2
        #     else:
        #         link_color = "green"
        #         link_width = 2
        #     for bone in part_links:
        #         ax.plot(encoder_inputs_sample_x_time_t[bone, 0],
        #             encoder_inputs_sample_x_time_t[bone, 1],
        #             encoder_inputs_sample_x_time_t[bone, 2],
        #             color = link_color,
        #             linewidth = link_width)
        for part_name, part_links in neighbor_link_partition.items():
            for bone in part_links:
                ax.plot(encoder_inputs_sample_x_time_t[bone, 0],
                    encoder_inputs_sample_x_time_t[bone, 1],
                    encoder_inputs_sample_x_time_t[bone, 2],
                    color = "red",
                    linewidth = 2)
        #label joints
        for joint_index in range(encoder_inputs_sample_x_time_t.shape[0]):
            ax.text(encoder_inputs_sample_x_time_t[joint_index, 0],
                encoder_inputs_sample_x_time_t[joint_index, 1],
                encoder_inputs_sample_x_time_t[joint_index, 2],
                f"{joint_index}",
                fontsize = 12,
                color="blue")
        plt.savefig(f"skeletons_38_joints_walking/skeleton_<{time_id}>.png")
        plt.close(fig)
    print(f"plotted {encoder_inputs_cartesian.shape[1]} skeletons")