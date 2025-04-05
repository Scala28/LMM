import sys
import os
import main_settings as ms
sys.path.append('my_modules')
from my_modules import Bvh
import my_modules.quat as quat
from trainings.train_common import load_database
import struct
import numpy as np
import math


def forward_kinematics(out_bone_position,
                       out_bone_rotation,
                       in_bone_positions,
                       in_bone_rotations,
                       in_bone_parents,
                       in_bone):
    if in_bone_parents[in_bone] != -1:
        out_parent_position, out_parent_rotation = forward_kinematics(np.ndarray(3),
                                                                     np.ndarray(4),
                                                                     in_bone_positions,
                                                                     in_bone_rotations,
                                                                     in_bone_parents,
                                                                     in_bone_parents[in_bone])

        out_bone_position = quat.mul_vec(out_parent_rotation, in_bone_positions[in_bone]) + out_parent_position
        out_bone_rotation = quat.mul(out_parent_rotation, in_bone_rotations[in_bone])
    else:
        out_bone_position = in_bone_positions[in_bone]
        out_bone_rotation = in_bone_rotations[in_bone]
    return out_bone_position, out_bone_rotation


def forward_kinematics_velocity(in_bone_positions,
                                in_bone_velocities,
                                in_bone_rotations,
                                in_bone_angular_velocities,
                                in_bone_parents,
                                in_bone):
    if in_bone_parents[in_bone] != -1:

        parent_position, parent_velocity, \
        parent_rotation,parent_angular_velocity = forward_kinematics_velocity(in_bone_positions,
                                                                              in_bone_velocities,
                                                                              in_bone_rotations,
                                                                              in_bone_angular_velocities,
                                                                              in_bone_parents,
                                                                              in_bone_parents[in_bone])
        bone_position = quat.mul_vec(parent_rotation, in_bone_positions[in_bone]) + parent_position
        bone_velocity = parent_velocity + quat.mul_vec(parent_rotation, in_bone_velocities[in_bone]) +\
                        quat._fast_cross(parent_angular_velocity, quat.mul_vec(parent_rotation, in_bone_positions[in_bone]))
        bone_rotation = quat.mul(parent_rotation, in_bone_rotations[in_bone])
        bone_angular_velocity = quat.mul_vec(parent_rotation, in_bone_angular_velocities[in_bone]) + parent_angular_velocity
    else:
        bone_position = in_bone_positions[in_bone]
        bone_velocity = in_bone_velocities[in_bone]
        bone_rotation = in_bone_rotations[in_bone]
        bone_angular_velocity = in_bone_angular_velocities[in_bone]
    return bone_position, bone_velocity, bone_rotation, bone_angular_velocity


def normalize_feature(offset, size, weight):
    global features_offset, features_scale, features

    for j in range(size):
        features_offset[offset + j] = 0
    # First compute what is essentially the mean value for each feature dimension
    for i in range(nframes):
        for j in range(size):
            features_offset[offset + j] += features[i, offset + j] / nframes

    # Now compute the variance of each feature dimension
    vars = np.zeros(size)
    for i in range(nframes):
        for j in range(size):
            vars[j] += (features[i, offset + j] - features_offset[offset + j]) ** 2 / nframes

    # We compute the overall std of the feature as the average std across all dimensions
    std = 0
    for j in range(size):
        std += math.sqrt(vars[j]) / size
    # Features with no variation can have zero std which is almost always a bug.
    assert std > 0

    # The scale of a feature is just the std divided by the weight
    for j in range(size):
        features_scale[offset + j] = std / weight

    # Using the offset and scale we can then normalize the features
    for i in range(nframes):
        for j in range(size):
            features[i, offset + j] = (features[i, offset + j] - features_offset[offset + j]) / features_scale[offset + j]


def compute_bone_position_feature(offset, bone, weight):
    global features

    for i in range(nframes):
        bone_position, bone_rotation = forward_kinematics(np.ndarray(3),
                           np.ndarray(4),
                           bone_positions[i],
                           bone_rotations[i],
                           bone_parents,
                           bone)
        bone_position = quat.mul_vec(quat.inv(bone_rotations[i,0]), bone_position - bone_positions[i,0])

        features[i, offset + 0] = bone_position[0]
        features[i, offset + 1] = bone_position[1]
        features[i, offset + 2] = bone_position[2]

    normalize_feature(offset, 3, weight)
    return offset + 3


def compute_bone_velocity_feature(offset, bone, weight):
    global features

    for i in range(nframes):
        bone_position, bone_velocity, \
        bone_rotation, bone_angular_velocity = forward_kinematics_velocity(bone_positions[i],
                                                                                bone_velocities[i],
                                                                                bone_rotations[i],
                                                                                bone_angular_velocities[i],
                                                                                bone_parents,
                                                                                bone)
        bone_velocity = quat.mul_vec(quat.inv(bone_rotations[i,0]), bone_velocity)
        features[i, offset + 0] = bone_velocity[0];
        features[i, offset + 1] = bone_velocity[1];
        features[i, offset + 2] = bone_velocity[2];

    normalize_feature(offset, 3, weight)
    return offset + 3


def database_trajectory_index_clamp(frame, offset):
    for i in range(nranges):
        if range_starts[i] <= frame < range_stops[i]:
            return max(min(frame + offset, range_stops[i] - 1), range_starts[i])
    assert False
    return -1


def compute_trajectory_position_feature(offset, weight):
    global features

    for i in range(nframes):
        t0 = database_trajectory_index_clamp(i, 20)
        t1 = database_trajectory_index_clamp(i, 40)
        t2 = database_trajectory_index_clamp(i, 60)

        trajectory_pos0 = quat.mul_vec(quat.inv(bone_rotations[i, 0]), bone_positions[t0, 0] - bone_positions[i, 0])
        trajectory_pos1 = quat.mul_vec(quat.inv(bone_rotations[i, 0]), bone_positions[t1, 0] - bone_positions[i, 0])
        trajectory_pos2 = quat.mul_vec(quat.inv(bone_rotations[i, 0]), bone_positions[t2, 0] - bone_positions[i, 0])

        features[i, offset + 0] = trajectory_pos0[0]
        features[i, offset + 1] = trajectory_pos0[2]
        features[i, offset + 2] = trajectory_pos1[0]
        features[i, offset + 3] = trajectory_pos1[2]
        features[i, offset + 4] = trajectory_pos2[0]
        features[i, offset + 5] = trajectory_pos2[2]

    normalize_feature(offset, 6, weight)
    return offset + 6


def compute_trajectory_direction_feature(offset, weight):
    global features

    for i in range(nframes):
        t0 = database_trajectory_index_clamp(i, 20)
        t1 = database_trajectory_index_clamp(i, 40)
        t2 = database_trajectory_index_clamp(i, 60)

        trajectory_dir0 = quat.mul_vec(quat.inv(bone_rotations[i, 0]), quat.mul_vec(bone_rotations[t0, 0], np.array([0, 0, 1])))
        trajectory_dir1 = quat.mul_vec(quat.inv(bone_rotations[i, 0]), quat.mul_vec(bone_rotations[t1, 0], np.array([0, 0, 1])))
        trajectory_dir2 = quat.mul_vec(quat.inv(bone_rotations[i, 0]), quat.mul_vec(bone_rotations[t2, 0], np.array([0, 0, 1])))

        features[i, offset + 0] = trajectory_dir0[0]
        features[i, offset + 1] = trajectory_dir0[2]
        features[i, offset + 2] = trajectory_dir1[0]
        features[i, offset + 3] = trajectory_dir1[2]
        features[i, offset + 4] = trajectory_dir2[0]
        features[i, offset + 5] = trajectory_dir2[2]

    normalize_feature(offset, 6, weight)
    return offset + 6


def compute_future_terrain_feature(offset, bone, weight):
    global features

    for i in range(nframes):
        bone_position, bone_rotation = forward_kinematics(np.ndarray(3),
                           np.ndarray(4),
                           bone_positions[i],
                           bone_rotations[i],
                           bone_parents,
                           bone)
        bone_position = quat.mul_vec(quat.inv(bone_rotations[i,0]), bone_position - bone_positions[i,0])

        features[i, offset + 0] = bone_position[1]

        t0 = database_trajectory_index_clamp(i, 15)
        t1 = database_trajectory_index_clamp(i, 30)
        t2 = database_trajectory_index_clamp(i, 45)

        bone_position_15, bone_rotation_15 = forward_kinematics(np.ndarray(3),
                                                                np.ndarray(4),
                                                                bone_positions[t0],
                                                                bone_rotations[t0],
                                                                bone_parents,
                                                                bone)

        bone_position_15 = quat.mul_vec(quat.inv(bone_rotations[i, 0]), bone_position_15 - bone_positions[i, 0])

        features[i, offset + 1] = bone_position_15[1]

        bone_position_30, bone_rotation_30 = forward_kinematics(np.ndarray(3),
                                                                np.ndarray(4),
                                                                bone_positions[t1],
                                                                bone_rotations[t1],
                                                                bone_parents,
                                                                bone)

        bone_position_30 = quat.mul_vec(quat.inv(bone_rotations[i, 0]), bone_position_30 - bone_positions[i, 0])

        features[i, offset + 2] = bone_position_30[1]

        bone_position_45, bone_rotation_45 = forward_kinematics(np.ndarray(3),
                                                                np.ndarray(4),
                                                                bone_positions[t2],
                                                                bone_rotations[t2],
                                                                bone_parents,
                                                                bone)

        bone_position_45 = quat.mul_vec(quat.inv(bone_rotations[i, 0]), bone_position_45 - bone_positions[i, 0])

        features[i, offset + 3] = bone_position_45[1]
    normalize_feature(offset, 4, weight)
    return offset + 4


def compute_torso_local_position(frame_indx, bone):
    hips_gp, hips_gr = forward_kinematics(np.ndarray(3),
                                          np.ndarray(4),
                                          bone_positions[frame_indx],
                                          bone_rotations[frame_indx],
                                          bone_parents,
                                          Bone_Hips)

    torso_gp, torso_gr = forward_kinematics(np.ndarray(3),
                                            np.ndarray(4),
                                            bone_positions[frame_indx],
                                            bone_rotations[frame_indx],
                                            bone_parents,
                                            bone)

    torso_local = quat.mul_vec(quat.inv(hips_gr), torso_gp - hips_gp)
    return torso_local


def compute_future_torso_feature(offset, bone,  weight):
    global features

    for i in range(nframes):
        torso_local = compute_torso_local_position(i, bone)

        features[i, offset + 0] = torso_local[0]
        features[i, offset + 1] = torso_local[2]

        t0 = database_trajectory_index_clamp(i, 15)
        t1 = database_trajectory_index_clamp(i, 30)
        # t2 = database_trajectory_index_clamp(i, 45)

        torso_local_15 = compute_torso_local_position(t0, bone)
        features[i, offset + 2] = torso_local_15[0]
        features[i, offset + 3] = torso_local_15[2]

        torso_local_30 = compute_torso_local_position(t1, bone)
        features[i, offset + 4] = torso_local_30[0]
        features[i, offset + 5] = torso_local_30[2]

        # torso_local_45 = compute_torso_local_position(t2)
        # features[i, offset + 6] = torso_local_45[0]
        # features[i, offset + 7] = torso_local_45[2]

    normalize_feature(offset, 6, weight)
    return offset + 6


# Build all motion matching features and acceleration structure
def database_build_matching_features():
    feature_weight_foot_position = 0.75
    feature_weight_foot_velocity = 1.0
    feature_weight_hip_velocity = 1.0
    feature_weight_trajectory_positions = 1.0
    feature_weight_trajectory_directions = 1.5
    feature_weight_torso_position = 1.5
    offset = 0
    offset = compute_bone_position_feature(offset, Bone_LeftFoot, feature_weight_foot_position)
    sys.stdout.write('\rOffset: %2i / %2i' % (offset, nfeatures))
    offset = compute_bone_position_feature(offset, Bone_RightFoot, feature_weight_foot_position)
    sys.stdout.write('\rOffset: %2i / %2i' % (offset, nfeatures))
    offset = compute_bone_velocity_feature(offset, Bone_LeftFoot, feature_weight_foot_velocity)
    sys.stdout.write('\rOffset: %2i / %2i' % (offset, nfeatures))
    offset = compute_bone_velocity_feature(offset, Bone_RightFoot, feature_weight_foot_velocity)
    sys.stdout.write('\rOffset: %2i / %2i' % (offset, nfeatures))
    offset = compute_bone_velocity_feature(offset, Bone_Hips, feature_weight_hip_velocity)
    sys.stdout.write('\rOffset: %2i / %2i' % (offset, nfeatures))
    offset = compute_trajectory_position_feature(offset, feature_weight_trajectory_positions)
    sys.stdout.write('\rOffset: %2i / %2i' % (offset, nfeatures))
    offset = compute_trajectory_direction_feature(offset, feature_weight_trajectory_directions)
    sys.stdout.write('\rOffset: %2i / %2i' % (offset, nfeatures))
    offset = compute_future_torso_feature(offset, Bone_Spine2,  feature_weight_torso_position)
    sys.stdout.write('\rOffset: %2i / %2i' % (offset, nfeatures))
    print("\n")
    if nfeatures != offset:
        print("\nAssertion error!")
        exit()


Bone_Entity = 0
Bone_Hips = 1
Bone_Spine = 2
Bone_Spine1 = 3
Bone_Spine2 = 4
Bone_Neck = 5
Bone_Head = 6
Bone_LeftShoulder = 7
Bone_LeftArm = 8
Bone_LeftForeArm = 9
Bone_LeftHand = 10
Bone_RightShoulder = 11
Bone_RightArm = 12
Bone_RightForeArm = 13
Bone_RightHand = 14
Bone_LeftUpLeg = 15
Bone_LeftLeg = 16
Bone_LeftFoot = 17
Bone_LeftToeBase = 18
Bone_RightUpLeg = 19
Bone_RightLeg = 20
Bone_RightFoot = 21
Bone_RightToeBase = 22

database = load_database('generate/data/fight/{0}/database.bin'.format(ms.settings_animation_type))
bone_positions = database['bone_positions']
bone_rotations = database['bone_rotations']
bone_velocities = database['bone_velocities']
bone_angular_velocities = database['bone_angular_velocities']
bone_parents = database['bone_parents']
range_starts = database['range_starts']
range_stops = database['range_stops']
nranges = range_starts.shape[0]
nframes = bone_positions.shape[0]
nfeatures = 3 + 3 + 3 + 3 + 3 + 6 + 6 + 6
features = np.zeros((nframes, nfeatures))
features_offset = np.zeros(nfeatures)
features_scale = np.zeros(nfeatures)

print("Writing features generate/data/fight/{0}/features.bin ...".format(ms.settings_animation_type))
database_build_matching_features()

features_32 = np.concatenate(features, axis=0).astype(np.float32)
features_offset_32 = features_offset.astype(np.float32)
features_scale_32 = features_scale.astype(np.float32)

with open('generate/data/fight/{0}/features.bin'.format(ms.settings_animation_type), 'wb') as f:
    f.write(struct.pack('II', nframes, nfeatures) + features_32.ravel().tobytes())
    f.write(struct.pack('I', nfeatures) + features_offset_32.ravel().tobytes())
    f.write(struct.pack('I', nfeatures) + features_scale_32.ravel().tobytes())
