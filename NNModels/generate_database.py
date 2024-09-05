import numpy as np
import torch
import struct
import my_modules.quat_functions as quat
import bvh

from scipy.interpolate import griddata
from scipy import signal
import scipy.ndimage as ndimage


files = [
    # We just use a small section of this clip for the standing idle
    ('pushAndStumble1_subject5.bvh', 194,  351),
    ('run1_subject5.bvh', 90, 7086),
    ('walk1_subject5.bvh', 80, 7791),
]


def animation_mirror(lrot, lpos, names, parents):
    joints_mirror = np.array([(
        names.index('Left' + n[5:]) if n.startswith('Right') else (
            names.index('Right' + n[4:]) if n.startswith('Left') else
            names.index(n)))
        for n in names])
    mirror_pos = np.array([-1, 1, 1])
    mirror_rot = np.array([[-1, -1, 1], [1, 1, -1], [1, 1, -1]])

    gpos, grot = quat.fk(lpos, lrot, parents)

    gpos = torch.as_tensor(gpos)
    grot = torch.as_tensor(grot)

    gpos_mirror = torch.as_tensor(mirror_pos) * gpos[:, joints_mirror]
    grot_mirror = quat.from_xform(torch.tensor(mirror_rot) * quat.to_xform(grot[:, joints_mirror]))

    return quat.ik(gpos_mirror, grot_mirror, parents)


# We will accumulate data in these lists

bone_positions = []
bone_velocities = []
bone_rotations = []
bone_angular_velocities = []
bone_parents = []
bone_names = []

range_starts = []
range_stops = []

contact_states = []

# Loop over files
for filename, start, stop in files:
    for mirror in [False, True]:
        # Load data
        print('Loading "%s" %s ...' % (filename, "(Mirrored)" if mirror else ""))

        bvh_data = bvh.load('animations/%s' % filename)
        bvh_data['positions'] = bvh_data['positions'][start:stop]
        bvh_data['rotations'] = bvh_data['rotations'][start:stop]

        positions = torch.as_tensor(bvh_data['positions'])
        rotations = quat.unroll(
            quat.from_euler(torch.as_tensor(np.radians(bvh_data['rotations']).copy()), order=bvh_data['order']))

        # Convert from cm to m
        positions *= 0.01

        if mirror:
            rotations, positions = animation_mirror(rotations, positions, bvh_data['names'], bvh_data['parents'])
            rotations = quat.unroll(torch.as_tensor(rotations))

        nframes = positions.shape[0]
        nbones = positions.shape[1]

        original_times = np.linspace(0, nframes - 1, nframes)
        sample_times = np.linspace(0, nframes - 1, int(0.9 * (nframes * 2 - 1)))

        # This does a cubic interpolation of the data for supersampling and also speeding up by 10%
        positions = torch.as_tensor(
            griddata(original_times, positions.reshape([nframes, -1]), sample_times, method='cubic')
            .reshape([len(sample_times), nbones, 3]))
        rotations = torch.as_tensor(
            griddata(original_times, rotations.reshape([nframes, -1]), sample_times, method='cubic')
            .reshape([len(sample_times), nbones, 4]))

        rotations = quat.normalize(rotations)

        """ Extract Simulation Bone """
        # First compute world space positions/rotations
        global_positions, global_rotations = quat.fk(positions, rotations, bvh_data['parents'])

        global_rotations = torch.as_tensor(global_rotations)
        global_positions = torch.as_tensor(global_positions)

        sim_position_joint = bvh_data['names'].index("Spine2")
        sim_rotation_joint = bvh_data['names'].index("Hips")

        # A Savitzky–Golay filter is a digital filter that can be applied to a set of digital data points for the
        # purpose of smoothing the data, that is, to increase the precision of the data without distorting the signal
        # tendency.
        sim_position = torch.tensor([1.0, 0.0, 1.0]) * global_positions[:, sim_position_joint:sim_position_joint + 1]
        sim_position = torch.as_tensor(signal.savgol_filter(sim_position, 31, 3, axis=0, mode='interp'))

        sim_direction = torch.tensor([1.0, 0.0, 1.0]) * quat.mul_vec(
            global_rotations[:, sim_rotation_joint:sim_rotation_joint + 1],
            torch.tensor([0.0, 1.0, 0.0]))

        # We need to re-normalize the direction after both projection and smoothing
        sim_direction = sim_direction / torch.sqrt(torch.sum(torch.square(sim_direction), dim=-1)[..., np.newaxis])
        sim_direction = torch.as_tensor(signal.savgol_filter(sim_direction, 61, 3, axis=0, mode='interp'))
        sim_direction = sim_direction / torch.sqrt(torch.sum(torch.square(sim_direction), dim=-1)[..., np.newaxis])

        # Extract rotation from direction
        sim_rotation = quat.normalize(quat.between(torch.tensor([0, 0, 1]), sim_direction))

        # Transform first joints to be local to sim and append sim as root bone
        positions[:, 0:1] = quat.mul_vec(quat.inv(sim_rotation), positions[:, 0:1] - sim_position)
        rotations[:, 0:1] = quat.mul(quat.inv(sim_rotation), rotations[:, 0:1])

        positions = torch.cat([sim_position, positions], dim=1)
        rotations = torch.cat([sim_rotation, rotations], dim=1)

        bone_parents = np.concatenate([[-1], bvh_data['parents'] + 1])

        bone_names = ['Simulation'] + bvh_data['names']

        """ Compute Velocities """

        # Compute velocities via central difference
        velocities = torch.empty_like(positions)
        velocities[1:-1] = (
                0.5 * (positions[2:] - positions[1:-1]) * 60.0 +
                0.5 * (positions[1:-1] - positions[:-2]) * 60.0)
        velocities[0] = velocities[1] - (velocities[3] - velocities[2])
        velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])

        # Same for angular velocities
        angular_velocities = torch.zeros_like(positions)
        angular_velocities[1:-1] = (
                0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[2:], rotations[1:-1]))) * 60.0 +
                0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[1:-1], rotations[:-2]))) * 60.0)
        angular_velocities[0] = angular_velocities[1] - (angular_velocities[3] - angular_velocities[2])
        angular_velocities[-1] = angular_velocities[-2] + (angular_velocities[-2] - angular_velocities[-3])

        """ Compute Contact Data """

        global_positions, global_rotations, global_velocities, global_angular_velocities = quat.fk_vel(
            positions,
            rotations,
            velocities,
            angular_velocities,
            bone_parents
        )

        contact_velocity_threshold = 0.15

        contact_velocity = torch.sqrt(torch.sum(global_velocities[:, np.array([
            bone_names.index("LeftToe"),
            bone_names.index("RightToe")
        ])] ** 2, dim=-1))  # ** 2 => exponential ^2

        contacts = contact_velocity < contact_velocity_threshold
        print(contacts.shape)
        for ci in range(contacts.shape[1]):
            contacts[:, ci] = torch.as_tensor(ndimage.median_filter(
                contacts[:, ci],
                size=6,
                mode='nearest'
            ))

        """ Append to Database """
        bone_positions.append(positions)
        bone_velocities.append(velocities)
        bone_rotations.append(rotations)
        bone_angular_velocities.append(angular_velocities)

        offset = 0 if len(range_starts) == 0 else range_stops[-1]

        range_starts.append(offset)
        range_stops.append(offset + len(positions))

        contact_states.append(contacts)

""" Concatenate Data """
bone_positions = np.concatenate(bone_positions, axis=0).astype(np.float32)
bone_velocities = np.concatenate(bone_velocities, axis=0).astype(np.float32)
bone_rotations = np.concatenate(bone_rotations, axis=0).astype(np.float32)
bone_angular_velocities = np.concatenate(bone_angular_velocities, axis=0).astype(np.float32)
bone_parents = bone_parents.astype(np.int32)

range_starts = np.array(range_starts).astype(np.int32)
range_stops = np.array(range_stops).astype(np.int32)

contact_states = np.concatenate(contact_states, axis=0).astype(np.uint8)

""" Write Database """

print("Writing Database...")

with open('data/database2.bin', 'wb') as f:
    nframes = bone_positions.shape[0]
    nbones = bone_positions.shape[1]
    nranges = range_starts.shape[0]
    ncontacts = contact_states.shape[1]

    f.write(struct.pack('II', nframes, nbones) + bone_positions.ravel().tobytes())
    f.write(struct.pack('II', nframes, nbones) + bone_velocities.ravel().tobytes())
    f.write(struct.pack('II', nframes, nbones) + bone_rotations.ravel().tobytes())
    f.write(struct.pack('II', nframes, nbones) + bone_angular_velocities.ravel().tobytes())
    f.write(struct.pack('I', nbones) + bone_parents.ravel().tobytes())

    f.write(struct.pack('I', nranges) + range_starts.ravel().tobytes())
    f.write(struct.pack('I', nranges) + range_stops.ravel().tobytes())

    f.write(struct.pack('II', nframes, ncontacts) + contact_states.ravel().tobytes())


