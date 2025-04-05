import my_modules.quat as quat
from my_modules import Bvh
from scipy.interpolate import griddata
import scipy.signal as signal
import scipy.ndimage as ndimage
import struct
import numpy as np
import main_settings as ms
from scipy.ndimage import gaussian_filter1d


anim_path = 'animations/plane/{0}/'.format(ms.settings_animation_type)
files = ms.settings_animations

""" Basic function for mirroring animation data with this particular skeleton structure """


def animation_mirror(lrot, lpos, names, parents):
    joints_mirror = np.array([(
        names.index('Left' + n[5:]) if n.startswith('Right') else (
            names.index('Right' + n[4:]) if n.startswith('Left') else
            names.index(n))) for n in names])

    mirror_pos = np.array([-1, 1, 1])
    mirror_rot = np.array([[1, -1, -1], [-1, 1, 1], [-1, 1, 1]])

    grot, gpos = quat.fk(lrot, lpos, parents)

    gpos_mirror = mirror_pos * gpos[:, joints_mirror]
    grot_mirror = quat.from_xform(mirror_rot * quat.to_xform(grot[:, joints_mirror]))

    return quat.ik(grot_mirror, gpos_mirror, parents)


""" We will accumulate data in these lists """

bone_positions = []
bone_velocities = []
bone_rotations = []
bone_angular_velocities = []
contact_states = []

bone_parents = []
bone_names = []

range_starts = []
range_stops = []

""" Loop Over Files """

for filename, start, stop, root_approach, toe_info, action in files:

    # For each file we process it mirrored and not mirrored
    for mirror in [False, True]:

        """ Load Data """
        anim = anim_path + filename.split('/')[-1]
        print('Loading "%s" %s...' % (anim, "(Mirrored)" if mirror else ""))

        bvh_data = Bvh.load(anim)
        bvh_data['positions'] = bvh_data['positions'][start:stop]
        bvh_data['rotations'] = bvh_data['rotations'][start:stop]

        positions = bvh_data['positions']
        rotations = quat.unroll(quat.from_euler(np.radians(bvh_data['rotations']), order=bvh_data['order']))

        # Convert from cm to m
        positions *= 0.01

        if mirror:
            rotations, positions = animation_mirror(rotations, positions, bvh_data['names'], bvh_data['parents'])
            rotations = quat.unroll(rotations)

        """ Supersample """

        nframes = positions.shape[0]
        nbones = positions.shape[1]

        # Supersample data to 60 fps
        original_times = np.linspace(0, nframes - 1, nframes)
        sample_times = np.linspace(0, nframes - 1, int(0.9 * (nframes * 2 - 1)))  # Speed up data by 10%

        # This does a cubic interpolation of the data for supersampling and also speeding up by 10%
        positions = griddata(original_times, positions.reshape([nframes, -1]), sample_times, method='cubic').reshape(
            [len(sample_times), nbones, 3])
        rotations = griddata(original_times, rotations.reshape([nframes, -1]), sample_times, method='cubic').reshape(
            [len(sample_times), nbones, 4])

        # Need to re-normalize after super-sampling
        rotations = quat.normalize(rotations)

        """ Extract Simulation Bone """

        # First compute world space positions/rotations
        global_rotations, global_positions = quat.fk(rotations, positions, bvh_data['parents'])

        # Specify joints to use for simulation bone
        pos_joint_ref = root_approach.split(':')[1].split('/')[0]
        rot_joint_ref = root_approach.split(':')[1].split('/')[1]

        sim_position_joint = bvh_data['names'].index(pos_joint_ref)
        sim_rotation_joint = bvh_data['names'].index(rot_joint_ref)

        sim_position = np.array([1.0, 0.0, 1.0]) * global_positions[:, sim_position_joint:sim_position_joint + 1]
        sim_direction = np.array([1.0, 0.0, 1.0]) * quat.mul_vec(
            global_rotations[:, sim_rotation_joint:sim_rotation_joint + 1], np.array([0.0, 0.0, 1.0]))
        if 'root_simple' in root_approach:
            if len(root_approach.split(':')) > 2:
                savgol_filter_param = root_approach.split(':')[2]
            else:
                savgol_filter_param = 61

            sim_position = signal.savgol_filter(sim_position, savgol_filter_param, 3, axis=0, mode='interp')

            # We need to re-normalize the direction after both projection and smoothing
            sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1))[..., np.newaxis]
            sim_direction = signal.savgol_filter(sim_direction, savgol_filter_param, 3, axis=0, mode='interp')
            sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1)[..., np.newaxis])

        elif 'root_smoothed:' in root_approach:
            smoothing_type = root_approach.split(':')[2]
            window_size = int(root_approach.split(':')[-1])

            sim_position = signal.savgol_filter(sim_position, 31, 3, axis=0, mode='interp')

            smoothed = np.copy(sim_direction)
            if smoothing_type == 'ma':
                kernel = np.ones(window_size) / window_size
                smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=sim_direction)
            elif smoothing_type == 'ema':
                alpha = window_size / 100
                for i in range(3):  # Process each Euler angle separately
                    for t in range(1, len(sim_direction)):
                        smoothed[t][0][i] = alpha * sim_direction[t][0][i] + (1 - alpha) * smoothed[t - 1][0][i]
            elif smoothing_type == 'cma':
                if window_size % 2 == 0:
                    window_size += 1  # Ensure the window size is odd
                for i in range(3):  # Process each Euler axis separately
                    smoothed[:, 0, i] = np.convolve(sim_direction[:, 0, i], np.ones(window_size) / window_size,
                                                    mode='same')
            elif smoothing_type == 'gf':
                for i in range(3):  # Process each Euler axis separately
                    smoothed[:, 0, i] = gaussian_filter1d(sim_direction[:, 0, i].flatten(), sigma=window_size,
                                                          mode="nearest")

            sim_direction = smoothed / np.sqrt(np.sum(np.square(smoothed), axis=-1))[..., np.newaxis]
            sim_direction = signal.savgol_filter(sim_direction, 61, 3, axis=0, mode='interp')
            sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1)[..., np.newaxis])
        elif 'root_locked:' in root_approach:
            global_target_offset = root_approach.split(':')[-1]
            sim_position_joint = bvh_data['names'].index(pos_joint_ref)
            sim_position = signal.savgol_filter(sim_position, 61, 3, axis=0, mode='interp')
            nframes = sim_position.shape[0]
            smoothed = np.zeros((nframes, 1, 3))
            target_position = np.copy(sim_position[0][0])
            target_position[0] += float(global_target_offset.split(',')[0])
            target_position[1] += float(global_target_offset.split(',')[1])
            target_position[2] += float(global_target_offset.split(',')[2])

            for t in range(nframes):
                direction = target_position - sim_position[t, 0, :]
                direction /= np.linalg.norm(direction)
                yaw = np.arctan2(direction[0], direction[2])
                pitch = np.arcsin(-direction[1])
                roll = 0.0
                smoothed[t, 0, :] = target_position - sim_position[t, 0, :]

        # Extract rotation from direction
        sim_rotation = quat.normalize(quat.between(np.array([0, 0, 1]), sim_direction))

        # Transform first joints to be local to sim and append sim as root bone
        positions[:, 0:1] = quat.mul_vec(quat.inv(sim_rotation), positions[:, 0:1] - sim_position)
        rotations[:, 0:1] = quat.mul(quat.inv(sim_rotation), rotations[:, 0:1])

        positions = np.concatenate([sim_position, positions], axis=1)
        rotations = np.concatenate([sim_rotation, rotations], axis=1)

        bone_parents = np.concatenate([[-1], bvh_data['parents'] + 1])

        bone_names = ['Simulation'] + bvh_data['names']

        """ Compute Velocities """

        # Compute velocities via central difference
        velocities = np.empty_like(positions)
        velocities[1:-1] = (
                0.5 * (positions[2:] - positions[1:-1]) * 60.0 +
                0.5 * (positions[1:-1] - positions[:-2]) * 60.0)
        velocities[0] = velocities[1] - (velocities[3] - velocities[2])
        velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])

        # Same for angular velocities
        angular_velocities = np.zeros_like(positions)
        angular_velocities[1:-1] = (
                0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[2:], rotations[1:-1]))) * 60.0 +
                0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[1:-1], rotations[:-2]))) * 60.0)
        angular_velocities[0] = angular_velocities[1] - (angular_velocities[3] - angular_velocities[2])
        angular_velocities[-1] = angular_velocities[-2] + (angular_velocities[-2] - angular_velocities[-3])

        """ Compute Contact Data """

        global_rotations, global_positions, global_velocities, global_angular_velocities = quat.fk_vel(
            rotations,
            positions,
            velocities,
            angular_velocities,
            bone_parents)

        contact_velocity_threshold = 0.15

        contact_velocity = np.sqrt(np.sum(global_velocities[:, np.array([
            bone_names.index("LeftToeBase"),
            bone_names.index("RightToeBase")])] ** 2, axis=-1))

        # Contacts are given for when contact bones are below velocity threshold
        contacts = contact_velocity < contact_velocity_threshold

        # Median filter here acts as a kind of "majority vote", and removes
        # small regions  where contact is either active or inactive
        for ci in range(contacts.shape[1]):
            contacts[:, ci] = ndimage.median_filter(
                contacts[:, ci],
                size=6,
                mode='nearest')

        offset = 0 if len(range_starts) == 0 else range_stops[-1]

        range_starts.append(offset)
        range_stops.append(offset + len(positions))

        # Compute world space positions/rotations
        global_rotations, global_positions = quat.fk(rotations, positions, bone_parents)

        """ Append to Database """

        bone_positions.append(positions)
        bone_velocities.append(velocities)
        bone_rotations.append(rotations)
        bone_angular_velocities.append(angular_velocities)
        contact_states.append(contacts)

""" Concatenate Data """

bone_positions = np.concatenate(bone_positions, axis=0).astype(np.float32)
bone_velocities = np.concatenate(bone_velocities, axis=0).astype(np.float32)
bone_rotations = np.concatenate(bone_rotations, axis=0).astype(np.float32)
bone_angular_velocities = np.concatenate(bone_angular_velocities, axis=0).astype(np.float32)
bone_parents = bone_parents.astype(np.int32)
contact_states = np.concatenate(contact_states, axis=0).astype(np.uint8)

range_starts = np.array(range_starts).astype(np.int32)
range_stops = np.array(range_stops).astype(np.int32)

""" Write Database """

print("Writing Database generate/data/plane/{0}/database.bin ...".format(ms.settings_animation_type))

with open('generate/data/plane/{0}/database.bin'.format(ms.settings_animation_type), 'wb') as f:
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

Bvh.save('generate/data/plane/{0}/database.bvh'.format(ms.settings_animation_type), {
    'rotations': np.degrees(quat.to_euler(bone_rotations)),
    'positions': 100.0 * bone_positions,
    'offsets': 100.0 * bone_positions[0],
    'parents': bone_parents,
    'names': ['joint_%i' % i for i in range(nbones)],
    'order': 'yxz'
})

