import numpy as np
import struct
import my_modules.quat_functions as quat
import bvh

files = [
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

    gpos_mirror = mirror_pos * gpos[:, joints_mirror]
    grot_mirror = quat.from_xform(mirror_rot * quat.to_xform(grot[:, joints_mirror]))

    return quat.ik(gpos_mirror, grot_mirror, parents)


# We will accumulate data in these lists

positions = []
velocities = []
rotations = []
angular_velocities = []
parents = []
names = []

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

        positions = bvh_data['positions']
        rotations = quat.unroll(quat.from_euler(np.radians(bvh_data['rotations']), order=bvh_data['order']))

        # Convert from cm to m
        positions *= 0.01

        if mirror:
            rotations, positions = animation_mirror(rotations, positions, bvh_data['names'], bvh_data['parents'])
            rotations = quat.unroll(rotations)

        nframes = positions.shape[0]
        nbones = positions.shape[1]
