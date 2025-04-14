import csv
import numpy as np
import re


def parse_hierarchy(lines):
    names = []
    parents = []

    joint_stack = [-1]
    joint_index = -1

    for i, line in enumerate(lines):
        stripped_line = line.lstrip()
        indentation = line[:len(line) - len(stripped_line)]  # Preserve indentation
        trimmed = stripped_line.strip()

        if trimmed.find(':') != -1:
            joint_index += 1
            parents.append(joint_stack[-1])
            joint_stack.append(joint_index)

            name = trimmed.split(':')[-1][:-2]
            names.append(name)
        elif trimmed.startswith(']'):
            joint_stack.pop()

    return {
        'names': names,
        'parents': parents
    }


def parse_motion(lines, joints):
    positions = np.zeros([len(lines), len(joints), 3])
    rotations = np.zeros([len(lines), len(joints), 4])

    dt = lines[0].split(',')[0]

    for (i, l) in enumerate(lines):
        data = l.split(',')
        frame_pose = data[1:]
        for k in range(len(joints)-1):
            joint_pos = frame_pose[k*7:k*7+3]
            joint_rot = frame_pose[k*7+3:k*7+7]

            positions[i, k] = joint_pos
            # Switch quat order (x, y, z, w) -> (w, x, y, z)
            rotations[i, k] = np.concatenate([joint_rot[3:4], joint_rot[:3]], axis=0)

    positions = positions.astype(np.float32)
    rotations = rotations.astype(np.float32)

    return {
        'positions': positions,
        'rotations': rotations,
        'offsets': positions[0],
        'dt': dt
    }


def load_csv(animation):
    with open(animation, 'r') as f:
        lines = f.readlines()

    # Split hierarchy and motion sections
    header_line = next(i for i, l in enumerate(lines) if l.strip() == "HEADER")
    hierarchy_lines = lines[1:header_line]
    motion_lines = lines[header_line+3:]

    hierarchy_data = parse_hierarchy(hierarchy_lines)
    motion_data = parse_motion(motion_lines, hierarchy_data['names'])

    return {
        'positions': motion_data['positions'],
        'rotations': motion_data['rotations'],
        'names': hierarchy_data['names'],
        'parents': hierarchy_data['parents'],
        'offsets': motion_data['offsets']
    }

