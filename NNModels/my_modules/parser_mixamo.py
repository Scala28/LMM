import re


def is_closing_brace_line(line):
    """Return True if the line consists solely of whitespace and a closing brace."""
    return re.match(r'^\s*}\s*$', line) is not None


def parse_hierarchy(lines, joints_to_remove, add_toe):
    """Parse BVH hierarchy and remove unwanted joints."""
    new_lines = []
    joint_stack = []  # Keeps track of nested joints
    remove_mode = False  # If True, we are inside an unwanted joint
    removed_indices = []  # Stores motion indices to remove
    removed_stack = []
    joint_channel_counts = []  # Stores number of channels per joint
    end_site = False
    joint_index = -1  # Keeps track of joint index in motion data
    toe_indices = []
    toe_offsets = []

    for i, line in enumerate(lines):
        stripped_line = line.lstrip()
        indentation = line[:len(line) - len(stripped_line)]  # Preserve indentation
        trimmed = stripped_line.strip()

        if trimmed.startswith("ROOT") or trimmed.startswith("JOINT"):
            joint_index += 1  # Increase joint index for each joint encountered

            # For ROOT and JOINT, the name is the second token.
            joint_name = trimmed.split()[1]
            # Remove the part before ':' (if any)
            joint_name = joint_name.split(":")[-1]

            # fingers_list = ["HandPinky", "HandIndex", "HandRing", "HandMiddle", "HandThumb", "HeadTop"]
            # Remove joints
            if any(word in joint_name for word in joints_to_remove):
                remove_mode = True
                removed_indices.append(joint_index)
                removed_stack.append(joint_name)
            else:
                new_lines.append(f"{indentation}{trimmed.split()[0]} {joint_name}\n")
                joint_stack.append(joint_name)
        elif trimmed.startswith("End"):
            end_site = True
        elif "CHANNELS" in trimmed:
            num_channels = int(trimmed.split()[1])  # Extract number of channels for this joint
            joint_channel_counts.append(num_channels)
            if not remove_mode:
                new_lines.append(line)
        elif "OFFSET" in trimmed:
            if add_toe:
                if not remove_mode and not end_site:
                    new_lines.append(line)
                elif not remove_mode:
                    if joint_stack[-1] == "LeftFoot":
                        toe_indices.append(joint_index + 1)
                        toe_offsets.append(trimmed)
                    elif joint_stack[-1] == "RightFoot":
                        toe_indices.append(joint_index + 2)
                        toe_offsets.append(trimmed)
            else:
                if not remove_mode and not end_site:
                    new_lines.append(line)

        else:
            if not remove_mode and not end_site:
                new_lines.append(line)
                if trimmed == "}":
                    joint_stack.pop()
            else:
                if trimmed == "}":
                    if end_site:
                        end_site = False
                    else:
                        if remove_mode:
                            removed_stack.pop()
                            if len(removed_stack) == 0:
                                remove_mode = False
    print("joints removed")

    if add_toe:
        new_lines, joint_channel_counts = add_toe_hierachy(new_lines, toe_indices, toe_offsets, joint_channel_counts)
        for i, idx in enumerate(removed_indices):
            if idx > toe_indices[0]:
                removed_indices[i] = idx + 1
            if idx > toe_indices[1]:
                removed_indices[i] = idx + 1
        print("added toe joints")
    return add_end_site(new_lines), removed_indices, joint_channel_counts, toe_indices


def add_end_site(lines):
    """
        second loop to hierarchy lines in order to add
            End Site
            {
                OFFSET 0.000000 0.000000 0.000000
            }
        when find
            CHANNELS ...
        }
    """
    new_lines = []
    last_line_channel = False
    for i, line in enumerate(lines):
        stripped_line = line.lstrip()
        indentation = line[:len(line) - len(stripped_line)]  # Preserve indentation
        trimmed = stripped_line.strip()

        if trimmed.startswith("CHANNELS"):
            new_lines.append(line)
            last_line_channel = True
        elif trimmed.startswith("}"):
            if last_line_channel:
                """"Add End Site with an + indentation and then reclose }"""
                indent = indentation + "\t"
                new_lines.append(f"{indent}End Site\n")
                new_lines.append(f"{indent}{{\n")
                new_lines.append(f"{indent}\tOFFSET 0.000000 0.000000 0.000000\n")
                new_lines.append(f"{indent}}}\n")

            new_lines.append(line)
            last_line_channel = False
        else:
            new_lines.append(line)
            last_line_channel = False

    return new_lines


def add_toe_hierachy(lines, toe_indices, toe_offsets, joints_channel_count):
    new_lines = []
    i = 0
    joint_index = -1  # Track joint indices for motion data
    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()
        new_lines.append(line)
        # Track joint index for motion updates
        if "JOINT" in stripped_line or "ROOT" in stripped_line:
            joint_index += 1
        is_channel = "CHANNELS" in stripped_line

        if joint_index == (toe_indices[0] - 1) and is_channel:  # LeftFoot
            indent = line[:len(line) - len(line.lstrip())]  # Extract indentation
            new_lines.append(f"{indent}JOINT LeftToe\n")
            new_lines.append(f"{indent}{{\n")
            new_lines.append(f"{indent}\t{toe_offsets[0]}\n")
            new_lines.append(f"{indent}\tCHANNELS	3	Yrotation	Xrotation	Zrotation\n")
            new_lines.append(f"{indent}}}\n")
            joint_index += 1
        elif joint_index == (toe_indices[1] - 1) and is_channel:  # RightFoot
            indent = line[:len(line) - len(line.lstrip())]  # Extract indentation
            new_lines.append(f"{indent}JOINT RighToe\n")
            new_lines.append(f"{indent}{{\n")
            new_lines.append(f"{indent}\t{toe_offsets[1]}\n")
            new_lines.append(f"{indent}\tCHANNELS	3	Yrotation	Xrotation	Zrotation\n")
            new_lines.append(f"{indent}}}\n")
            joint_index += 1

        i += 1  # Move to the next line

    joints_channel_count.insert(toe_indices[0], 3)
    joints_channel_count.insert(toe_indices[1], 3)

    return new_lines, joints_channel_count


def process_motion(lines, removed_indices, joint_channel_counts, add_toe, toe_indices):
    print("processing motion")
    """Modify motion data by removing corresponding channels."""
    new_lines = []
    for i, line in enumerate(lines):
        if i < 3:  # First two lines contain 'MOTION' and 'Frames:'
            new_lines.append(line)
            continue
        values = line.split()
        # Compute indices of columns to remove based on varying channel counts
        column_indices_to_remove = []
        column_index = 0
        values_index = 0
        updated_values = []
        for joint_idx, num_channels in enumerate(joint_channel_counts):
            if joint_idx in removed_indices:
                column_indices_to_remove.extend(range(column_index, column_index + num_channels))
            if add_toe and (joint_idx == toe_indices[0] or joint_idx == toe_indices[1]):
                updated_values.extend(["0.000", "0.000", "0.000"])
            else:
                updated_values.extend(values[values_index: values_index + num_channels])
                values_index += num_channels
            column_index += num_channels
        filtered_values = [val for idx, val in enumerate(updated_values) if idx not in column_indices_to_remove]
        new_lines.append(" ".join(filtered_values) + "\n")

    return new_lines


def process_bvh(input_bvh, output_bvh, add_toe=False):
    with open(input_bvh, 'r') as f:
        lines = f.readlines()

    # Split hierarchy and motion sections
    motion_start = next(i for i, l in enumerate(lines) if l.strip() == "MOTION")
    hierarchy_lines = lines[:motion_start]
    motion_lines = lines[motion_start:]

    # Process hierarchy
    joints_to_remove = ["HandPinky", "HandIndex", "HandRing", "HandMiddle", "HandThumb", "_End"]
    new_hierarchy, removed_indices, joint_channel_counts, toe_indices = (
        parse_hierarchy(hierarchy_lines, joints_to_remove, add_toe))

    # Process motion
    new_motion = process_motion(motion_lines, removed_indices, joint_channel_counts, add_toe, toe_indices)

    print("Writing file")
    # Write new BVH file
    with open(output_bvh, 'w') as f:
        f.writelines(new_hierarchy + new_motion)

