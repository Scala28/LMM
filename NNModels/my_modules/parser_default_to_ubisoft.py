import re

def is_closing_brace_line(line):
    """Return True if the line consists solely of whitespace and a closing brace."""
    return re.match(r'^\s*}\s*$', line) is not None


def parse_hierarchy(lines):
    """Parse BVH hierarchy and remove unwanted joints."""
    new_lines = []
    joint_stack = []  # Keeps track of nested joints
    remove_mode = False  # If True, we are inside an unwanted joint
    removed_indices = []  # Stores motion indices to remove
    joint_channel_counts = []  # Stores number of channels per joint
    joint_index = -1  # Keeps track of joint index in motion data

    for i, line in enumerate(lines):
        stripped_line = line.lstrip()
        indentation = line[:len(line) - len(stripped_line)]  # Preserve indentation
        trimmed = stripped_line.strip()

        if trimmed.startswith("ROOT") or trimmed.startswith("JOINT"):
            # For ROOT and JOINT, the name is the second token.
            # For End Site, we simply use "End Site" as the name.
            joint_name = trimmed.split()[1] if (
                        trimmed.startswith("ROOT") or trimmed.startswith("JOINT")) else "End Site"
            joint_index += 1  # Increase joint index for each joint encountered

            # Remove the part before ':' (if any)
            joint_name = joint_name.split(":")[-1]

            if "Finger" in joint_name:  # Match RightHandXXX / LeftHandXXX
                remove_mode = True
                removed_indices.append(joint_index)
            else:
                new_lines.append(f"{indentation}{trimmed.split()[0]} {joint_name}\n")
                joint_stack.append(joint_name)

        elif "CHANNELS" in trimmed:
            num_channels = int(trimmed.split()[1])  # Extract number of channels for this joint
            joint_channel_counts.append(num_channels)
            if not remove_mode:
                new_lines.append(line)

        elif trimmed == "}":
            if remove_mode:
                remove_mode = False  # End of removed section
            else:
                new_lines.append(line)
                if joint_stack:
                    joint_stack.pop()
        else:
            if not remove_mode:
                new_lines.append(line)

    return new_lines, removed_indices, joint_channel_counts


def cleanup_hierarchy(lines):
    """
    Second loop to clean up hierarchy lines.

    - If a line (after stripping) is "End End Site" and it is immediately followed by two lines
      that consist solely of a closing brace ("}") (ignoring whitespace), then drop these three lines.
    - Otherwise, replace "End End Site" with "End Site", preserving indentation.
    """
    cleaned = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()
        if "End End Site" in stripped_line:
            if stripped_line == "End End Site":
                if i + 2 < len(lines) and lines[i + 1].strip() == "}" and lines[i + 2].strip() == "}":
                    i += 3
                    continue
                else:
                    new_line = line.replace("End End Site", "End Site")
                    cleaned.append(new_line)
            else:
                new_line = line.replace("End End Site", "End Site")
                cleaned.append(new_line)
        else:
            cleaned.append(line)
        i += 1
    return cleaned


def insert_end_site_after_channels(lines):
    """
    Third loop:
    If a line starting with "CHANNEL" (or "CHANNELS") is immediately followed by a line that consists
    only of whitespace and a "}", then insert the following three lines after the CHANNEL line (and before the "}")
    at the same indentation level as the CHANNEL line:

        End Site
        {
        	OFFSET   0.000	0.000	0.000
    """
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        new_lines.append(line)
        # Check if the line starts with CHANNEL (or CHANNELS)
        if line.lstrip().startswith("CHANNEL"):
            # If the next line exists and is solely a closing brace "}"
            if i + 1 < len(lines) and lines[i + 1].strip() == "}":
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}End Site\n")
                new_lines.append(f"{indent}{{\n")
                new_lines.append(f"{indent}\tOFFSET   0.000\t0.000\t0.000\n")
                i += 1  # Skip the "}" line after insertion
        i += 1
    return new_lines

def remove_repeated_brace_couplets(lines):
    """Remove repeated closing brace groups of 4."""
    cleaned = []
    i = 0
    while i < len(lines):
        if i + 3 < len(lines) and all(is_closing_brace_line(lines[j]) for j in range(i, i + 4)):
            base_lines = [lines[j].rstrip('\n') for j in range(i, i + 4)]
            j = i
            repetition_count = 0
            while j + 3 < len(lines) and all(is_closing_brace_line(lines[j + k]) and lines[j + k].rstrip('\n') == base_lines[k] for k in range(4)):
                repetition_count += 1
                j += 4
            if repetition_count > 1:
                i = j
                continue
            else:
                cleaned.extend(lines[i:i+4])
                i += 4
                continue
        else:
            cleaned.append(lines[i])
            i += 1
    return cleaned

def remove_closing_brace_after_offset(lines):
    """Remove a closing brace '}' if it directly follows an 'OFFSET' line at the same indentation level."""
    cleaned = []
    delete_flag = False;
    for i in range(len(lines) - 1):
        if delete_flag:
            delete_flag = False
            continue
        if lines[i].lstrip().startswith("OFFSET"):
            indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
            if lines[i + 1].strip() == "}" and lines[i + 1].startswith(indent):
                delete_flag = True  # Skip adding the closing brace
        cleaned.append(lines[i])
    cleaned.append(lines[-1])  # Add the last line
    return cleaned



def filter_motion(lines, removed_indices, joint_channel_counts):
    """Modify motion data by removing corresponding channels."""
    new_lines = []
    for i, line in enumerate(lines):
        if i < 2:  # First two lines contain 'MOTION' and 'Frames:'
            new_lines.append(line)
            continue
        values = line.split()

        # Compute indices of columns to remove based on varying channel counts
        column_indices_to_remove = []
        column_index = 0
        for joint_idx, num_channels in enumerate(joint_channel_counts):
            if joint_idx in removed_indices:
                column_indices_to_remove.extend(range(column_index, column_index + num_channels))
            column_index += num_channels

        filtered_values = [val for idx, val in enumerate(values) if idx not in column_indices_to_remove]
        new_lines.append(" ".join(filtered_values) + "\n")
    return new_lines


def process_bvh(input_bvh, output_bvh):
    with open(input_bvh, 'r') as f:
        lines = f.readlines()

    # Split hierarchy and motion sections
    motion_start = next(i for i, l in enumerate(lines) if l.strip() == "MOTION")
    hierarchy_lines = lines[:motion_start]
    motion_lines = lines[motion_start:]

    # Process hierarchy
    new_hierarchy, removed_indices, joint_channel_counts = parse_hierarchy(hierarchy_lines)
    new_hierarchy = cleanup_hierarchy(new_hierarchy)
    new_hierarchy = insert_end_site_after_channels(new_hierarchy)
    new_hierarchy = remove_repeated_brace_couplets(new_hierarchy)
    new_hierarchy = remove_closing_brace_after_offset(new_hierarchy)
    # Process motion
    new_motion = filter_motion(motion_lines, removed_indices, joint_channel_counts)

    # Write new BVH file
    with open(output_bvh, 'w') as f:
        f.writelines(new_hierarchy + new_motion)