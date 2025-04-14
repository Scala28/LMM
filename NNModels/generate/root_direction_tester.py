import sys
import os
import main_settings as ms


def extract_root_joint_bvh(input_bvh, output_bvh):
    with open(input_bvh, 'r') as file:
        lines = file.readlines()

    hierarchy = lines[:7]  # Keep only the first 5 lines of the hierarchy
    hierarchy.append("\t\tOFFSET 0.0000 0.00000 50.0000\n")
    hierarchy.append(lines[8])  # Copy original line 8 from the BVH file
    hierarchy.append("\t\tEnd Site\n")  # Add End Site with 1 tab indentation
    hierarchy.append("\t\t{\n")  # Add opening brace with 1 tab indentation
    hierarchy.append("\t\t\tOFFSET 0.0000 0.00000 0.0000\n")
    hierarchy.append("\t\t}\n")  # Add closing brace with 1 tab indentation
    hierarchy.append("\t}\n")  # Add opening brace with 1 tab indentation
    hierarchy.append("}\n")  # Add final closing brace

    motion = ["MOTION\n"]
    recording_motion = False

    for line in lines:
        if "Frames:" in line:
            motion.append(line)
            continue
        elif "Frame Time:" in line:
            recording_motion = True
            motion.append(line)
            continue

        if recording_motion:
            parts = line.strip().split()
            if len(parts) >= 9:  # Ensure at least 6 channels exist
                motion.append(" ".join(parts[:9]) + "\n")  # Keep only first 6 entries

    # Write new BVH file
    with open(output_bvh, 'w') as file:
        file.writelines(hierarchy)
        file.writelines(motion)


anim_target_dir = "generate/data/{0}/{1}/".format(ms.controller_type, ms.animation_type)
extract_root_joint_bvh(anim_target_dir + "database.bvh".format(ms.controller_type),
                       (anim_target_dir + "root_direction.bvh").format(ms.controller_type))
