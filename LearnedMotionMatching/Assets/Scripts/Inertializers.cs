using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class Inertializers
{
    public static void inertialize_pose_reset(ref Vector3[] bone_offset_positions, ref Vector4[] bone_offset_rotations, ref Vector3[] bone_offset_velocities, ref Vector3[] bone_offset_angular_velocities,
        ref Vector3 transition_src_position, ref Vector4 transition_src_rotation, ref Vector3 transition_dst_position, ref Vector4 transition_dst_rotation, Pose pose, DataManager.database db)
    {
        for (int i = 0; i < db.nbones(); i++)
        {
            bone_offset_positions[i] = Vector3.zero;
            bone_offset_rotations[i] = new Vector4(1.0f, .0f, .0f, .0f);
            bone_offset_velocities[i] = Vector3.zero;
            bone_offset_angular_velocities[i] = Vector3.zero;
        }

        transition_src_position = pose.root_position;
        transition_src_rotation = pose.root_rotation;
        transition_dst_position = Vector3.zero;
        transition_dst_rotation = new Vector4(1.0f, .0f, .0f, .0f);
    }
    public static void inertialize_pose_transition(ref Vector3[] bone_offset_positions, ref Vector4[] bone_offset_rotations, ref Vector3[] bone_offset_velocities, ref Vector3[] bone_offset_angular_velocities,
        ref Vector3 transition_src_position, ref Vector4 transition_src_rotation, ref Vector3 transition_dst_position, ref Vector4 transition_dst_rotation, Pose pose, Pose current_pose, Pose trns_pose, DataManager.database db)
    {
        transition_dst_position = pose.root_position;
        transition_dst_rotation = pose.root_rotation;
        transition_src_position = trns_pose.root_position;
        transition_src_rotation = trns_pose.root_rotation;

        Vector3 world_space_dst_vel = Quat.quat_mul_vec(transition_dst_rotation,
            Quat.quat_inv_mul_vec(transition_src_rotation, trns_pose.root_velocity));
        Vector3 world_space_dst_angular_vel = Quat.quat_mul_vec(transition_dst_rotation,
            Quat.quat_inv_mul_vec(transition_src_rotation, trns_pose.root_angular_velocity));

        Spring.inertialize_transition(
            ref bone_offset_positions[0],
            ref bone_offset_velocities[0],
            pose.root_position,
            pose.root_velocity,
            pose.root_position,
            world_space_dst_vel);
        Spring.inertialize_transition(
            ref bone_offset_rotations[0],
            ref bone_offset_angular_velocities[0],
            pose.root_rotation,
            pose.root_angular_velocity,
            pose.root_rotation,
            world_space_dst_angular_vel);

        for (int i = 1; i < db.nbones(); i++)
        {
            Spring.inertialize_transition(
                ref bone_offset_positions[i],
                ref bone_offset_velocities[i],
                current_pose.joints[i - 1].position,
                current_pose.joints[i - 1].velocity,
                trns_pose.joints[i - 1].position,
                trns_pose.joints[i - 1].velocity);
            Spring.inertialize_transition(
                ref bone_offset_rotations[i],
                ref bone_offset_angular_velocities[i],
                current_pose.joints[i - 1].rotation,
                current_pose.joints[i - 1].angular_velocity,
                trns_pose.joints[i - 1].rotation,
                trns_pose.joints[i - 1].angular_velocity);
        }
    }
    public static void inertialize_pose_update(ref Vector3[] bone_offset_positions, ref Vector4[] bone_offset_rotations, ref Vector3[] bone_offset_velocities, ref Vector3[] bone_offset_angular_velocities,
        ref Vector3 transition_src_position, ref Vector4 transition_src_rotation, ref Vector3 transition_dst_position, ref Vector4 transition_dst_rotation, Pose pose, DataManager.database db, Pose input_pose, float _dt, float inertialize_blending_halflife)
    {

        Vector3 world_space_pos = Quat.quat_mul_vec(transition_dst_rotation,
            Quat.quat_inv_mul_vec(transition_src_rotation, input_pose.root_position - transition_src_position)) + transition_dst_position;
        Vector3 world_space_vel = Quat.quat_mul_vec(transition_dst_rotation,
            Quat.quat_inv_mul_vec(transition_src_rotation, input_pose.root_velocity));

        Vector4 world_space_rot = Quat.quat_normalize(Quat.quat_mul(transition_dst_rotation,
            Quat.quat_inv_mul(transition_src_rotation, input_pose.root_rotation)));
        Vector3 world_space_angular_vel = Quat.quat_mul_vec(transition_dst_rotation,
            Quat.quat_inv_mul_vec(transition_src_rotation, input_pose.root_angular_velocity));

        Spring.inertialize_update(
            ref pose.root_position,
            ref pose.root_velocity,
            ref bone_offset_positions[0],
            ref bone_offset_velocities[0],
            world_space_pos,
            world_space_vel,
            inertialize_blending_halflife,
            _dt);
        Spring.inertialize_update(
            ref pose.root_rotation,
            ref pose.root_angular_velocity,
            ref bone_offset_rotations[0],
            ref bone_offset_angular_velocities[0],
            world_space_rot,
            world_space_angular_vel,
            inertialize_blending_halflife,
            _dt);

        for (int i = 1; i < db.nbones(); i++)
        {
            Spring.inertialize_update(
                ref pose.joints[i - 1].position,
                ref pose.joints[i - 1].velocity,
                ref bone_offset_positions[i],
                ref bone_offset_velocities[i],
                input_pose.joints[i - 1].position,
                input_pose.joints[i - 1].velocity,
                inertialize_blending_halflife,
                _dt);
            Spring.inertialize_update(
                ref pose.joints[i - 1].rotation,
                ref pose.joints[i - 1].angular_velocity,
                ref bone_offset_rotations[i],
                ref bone_offset_angular_velocities[i],
                input_pose.joints[i - 1].rotation,
                input_pose.joints[i - 1].angular_velocity,
                inertialize_blending_halflife,
                _dt);
        }
    }
    public static void inertialize_root_adjust(ref Pose pose, ref Vector3 transition_src_position, ref Vector3 transition_dst_position, Vector4 transition_src_rotation, ref Vector4 transition_dst_rotation, 
        ref Vector3[] bone_offset_positions, Vector3 input_position, Vector4 input_rotation)
    {
        // Find the position difference and add it to the state and transition location
        Vector3 position_difference = input_position - pose.root_position;
        pose.root_position += position_difference;
        transition_dst_position += position_difference;

        // Find the point at which we want to now transition from in the src data
        transition_src_position = transition_src_position + Quat.quat_mul_vec(transition_src_rotation,
            Quat.quat_inv_mul_vec(transition_dst_rotation, pose.root_position - bone_offset_positions[0] - transition_dst_position));

        transition_dst_position = pose.root_position;
        bone_offset_positions[0] = new Vector3();

        // Find the rotation difference. We need to normalize here or some error can accumulate 
        // over time during adjustment.
        Vector4 rotation_difference = Quat.quat_normalize(Quat.quat_mul_inv(input_rotation, pose.root_rotation));

        // Apply the rotation difference to the current rotation and transition location
        pose.root_rotation = Quat.quat_mul(rotation_difference, pose.root_rotation);
        transition_dst_rotation = Quat.quat_mul(rotation_difference, transition_dst_rotation);
    }
}
