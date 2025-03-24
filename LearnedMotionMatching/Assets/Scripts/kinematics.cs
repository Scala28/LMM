using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class kinematics
{

    #region IKs
    public static void ik_look_at(ref Vector4 bone_rotation, Pose global_pose,
        Vector3 child_position,
        Vector3 target_position,
        int indx_bone,
        int indx_parent,
        float eps = 1e-5f)
    {
        Vector4 global_parent_rotation = global_pose.joints[indx_parent - 1].rotation;
        Vector4 global_rotation = global_pose.joints[indx_bone - 1].rotation;
        Vector3 global_position = global_pose.joints[indx_bone - 1].position;

        Vector3 curr_dir = Quat.vec_normalize(child_position - global_position);
        Vector3 targ_dir = Quat.vec_normalize(target_position - global_position);

        if (Mathf.Abs(1.0f - Quat.vec_dot(curr_dir, targ_dir)) > eps)
        {
            bone_rotation = Quat.quat_inv_mul(global_parent_rotation, Quat.quat_mul(Quat.quat_between(curr_dir, targ_dir), global_rotation));
        }
    }

    // Basic two-joint IK in the style of https://theorangeduck.com/page/simple-two-joint
    // Here I add a basic "forward vector" which acts like a kind of pole-vetor
    // to control the bending direction
    public static void ik_two_bone(Pose global_pose, ref Pose adjusted_bones_pose,
        Vector3 contact_position_clamp,
        int indx_hips,
        int indx_mid,
        int indx_end,
        int indx_toe,
        int indx_root,
        float max_length_buffer)
    {
        Vector3 bone_root = global_pose.joints[indx_hips - 1].position;
        Vector3 bone_mid = global_pose.joints[indx_mid - 1].position;
        Vector3 bone_end = global_pose.joints[indx_end - 1].position;
        Vector3 target = contact_position_clamp + (global_pose.joints[indx_end - 1].position - global_pose.joints[indx_toe - 1].position);
        Vector3 fwd = Quat.quat_mul_vec(global_pose.joints[indx_mid - 1].rotation, new Vector3(0.0f, 1.0f, 0.0f));
        Vector4 bone_root_gr = global_pose.joints[indx_hips - 1].rotation;
        Vector4 bone_mid_gr = global_pose.joints[indx_mid - 1].rotation;
        Vector4 bone_par_gr = global_pose.joints[indx_root - 1].rotation;

        float max_extension = length(bone_root - bone_mid) + length(bone_mid - bone_end) - max_length_buffer;
        Vector3 target_clamp = target;

        if (length(target - bone_root) > max_extension)
        {
            target_clamp = bone_root + max_extension * Quat.vec_normalize(target - bone_root);
        }

        Vector3 axis_dwn = Quat.vec_normalize(bone_end - bone_root);
        Vector3 axis_rot = Quat.vec_normalize(Quat._cross(axis_dwn, fwd));

        Vector3 a = bone_root;
        Vector3 b = bone_mid;
        Vector3 c = bone_end;
        Vector3 t = target_clamp;

        float lab = length(b - a);
        float lcb = length(b - c);
        float lat = length(t - a);

        float ac_ab_0 = Mathf.Acos(clampf(Quat.vec_dot(Quat.vec_normalize(c - a), Quat.vec_normalize(b - a)), -1.0f, 1.0f));
        float ba_bc_0 = Mathf.Acos(clampf(Quat.vec_dot(Quat.vec_normalize(a - b), Quat.vec_normalize(c - b)), -1.0f, 1.0f));

        float ac_ab_1 = Mathf.Acos(clampf((lab * lab + lat * lat - lcb * lcb) / (2.0f * lab * lat), -1.0f, 1.0f));
        float ba_bc_1 = Mathf.Acos(clampf((lab * lab + lcb * lcb - lat * lat) / (2.0f * lab * lcb), -1.0f, 1.0f));

        Vector4 r0 = Quat.quat_from_angle_axis(ac_ab_1 - ac_ab_0, axis_rot);
        Vector4 r1 = Quat.quat_from_angle_axis(ba_bc_1 - ba_bc_0, axis_rot);

        Vector3 c_a = Quat.vec_normalize(bone_end - bone_root);
        Vector3 t_a = Quat.vec_normalize(target_clamp - bone_root);

        Vector4 r2 = Quat.quat_from_angle_axis(Mathf.Acos(clampf(Quat.vec_dot(c_a, t_a), -1.0f, 1.0f)), Quat.vec_normalize(Quat._cross(c_a, t_a)));

        adjusted_bones_pose.joints[indx_hips - 1].rotation = Quat.quat_inv_mul(bone_par_gr, Quat.quat_mul(r2, Quat.quat_mul(r0, bone_root_gr)));
        adjusted_bones_pose.joints[indx_mid - 1].rotation = Quat.quat_inv_mul(bone_root_gr, Quat.quat_mul(r1, bone_mid_gr));
    }
    #endregion

    #region FKs
    public static void forward_kinamatic_full(DataManager.database db, ref Pose global_pose, Pose adjusted_bones_pose)
    {
        for (int i = 0; i < db.bone_parents.Length; i++)
        {
            Debug.Assert(db.bone_parents[i] < i);
            if (db.bone_parents[i] == -1)
            {
                global_pose.root_position = adjusted_bones_pose.root_position;
                global_pose.root_rotation = adjusted_bones_pose.root_rotation;
            }
            else
            {
                Vector3 parent_position = db.bone_parents[i] == 0 ? global_pose.root_position :
                    global_pose.joints[db.bone_parents[i] - 1].position;
                Vector4 parent_rotation = db.bone_parents[i] == 0 ? global_pose.root_rotation :
                    global_pose.joints[db.bone_parents[i] - 1].rotation;

                global_pose.joints[i - 1].position = Quat.quat_mul_vec(parent_rotation, adjusted_bones_pose.joints[i - 1].position) + parent_position;
                global_pose.joints[i - 1].rotation = Quat.quat_mul(parent_rotation, adjusted_bones_pose.joints[i - 1].rotation);
            }
        }
    }
    public static void forward_kinematics_velocity(out Vector3 bone_pos,
                                             out Vector3 bone_vel,
                                             out Vector4 bone_rot,
                                             out Vector3 bone_ang_vel,
                                             int bone,
                                             DataManager.database db,
                                             Pose pose)
    {
        if (db.bone_parents[bone] != -1)
        {
            Vector3 parent_pos;
            Vector3 parent_vel;
            Vector4 parent_rot;
            Vector3 parent_ang_vel;

            forward_kinematics_velocity(out parent_pos, out parent_vel, out parent_rot, out parent_ang_vel,
                db.bone_parents[bone], db, pose);

            bone_pos = Quat.quat_mul_vec(parent_rot, pose.joints[bone - 1].position) + parent_pos;
            bone_vel = parent_vel + Quat.quat_mul_vec(parent_rot, pose.joints[bone - 1].velocity) +
                Quat._cross(parent_ang_vel, Quat.quat_mul_vec(parent_rot, pose.joints[bone - 1].position));
            bone_rot = Quat.quat_mul(parent_rot, pose.joints[bone - 1].rotation);
            bone_ang_vel = Quat.quat_mul_vec(parent_rot, pose.joints[bone - 1].angular_velocity) + parent_ang_vel;
        }
        else
        {
            bone_pos = pose.root_position;
            bone_vel = pose.root_velocity;
            bone_rot = pose.root_rotation;
            bone_ang_vel = pose.root_angular_velocity;
        }
    }
    public static void forward_kinematic_partial(Pose input_pose, int bone, ref Pose global_pose, ref bool[] global_bone_computed, DataManager.database db)
    {
        if (db.bone_parents[bone] == -1)
        {
            global_pose.root_position = input_pose.root_position;
            global_pose.root_rotation = input_pose.root_rotation;
            global_bone_computed[bone] = true;
            return;
        }
        if (!global_bone_computed[db.bone_parents[bone]])
        {
            forward_kinematic_partial(input_pose, db.bone_parents[bone], ref global_pose, ref global_bone_computed, db);
        }
        if (db.bone_parents[bone] == 0)
        {
            Vector3 parent_pos = global_pose.root_position;
            Vector4 parent_rot = global_pose.root_rotation;
            global_pose.joints[bone - 1].position = Quat.quat_mul_vec(parent_rot, input_pose.joints[bone - 1].position)
                + parent_pos;
            global_pose.joints[bone - 1].rotation = Quat.quat_mul(parent_rot, input_pose.joints[bone - 1].rotation);
        }
        else
        {
            Vector3 parent_pos = global_pose.joints[db.bone_parents[bone] - 1].position;
            Vector4 parent_rot = global_pose.joints[db.bone_parents[bone] - 1].rotation;
            global_pose.joints[bone - 1].position = Quat.quat_mul_vec(parent_rot, input_pose.joints[bone - 1].position)
                + parent_pos;
            global_pose.joints[bone - 1].rotation = Quat.quat_mul(parent_rot, input_pose.joints[bone - 1].rotation);
        }
        global_bone_computed[bone] = true;
    }
    #endregion

    private static float clampf(float x, float min, float max) { return x > max ? max : x < min ? min : x; }
    private static float length(Vector3 v) { return Mathf.Sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
}
