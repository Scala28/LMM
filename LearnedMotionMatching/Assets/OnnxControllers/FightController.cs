using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

[CreateAssetMenu(fileName = "FightController", menuName = "OnnxMotionController/Fight")]
public class FightController : MotionController
{
    [Header("Fight-stance")]
    // All speeds in m/s
    public float simulation_std_frwd_speed = 1.0f;
    public float simulation_std_side_speed = .8f;
    public float simulation_std_back_speed = .5f;

    float simulation_fwrd_speed;
    float simulation_side_speed;
    float simulation_back_speed;

    public ControllerOrchestrator.character busto_bone;
    private Vector3 offset_busto_zero = new Vector3(.02f, 0, .04f);

    private float lock_target = 0.0f;
    private float lock_target_velocity = 0.0f;

    public LayerMask whatIsTerrain;

    [Header("Fight")]
    public float std_camera_altitude = .4f;
    private Transform Target;


    #region Trajectory & Gameplay Data
    public void lock_spring(bool target_lock, float lock_velocity = 0.1f)
    {
        Spring.simple_spring_damper_exact(
            ref lock_target,
            ref lock_target_velocity,
            target_lock ? 1.0f : 0.0f,
            lock_velocity,
            dt);
    }

    public Vector3 desired_velocity_update(Vector3 gamepad_stickleft, float camera_azimuth, Vector3 simulation_rotation)
    {
        // Find stick position in world space by rotating using camera azimuth
        Vector3 global_stick_direction = Quat.quat_mul_vec(
            Quat.quat_from_angle_axis(camera_azimuth, new Vector3(0f, 1.0f, 0f)), gamepad_stickleft);

        // Find stick position local to current facing direction
        Vector3 local_stick_direction = Quat.quat_inv_mul_vec(
            simulation_rotation, global_stick_direction);

        // Scale stick by forward, sideways and backwards speeds
        Vector3 local_desired_velocity = local_stick_direction.z > 0.0 ?
            new Vector3(simulation_side_speed * local_stick_direction.x, 0.0f, simulation_fwrd_speed * local_stick_direction.z) :
            new Vector3(simulation_side_speed * local_stick_direction.x, 0.0f, simulation_back_speed * local_stick_direction.z);

        return Quat.quat_mul_vec(simulation_rotation, local_desired_velocity);
    }
    public Vector4 desired_rotation_update(Vector4 desired_rotation, Vector3 gamepad_stickleft, float camera_azimuth, bool target_lock, Vector3 desired_velocity)
    {
        Vector4 desired_rotation_curr = desired_rotation;
        if (target_lock)
        {
            Vector3 desired_dir = Quat.quat_mul_vec(Quat.quat_from_angle_axis(camera_azimuth, new Vector3(0f, 1f, 0f)), new Vector3(0f, 0f, 1f));
            return Quat.quat_from_angle_axis(Mathf.Atan2(desired_dir.x, desired_dir.z), new Vector3(0f, 1f, 0f));
        }
        // stick as long as that stick is being used
        else if (length(gamepad_stickleft) > 0.01f)
        {
            Vector3 desired_dir = Quat.vec_normalize(desired_velocity);
            return Quat.quat_from_angle_axis(Mathf.Atan2(desired_dir.x, desired_dir.z), new Vector3(0f, 1f, 0f));
        }
        // Otherwise desired direction remains the same
        else
        {
            return desired_rotation_curr;
        }
    }
    private void simulation_rotation_update(ref Vector4 rotation, ref Vector3 angular_velocity, Vector4 desired_rotation, float halflife, float dt)
    {
        Spring.simple_spring_damper_exact(
            ref rotation,
            ref angular_velocity,
            desired_rotation,
            halflife,
            dt);
    }
    private void simulation_position_update(ref Vector3 position, ref Vector3 velocity, ref Vector3 acceleration, Vector3 desired_velocity, float halflife, float dt)
    {
        float y = Spring.halflife_to_damping(halflife) / 2.0f;
        Vector3 j0 = velocity - desired_velocity;
        Vector3 j1 = acceleration + j0 * y;
        float eydt = Spring.fast_negexpf(y * dt);

        Vector3 position_prev = position;

        position = eydt * (((-j1) / (y * y)) + ((-j0 - j1 * dt) / y)) +
            (j1 / (y * y)) + j0 / y + desired_velocity * dt + position_prev;
        velocity = eydt * (j0 + j1 * dt) + desired_velocity;
        acceleration = eydt * (acceleration - j1 * y * dt);
    }
    public void trajectory_desired_rotations_predict(Vector3 gamepadstick_left, Vector3 gamepadstick_right, float camera_azimuth, bool target_lock, float dt)
    {
        trajectory_desired_rotations[0] = desired_rotation;

        for (int i = 1; i < trajectory_desired_rotations.Length; i++)
        {
            trajectory_desired_rotations[i] = desired_rotation_update(
                trajectory_desired_rotations[i - 1],
                gamepadstick_left,
                orbit_camera_azimuth(camera_azimuth, target_lock, gamepadstick_right, dt),
                target_lock,
                trajectory_desired_velocities[i]);
        }
    }
    public void trajectory_rotations_predict(float dt)
    {
        for (int i = 0; i < trajectory_rotations.Length; i++)
        {
            trajectory_rotations[i] = simulation_rotation;
            trajectory_angular_velocities[i] = simulation_angular_velocity;
        }

        for (int i = 1; i < trajectory_rotations.Length; i++)
        {
            simulation_rotation_update(
                ref trajectory_rotations[i],
                ref trajectory_angular_velocities[i],
                trajectory_desired_rotations[i],
                simulation_rotation_halflife,
                i * dt);
        }
    }
    public void trajectory_desired_velocities_predict(Vector3 gamepadstick_left, Vector3 gamepadstick_right, float camera_azimuth, bool target_lock, float dt)
    {
        trajectory_desired_velocities[0] = desired_velocity;
        for (int i = 1; i < trajectory_desired_velocities.Length; i++)
        {
            trajectory_desired_velocities[i] = desired_velocity_update(
                gamepadstick_left,
                orbit_camera_azimuth(camera_azimuth, target_lock, gamepadstick_right, dt),
                trajectory_rotations[i]);
        }
    }
    public void trajectory_positions_predict(float dt)
    {
        trajectory_positions[0] = simulation_position;
        trajectory_velocities[0] = simulation_velocity;
        trajectory_accelerations[0] = simulation_acceleration;

        for (int i = 1; i < trajectory_positions.Length; i++)
        {
            trajectory_positions[i] = trajectory_positions[i - 1];
            trajectory_velocities[i] = trajectory_velocities[i - 1];
            trajectory_accelerations[i] = trajectory_accelerations[i - 1];

            simulation_position_update(
                ref trajectory_positions[i],
                ref trajectory_velocities[i],
                ref trajectory_accelerations[i],
                trajectory_desired_velocities[i],
                simulation_velocity_halflife,
                dt);
        }
    }
    public (float[], int) compute_query_vector(Vector3 rightStick, bool target_lock)
    {
        float[] query = new float[db.nfeatures()];
        int offset = 0;

        // query_copy_denormalized_feature
        // Left foot pos
        for (int i = 0; i < 3; i++)
        {
            query[offset + i] = feature_curr[offset + i] * db.features_scale[offset + i] + db.features_offset[offset + i];
        }
        offset += 3;

        // Right foot pos
        for (int i = 0; i < 3; i++)
        {
            query[offset + i] = feature_curr[offset + i] * db.features_scale[offset + i] + db.features_offset[offset + i];
        }
        offset += 3;

        // Left foot velocity
        for (int i = 0; i < 3; i++)
        {
            query[offset + i] = feature_curr[offset + i] * db.features_scale[offset + i] + db.features_offset[offset + i];
        }
        offset += 3;

        // Right foot velocity
        for (int i = 0; i < 3; i++)
        {
            query[offset + i] = feature_curr[offset + i] * db.features_scale[offset + i] + db.features_offset[offset + i];
        }
        offset += 3;

        // Hip velocity
        for (int i = 0; i < 3; i++)
        {
            query[offset + i] = feature_curr[offset + i] * db.features_scale[offset + i] + db.features_offset[offset + i];
        }
        offset += 3;

        // query_compute_trajectory_position_feature
        Vector3 traj0 = Quat.quat_inv_mul_vec(base.pose.root_rotation, trajectory_positions[1] - base.pose.root_position);
        Vector3 traj1 = Quat.quat_inv_mul_vec(base.pose.root_rotation, trajectory_positions[2] - base.pose.root_position);
        Vector3 traj2 = Quat.quat_inv_mul_vec(base.pose.root_rotation, trajectory_positions[3] - base.pose.root_position);

        query[offset + 0] = traj0.x;
        query[offset + 1] = traj0.z;
        query[offset + 2] = traj1.x;
        query[offset + 3] = traj1.z;
        query[offset + 4] = traj2.x;
        query[offset + 5] = traj2.z;

        offset += 6;

        // query_compute_trajectory_direction_feature
        Vector3 dir0 = Quat.quat_inv_mul_vec(base.pose.root_rotation, Quat.quat_mul_vec(trajectory_rotations[1], new Vector3(0, 0, 1f)));
        Vector3 dir1 = Quat.quat_inv_mul_vec(base.pose.root_rotation, Quat.quat_mul_vec(trajectory_rotations[2], new Vector3(0, 0, 1f)));
        Vector3 dir2 = Quat.quat_inv_mul_vec(base.pose.root_rotation, Quat.quat_mul_vec(trajectory_rotations[3], new Vector3(0, 0, 1f)));

        query[offset + 0] = dir0.x;
        query[offset + 1] = dir0.z;
        query[offset + 2] = dir1.x;
        query[offset + 3] = dir1.z;
        query[offset + 4] = dir2.x;
        query[offset + 5] = dir2.z;

        offset += 6;

        // Compute torso local position
        Vector3 hips_gp = global_pose.joints[(int)ControllerOrchestrator.character.Bone_Hips - 1].position;
        Vector4 hips_gr = global_pose.joints[(int)ControllerOrchestrator.character.Bone_Hips - 1].rotation;
        Vector3 busto_gp = global_pose.joints[(int)busto_bone - 1].position;
        hips_gp.y = 0f;
        busto_gp.y = 0f;

        Vector3 offset_busto_zero = new Vector3(0, 0, 0.02f);
        Vector3 input_torso;
        if (!target_lock)
            input_torso = Quat.quat_mul_vec(hips_gr, offset_busto_zero);
        else
            input_torso = Quat.quat_mul_vec(hips_gr, rightStick / 5f + offset_busto_zero);

        Vector3 torso_relative_position = Quat.quat_inv_mul_vec(hips_gr, busto_gp - hips_gp);
        Vector3 input_torso_relative_position = Quat.quat_inv_mul_vec(hips_gr, input_torso);

        query[offset + 0] = torso_relative_position.x;
        query[offset + 1] = torso_relative_position.z;
        query[offset + 2] = torso_relative_position.x + (input_torso_relative_position.x - torso_relative_position.x) * 2f / 3f;
        query[offset + 3] = torso_relative_position.z + (input_torso_relative_position.z - torso_relative_position.z) * 2f / 3f;
        query[offset + 4] = input_torso_relative_position.x;
        query[offset + 5] = input_torso_relative_position.z;
        offset += 6;

        return (query, offset);
    }
    private float orbit_camera_azimuth(float azimuth, bool target_lock, Vector3 gamepadstick_right, float dt)
    {
        if (target_lock)
        {
            // Compute the vector from camera to target
            Vector3 direction = (Target.position - controller.vcam.transform.position);
            direction.y = 0; // Project onto the XZ plane to get the azimuthal direction

            if (direction.sqrMagnitude > 0.0001f) // Avoid division by zero
            {
                direction = direction.normalized;
                azimuth = Mathf.Atan2(direction.x, direction.z); // Compute azimuth angle
            }

            return azimuth;
        }
        else
        {
            Vector3 gamepadaxis = gamepadstick_right;
            return azimuth + 2.0f * dt * gamepadaxis.x;
        }
    }
    private float orbit_camera_altitude(float altitude,  bool target_lock, Vector3 gamepadstick_right, float dt)
    {
        if(target_lock)
        {
            Vector3 gamepadaxis = gamepadstick_right;
            return clampf(std_camera_altitude + 2.0f * dt * gamepadaxis.z, 0.0f, 0.4f * Mathf.PI);
        }
        else
        {
            Vector3 gamepadaxis = gamepadstick_right;
            return clampf(altitude + 2.0f * dt * gamepadaxis.z, 0.0f, 0.4f * Mathf.PI);
        }
    }
    private float orbit_camera_distance(float distance, float dt)
    {
        float gamepadzoom = 0.0f;
        return clampf(distance + 10f * dt * gamepadzoom, 0.1f, 100.0f);
    }
    public (Vector3, Vector3) orbit_camera_update(Vector3 target, Vector3 gamepadstick_right, bool target_lock, float dt)
    {
        camera_azimuth = orbit_camera_azimuth(camera_azimuth, target_lock, gamepadstick_right, dt);
        camera_altitude = orbit_camera_altitude(camera_altitude, target_lock, gamepadstick_right, dt);
        camera_distance = orbit_camera_distance(camera_distance, dt);

        Vector4 rotation_azimuth = Quat.quat_from_angle_axis(camera_azimuth, new Vector3(0, 1f, 0));

        Vector3 position = Quat.quat_mul_vec(rotation_azimuth, new Vector3(0, 0, -camera_distance));
        Vector3 axis = Quat.vec_normalize(Quat._cross(position, new Vector3(0, 1f, 0)));
        Vector4 rotation_altitude = Quat.quat_from_angle_axis(camera_altitude, axis);
        Vector3 eye = target + Quat.quat_mul_vec(rotation_altitude, position);

        return (eye, target);
    }
    #endregion


    public override (Pose, float[], float[]) perform_cycle()
    {
        Vector3 stickLeft = controller.input_handler.StickLeft;
        Vector3 stickRight = controller.input_handler.StickRight;
        bool target_lock = false;
        if(controller.input_handler.LeftTrigger)
            try
            {
                Target = GameObject.FindGameObjectWithTag("target").transform;
                target_lock = true;

            }
            catch { }

        lock_spring(target_lock);

        simulation_fwrd_speed = simulation_std_frwd_speed;
        simulation_side_speed = simulation_std_side_speed;
        simulation_back_speed = simulation_std_back_speed;

        // Get the desired velocity
        Vector3 desired_velocity_curr = desired_velocity_update(stickLeft, camera_azimuth, simulation_rotation);

        // Get the desired rotation/direction
        Vector4 desired_rotation_curr = desired_rotation_update(desired_rotation, stickLeft, camera_azimuth, target_lock, desired_velocity_curr); 

        desired_velocity_change_prev = desired_velocity_change_curr;
        desired_velocity_change_curr = (desired_velocity_curr - desired_velocity) / dt;
        desired_velocity = desired_velocity_curr;

        desired_rotation_change_prev = desired_rotation_change_curr;
        desired_rotation_change_curr = Quat.quat_to_scaled_angle_axis(Quat.quat_abs(Quat.quat_mul_inv(desired_rotation_curr, desired_rotation))) / dt;
        desired_rotation = desired_rotation_curr;

        bool force_search = false;

        if (force_search_timer <= 0.0f && (
            (length(desired_velocity_change_prev) >= desired_velocity_change_threshold &&
            length(desired_velocity_change_curr) < desired_velocity_change_threshold) ||
            (length(desired_rotation_change_prev) >= desired_rotation_change_threshold &&
            length(desired_rotation_change_curr) < desired_rotation_change_threshold)))
        {
            force_search = true;
            force_search_timer = search_time;
        }
        else if (force_search_timer > 0f)
            force_search_timer -= dt;

        trajectory_desired_rotations_predict(stickLeft, stickRight, camera_azimuth, target_lock, 0);
        trajectory_rotations_predict(20.0f * dt);

        trajectory_desired_velocities_predict(stickLeft, stickLeft, camera_azimuth, target_lock, 0);
        trajectory_positions_predict(20.0f * dt);

        // Do we need to search?
        if (force_search || search_timer <= 0.0f)
        {
            // Compute the features of the query vector
            (float[] query, int offset) = compute_query_vector(stickRight, target_lock);

            Debug.Assert(offset == db.nfeatures());

            bool transition = compute_projection_distance(query);
            if (transition)
            {
                evaluate_projector(query);
                evaluate_decompressor(ref trns_pose, feature_proj, latent_proj);
                Inertializers.inertialize_pose_transition(ref bone_offset_positions, ref bone_offset_rotations, ref bone_offset_velocities, ref bone_offset_angular_velocities,
                    ref transition_src_position, ref transition_src_rotation, ref transition_dst_position, ref transition_dst_rotation, pose, current_pose, trns_pose, db);
                Array.Copy(feature_proj, feature_curr, db.nfeatures());
                Array.Copy(latent_proj, latent_curr, latent_curr.Length);
            }
            search_timer = search_time;
        }
        search_timer -= dt;
        evaluate_stepper();

        evaluate_decompressor(ref current_pose, feature_curr, latent_curr);

        Inertializers.inertialize_pose_update(ref bone_offset_positions, ref bone_offset_rotations, ref bone_offset_velocities, ref bone_offset_angular_velocities,
            ref transition_src_position, ref transition_src_rotation, ref transition_dst_position, ref transition_dst_rotation, pose, db, current_pose, dt, inertialize_blending_halflife);

        simulation_position_update(ref simulation_position, ref simulation_velocity, ref simulation_acceleration,
            desired_velocity, simulation_velocity_halflife, dt);
        simulation_rotation_update(ref simulation_rotation, ref simulation_angular_velocity,
            desired_rotation, simulation_rotation_halflife, dt);

        //Adjustment
        if (adjustment_enabled)
        {
            Vector3 adjusted_position = pose.root_position;
            Vector4 adjusted_rotation = pose.root_rotation;

            if (adjustment_by_velocity)
            {
                adjusted_position = adjust_character_position_by_velocity(
                    pose.root_position,
                    pose.root_velocity,
                    simulation_position,
                    adjustment_position_halflife,
                    dt);
                adjusted_rotation = adjust_character_rotation_by_velocity(
                    pose.root_rotation,
                    pose.root_angular_velocity,
                    simulation_rotation,
                    adjustment_rotation_halflife,
                    dt);
            }
            Inertializers.inertialize_root_adjust(ref pose, ref transition_src_position, ref transition_dst_position, transition_src_rotation, ref transition_dst_rotation, ref bone_offset_positions,
                adjusted_position, adjusted_rotation);
        }
        //Clamping
        if (clamping_enabled)
        {
            Vector3 adjusted_position = pose.root_position;
            Vector4 adjusted_rotation = pose.root_rotation;

            adjusted_position = clamp_character_position(
                adjusted_position,
                simulation_position,
                clamping_max_distance);
            adjusted_rotation = clamp_character_rotation(
                adjusted_rotation,
                simulation_rotation,
                clamping_max_angle);

            Inertializers.inertialize_root_adjust(ref pose, ref transition_src_position, ref transition_dst_position, transition_src_rotation, ref transition_dst_rotation, ref bone_offset_positions,
                adjusted_position, adjusted_rotation);
        }

        adjusted_bones_pose = pose.DeepClone();
        if (ik_enabled)
        {
            compute_feet_positions(whatIsTerrain);
        }

        kinematics.forward_kinamatic_full(db, ref global_pose, adjusted_bones_pose);

        (Vector3 eye, Vector3 target) = orbit_camera_update(pose.root_position + Vector3.up, stickRight, target_lock, dt);

        if (controller.set_vcam)
            controller.SetVcam(eye, target_lock ? Target.position : target); 

        return (global_pose, feature_curr, latent_curr);
    }

    public (Vector3[], Vector4[]) Gizmos() => (trajectory_positions, trajectory_rotations);
}
