using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem.XR;

[CreateAssetMenu(fileName ="TerrainController", menuName = "OnnxMotionController/Terrain")]
public class TerrainController : MotionController
{
    [Header("Locomotion")]
    // All speeds in m/s
    public float simulation_run_fwrd_speed = 3.3f;
    public float simulation_run_side_speed = 2.8f;
    public float simulation_run_back_speed = 2.3f;

    public float simulation_walk_fwrd_speed = 1.75f;
    public float simulation_walk_side_speed = 1.5f;
    public float simulation_walk_back_speed = 1.25f;

    float simulation_fwrd_speed;
    float simulation_side_speed;
    float simulation_back_speed;

    private Vector2[][] trajectory_toe_position = new Vector2[3][];

    private float terrain_speed_multiplier = 1.0f;
    private Vector3[] terrain_root_positions;
    private Vector4[] terrain_root_rotations;
    private Vector3[][] terrain_toe_positions = new Vector3[3][];

    public LayerMask whatIsTerrain;

    private float desired_gait = 0.0f;
    private float desired_gait_velocity = 0.0f;

    [ConditionalField("controller_oriented", true)]
    public bool clamp_character_y = true;

    #region Trajectory & Gameplay Data
    public void desired_gait_update(bool gait, float gait_change_halflife = 0.1f)
    {
        Spring.simple_spring_damper_exact(
            ref desired_gait,
            ref desired_gait_velocity,
            gait ? 1.0f : 0.0f,
            gait_change_halflife,
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
    public Vector4 desired_rotation_update(Vector4 desired_rotation, Vector3 gamepad_stickleft, Vector3 gamepad_stickright, float camera_azimuth, bool desired_strafe, Vector3 desired_velocity)
    {
        Vector4 desired_rotation_curr = desired_rotation;
        // If strafe is active then desired direction is coming from right
        // stick as long as that stick is being used, otherwise we assume
        // forward facing
        if (desired_strafe)
        {
            Vector3 desired_dir = Quat.quat_mul_vec(Quat.quat_from_angle_axis(camera_azimuth, new Vector3(0f, 1f, 0f)), new Vector3(0f, 0f, 1f));
            if (length(gamepad_stickright) > 0.01f)
            {
                desired_dir = Quat.quat_mul_vec(Quat.quat_from_angle_axis(camera_azimuth, new Vector3(0f, 1f, 0f)), Quat.vec_normalize(gamepad_stickright));
            }
            return Quat.quat_from_angle_axis(Mathf.Atan2(desired_dir.x, desired_dir.z), new Vector3(0f, 1f, 0f));
        }
        // If strafe is not active the desired direction comes from the left 
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
    public void trajectory_desired_rotations_predict(Vector3 gamepadstick_left, Vector3 gamepadstick_right, float camera_azimuth, bool desired_strafe, float dt)
    {
        trajectory_desired_rotations[0] = desired_rotation;

        for (int i = 1; i < trajectory_desired_rotations.Length; i++)
        {
            trajectory_desired_rotations[i] = desired_rotation_update(
                trajectory_desired_rotations[i - 1],
                gamepadstick_left,
                gamepadstick_right,
                orbit_camera_azimuth(camera_azimuth, gamepadstick_right, desired_strafe, i * dt),
                desired_strafe,
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
    public void trajectory_desired_velocities_predict(Vector3 gamepadstick_left, Vector3 gamepadstick_right, float camera_azimuth, bool desired_strafe, float dt)
    {
        trajectory_desired_velocities[0] = desired_velocity;
        for (int i = 1; i < trajectory_desired_velocities.Length; i++)
        {
            trajectory_desired_velocities[i] = desired_velocity_update(
                gamepadstick_left,
                orbit_camera_azimuth(camera_azimuth, gamepadstick_right, desired_strafe, i * dt),
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
    private void compute_trajectory_toe_position()
    {
        trajectory_toe_position[0] = new Vector2[2];
        trajectory_toe_position[1] = new Vector2[2];
        trajectory_toe_position[2] = new Vector2[2];

        // Find root position and direction at 15, 30, 45
        float mag = (trajectory_positions[1] - trajectory_positions[0]).magnitude;
        Vector3 root_pos_15 = (trajectory_positions[1] - trajectory_positions[0]).normalized * mag * .75f + trajectory_positions[0];

        mag = (trajectory_positions[2] - trajectory_positions[1]).magnitude;
        Vector3 root_pos_30 = (trajectory_positions[1] - trajectory_positions[0]).normalized * mag * .5f + trajectory_positions[1];

        mag = (trajectory_positions[3] - trajectory_positions[2]).magnitude;
        Vector3 root_pos_45 = (trajectory_positions[1] - trajectory_positions[0]).normalized * mag * .25f + trajectory_positions[2];

        terrain_root_positions = new Vector3[] { root_pos_15, root_pos_30, root_pos_45 };

        Vector4 root_rot_15 = Quat.quat_nlerp(trajectory_rotations[0], trajectory_rotations[1], .75f);
        Vector4 root_rot_30 = Quat.quat_nlerp(trajectory_rotations[1], trajectory_rotations[2], .5f);
        Vector4 root_rot_45 = Quat.quat_nlerp(trajectory_rotations[2], trajectory_rotations[3], .25f);

        terrain_root_rotations = new Vector4[] { root_rot_15, root_rot_30, root_rot_45 };

        // Compute global toe positions at 15, 30, 45

        for (int i = 0; i < trajectory_toe_position.Length; i++)
        {
            Vector3 left_pos = Quat.quat_mul_vec(terrain_root_rotations[i], current_pose.traj_toe_position[i][0]) + terrain_root_positions[i];
            Vector3 right_pos = Quat.quat_mul_vec(terrain_root_rotations[i], current_pose.traj_toe_position[i][1]) + terrain_root_positions[i];

            trajectory_toe_position[i][0] = new Vector2(left_pos.x, left_pos.z);
            trajectory_toe_position[i][1] = new Vector2(right_pos.x, right_pos.z);
        }
    }
    private (float[][], float) cast_terrain_height(Pose global_pose)
    {
        float[][] hit_y = new float[trajectory_toe_position.Length][];
        float variance = 0f;
        for (int i = 0; i < trajectory_toe_position.Length; i++)
        {
            hit_y[i] = new float[2];
            terrain_toe_positions[i] = new Vector3[2];
            // Record terrain height at 15, 30, 45 local to root now
            for (int j = 0; j < trajectory_toe_position[0].Length; j++)
            {

                RaycastHit hit_point = new RaycastHit();
                Ray ray = new Ray(new Vector3(
                    trajectory_toe_position[i][j].x,
                    100f,
                    trajectory_toe_position[i][j].y), -Vector3.up);
                Debug.Assert(Physics.Raycast(ray, out hit_point, float.MaxValue, whatIsTerrain));

                Vector3 chr_hit_point = Quat.quat_inv_mul_vec(global_pose.root_rotation,
                    hit_point.point - global_pose.root_position);

                hit_y[i][j] = chr_hit_point.y;
                terrain_toe_positions[i][j] = hit_point.point;
                variance += chr_hit_point.y * chr_hit_point.y;
            }
        }
        return (hit_y, variance);
    }
    public (float[], int) compute_query_vector(bool gait_input)
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

        // terrain heights at 0, 15, 30, 45 local to root now
        RaycastHit hit;
        Physics.Raycast(global_pose.joints[(int)ControllerOrchestrator.character.Bone_LeftToe - 1].position,
                    -Vector3.up, out hit, float.MaxValue, whatIsTerrain);

        Vector3 terrain_height_0_left = Quat.quat_inv_mul_vec(global_pose.root_rotation,
            hit.point - global_pose.root_position);

        Physics.Raycast(global_pose.joints[(int)ControllerOrchestrator.character.Bone_RightToe - 1].position,
                    -Vector3.up, out hit, float.MaxValue, whatIsTerrain);
        Vector3 terrain_height_0_right = Quat.quat_inv_mul_vec(global_pose.root_rotation,
            hit.point - global_pose.root_position);

        compute_trajectory_toe_position();

        (float[][] terrain_heights, float height_variance) = cast_terrain_height(pose);

        height_variance += terrain_height_0_left.y * terrain_height_0_left.y;
        height_variance += terrain_height_0_right.y * terrain_height_0_right.y;

        float multiplier_min_value = gait_input ? .175f : .4f;

        if (height_variance >= .15f)
            terrain_speed_multiplier = lerpf(multiplier_min_value, 1f, clampf(1 - height_variance, 0f, 1f));
        else
            terrain_speed_multiplier = 1f;

        query[offset + 0] = terrain_height_0_left.y;
        query[offset + 1] = terrain_height_0_right.y;
        query[offset + 2] = terrain_heights[0][0];
        query[offset + 3] = terrain_heights[0][1];
        query[offset + 4] = terrain_heights[1][0];
        query[offset + 5] = terrain_heights[1][1];
        query[offset + 6] = terrain_heights[2][0];
        query[offset + 7] = terrain_heights[2][1];

        offset += 8;

        return (query, offset);
    }
    private float orbit_camera_azimuth(float azimuth, Vector3 gamepadstick_right, bool desired_strafe, float dt)
    {
        Vector3 gamepadaxis = desired_strafe ? Vector3.zero : gamepadstick_right;
        return azimuth + 2.0f * dt * gamepadaxis.x;
    }
    private float orbit_camera_altitude(float altitude, Vector3 gamepadstick_right, bool desired_strafe, float dt)
    {
        Vector3 gamepadaxis = desired_strafe ? Vector3.zero : gamepadstick_right;
        return clampf(altitude + 2.0f * dt * gamepadaxis.z, 0.0f, 0.4f * Mathf.PI);
    }
    private float orbit_camera_distance(float distance, float dt)
    {
        float gamepadzoom = 0.0f;
        return clampf(distance + 10f * dt * gamepadzoom, 0.1f, 100.0f);
    }
    public (Vector3, Vector3) orbit_camera_update(Vector3 target, Vector3 gamepadstick_right, bool desired_strafe, float dt)
    {
        camera_azimuth = orbit_camera_azimuth(camera_azimuth, gamepadstick_right, desired_strafe, dt);
        camera_altitude = orbit_camera_altitude(camera_altitude, gamepadstick_right, desired_strafe, dt);
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
        bool gait = controller.input_handler.RightShoulder;
        bool strafe = false;//controller.input_handler.LeftTrigger;

        desired_gait_update(gait);

        simulation_fwrd_speed = lerpf(simulation_walk_fwrd_speed, simulation_run_fwrd_speed, desired_gait) * terrain_speed_multiplier;
        simulation_side_speed = lerpf(simulation_walk_side_speed, simulation_run_side_speed, desired_gait) * terrain_speed_multiplier;
        simulation_back_speed = lerpf(simulation_walk_back_speed, simulation_run_back_speed, desired_gait) * terrain_speed_multiplier;

        // Get the desired velocity
        Vector3 desired_velocity_curr = desired_velocity_update(stickLeft, camera_azimuth, simulation_rotation);

        // Get the desired rotation/direction
        Vector4 desired_rotation_curr = desired_rotation_update(desired_rotation, stickLeft, stickRight, camera_azimuth, strafe, desired_velocity_curr); ;

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

        trajectory_desired_rotations_predict(stickLeft, stickRight, camera_azimuth, strafe, 20.0f * dt);
        trajectory_rotations_predict(20.0f * dt);

        trajectory_desired_velocities_predict(stickLeft, stickRight, camera_azimuth, strafe, 20.0f * dt);
        trajectory_positions_predict(20.0f * dt);

        // Do we need to search?
        if (force_search || search_timer <= 0.0f)
        {
            // Compute the features of the query vector
            (float[] query, int offset) = compute_query_vector(gait);

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

        // Project simulation position on terrain
        RaycastHit hit = new RaycastHit();
        Debug.Assert(Physics.Raycast(new Vector3(simulation_position.x, 100f, simulation_position.z), -Vector3.up, out hit, float.MaxValue, whatIsTerrain));
        simulation_position.y = hit.point.y;

        if (controller_oriented)
        {
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

                if (!clamp_character_y)
                    adjusted_position = this.clamp_character_position(
                    adjusted_position,
                    simulation_position,
                    clamping_max_distance);
                else
                    adjusted_position = base.clamp_character_position(
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
        }
        else
        {
            simulation_position.x = pose.root_position.x;
            simulation_position.z = pose.root_position.z;
            Vector4 adjusted_rotation = pose.root_rotation;
            Inertializers.inertialize_root_adjust(ref pose, ref transition_src_position, ref transition_dst_position, transition_src_rotation, ref transition_dst_rotation, ref bone_offset_positions,
                    simulation_position, adjusted_rotation);
        }

        adjusted_bones_pose = pose.DeepClone();
        if (ik_enabled)
        {
            compute_feet_positions(whatIsTerrain);
        }

        kinematics.forward_kinamatic_full(db, ref global_pose, adjusted_bones_pose);

        (Vector3 eye, Vector3 target) = orbit_camera_update(pose.root_position + Vector3.up, stickRight, strafe, dt);

        if (controller.set_vcam)
            controller.SetVcam(eye, target);

        return (global_pose, feature_curr, latent_curr);
    }

    #region clamping
    protected override Vector3 clamp_character_position(Vector3 character_position, Vector3 simulation_position, float max_distance)
    {
        Vector3 distance = (character_position - simulation_position);
        distance.y = 0;
        if (length(distance) > max_distance)
        {
            Vector3 dir = (character_position - simulation_position);
            dir.y = 0;
            return max_distance * Quat.vec_normalize(dir) + simulation_position;
        }
        else
        {
            return character_position;
        }
    }
    #endregion

    public (Vector3[], Vector4[], Vector3[][]) Gizmos() => (trajectory_positions, trajectory_rotations, terrain_toe_positions);


}
