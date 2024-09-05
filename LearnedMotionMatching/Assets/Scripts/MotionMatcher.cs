using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;
using UnityEditor;

public class MotionMatcher : MonoBehaviour
{
    #region NN Inference
    [Header("NN Inference")]
    [SerializeField]
    private NNModel stepper;

    [SerializeField]
    private NNModel decompressor;

    [SerializeField]
    private NNModel projector;

    private IWorker stepper_inference;
    private IWorker decompressor_inference;
    private IWorker projector_inference;

    private Model decompressor_nn;
    private Model stepper_nn;
    private Model projector_nn;
    #endregion

    #region LMM
    private float[] feature_curr;
    private float[] feature_proj;
    private float[] latent_curr;
    private float[] latent_proj;

    private float feature_weight_foot_position = 0.75f;
    private float feature_weight_foot_velocity = 1.0f;
    private float feature_weight_hip_velocity = 1.0f;
    private float feature_weight_trajectory_positions = 1.0f;
    private float feature_weight_trajectory_directions = 1.5f;
    private float feature_weight_trajectory_toe_height = 1.0f;
    #endregion
    public enum character
    {
        Bone_Entity = 0,
        Bone_Hips = 1,
        Bone_LeftUpLeg = 2,
        Bone_LeftLeg = 3,
        Bone_LeftFoot = 4,
        Bone_LeftToe = 5,
        Bone_RightUpLeg = 6,
        Bone_RightLeg = 7,
        Bone_RightFoot = 8,
        Bone_RightToe = 9,
        Bone_Spine = 10,
        Bone_Spine1 = 11,
        Bone_Spine2 = 12,
        Bone_Neck = 13,
        Bone_Head = 14,
        Bone_LeftShoulder = 15,
        Bone_LeftArm = 16,
        Bone_LeftForeArm = 17,
        Bone_LeftHand = 18,
        Bone_RightShoulder = 19,
        Bone_RightArm = 20,
        Bone_RightForeArm = 21,
        Bone_RightHand = 22
    };

    private float camera_azimuth = 0.0f;
    private float camera_altitude = .4f;
    private float camera_distance = 4.0f;

    private DataManager.database db;
    private DataManager.character ch;

    private float inertialize_blending_halflife = .1f;
    private Pose pose;
    private Pose current_pose;
    private Pose trns_pose;
    private Pose global_pose;
    private Pose adjusted_bones_pose;

    private bool[] global_bone_computed;

    private Vector3[] bone_offset_positions;
    private Vector4[] bone_offset_rotations;
    private Vector3[] bone_offset_velocities;
    private Vector3[] bone_offset_angular_velocities;

    Vector3 transition_src_position;
    Vector4 transition_src_rotation;
    Vector3 transition_dst_position;
    Vector4 transition_dst_rotation;

    [Header("Animation")]

    #region Trajectory & gameplay

    public float search_time = 0.1f;
    private float search_timer;
    private float force_search_timer;

    private InputHandler input_handler;

    private Vector3 desired_velocity;
    private Vector3 desired_velocity_change_curr;
    private Vector3 desired_velocity_change_prev;
    private float desired_velocity_change_threshold = 50.0f;

    private Vector4 desired_rotation = new Vector4(1f, 0f, 0f, 0f);
    private Vector3 desired_rotation_change_curr;
    private Vector3 desired_rotation_change_prev;
    private float desired_rotation_change_threshold = 50.0f;

    private float desired_gait = 0.0f;
    private float desired_gait_velocity = 0.0f;

    private Vector3 simulation_position;
    private Vector3 simulation_velocity;
    private Vector3 simulation_acceleration;
    private Vector4 simulation_rotation = new Vector4(1f, 0f, 0f, 0f);
    private Vector3 simulation_angular_velocity;

    private float simulation_velocity_halflife = 0.27f;
    private float simulation_rotation_halflife = 0.27f;

    // All speeds in m/s
    private float simulation_run_fwrd_speed = 4.0f;
    private float simulation_run_side_speed = 3.0f;
    private float simulation_run_back_speed = 2.5f;

    private float simulation_walk_fwrd_speed = 1.75f;
    private float simulation_walk_side_speed = 1.5f;
    private float simulation_walk_back_speed = 1.25f;

    private Vector3[] trajectory_desired_velocities = new Vector3[4];
    private Vector4[] trajectory_desired_rotations = new Vector4[4];
    private Vector3[] trajectory_positions = new Vector3[4];
    private Vector3[] trajectory_velocities = new Vector3[4];
    private Vector3[] trajectory_accelerations = new Vector3[4];
    private Vector4[] trajectory_rotations = new Vector4[4];
    private Vector3[] trajectory_angular_velocities = new Vector3[4];
    #endregion

    #region Contact states and foot locking

    public bool ik_enabled = true;
    private float ik_max_length_buffer = 0.015f;
    private float ik_foot_height = 0.02f;
    private float ik_toe_length = 0.15f;
    private float ik_unlock_radius = 0.2f;
    private float ik_blending_halflife = 0.1f;

    private int[] contact_bones = new int[2];

    private bool[] contact_states;
    private bool[] contact_locks;
    private Vector3[] contact_positions;
    private Vector3[] contact_velocities;
    private Vector3[] contact_points;
    private Vector3[] contact_targets;
    private Vector3[] contact_offset_positions;
    private Vector3[] contact_offset_velocities;
    #endregion

    #region Adjustments
    public bool adjustment_enabled = true;
    private bool adjustment_by_velocity = true;
    private float adjustment_position_halflife = 0.1f;
    private float adjustment_rotation_halflife = 0.2f;
    private float adjustment_position_max_ratio = 0.5f;
    private float adjustment_rotation_max_ratio = 0.5f;
    #endregion

    #region Clamping
    public bool clamping_enabled = true;
    private float clamping_max_distance = .15f;
    private float clamping_max_angle = .5f * Mathf.PI;
    #endregion

    public bool gizmos = false;

    private int frame_index;

    private const float dt = 1 / 60f;

    private List<Transform> bones = new List<Transform>();
    private Mesh mesh;

    public bool rigged = false;

    // Start is called before the first frame update
    void Start()
    {
        input_handler = GetComponent<InputHandler>();
        db = DataManager.load_database("Assets/Resources/database.bin");
        ch = DataManager.load_character("Assets/Resources/character.bin");
        if (!rigged)
        {
            mesh = DataManager.gen_mesh_from_character(ch);
            transform.GetComponent<MeshFilter>().mesh = mesh;
        }else
            initialize_skeleton(this.transform);

        Debug.Assert(db.nbones() == ch.nbones());

        (db.features, db.features_offset, db.features_scale) = DataManager.load_features("Assets/Resources/features.bin");

        frame_index = db.range_starts[0];

        initialize_pose();

        inertialize_pose_reset();
        inertialize_pose_update(pose.DeepClone(), 0.0f);

        #region contacts
        search_timer = search_time;
        force_search_timer = search_time;

        contact_bones[0] = (int)character.Bone_LeftToe;
        contact_bones[1] = (int)character.Bone_RightToe;
        
        contact_states = new bool[contact_bones.Length];
        contact_locks = new bool[contact_bones.Length];
        contact_positions = new Vector3[contact_bones.Length];
        contact_velocities = new Vector3[contact_bones.Length];
        contact_points = new Vector3[contact_bones.Length];
        contact_targets = new Vector3[contact_bones.Length];
        contact_offset_positions = new Vector3[contact_bones.Length];
        contact_offset_velocities = new Vector3[contact_bones.Length];

        for (int i = 0; i < contact_bones.Length; i++)
        {
            Vector3 bone_position;
            Vector3 bone_velocity;
            Vector4 bone_rotation;
            Vector3 bone_angular_rotation;

            forward_kinematics_velocity(out bone_position, out bone_velocity, out bone_rotation, out bone_angular_rotation,
                contact_bones[i]);

            contact_states[i] = false;
            contact_locks[i] = false;
            contact_positions[i] = bone_position;
            contact_velocities[i] = bone_velocity;
            contact_points[i] = bone_position;
            contact_targets[i] = bone_position;
            contact_offset_positions[i] = Vector3.zero;
            contact_offset_velocities[i] = Vector3.zero;
        }
        #endregion

        initialize_models();

        feature_curr = new float[db.nfeatures()];
        feature_proj = new float[db.nfeatures()];

        Array.Copy(db.features[frame_index], feature_curr, db.nfeatures());
        Array.Copy(db.features[frame_index], feature_proj, db.nfeatures());

        latent_curr = new float[32];
        latent_proj = new float[32];
    }
    #region Initialize
    private void initialize_models()
    {

        stepper_inference = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled,
            ModelLoader.Load(stepper));
        decompressor_inference = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled,
            ModelLoader.Load(decompressor));
        projector_inference = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled,
            ModelLoader.Load(projector));

        stepper_nn = DataManager.Load_net_fromParameters("Assets/NNModels/stepper.bin");
        decompressor_nn = DataManager.Load_net_fromParameters("Assets/NNModels/decompressor.bin");
        projector_nn = DataManager.Load_net_fromParameters("Assets/NNModels/projector.bin");
    }
    private void initialize_skeleton(Transform bone)
    {
        if (bone.CompareTag("joint"))
        {
            bones.Add(bone);
        }
        foreach (Transform child in bone)
            initialize_skeleton(child);
    }
    private void initialize_pose()
    {
        pose = new Pose(db.nbones(), db.ncontacts());

        pose.root_position = db.bone_positions[frame_index][0];
        pose.root_rotation = db.bone_rotations[frame_index][0];
        pose.root_velocity = db.bone_velocities[frame_index][0];
        pose.root_angular_velocity = db.bone_angular_velocities[frame_index][0];

        for (int i = 1; i < db.nbones(); i++)
        {
            pose.joints[i - 1].position = db.bone_positions[frame_index][i];
            pose.joints[i - 1].rotation = db.bone_rotations[frame_index][i];
            pose.joints[i - 1].velocity = db.bone_velocities[frame_index][i];
            pose.joints[i - 1].angular_velocity = db.bone_angular_velocities[frame_index][i];
        }

        current_pose = pose.DeepClone();
        trns_pose = pose.DeepClone();

        bone_offset_positions = new Vector3[db.nbones()];
        bone_offset_rotations = new Vector4[db.nbones()];
        bone_offset_velocities = new Vector3[db.nbones()];
        bone_offset_angular_velocities = new Vector3[db.nbones()];

        global_pose = new Pose(db.nbones(), db.ncontacts());

        global_bone_computed = new bool[db.nbones()];
    }
    #endregion

    // Update is called once per frame
    void Update()
    {
        Vector3 gamepad_stickleft = input_handler.MoveInput;
        Vector3 gamepad_stickright = input_handler.LookInput;

        bool desired_strafe = input_handler.StrafeInput;

        // Get the desired gait (walk / run)
        desired_gait_update();

        // Get the desired simulation speeds based on the gait
        float simulation_fwrd_speed = lerpf(simulation_run_fwrd_speed, simulation_walk_fwrd_speed, desired_gait);
        float simulation_side_speed = lerpf(simulation_run_side_speed, simulation_walk_side_speed, desired_gait);
        float simulation_back_speed = lerpf(simulation_run_back_speed, simulation_walk_back_speed, desired_gait);

        // Get the desired velocity
        Vector3 desired_velocity_curr =
            desired_velocity_update(gamepad_stickleft, camera_azimuth, simulation_rotation,
            simulation_fwrd_speed, simulation_side_speed, simulation_back_speed);


        // Get the desired rotation/direction
        Vector4 desired_rotation_curr =
            desired_rotation_update(desired_rotation, gamepad_stickleft, gamepad_stickright, camera_azimuth, desired_strafe, desired_velocity_curr);

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

        trajectory_desired_rotations_predict(gamepad_stickleft, gamepad_stickright, camera_azimuth, desired_strafe, 20.0f * dt);
        trajectory_rotations_predict(simulation_rotation_halflife, 20.0f * dt);

        trajectory_desired_velocities_predict(gamepad_stickleft, gamepad_stickright, camera_azimuth, desired_strafe,
            simulation_fwrd_speed, simulation_side_speed, simulation_back_speed, 20.0f * dt);
        trajectory_positions_predict(simulation_velocity_halflife, 20.0f * dt);

        // Do we need to search?
        if (force_search || search_timer <= 0.0f)
        {
            // Compute the features of the query vector
            (float[] query, int offset) = compute_query_vector();

            Debug.Assert(offset == db.nfeatures());

            bool transition = compute_projection_distance(query);
            if (transition)
            {
                evaluate_projector(query);
                evaluate_decompressor(ref trns_pose, feature_proj, latent_proj);
                inertialize_pose_transition();
                Array.Copy(feature_proj, feature_curr, db.nfeatures());
                Array.Copy(latent_proj, latent_curr, latent_curr.Length);
            }

            search_timer = search_time;
        }
        search_timer -= dt;

        evaluate_stepper();

        evaluate_decompressor(ref current_pose, feature_curr, latent_curr);

        Debug.Log(current_pose.contact_states[0]);
        Debug.Log(current_pose.contact_states[1]);

        inertialize_pose_update(current_pose, dt);

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
            inertialize_root_adjust(adjusted_position, adjusted_rotation);
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

            inertialize_root_adjust(adjusted_position, adjusted_rotation);
        }

        adjusted_bones_pose = pose.DeepClone();
        if (ik_enabled)
        {
            compute_feet_positions();
        }

        forward_kinamatic_full();
        camera_azimuth = orbit_camera_azimuth(camera_azimuth, gamepad_stickright, desired_strafe, dt);

        if(!rigged)
            deform_character_mesh();
        else
            display_frame_pose();
    }
    #region NN inferences
    private void evaluate_stepper()
    {
        Tensor stepper_in = new Tensor(new TensorShape(1, 1, 1, feature_curr.Length + latent_curr.Length));
        for (int i = 0; i < feature_curr.Length; i++)
            stepper_in[i] = feature_curr[i];
        for (int i = 0; i < latent_curr.Length; i++)
            stepper_in[i + feature_curr.Length] = latent_curr[i];

        stepper_nn.nnLayer_normalize(stepper_in);
        stepper_inference.Execute(stepper_in);
        Tensor stepper_out = stepper_inference.PeekOutput();
        stepper_nn.nnLayer_denormalize(stepper_out);

        for (int i = 0; i < feature_curr.Length; i++)
            feature_curr[i] += dt * stepper_out[i];
        for (int i = 0; i < latent_curr.Length; i++)
            latent_curr[i] += dt * stepper_out[feature_curr.Length + i];

        stepper_in.Dispose();
        stepper_out.Dispose();
    }
    private void evaluate_decompressor(ref Pose target_pose, float[] features, float[] latents)
    {
        Tensor decompressor_in = new Tensor(new TensorShape(1, 1, 1, features.Length + latents.Length));
        for (int i = 0; i < features.Length; i++)
            decompressor_in[i] = features[i];
        for (int i = 0; i < latents.Length; i++)
            decompressor_in[i + features.Length] = latents[i];

        //nnLayer_normalize(decompressor_in, decompressor_nn);
        decompressor_inference.Execute(decompressor_in);
        Tensor decompressor_out = decompressor_inference.PeekOutput();
        decompressor_nn.nnLayer_denormalize(decompressor_out);

        target_pose = Parser.parse_decompressor_out(decompressor_out, current_pose, db.nbones(), db.ncontacts());

        decompressor_in.Dispose();
        decompressor_out.Dispose();
    }
    private void evaluate_projector(float[] query)
    {
        Tensor projector_in = new Tensor(new TensorShape(1, 1, 1, query.Length));
        for (int i = 0; i < query.Length; i++)
            projector_in[i] = (query[i] - db.features_offset[i]) / db.features_scale[i];

        projector_nn.nnLayer_normalize(projector_in);
        projector_inference.Execute(projector_in);
        Tensor projector_out = projector_inference.PeekOutput();
        projector_nn.nnLayer_denormalize(projector_out);

        for (int i = 0; i < feature_proj.Length; i++)
            feature_proj[i] = projector_out[i];
        for (int i = 0; i < latent_proj.Length; i++)
            latent_proj[i] = projector_out[feature_proj.Length + i];

        projector_in.Dispose();
        projector_out.Dispose();
    }
    private bool compute_projection_distance(float[] query, float transition_cost = 0.0f)
    {
        bool transition;

        float best_cost = 0.0f;
        for (int i = 0; i < feature_proj.Length; i++)
        {
            best_cost += squaref(query[i] - feature_proj[i]);
        }
        best_cost = Mathf.Sqrt(best_cost);

        float trns_dist_squared = 0.0f;
        for (int i = 0; i < feature_proj.Length; i++)
        {
            trns_dist_squared += squaref(feature_curr[i] - feature_proj[i]);
        }

        if (trns_dist_squared > squaref(transition_cost))
        {
            transition = true;
            best_cost += transition_cost;
        }
        else
        {
            transition = false;
            for (int i = 0; i < feature_proj.Length; i++)
            {
                feature_proj[i] = feature_curr[i];
            }

            best_cost = 0.0f;
            for (int i = 0; i < feature_curr.Length; i++)
            {
                best_cost += squaref(query[i] - feature_curr[i]);
            }
            best_cost = Mathf.Sqrt(best_cost);
        }
        return transition;
    }
    #endregion

    #region Inertializers
    private void inertialize_pose_reset()
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
    private void inertialize_pose_transition()
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
    private void inertialize_pose_update(Pose input_pose, float _dt)
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
    private void inertialize_root_adjust(Vector3 input_position, Vector4 input_rotation)
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
    #endregion

    #region Trajectory & Gameplay Data
    private void desired_gait_update(float gait_change_halflife = 0.1f)
    {
        Spring.simple_spring_damper_exact(
            ref desired_gait,
            ref desired_gait_velocity,
            input_handler.GaitInput ? 0.0f : 1.0f,
            gait_change_halflife,
            dt);
    }
    private Vector3 desired_velocity_update(Vector3 gamepad_stickleft, float camera_azimuth, Vector3 simulation_rotation, float fwrd_speed, float side_speed, float back_speed)
    {
        // Find stick position in world space by rotating using camera azimuth
        Vector3 global_stick_direction = Quat.quat_mul_vec(
            Quat.quat_from_angle_axis(camera_azimuth, new Vector3(0f, 1.0f, 0f)), gamepad_stickleft);

        // Find stick position local to current facing direction
        Vector3 local_stick_direction = Quat.quat_inv_mul_vec(
            simulation_rotation, global_stick_direction);

        // Scale stick by forward, sideways and backwards speeds
        Vector3 local_desired_velocity = local_stick_direction.z > 0.0 ?
            new Vector3(side_speed * local_stick_direction.x, 0.0f, fwrd_speed * local_stick_direction.z) :
            new Vector3(side_speed * local_stick_direction.x, 0.0f, back_speed * local_stick_direction.z);

        return Quat.quat_mul_vec(simulation_rotation, local_desired_velocity);
    }
    private Vector4 desired_rotation_update(Vector4 desired_rotation, Vector3 gamepad_stickleft, Vector3 gamepad_stickright, float camera_azimuth, bool desired_strafe, Vector3 desired_velocity)
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
    private void trajectory_desired_rotations_predict(Vector3 gamepadstick_left, Vector3 gamepadstick_right, float camera_azimuth, bool desired_strafe, float dt)
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
    private void trajectory_rotations_predict(float halflife, float dt)
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
                halflife,
                i * dt);
        }
    }
    private void trajectory_desired_velocities_predict(Vector3 gamepadstick_left, Vector3 gamepadstick_right, float camera_azimuth, bool desired_strafe,
        float fwrd_speed, float side_speed, float back_speed, float dt)
    {
        trajectory_desired_velocities[0] = desired_velocity;
        for (int i = 1; i < trajectory_desired_velocities.Length; i++)
        {
            trajectory_desired_velocities[i] = desired_velocity_update(
                gamepadstick_left,
                orbit_camera_azimuth(camera_azimuth, gamepadstick_right, desired_strafe, i * dt),
                trajectory_rotations[i],
                fwrd_speed,
                side_speed,
                back_speed);
        }
    }
    private void trajectory_positions_predict(float halflife, float dt)
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
                halflife,
                dt);
        }
    }
    private (float[], int) compute_query_vector()
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
        Vector3 traj0 = Quat.quat_inv_mul_vec(pose.root_rotation, trajectory_positions[1] - pose.root_position);
        Vector3 traj1 = Quat.quat_inv_mul_vec(pose.root_rotation, trajectory_positions[2] - pose.root_position);
        Vector3 traj2 = Quat.quat_inv_mul_vec(pose.root_rotation, trajectory_positions[3] - pose.root_position);

        query[offset + 0] = traj0.x;
        query[offset + 1] = traj0.z;
        query[offset + 2] = traj1.x;
        query[offset + 3] = traj1.z;
        query[offset + 4] = traj2.x;
        query[offset + 5] = traj2.z;

        offset += 6;

        // query_compute_trajectory_direction_feature
        Vector3 dir0 = Quat.quat_inv_mul_vec(pose.root_rotation, Quat.quat_mul_vec(trajectory_rotations[1], new Vector3(0, 0, 1f)));
        Vector3 dir1 = Quat.quat_inv_mul_vec(pose.root_rotation, Quat.quat_mul_vec(trajectory_rotations[2], new Vector3(0, 0, 1f)));
        Vector3 dir2 = Quat.quat_inv_mul_vec(pose.root_rotation, Quat.quat_mul_vec(trajectory_rotations[3], new Vector3(0, 0, 1f)));

        query[offset + 0] = dir0.x;
        query[offset + 1] = dir0.z;
        query[offset + 2] = dir1.x;
        query[offset + 3] = dir1.z;
        query[offset + 4] = dir2.x;
        query[offset + 5] = dir2.z;

        offset += 6;

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
    private void orbit_camera_update(Vector3 target, Vector3 gamepadstick_right, bool desired_strafe, float dt)
    {
        camera_azimuth = orbit_camera_azimuth(camera_azimuth, gamepadstick_right, desired_strafe, dt);
        camera_altitude = orbit_camera_altitude(camera_altitude, gamepadstick_right, desired_strafe, dt);
        camera_distance = orbit_camera_distance(camera_distance, dt);

        Vector4 rotation_azimuth = Quat.quat_from_angle_axis(camera_azimuth, new Vector3(0, 1f, 0));

        Vector3 position = Quat.quat_mul_vec(rotation_azimuth, new Vector3(0, 0, camera_distance));
        Vector3 axis = Quat.vec_normalize(Quat._cross(position, new Vector3(0, 1f, 0)));

        Vector4 rotation_altitude = Quat.quat_from_angle_axis(camera_altitude, axis);

        Vector3 eye = target + Quat.quat_mul_vec(rotation_altitude, position);


    }
    #endregion

    #region FKs
    private void forward_kinamatic_full()
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
    private void forward_kinematics_velocity(out Vector3 bone_pos,
                                             out Vector3 bone_vel,
                                             out Vector4 bone_rot,
                                             out Vector3 bone_ang_vel,
                                             int bone)
    {
        if (db.bone_parents[bone] != -1)
        {
            Vector3 parent_pos;
            Vector3 parent_vel;
            Vector4 parent_rot;
            Vector3 parent_ang_vel;

            forward_kinematics_velocity(out parent_pos, out parent_vel, out parent_rot, out parent_ang_vel,
                db.bone_parents[bone]);

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
    private void forward_kinematic_partial(Pose input_pose, int bone)
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
            forward_kinematic_partial(input_pose, db.bone_parents[bone]);
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

    #region Contact & feet 
    private void compute_feet_positions()
    {

        for (int i = 0; i < contact_bones.Length; i++)
        {
            // Find all the relevant bone indices
            int toe_bone = contact_bones[i];
            int heel_bone = db.bone_parents[toe_bone];
            int knee_bone = db.bone_parents[heel_bone];
            int hip_bone = db.bone_parents[knee_bone];
            int root_bone = db.bone_parents[hip_bone];
            // Compute the world space position for the toe
            global_bone_computed = new bool[db.nbones()];

            forward_kinematic_partial(pose, toe_bone);
            // Update the contact state
            contact_update(i, global_pose.joints[toe_bone - 1].position);

            // Ensure contact position never goes through floor
            Vector3 contact_position_clamp = contact_positions[i];
            contact_position_clamp.y = Mathf.Max(contact_position_clamp.y, ik_foot_height);

            // Re-compute toe, heel, knee, hip, and root bone positions
            int[] bones = new int[] { heel_bone, knee_bone, hip_bone, root_bone };

            for (int bone_indx = 0; bone_indx < bones.Length; bone_indx++)
            {
                forward_kinematic_partial(pose, bones[bone_indx]);
            }
            // Perform simple two-joint IK to place heel
            // Qua lascio piu input variables in caso dobbiamo fare mani in futuro (per combattimento o altre cose)

            ik_two_bone(contact_position_clamp,
                hip_bone,
                knee_bone,
                heel_bone,
                toe_bone,
                root_bone,
                ik_max_length_buffer
                );
            // Re-compute toe, heel, and knee positions 
            global_bone_computed = new bool[db.nbones()];

            int[] bones_stptwo = new int[] { toe_bone, heel_bone, knee_bone };
            for (int bone_indx = 0; bone_indx < bones_stptwo.Length; bone_indx++)
            {
                forward_kinematic_partial(adjusted_bones_pose, bones_stptwo[bone_indx]);
            }

            // Rotate heel so toe is facing toward contact point
            ik_look_at(ref adjusted_bones_pose.joints[heel_bone - 1].rotation, global_pose.joints[toe_bone - 1].position, contact_position_clamp, heel_bone, knee_bone);

            // Re-compute toe and heel positions 
            global_bone_computed = new bool[db.nbones()];

            int[] bones_stptree = new int[] { toe_bone, heel_bone };
            for (int bone_indx = 0; bone_indx < bones_stptree.Length; bone_indx++)
            {
                forward_kinematic_partial(adjusted_bones_pose, bones_stptree[bone_indx]);
            }

            // Rotate toe bone so that the end of the toe
            // does not intersect with the ground
            Vector3 toe_end_curr = Quat.quat_mul_vec(global_pose.joints[toe_bone - 1].rotation, new Vector3(ik_toe_length, 0.0f, 0.0f)) +
                    global_pose.joints[toe_bone - 1].position;

            Vector3 toe_end_targ = toe_end_curr;
            toe_end_targ.y = Mathf.Max(toe_end_targ.y, ik_foot_height);

            ik_look_at(ref adjusted_bones_pose.joints[toe_bone - 1].rotation, toe_end_curr, toe_end_targ, toe_bone, heel_bone);

        }
    }

    private void contact_update(int indx, Vector3 input_contact_position, float eps = 1e-8f)
    {
        Vector3 input_contact_velocity = (input_contact_position - contact_targets[indx]) / (dt + eps);
        contact_targets[indx] = input_contact_position;

        // Update the inertializer to tick forward in time
        Spring.inertialize_update(
            ref contact_positions[indx],
            ref contact_velocities[indx],
            ref contact_offset_positions[indx],
            ref contact_offset_velocities[indx],
            // If locked we feed the contact point and zero velocity,    
            // otherwise we feed the input from the animation
            contact_locks[indx] ? contact_points[indx] : input_contact_position,
            contact_locks[indx] ? new Vector3() : input_contact_velocity,
            ik_blending_halflife,
            dt);

        // If the contact point is too far from the current input position 
        // then we need to unlock the contact
        bool unlock_contact = contact_locks[indx] && (length(contact_points[indx] - input_contact_position) > ik_unlock_radius);

        // If the contact was previously inactive but is now active we 
        // need to transition to the locked contact state
        if (!contact_states[indx] && current_pose.contact_states[indx])
        {
            // Contact point is given by the current position of 
            // the foot projected onto the ground plus foot height
            contact_locks[indx] = true;
            contact_points[indx] = contact_positions[indx];
            contact_points[indx].y = ik_foot_height;

            Spring.inertialize_transition(
                ref contact_offset_positions[indx],
                ref contact_offset_velocities[indx],
                input_contact_position,
                input_contact_velocity,
                contact_points[indx],
                new Vector3());
        }
        // Otherwise if we need to unlock or we were previously in 
        // contact but are no longer we transition to just taking 
        // the input position as-is
        else if ((contact_locks[indx] && contact_states[indx] && !current_pose.contact_states[indx]) || unlock_contact)
        {
            contact_locks[indx] = false;

            Spring.inertialize_transition(
                ref contact_offset_positions[indx],
                ref contact_offset_velocities[indx],
                contact_points[indx],
                new Vector3(),
                input_contact_position,
                input_contact_velocity);
        }
        // Update contact state
        contact_states[indx] = current_pose.contact_states[indx];
    }
    #endregion

    #region IKs
    private void ik_look_at(ref Vector4 bone_rotation,
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
    private void ik_two_bone(
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

    #region adjustments
    private Vector3 adjust_character_position_by_velocity(Vector3 character_pos, Vector3 character_vel, Vector3 simulation_pos,
        float halflife, float dt)
    {
        Vector3 adjustment_position = Spring.damp_adjustment_exact(
            simulation_pos - character_pos,
            halflife,
            dt);
        // If the length of the adjustment is greater than the character velocity 
        // multiplied by the ratio then we need to clamp it to that length
        float max_length = adjustment_position_max_ratio * length(character_vel) * dt;

        if(length(adjustment_position) > max_length)
        {
            adjustment_position = max_length * Quat.vec_normalize(adjustment_position);
        }

        return adjustment_position + character_pos;
    }
    private Vector3 adjust_character_rotation_by_velocity(Vector4 character_rot, Vector3 character_angular_vel, Vector4 simulation_rot,
        float halflife, float dt)
    {
        Vector4 adjustment_rotation = Spring.damp_adjustment_exact(
            Quat.quat_abs(Quat.quat_normalize(Quat.quat_mul_inv(
                simulation_rot, character_rot))),
            halflife,
            dt);

        float max_length = adjustment_rotation_max_ratio * length(character_angular_vel) * dt;

        if(length(Quat.quat_to_scaled_angle_axis(adjustment_rotation)) > max_length)
        {
            adjustment_rotation = Quat.quat_from_scaled_angle_axis(max_length *
                Quat.vec_normalize(Quat.quat_to_scaled_angle_axis(adjustment_rotation)));
        }

        return Quat.quat_mul(adjustment_rotation, character_rot);
    }
    #endregion

    #region clamping
    private Vector3 clamp_character_position(Vector3 character_position, Vector3 simulation_position, float max_distance)
    {
        if(length(character_position - simulation_position) > max_distance)
        {
            return max_distance * Quat.vec_normalize(character_position - simulation_position) + simulation_position;
        }
        else
        {
            return character_position;
        }
    }
    private Vector4 clamp_character_rotation(Vector4 character_rotation, Vector4 simulation_rotation, float max_angle)
    {
        if(Quat.quat_angle_between(character_rotation, simulation_rotation) > max_angle)
        {
            Vector4 diff = Quat.quat_abs(Quat.quat_mul_inv(character_rotation, simulation_rotation));
            float diff_angle; Vector3 diff_axis;
            Quat.quat_to_angle_axis(diff, out diff_angle, out diff_axis);

            diff_angle = clampf(diff_angle, -max_angle, max_angle);

            return Quat.quat_mul(
                Quat.quat_from_angle_axis(diff_angle, diff_axis), simulation_rotation);
        }
        else
        {
            return character_rotation;
        }
    }
    #endregion
    private void deform_character_mesh()
    {
        Vector3[] mesh_vertices = new Vector3[mesh.vertices.Length];
        Vector3[] mesh_normals = new Vector3[mesh.normals.Length];
        DataManager.character.liner_blend_skinning_positions(ch, global_pose, ref mesh_vertices);
        DataManager.character.liner_blend_skinning_normals(ch, global_pose, ref mesh_normals);

        mesh.vertices = mesh_vertices;
        mesh.normals = mesh_normals;

        mesh.RecalculateBounds();
        mesh.RecalculateTangents();
        mesh.UploadMeshData(false);

    }
    private void display_frame_pose()
    {
        transform.position = new Vector3(global_pose.root_position.x, global_pose.root_position.y, global_pose.root_position.z);
        Vector3 ang = Quat.convert_ToEuler(global_pose.root_rotation) * Mathf.Rad2Deg;
        transform.rotation = Quaternion.Euler(0f, 0f, ang.z) * Quaternion.Euler(0f, ang.y, 0f) * Quaternion.Euler(ang.x, 0f, 0f);
        for (int i = 1; i < db.nbones(); i++)
        {
            Transform joint = bones[i];
            JointMotionData jdata = global_pose.joints[i - 1];

            ang = Quat.convert_ToEuler(jdata.rotation) * Mathf.Rad2Deg;

            joint.rotation = Quaternion.Euler(0f, 0f, -ang.z) *
                Quaternion.Euler(0f, -ang.y, 0f) * Quaternion.Euler(ang.x, 0f, 0f);
        }
    }

    private float lerpf(float x, float y, float a) { return (1.0f - a) * x + a * y; }
    private float clampf(float x, float min, float max) { return x > max ? max : x < min ? min : x; }
    private float length(Vector3 v) { return Mathf.Sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
    private float length(Vector4 v) { return Mathf.Sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w); }
    private float squaref(float x) { return x * x; }
    private void OnDestroy()
    {
        if (stepper_inference != null)
            stepper_inference.Dispose();
        if (decompressor_inference != null)
            decompressor_inference.Dispose();
        if (projector_inference != null)
            projector_inference.Dispose();
    }
    private void OnDrawGizmosSelected()
    {
        if(gizmos)
            for (int i = 0; i < trajectory_positions.Length; i++)
            {
                Gizmos.DrawSphere(trajectory_positions[i], .2f);
            }
    }
}