using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;
using UnityEditor;
using UnityEngine.XR;
using UnityEditor.Experimental.GraphView;
using System.ComponentModel;
using Unity.VisualScripting;
using UnityEngine.Assertions;
using Google.Protobuf.WellKnownTypes;
using UnityEngine.UIElements;

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
    #endregion

    #region Animation
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

    private DataManager.database db;
    private DataManager.character ch;

    private float inertialize_blending_halflife = .1f;
    private Pose pose;
    private Pose current_pose;
    private Pose trns_pose;
    private Pose global_pose;

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
    #region Trajectory and gameplay data
    public float search_time = 0.1f;
    private float search_timer;
    private float force_search_timer;

    private InputHandler input_handler;

    private Vector3 desired_velocity;
    private Vector3 desired_velocity_change_curr;
    private Vector3 desired_velocity_change_prev;
    private float desired_velocity_change_threshold = 50.0f;

    private Vector4 desired_rotation;
    private Vector3 desired_rotation_change_curr;
    private Vector3 desired_rotation_change_prev;
    private float desired_rotation_change_threshold = 50.0f;

    private float desired_gait = 0.0f;
    private float desired_gait_velocity = 0.0f;

    private Vector3 simulation_position;
    private Vector3 simulation_velocity;
    private Vector3 simulation_acceleration;
    private Vector4 simulation_rotation;
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

    public bool ik_enabled = true;
    private float ik_foot_height = 0.02f;
    private float ik_toe_length = 0.15f;
    private float ik_unlock_radius = 0.2f;
    private float ik_blending_halflife = 0.1f;

    #region Contact states and foot locking
    public int[] contact_bones = new int[2];

    private bool[] contact_states;
    private bool[] contact_locks;
    private Vector3[] contact_positions;
    private Vector3[] contact_velocities;
    private Vector3[] contact_points;
    private Vector3[] contact_targets;
    private Vector3[] contact_offset_positions;
    private Vector3[] contact_offset_velocities;

    private Vector3[] adjusted_bone_positions;
    private Vector4[] adjusted_bone_rotations;
    #endregion

    private int frame_index;
    private int window = 20;
    private int frame_count = 0;
    private float frame_time = 0.0f;

    private const float dt = 1 / 60f;

    private List<Transform> bones = new List<Transform>();
    private Mesh mesh;
    #endregion

    // Start is called before the first frame update
    void Start()
    {
        input_handler = GetComponent<InputHandler>();
        db = DataManager.load_database("Assets/Resources/database.bin");
        ch = DataManager.load_character("Assets/Resources/character.bin");
        mesh = DataManager.gen_mesh_from_character(ch);
        transform.GetComponent<MeshFilter>().mesh = mesh;

        Debug.Log(db.nbones());
        Debug.Log(ch.nbones());

        if (db.nbones() != ch.nbones())
        {
            Debug.LogError("Database and skeleton do not match!");
            return;
        }

        (db.features, db.features_offset, db.features_scale) = DataManager.load_features("Assets/Resources/features.bin");

        frame_index = db.range_starts[2];
        Debug.Log(frame_index);

        initialize_skeleton(this.transform);
        initialize_pose();

        //inertialize_pose_reset();
        //inertialize_pose_update(pose.DeepClone(), 0.0f);

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

        initialize_models();

        feature_curr = db.features[frame_index];
        feature_proj = db.features[frame_index];

        latent_curr = new float[32];
        latent_proj = new float[32];

    }
    #region Initialize
    private void initialize_models() {

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
        pose = new Pose(db.nbones());

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

        global_pose = new Pose(db.nbones());
        global_bone_computed = new bool[db.nbones()];
    }
    #endregion
    private void set_frame(int frame_index)
    {
        (int nframes1, int nfeatures, float[] features) = DataManager.Load_database_fromResources("features");
        (int nframes2, int nlatent, float[] latent) = DataManager.Load_database_fromResources("latent");

        if (features == null || latent == null)
            return;

        if (nframes1 != nframes2)
        {
            Debug.LogError("Mismatch in the number of frames between the two datasets.");
            return;
        }
        Array.Copy(features, frame_index * nfeatures, feature_curr, 0, nfeatures);
        Array.Copy(latent, frame_index * nlatent, latent_curr, 0, nlatent);
    }
    private float[] gen_query(int offset = 5)
    {
        float[] Xhat = new float[feature_curr.Length];
        (int nframes1, int nfeatures, float[] features) = DataManager.Load_database_fromResources("features");
        if (features == null)
            return null;
        Array.Copy(features, (offset + frame_count + frame_index) * nfeatures, Xhat, 0, nfeatures);
        return Xhat;
    }

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
        Debug.Log("Desired vel: " + desired_velocity_curr);
        Debug.Log("Desired rot: " + Quat.convert_ToEuler(desired_rotation_curr) * Mathf.Rad2Deg);
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

        Debug.Log("Trajectory desired Rotations");
        Debug.Log(Quat.convert_ToEuler(trajectory_desired_rotations[0]) * Mathf.Rad2Deg);
        Debug.Log(Quat.convert_ToEuler(trajectory_desired_rotations[1]) * Mathf.Rad2Deg);
        Debug.Log(Quat.convert_ToEuler(trajectory_desired_rotations[2]) * Mathf.Rad2Deg);
        Debug.Log(Quat.convert_ToEuler(trajectory_desired_rotations[3]) * Mathf.Rad2Deg);

        Debug.Log("Trajectory Rotations");
        Debug.Log(Quat.convert_ToEuler(trajectory_rotations[0]) * Mathf.Rad2Deg);
        Debug.Log(Quat.convert_ToEuler(trajectory_rotations[1]) * Mathf.Rad2Deg);
        Debug.Log(Quat.convert_ToEuler(trajectory_rotations[2]) * Mathf.Rad2Deg);
        Debug.Log(Quat.convert_ToEuler(trajectory_rotations[3]) * Mathf.Rad2Deg);

        trajectory_desired_velocities_predict(gamepad_stickleft, gamepad_stickright, camera_azimuth, desired_strafe, 
            simulation_fwrd_speed, simulation_side_speed, simulation_back_speed, 20.0f * dt);
        trajectory_positions_predict(simulation_velocity_halflife, 20.0f * dt);

        Debug.Log("Trajectory desired velocities");
        Debug.Log(trajectory_desired_velocities[0]);
        Debug.Log(trajectory_desired_velocities[1]);
        Debug.Log(trajectory_desired_velocities[2]);
        Debug.Log(trajectory_desired_velocities[3]);
        Debug.Log("Trajectory velocities");
        Debug.Log(trajectory_velocities[0]);
        Debug.Log(trajectory_velocities[1]);
        Debug.Log(trajectory_velocities[2]);
        Debug.Log(trajectory_velocities[3]);
        Debug.Log("Trajectory positions");
        Debug.Log(trajectory_positions[0]);
        Debug.Log(trajectory_positions[1]);
        Debug.Log(trajectory_positions[2]);
        Debug.Log(trajectory_positions[3]);


        frame_time += Time.deltaTime;
        if (frame_time >= dt)
        {
            if (frame_count % window == 0)
            {
                float[] query = gen_query();
                evaluate_projector(query);
                feature_curr = feature_proj;
                latent_curr = latent_proj;
            }

            //Set new features_curr and latent_curr
            evaluate_stepper();

            //Set new curr_pose
            evaluate_decompressor(ref current_pose);

            //Set new pose
            //inertialize_pose_update(current_pose, dt);

            // Full pass of forward kinematics to compute 
            // all bone positions and rotations in the world
            // space ready for rendering (set global_pose)
            forward_kinamatic_full();

            //display_frame_pose();
            deform_character_mesh();
            pose = current_pose;
            frame_time = 0f;
            frame_count++;
        }
    }
    #region NN inferences
    private void evaluate_stepper()
    {
        Tensor stepper_in = new Tensor(new TensorShape(1, 1, 1, feature_curr.Length + latent_curr.Length));
        for (int i = 0; i < feature_curr.Length; i++)
            stepper_in[i] = feature_curr[i];
        for (int i = 0; i < latent_curr.Length; i++)
            stepper_in[i + feature_curr.Length] = latent_curr[i];

        nnLayer_normalize(stepper_in, stepper_nn);
        stepper_inference.Execute(stepper_in);
        Tensor stepper_out = stepper_inference.PeekOutput();
        nnLayer_denormalize(stepper_out, stepper_nn);

        for (int i = 0; i < feature_curr.Length; i++)
            feature_curr[i] += dt * stepper_out[i];
        for (int i = 0; i < latent_curr.Length; i++)
            latent_curr[i] += dt * stepper_out[feature_curr.Length + i];

        stepper_in.Dispose();
        stepper_out.Dispose();
    }
    private void evaluate_decompressor(ref Pose target_pose)
    {
        Tensor decompressor_in = new Tensor(new TensorShape(1, 1, 1, feature_curr.Length + latent_curr.Length));
        for (int i = 0; i < feature_curr.Length; i++)
            decompressor_in[i] = feature_curr[i];
        for (int i = 0; i < latent_curr.Length; i++)
            decompressor_in[i + feature_curr.Length] = latent_curr[i];

        //nnLayer_normalize(decompressor_in, decompressor_nn);
        decompressor_inference.Execute(decompressor_in);
        Tensor decompressor_out = decompressor_inference.PeekOutput();
        nnLayer_denormalize(decompressor_out, decompressor_nn);

        target_pose = Parser.parse_decompressor_out(decompressor_out, pose, db.nbones());

        decompressor_in.Dispose();
        decompressor_out.Dispose();
    }
    private void evaluate_projector(float[] query)
    {
        Tensor projector_in = new Tensor(new TensorShape(1, 1, 1, feature_curr.Length));
        for (int i = 0; i < feature_curr.Length; i++)
            projector_in[i] = query[i];

        nnLayer_normalize(projector_in, projector_nn);
        projector_inference.Execute(projector_in);
        Tensor projector_out = projector_inference.PeekOutput();
        nnLayer_denormalize(projector_out, projector_nn);

        for (int i = 0; i < feature_curr.Length; i++)
            feature_proj[i] = projector_out[i];
        for (int i = 0; i < latent_curr.Length; i++)
            latent_proj[i] = projector_out[feature_curr.Length + i];

        projector_in.Dispose();
        projector_out.Dispose();
    }
    private void nnLayer_denormalize(Tensor _out, Model param)
    {
        for (int i = 0; i < param.Mean_out.Length; i++)
        {
            _out[i] = _out[i] * param.Std_out[i] + param.Mean_out[i];
        }

    }
    private void nnLayer_normalize(Tensor _out, Model param)
    {
        for (int i = 0; i < param.Mean_in.Length; i++)
        {
            _out[i] = (_out[i] - param.Mean_in[i]) / param.Std_in[i];
        }
    }
    #endregion

    #region Inertializers
    private void inertialize_pose_reset() {
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
    private void inertialize_pose_update(Pose input_pose, float _dt) {

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
    #endregion

    #region Player input
    private float lerpf(float x, float y, float a)
    {
        return (1.0f - a) * x + a * y;
    }
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
            if(length(gamepad_stickright) > 0.01f)
            {
                desired_dir = Quat.quat_mul_vec(Quat.quat_from_angle_axis(camera_azimuth, new Vector3(0f, 1f, 0f)), Quat.vec_normalize(gamepad_stickright));
            }
            return Quat.quat_from_angle_axis(Mathf.Atan2(desired_dir.x, desired_dir.z), new Vector3(0f, 1f, 0f));
        }
        // If strafe is not active the desired direction comes from the left 
        // stick as long as that stick is being used
        else if(length(gamepad_stickleft) > 0.01f)
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

        for(int i=1; i<trajectory_desired_rotations.Length; i++)
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
        for(int i=0; i < trajectory_rotations.Length; i++)
        {
            trajectory_rotations[i] = simulation_rotation;
            trajectory_angular_velocities[i] = simulation_angular_velocity;
        }

        for(int i=1; i<trajectory_rotations.Length; i++)
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
        for(int i=1; i<trajectory_desired_velocities.Length; i++)
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

        for(int i=1; i<trajectory_positions.Length; i++)
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
    private float orbit_camera_azimuth(float azimuth, Vector3 gamepadstick_right, bool desired_strafe, float dt)
    {
        Vector3 gamepadaxis = desired_strafe ? Vector3.zero : gamepadstick_right;
        return azimuth + 2.0f * dt * gamepadaxis.x;
    }
    #endregion

    #region FKs
    private void forward_kinamatic_full()
    {
        for(int i=0; i<db.bone_parents.Length; i++)
        {
            if (db.bone_parents[i] >= i)
            {
                Debug.LogError("DB bone_parents does not match");
                return;
            }
            if (db.bone_parents[i] == -1)
            {
                global_pose.root_position = current_pose.root_position;
                global_pose.root_rotation = current_pose.root_rotation;
            }
            else
            {
                Vector3 parent_position = db.bone_parents[i] == 0 ? global_pose.root_position : 
                    global_pose.joints[db.bone_parents[i] - 1].position;
                Vector4 parent_rotation = db.bone_parents[i] == 0 ? global_pose.root_rotation :
                    global_pose.joints[db.bone_parents[i] - 1].rotation;

                global_pose.joints[i-1].position = Quat.quat_mul_vec(parent_rotation, current_pose.joints[i-1].position) + parent_position;
                global_pose.joints[i - 1].rotation = Quat.quat_mul(parent_rotation, current_pose.joints[i - 1].rotation);
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

            bone_pos = Quat.quat_mul_vec(parent_rot, current_pose.joints[bone-1].position) + parent_pos;
            bone_vel = parent_vel + Quat.quat_mul_vec(parent_rot, current_pose.joints[bone-1].velocity) +
                Quat._cross(parent_ang_vel, Quat.quat_mul_vec(parent_rot, current_pose.joints[bone-1].position));
            bone_rot = Quat.quat_mul(parent_rot, current_pose.joints[bone-1].rotation);
            bone_ang_vel = Quat.quat_mul_vec(parent_rot, current_pose.joints[bone-1].angular_velocity) + parent_ang_vel;
        }
        else
        {
            bone_pos = current_pose.root_position;
            bone_vel = current_pose.root_velocity;
            bone_rot = current_pose.root_rotation;
            bone_ang_vel = current_pose.root_angular_velocity;
        }
    }
    private void forward_kinematic_partial(int bone)
    {
        if (db.bone_parents[bone] == -1)
        {
            global_pose.root_position = current_pose.root_position;
            global_pose.root_rotation = current_pose.root_rotation;
            global_bone_computed[bone] = true;
            return;
        }
        
        if (!global_bone_computed[db.bone_parents[bone]]){
            forward_kinematic_partial(db.bone_parents[bone]);
        }
        Vector3 parent_pos = global_pose.joints[db.bone_parents[bone]-1].position;
        Vector4 parent_rot = global_pose.joints[db.bone_parents[bone] - 1].rotation;
        global_pose.joints[bone - 1].position = Quat.quat_mul_vec(parent_rot, current_pose.joints[bone - 1].position)
            + parent_pos;
        global_pose.joints[bone - 1].rotation = Quat.quat_mul(parent_rot, current_pose.joints[bone - 1].rotation);
        global_bone_computed[bone] = true;
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
        transform.position = new Vector3(current_pose.root_position.x, current_pose.root_position.y, -current_pose.root_position.z);
        Quaternion q = Quaternion.Euler(0f, 180f, 0f);
        Vector3 ang = Quat.convert_ToEuler(Quat.quat_mul(current_pose.root_rotation, new Vector4(q.w, q.x, q.y, q.z)));
        Vector3 root_angle = new Vector3(ang.x, ang.y, ang.z) * Mathf.Rad2Deg;
        transform.rotation = Quaternion.Euler(0f, 0f, -root_angle.z) *
                    Quaternion.Euler(0f, -root_angle.y, 0f) * Quaternion.Euler(root_angle.x, 0f, 0f);
        for (int i = 1; i < db.nbones(); i++)
        {
            Transform joint = bones[i];
            JointMotionData jdata = current_pose.joints[i - 1];

            ang = Quat.convert_ToEuler(jdata.rotation) * Mathf.Rad2Deg;

            joint.localRotation = Quaternion.Euler(0f, 0f, -ang.z) *
                    Quaternion.Euler(0f, -ang.y, 0f) * Quaternion.Euler(ang.x, 0f, 0f);
            //joint.rotation = Quaternion.Euler(0f, 0f, ang.z) * Quaternion.Euler(ang.x, 0f, 0f) * Quaternion.Euler(0f, ang.y, 0f);
        }
    }
    private float length(Vector3 v)
    {
        return Mathf.Sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }
    private float length(Vector4 v)
    {
        return Mathf.Sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
    }
    private void OnDestroy()
    {
        if(stepper_inference != null)
            stepper_inference.Dispose();
        if(decompressor_inference != null)
            decompressor_inference.Dispose();
        if(projector_inference != null)
            projector_inference.Dispose();
    }
    private void OnDrawGizmosSelected()
    {
        for(int i=0; i<trajectory_positions.Length; i++)
        {
            Gizmos.DrawSphere(trajectory_positions[i], .2f);
        }
    }
}
