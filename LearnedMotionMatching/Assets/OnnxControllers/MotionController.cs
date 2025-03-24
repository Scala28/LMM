using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;
public abstract class MotionController : ScriptableObject
{
    protected ControllerOrchestrator controller;

    #region NN Inference
    [Header("NN Inference")]
    [SerializeField]
    private NNModel stepper;
    [SerializeField]
    private NNModel decompressor;
    [SerializeField]
    private NNModel projector;

    [SerializeField]
    private string stepper_name;
    [SerializeField]
    private string decompressor_name;
    [SerializeField]
    private string projector_name;

    private IWorker stepper_inference;
    private IWorker decompressor_inference;
    private IWorker projector_inference;

    private Model decompressor_nn;
    private Model stepper_nn;
    private Model projector_nn;
    #endregion
    public float search_time = 0.1f;
    protected float search_timer;
    protected float force_search_timer;

    #region LMM
    protected float[] feature_curr;
    protected float[] feature_proj;
    protected float[] latent_curr;
    protected float[] latent_proj;
    #endregion

    #region Data
    [Header("Data")]
    [SerializeField]
    private string db_filename;
    [SerializeField]
    private string features_filename;
    [SerializeField]
    private string latents_filename;
    protected DataManager.database db;
    private int frame_index;
    private float[][] latents;
    public DataManager.database getDB() => db;
    public int getFrameIndex() => frame_index;
    public float[][] getLatents() => latents;

    #endregion

    public float camera_azimuth = 0.0f;
    public float camera_altitude = .4f;
    public float camera_distance = 4.0f;

    #region Animation
    protected Pose pose;
    protected Pose current_pose;
    protected Pose trns_pose;
    protected Pose adjusted_bones_pose;
    protected Pose global_pose;

    protected bool[] global_bone_computed;

    protected Vector3[] bone_offset_positions;
    protected Vector4[] bone_offset_rotations;
    protected Vector3[] bone_offset_velocities;
    protected Vector3[] bone_offset_angular_velocities;

    protected Vector3 transition_src_position;
    protected Vector4 transition_src_rotation;
    protected Vector3 transition_dst_position;
    protected Vector4 transition_dst_rotation;

    protected Vector3 simulation_position;
    protected Vector3 simulation_velocity;
    protected Vector3 simulation_acceleration;
    protected Vector4 simulation_rotation = new Vector4(1f, 0f, 0f, 0f);
    protected Vector3 simulation_angular_velocity;

    public float simulation_velocity_halflife = 0.27f;
    public float simulation_rotation_halflife = 0.27f;

    public float inertialize_blending_halflife = .1f;

    #endregion

    #region Trajectories
    public Vector3 Desired_velocity { get { return desired_velocity; } }
    protected Vector3 desired_velocity;
    protected Vector3 desired_velocity_change_curr;
    protected Vector3 desired_velocity_change_prev;
    protected float desired_velocity_change_threshold = 50.0f;

    public Quaternion Desired_rotation { get { return new Quaternion(desired_rotation.y, desired_rotation.z, desired_rotation.w, desired_rotation.x); } }
    protected Vector4 desired_rotation = new Vector4(1f, 0f, 0f, 0f);
    protected Vector3 desired_rotation_change_curr;
    protected Vector3 desired_rotation_change_prev;
    protected float desired_rotation_change_threshold = 50.0f;

    protected Vector3[] trajectory_desired_velocities = new Vector3[4];
    protected Vector4[] trajectory_desired_rotations = new Vector4[4];
    protected Vector3[] trajectory_positions = new Vector3[4];
    protected Vector3[] trajectory_velocities = new Vector3[4];
    protected Vector3[] trajectory_accelerations = new Vector3[4];
    protected Vector4[] trajectory_rotations = new Vector4[4];
    protected Vector3[] trajectory_angular_velocities = new Vector3[4];
    #endregion

    #region Contact states and foot locking

    public bool ik_enabled = true;
    protected float ik_max_length_buffer = 0.015f;
    protected float ik_foot_height = 0.02f;
    protected float ik_toe_length = 0.15f;
    protected float ik_unlock_radius = 0.2f;
    protected float ik_blending_halflife = 0.1f;

    protected int[] contact_bones = new int[2];

    protected bool[] contact_states;
    protected bool[] contact_locks;
    protected Vector3[] contact_positions;
    protected Vector3[] contact_velocities;
    protected Vector3[] contact_points;
    protected Vector3[] contact_targets;
    protected Vector3[] contact_offset_positions;
    protected Vector3[] contact_offset_velocities;
    #endregion

    #region Adjustments
    public bool adjustment_enabled = true;
    protected bool adjustment_by_velocity = true;
    protected float adjustment_position_halflife = 0.1f;
    protected float adjustment_rotation_halflife = 0.2f;
    protected float adjustment_position_max_ratio = 0.5f;
    protected float adjustment_rotation_max_ratio = 0.5f;
    #endregion

    #region Clamping
    public bool clamping_enabled = true;
    protected float clamping_max_distance = .15f;
    protected float clamping_max_angle = .5f * Mathf.PI;
    #endregion

    protected const float dt = 1 / 60f;

    public void Setup(ControllerOrchestrator controller) {
        this.controller = controller;
        db = DataManager.load_database("Data/" + db_filename);
        (db.features, db.features_offset, db.features_scale) = DataManager.load_features("Data/" + features_filename);
        latents = DataManager.load_latent("Data/" + latents_filename);

        frame_index = db.range_starts[0];
        initialize_pose();

        Inertializers.inertialize_pose_reset(ref bone_offset_positions, ref bone_offset_rotations, ref bone_offset_velocities, ref bone_offset_angular_velocities, ref transition_src_position,
            ref transition_src_rotation, ref transition_dst_position, ref transition_dst_rotation, pose, db);
        Inertializers.inertialize_pose_update(ref bone_offset_positions, ref bone_offset_rotations, ref bone_offset_velocities, ref bone_offset_angular_velocities, ref transition_src_position,
            ref transition_src_rotation, ref transition_dst_position, ref transition_dst_rotation, pose, db, pose.DeepClone(), 0.0f, inertialize_blending_halflife);

        search_timer = search_time;
        force_search_timer = search_time;

        #region contacts

        contact_bones[0] = (int)ControllerOrchestrator.character.Bone_LeftToe;
        contact_bones[1] = (int)ControllerOrchestrator.character.Bone_RightToe;

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

            kinematics.forward_kinematics_velocity(out bone_position, out bone_velocity, out bone_rotation, out bone_angular_rotation,
                contact_bones[i], db, pose);

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

        latent_curr = new float[latents[0].Length];
        latent_proj = new float[latents[0].Length];
    }

    #region Init
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
        Array.Copy(db.contact_states[frame_index], pose.contact_states, db.contact_states[frame_index].Length);

        current_pose = pose.DeepClone();
        trns_pose = pose.DeepClone();
        adjusted_bones_pose = pose.DeepClone();

        bone_offset_positions = new Vector3[db.nbones()];
        bone_offset_rotations = new Vector4[db.nbones()];
        bone_offset_velocities = new Vector3[db.nbones()];
        bone_offset_angular_velocities = new Vector3[db.nbones()];

        global_pose = new Pose(db.nbones(), db.ncontacts()); 
    }
    private void initialize_models()
    {

        stepper_inference = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled,
            ModelLoader.Load(stepper));
        decompressor_inference = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled,
            ModelLoader.Load(decompressor));
        projector_inference = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled,
            ModelLoader.Load(projector));

        stepper_nn = DataManager.Load_net_fromParameters("NNModels/" + stepper_name);
        decompressor_nn = DataManager.Load_net_fromParameters("NNModels/" + decompressor_name);
        projector_nn = DataManager.Load_net_fromParameters("NNModels/" + projector_name);
    }
    #endregion

    public abstract (Pose, float[], float[]) perform_cycle(Vector3 stickLeft, Vector3 stickRight, bool gait, bool strafe);

    #region NNet inference
    public bool compute_projection_distance(float[] query, float transition_cost = 0.0f)
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
    public void evaluate_stepper()
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
    public void evaluate_decompressor(ref Pose target_pose, float[] features, float[] latents)
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
    public void evaluate_projector(float[] query)
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
    #endregion

    public void CleanUp()
    {
        if (stepper_inference != null)
            stepper_inference.Dispose();
        if (decompressor_inference != null)
            decompressor_inference.Dispose();
        if (projector_inference != null)
            projector_inference.Dispose();
    }

    protected float lerpf(float x, float y, float a) { return (1.0f - a) * x + a * y; }
    protected float clampf(float x, float min, float max) { return x > max ? max : x < min ? min : x; }
    protected float length(Vector3 v) { return Mathf.Sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
    protected float squaref(float x) { return x * x; }
}
