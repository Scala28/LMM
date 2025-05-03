using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEditor.TerrainTools;
using UnityEngine;
using UnityEngine.InputSystem.XR;
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

    public void SetFeatureCurr(float[] new_x) { Array.Copy(new_x, feature_curr, feature_curr.Length); }
    public void SetLatentCurr(float[] new_z) { Array.Copy(new_z, latent_curr, latent_curr.Length); }

    #region Data
    [Header("Data")]
    [SerializeField]
    private string db_filename;
    [SerializeField]
    private string features_filename;
    [SerializeField]
    private string latent_filename;
    protected DataManager.database db;
    protected int frame_index;
    private float[][] latents;
    public DataManager.database getDB() => db;
    public float[][] getLatents() => latents;

    #endregion


    [HideInInspector] public float camera_azimuth = 0.0f;
    [HideInInspector] public float camera_altitude = .4f;
    [HideInInspector] public float camera_distance = 4.0f;

    [Header("Animation")]
    public int FPS = 60;

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
    protected float ik_toe_length = 0.1f;
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

    public bool controller_oriented = true;

    #region Adjustments
    [ConditionalField("controller_oriented", true)]
    public bool adjustment_enabled = true;
    protected bool adjustment_by_velocity = true;
    protected float adjustment_position_halflife = 0.1f;
    protected float adjustment_rotation_halflife = 0.2f;
    protected float adjustment_position_max_ratio = 0.5f;
    protected float adjustment_rotation_max_ratio = 0.5f;
    #endregion

    #region Clamping
    [ConditionalField("controller_oriented", true)]
    public bool clamping_enabled = true;
    protected float clamping_max_distance = .15f;
    protected float clamping_max_angle = .5f * Mathf.PI;
    #endregion


    public List<Motion_Action> actions;
    protected int current_action_tag = 0;
    protected int input_action_tag = 0;
    protected bool first_action = true;

    protected float dt;

    public virtual void Setup(ControllerOrchestrator controller) {
        this.controller = controller;
        dt = 1f / FPS;
        db = DataManager.load_database("Data/" + db_filename, controller.controllers.Find(x => x.motion_controller == this).behaviour, false);
        (db.features, db.features_offset, db.features_scale) = DataManager.load_features("Data/" + features_filename);
        latents = DataManager.load_latent("Data/" + latent_filename);

        frame_index = db.range_starts[0];
        initialize_pose();

        camera_azimuth = 0.0f;
        camera_altitude = .4f;
        camera_distance = 4.0f;

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

        foreach (Motion_Action action in actions)
            action.SetUp(this, controller.controllers.Find(x => x.motion_controller == this).behaviour);

        input_action_tag = 0;
        current_action_tag = 0;
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

        simulation_position = Vector3.zero;
        simulation_rotation = new Vector4(1, 0, 0, 0);
        simulation_velocity = Vector3.zero;
        simulation_angular_velocity = Vector3.zero;

        desired_rotation = new Vector4(1, 0, 0, 0);
        desired_velocity = Vector3.zero;
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

    public abstract (Pose, float[], float[]) perform_cycle();

    #region NNet inference
    public virtual bool compute_projection_distance(float[] query, float transition_cost = 0.0f)
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
    public virtual void evaluate_stepper()
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
    public virtual void evaluate_decompressor(ref Pose target_pose, float[] features, float[] latents)
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

        target_pose = Parser.parse_decompressor_out(decompressor_out, current_pose, db.nbones(), db.ncontacts(), 
            controller.controllers.Find(x => x.motion_controller == this).behaviour);

        decompressor_in.Dispose();
        decompressor_out.Dispose();
    }
    public virtual void evaluate_projector(float[] query)
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

    #region adjustments
    protected virtual Vector3 adjust_character_position_by_velocity(Vector3 character_pos, Vector3 character_vel, Vector3 simulation_pos,
        float halflife, float dt)
    {
        Vector3 adjustment_position = Spring.damp_adjustment_exact(
            simulation_pos - character_pos,
            halflife,
            dt);
        // If the length of the adjustment is greater than the character velocity 
        // multiplied by the ratio then we need to clamp it to that length
        float max_length = adjustment_position_max_ratio * length(character_vel) * dt;

        if (length(adjustment_position) > max_length)
        {
            adjustment_position = max_length * Quat.vec_normalize(adjustment_position);
        }

        return adjustment_position + character_pos;
    }
    protected virtual Vector3 adjust_character_rotation_by_velocity(Vector4 character_rot, Vector3 character_angular_vel, Vector4 simulation_rot,
        float halflife, float dt)
    {
        Vector4 adjustment_rotation = Spring.damp_adjustment_exact(
            Quat.quat_abs(Quat.quat_normalize(Quat.quat_mul_inv(
                simulation_rot, character_rot))),
            halflife,
            dt);

        float max_length = adjustment_rotation_max_ratio * length(character_angular_vel) * dt;

        if (length(Quat.quat_to_scaled_angle_axis(adjustment_rotation)) > max_length)
        {
            adjustment_rotation = Quat.quat_from_scaled_angle_axis(max_length *
                Quat.vec_normalize(Quat.quat_to_scaled_angle_axis(adjustment_rotation)));
        }

        return Quat.quat_mul(adjustment_rotation, character_rot);
    }
    #endregion

    #region clamping
    protected virtual Vector3 clamp_character_position(Vector3 character_position, Vector3 simulation_position, float max_distance)
    {
        Vector3 distance = (character_position - simulation_position);
        if (length(distance) > max_distance)
        {
            return max_distance * Quat.vec_normalize(character_position - simulation_position) + simulation_position;
        }
        else
        {
            return character_position;
        }
    }
    protected virtual Vector4 clamp_character_rotation(Vector4 character_rotation, Vector4 simulation_rotation, float max_angle)
    {
        if (Quat.quat_angle_between(character_rotation, simulation_rotation) > max_angle)
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

    #region Contact & feet 
    protected virtual void compute_feet_positions(LayerMask whatIsTerrain)
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

            kinematics.forward_kinematic_partial(pose, toe_bone, ref global_pose, ref global_bone_computed, db);
            // Update the contact state
            contact_update(i, global_pose.joints[toe_bone - 1].position);

            RaycastHit hit = new RaycastHit();
            Debug.Assert(Physics.Raycast(new Vector3(contact_positions[i].x, 100f, contact_positions[i].z), -Vector3.up, out hit, float.MaxValue, whatIsTerrain));

            // Ensure contact position never goes through floor
            Vector3 contact_position_clamp = contact_positions[i];
            contact_position_clamp.y = Mathf.Max(contact_position_clamp.y, hit.point.y + ik_foot_height);

            // Re-compute toe, heel, knee, hip, and root bone positions
            int[] bones = new int[] { heel_bone, knee_bone, hip_bone, root_bone };

            for (int bone_indx = 0; bone_indx < bones.Length; bone_indx++)
            {
                kinematics.forward_kinematic_partial(pose, bones[bone_indx], ref global_pose, ref global_bone_computed, db);
            }
            // Perform simple two-joint IK to place heel
            // Qua lascio piu input variables in caso dobbiamo fare mani in futuro (per combattimento o altre cose)

            kinematics.ik_two_bone(global_pose, ref adjusted_bones_pose,
                contact_position_clamp,
                hip_bone,
                knee_bone,
                heel_bone,
                toe_bone,
                root_bone,
                ik_max_length_buffer);

            // Re-compute toe, heel, and knee positions 
            global_bone_computed = new bool[db.nbones()];

            int[] bones_stptwo = new int[] { toe_bone, heel_bone, knee_bone };
            for (int bone_indx = 0; bone_indx < bones_stptwo.Length; bone_indx++)
            {
                kinematics.forward_kinematic_partial(adjusted_bones_pose, bones_stptwo[bone_indx], ref global_pose, ref global_bone_computed, db);
            }

            // Rotate heel so toe is facing toward contact point
            kinematics.ik_look_at(ref adjusted_bones_pose.joints[heel_bone - 1].rotation, global_pose, global_pose.joints[toe_bone - 1].position, contact_position_clamp, heel_bone, knee_bone);

            // Re-compute toe and heel positions 
            global_bone_computed = new bool[db.nbones()];

            int[] bones_stptree = new int[] { toe_bone, heel_bone };
            for (int bone_indx = 0; bone_indx < bones_stptree.Length; bone_indx++)
            {
                kinematics.forward_kinematic_partial(adjusted_bones_pose, bones_stptree[bone_indx], ref global_pose, ref global_bone_computed, db);
            }

            // Rotate toe bone so that the end of the toe
            // does not intersect with the ground
            Vector3 toe_end_curr = Quat.quat_mul_vec(global_pose.joints[toe_bone - 1].rotation, new Vector3(ik_toe_length, 0.0f, 0.0f)) +
                    global_pose.joints[toe_bone - 1].position;

            Vector3 toe_end_targ = toe_end_curr;
            toe_end_targ.y = Mathf.Max(toe_end_targ.y, ik_foot_height);

            kinematics.ik_look_at(ref adjusted_bones_pose.joints[toe_bone - 1].rotation, global_pose, toe_end_curr, toe_end_targ, toe_bone, heel_bone);

        }
    }

    protected virtual void contact_update(int indx, Vector3 input_contact_position, float eps = 1e-8f)
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

    #region Read database
    public Pose GetFrameDatabase()
    {
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
        kinematics.forward_kinamatic_full(db, ref global_pose, pose);

        return global_pose;
    }
    public List<Vector3> GetFrameFeatures_trajPositions()
    {
        List<Vector3> _out = new List<Vector3>();
        int offset = 15;
        for(int i=0; i<6; i+=2)
        {
            float x = db.features[frame_index][offset + i];
            float z = db.features[frame_index][offset + i + 1];
            Vector3 loc_pos = new Vector3(x, 0, z);
            _out.Add(Quat.quat_mul_vec(db.bone_rotations[frame_index][0], loc_pos) + db.bone_positions[frame_index][0]);
        }
        return _out;
    }
    public Pose GetFrameAction(int action_idx)
    {
        int action_tag;
        (pose, action_tag) = actions[action_idx].GetFramePose();
        kinematics.forward_kinamatic_full(db, ref global_pose, pose);
        Debug.Log("action-tag: " + action_tag);

        return global_pose;
    }
    public void NextFrame() => frame_index += 1;
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

[System.Serializable]
public class Motion_Action
{
    private MotionController _controller;

    [SerializeField] private string database_filename;
    [SerializeField] private string features_filename;
    [SerializeField] private string latent_filename;

    private float[][] features;
    private float[] features_offset, features_scale;
    private float[][] latent;

    public int BOUND_LR_SIZE = 8;
    public int BOUND_SM_SIZE = 2;

    private DataManager.database database;
    private int frame_index;

    private float dt;

    public void SetUp(MotionController controller, Behaviour behav)
    {
        _controller = controller;
        dt = 1f / controller.FPS;
        database = DataManager.load_database("Data/" + database_filename, behav, true);
        (database.features, database.features_offset, database.features_scale) = DataManager.load_features("Data/" + features_filename);
        (features, features_offset, features_scale) = (database.features, database.features_offset, database.features_scale);
        DataManager.database_build_bounds(ref database, BOUND_LR_SIZE, BOUND_SM_SIZE);
        latent = DataManager.load_latent("Data/" + latent_filename);

        frame_index = database.range_starts[0];
    }

    public (Pose, int) GetFramePose()
    {
        Pose pose = new Pose(database.nbones(), database.ncontacts());
        pose.root_position = database.bone_positions[frame_index][0];
        pose.root_rotation = database.bone_rotations[frame_index][0];
        pose.root_velocity = database.bone_velocities[frame_index][0];
        pose.root_angular_velocity = database.bone_angular_velocities[frame_index][0];

        for (int i = 1; i < database.nbones(); i++)
        {
            pose.joints[i - 1].position = database.bone_positions[frame_index][i];
            pose.joints[i - 1].rotation = database.bone_rotations[frame_index][i];
            pose.joints[i - 1].velocity = database.bone_velocities[frame_index][i];
            pose.joints[i - 1].angular_velocity = database.bone_angular_velocities[frame_index][i];
        }

        return (pose, database.action_tags[frame_index]);
    }

    public bool NextFrame()
    {
        int next = database.database_trajectory_index_clamp(frame_index, 1);
        bool end_of_anim = next == frame_index;
        frame_index++;
        return end_of_anim;
    }

    public void tansform_local_pose_action(ref Pose pose, out float[] features_curr, out float[] latent_curr)
    {
        pose.root_velocity = database.bone_velocities[frame_index][0];
        pose.root_angular_velocity = database.bone_angular_velocities[frame_index][0];

        for (int i = 1; i < database.nbones(); i++)
        {
            pose.joints[i - 1].position = database.bone_positions[frame_index][i];
            pose.joints[i - 1].rotation = database.bone_rotations[frame_index][i];
            pose.joints[i - 1].velocity = database.bone_velocities[frame_index][i];
            pose.joints[i - 1].angular_velocity = database.bone_angular_velocities[frame_index][i];
        }
        features_curr = new float[features[frame_index].Length];
        Array.Copy(features[frame_index], features_curr, features_curr.Length);
        latent_curr = new float[latent[frame_index].Length];
        Array.Copy(latent[frame_index], latent_curr, latent_curr.Length);

        Debug.Log(features_curr[features_curr.Length - 1]);
    }

    #region Database search
    public void database_search(float[] query, bool first_action, float transition_cost = 0.0f)
    {
        Debug.Assert(query.Length == database.nfeatures());

        float[] query_normalized = new float[query.Length];
        for (int i = 0; i < database.nfeatures()-1; i++)
        {
            query_normalized[i] = (query[i] - features_offset[i]) / features_scale[i];
        }
        query_normalized[database.nfeatures() - 1] = query[database.nfeatures() - 1];

        int best_idx = frame_index;
        float best_cost = float.MaxValue;

        motion_matching(ref best_idx, ref best_cost, query, transition_cost);

        frame_index = first_action ? database.range_starts[database.database_get_animation_index(best_idx)] + 1 : best_idx;

    }

    private void motion_matching(ref int best_idx, ref float best_cost, float[] query_n, float transition_cost)
    {
        float ACTION_TAG = query_n[query_n.Length - 1];

        int curr_idx = best_idx;
        if (best_idx != -1)
        {
            best_cost = 0.0f;
            for (int i = 0; i < database.nfeatures(); i++)
            {
                best_cost += squaref(query_n[i] - features[best_idx][i]);
            }
        }

        float curr_cost = 0.0f;

        for (int r = 0; r < database.nranges(); r++)
        {
            int i = database.range_starts[r];
            int end_range = database.range_stops[r];
            while (i < end_range)
            {
                float action_tag = features[i][database.nfeatures() - 1];
                if (action_tag != ACTION_TAG) // Skip frames with action_tag != query ACTION_TAG
                    break;

                // Find index of current and next large box
                int i_lr = i / database.BOUND_LR_SIZE;
                int i_lr_next = (i_lr + 1) * database.BOUND_LR_SIZE;

                // Find distance to box
                curr_cost = transition_cost;
                for (int j = 0; j < database.nfeatures(); j++)
                {
                    curr_cost += squaref(query_n[j] - clampf(query_n[j],
                        database.bound_lr_min[i_lr][j], database.bound_lr_max[i_lr][j]));

                    if (curr_cost >= best_cost)
                    {
                        break;
                    }
                }

                // If distance is greater than current best jump to next box
                if (curr_cost >= best_cost)
                {
                    i = i_lr_next;
                    continue;
                }

                // Check against small box
                while (i < i_lr_next && i < end_range)
                {
                    // Find index of current and next small box
                    int i_sm = i / database.BOUND_SM_SIZE;
                    int i_sm_next = (i_sm + 1) * database.BOUND_SM_SIZE;

                    // Find distance to box
                    curr_cost = transition_cost;
                    for (int j = 0; j < database.nfeatures(); j++)
                    {
                        curr_cost += squaref(query_n[j] - clampf(query_n[j],
                            database.bound_sm_min[i_sm][j], database.bound_sm_max[i_sm][j]));

                        if (curr_cost >= best_cost)
                        {
                            break;
                        }
                    }
                    // If distance is greater than current best jump to next box
                    if (curr_cost >= best_cost)
                    {
                        i = i_sm_next;
                        continue;
                    }

                    // Search inside small box
                    while (i < i_sm_next && i < end_range)
                    {

                        // Check against each frame inside small box
                        curr_cost = transition_cost;
                        for (int j = 0; j < database.nfeatures(); j++)
                        {
                            curr_cost += squaref(query_n[j] - features[i][j]);
                            if (curr_cost >= best_cost)
                            {
                                break;
                            }
                        }

                        // If cost is lower than current best then update best
                        if (curr_cost < best_cost)
                        {
                            best_idx = i;
                            best_cost = curr_cost;
                        }

                        i++;

                    }

                }
            }
        }
    }
    #endregion
    protected float lerpf(float x, float y, float a) { return (1.0f - a) * x + a * y; }
    protected float clampf(float x, float min, float max) { return x > max ? max : x < min ? min : x; }
    protected float length(Vector3 v) { return Mathf.Sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
    protected float squaref(float x) { return x * x; }
}

