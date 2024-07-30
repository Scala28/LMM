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

public class MotionMatcher : MonoBehaviour
{
    #region NN Inference
    [Header("NN Inference")]
    [SerializeField]
    private NNModel stepper;

    [SerializeField]
    private NNModel decompressor;

    private IWorker stepper_inference;
    private IWorker decompressor_inference;

    private Model decompressor_nn;
    private Model stepper_nn;
    #endregion

    #region LMM
    private float[] feature_curr;
    private float[] feature_proj;
    private float[] latent_curr;
    private float[] latent_proj;
    #endregion

    #region Animation
    [Header("Animation")]
    public float inertialize_blending_halflife = .1f;
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
    private float halflife;
    private const float dt = 1 / 60f;

    private DataManager.database db;
    private DataManager.character ch;

    private Pose pose;
    private Pose current_pose;
    private Pose trns_pose;
    private Pose global_pose;

    private Vector3[] bone_offset_positions;
    private Vector4[] bone_offset_rotations;
    private Vector3[] bone_offset_velocities;
    private Vector3[] bone_offset_angular_velocities;

    Vector3 transition_src_position;
    Vector4 transition_src_rotation;
    Vector3 transition_dst_position;
    Vector4 transition_dst_rotation;

    private int frame_index;
    private int window = 20;
    private int frame_count = 0;
    private float frame_time= 0.0f;

    private List<int> bone_indexes = new List<int>();
    private List<Transform> bones = new List<Transform>();
    #endregion

    // Start is called before the first frame update
    void Start()
    {
        halflife = inertialize_blending_halflife;

        int root_index = 0;
        initialize_skeleton(this.transform, ref root_index);

        db = DataManager.load_database("Assets/Resources/database.bin");
        ch = DataManager.load_character("Assets/Resources/character.bin");

        Debug.Log(db.nbones());
        Debug.Log(bone_indexes.Count);

        if (db.nbones() != bone_indexes.Count && db.nbones() != bones.Count && db.nbones() != ch.nbones())
        {
            Debug.LogError("Database and skeleton do not match!");
            return;
        }

        (db.features, db.features_offset, db.features_scale) = DataManager.load_features("Assets/Resources/features.bin");

        frame_index = db.range_starts[2];
        Debug.Log(frame_index);

        initialize_pose();

        //inertialize_pose_reset();
        //inertialize_pose_update(pose.DeepClone(), 0.0f);

        initialize_models();

        feature_curr = db.features[frame_index];
        feature_proj = db.features[frame_index];

        latent_curr = new float[32];
        latent_proj = new float[32];
    }
    private void initialize_models() {

        stepper_inference = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled,
            ModelLoader.Load(stepper));
        decompressor_inference = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled,
            ModelLoader.Load(decompressor));

        stepper_nn = DataManager.Load_net_fromParameters("Assets/NNModels/stepper.bin");
        decompressor_nn = DataManager.Load_net_fromParameters("Assets/NNModels/decompressor.bin");
    }
    private void initialize_skeleton(Transform bone, ref int n)
    {
        if (bone.CompareTag("joint"))
        {
            bones.Add(bone);
            bone_indexes.Add(n);
        }
        n += 1;
        foreach (Transform child in bone)
            initialize_skeleton(child, ref n);
    }
    private void initialize_pose()
    {
        pose = new Pose(db.nbones());

        pose.root_position = db.bone_positions[frame_index][0];
        pose.root_rotation = db.bone_rotations[frame_index][0];
        pose.root_velocity = db.bone_velocities[frame_index][0];
        pose.root_angular_velocity = db.bone_angular_velocities[frame_index][0];

        for (int i=1; i<db.nbones(); i++)
        {
            pose.joints[i-1].position = db.bone_positions[frame_index][i];
            pose.joints[i-1].rotation = db.bone_rotations[frame_index][i];
            pose.joints[i-1].velocity = db.bone_velocities[frame_index][i];
            pose.joints[i-1].angular_velocity = db.bone_angular_velocities[frame_index][i];
        }

        current_pose = pose.DeepClone();
        trns_pose = pose.DeepClone();

        bone_offset_positions = new Vector3[bone_indexes.Count];
        bone_offset_rotations = new Vector4[bone_indexes.Count];
        bone_offset_velocities = new Vector3[bone_indexes.Count];
        bone_offset_angular_velocities = new Vector3[bone_indexes.Count];

        global_pose = new Pose(db.nbones());
    }
    private void OnDestroy()
    {
        stepper_inference.Dispose();
        decompressor_inference.Dispose();
    }
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

    // Update is called once per frame
    void Update()
    {
        frame_time += Time.deltaTime;
        if(frame_time >= dt)
        {
            if (frame_count % window == 0)
                set_frame(frame_index + frame_count);

            //Set new features_curr and latent_curr
            evaluate_stepper();

            //Set new curr_pose
            evaluate_decompressor(ref current_pose);
            Debug.Log("FRAME " + frame_count);
            Debug.Log(current_pose.root_position);
            Debug.Log(Quat.convert_ToEuler(current_pose.root_rotation) * Mathf.Rad2Deg);
            //Set new pose
            //inertialize_pose_update(current_pose, dt);

            // Full pass of forward kinematics to compute 
            // all bone positions and rotations in the world
            // space ready for rendering (set global_pose)
            forward_kinamatic_full();

            display_frame_pose();
            pose = current_pose;
            frame_time = 0f;
            frame_count++;
        }
    }
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
        nnLayer_denormalize(ref stepper_out, stepper_nn);

        for (int i = 0; i < feature_curr.Length; i++)
            feature_curr[i] += dt * stepper_out[i];
        for (int i = 0; i < latent_curr.Length; i++)
            latent_curr[i] += dt * stepper_out[feature_curr.Length + i];

        stepper_in.Dispose();
        stepper_out.Dispose();

        //float[] stepper_in = new float[feature_curr.Length + latent_curr.Length];
        //Array.Copy(feature_curr, stepper_in, feature_curr.Length);
        //Array.Copy(latent_curr, 0, stepper_in, feature_curr.Length, latent_curr.Length);
        //float[] stepper_out = new float[feature_curr.Length + latent_curr.Length];

        //stepper_nn.evaluate(stepper_in, ref stepper_out);

        //Array.Copy(stepper_out, feature_curr, feature_curr.Length);
        //Array.Copy(stepper_out, feature_curr.Length, latent_curr, 0, latent_curr.Length);
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
        nnLayer_denormalize(ref decompressor_out, decompressor_nn);

        target_pose = Parser.parse_decompressor_out(decompressor_out, pose, db.nbones());

        decompressor_in.Dispose();
        decompressor_out.Dispose();

        //float[] decompressor_in = new float[feature_curr.Length + latent_curr.Length];
        //Array.Copy(feature_curr, decompressor_in, feature_curr.Length);
        //Array.Copy(latent_curr, 0, decompressor_in, feature_curr.Length, latent_curr.Length);
        //float[] decompressor_out = new float[338];

        //decompressor_nn.evaluate(decompressor_in, ref decompressor_out);

        //Tensor _out = new Tensor(new int[] { 1, 1, 1, decompressor_out.Length }, decompressor_out);

        //target_pose = Parser.parse_decompressor_out(_out, pose, db.nbones());
    }
    private void nnLayer_denormalize(ref Tensor _out, Model param)
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

    #region Inertializers
    private void inertialize_pose_reset() {
        for(int i=0; i<db.nbones(); i++)
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

        for(int i=1; i<bone_indexes.Count; i++)
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
            halflife,
            _dt);
        Spring.inertialize_update(
            ref pose.root_rotation,
            ref pose.root_angular_velocity,
            ref bone_offset_rotations[0],
            ref bone_offset_angular_velocities[0],
            world_space_rot,
            world_space_angular_vel,
            halflife,
            _dt);

        for(int i=1; i<bone_indexes.Count; i++)
        {
            Spring.inertialize_update(
                ref pose.joints[i - 1].position,
                ref pose.joints[i - 1].velocity,
                ref bone_offset_positions[i],
                ref bone_offset_velocities[i],
                input_pose.joints[i - 1].position,
                input_pose.joints[i - 1].velocity,
                halflife,
                _dt);
            Spring.inertialize_update(
                ref pose.joints[i - 1].rotation,
                ref pose.joints[i - 1].angular_velocity,
                ref bone_offset_rotations[i],
                ref bone_offset_angular_velocities[i],
                input_pose.joints[i - 1].rotation,
                input_pose.joints[i - 1].angular_velocity,
                halflife,
                _dt);
        }
    }
    #endregion
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
                global_pose.root_position = pose.root_position;
                global_pose.root_rotation = pose.root_rotation;
            }
            else
            {
                Vector3 parent_position = db.bone_parents[i] == 0 ? global_pose.root_position : 
                    global_pose.joints[db.bone_parents[i] - 1].position;
                Vector4 parent_rotation = db.bone_parents[i] == 0 ? global_pose.root_rotation :
                    global_pose.joints[db.bone_parents[i] - 1].rotation;

                global_pose.joints[i-1].position = Quat.quat_mul_vec(parent_rotation, pose.joints[i-1].position) + parent_position;
                global_pose.joints[i - 1].rotation = Quat.quat_mul(parent_rotation, pose.joints[i - 1].rotation);
            }
        }
    }
}
