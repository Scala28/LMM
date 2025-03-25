using Cinemachine;
using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UIElements;

public class ControllerOrchestrator : MonoBehaviour
{
    public Transform[] rigToTransform;

    [Header("Behaviours")]
    public List<Controller> controllers;
    private Controller current_controller;
    private DataManager.database current_db;

    #region LMM
    private float[] feature_curr;
    private float[] feature_proj;
    private float[] latent_curr;
    private float[] latent_proj;

    #endregion

    private SyncFPS _sync60Fps;
    private const float dt = 1 / 60f;

    private InputHandler input_handler;
    private PlayerInput player_input;
    private Dictionary<Behaviour, string> action_maps = new Dictionary<Behaviour, string>()
    {
        {Behaviour.locomotion, "locomotion" },
        {Behaviour.fight, "combact" },
        {Behaviour.climb, "climb" },
    };
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

    private Pose pose;
    private Pose global_pose;

    [Header("Camera")]
    public CinemachineVirtualCamera vcam;
    [SerializeField] private Transform camera_follow;
    [SerializeField] private Transform camera_lookAt;
    private float camera_azimuth = 0.0f;
    private float camera_altitude = .4f;
    private float camera_distance = 4.0f;

    [Header("Others")]
    [SerializeField] private LayerMask whatIsTerrain;
    public bool lock60Fps = false;
    public bool gizmos = false;
    public bool set_vcam = true;
    public bool render_mesh = false;
    [SerializeField] private string ch_filename;
    private DataManager.character ch;
    private Mesh mesh;

    int nbones = Enum.GetValues(typeof(character)).Length;
    private void Awake()
    {
        Application.targetFrameRate = 60;
        _sync60Fps = SyncFPS.Instance;
        input_handler = GetComponent<InputHandler>();
        player_input = GetComponent<PlayerInput>();

        if (render_mesh)
        {
            ch = DataManager.load_character("Data/" + ch_filename);
            Debug.Assert(ch.nbones() == nbones);

            mesh = DataManager.gen_mesh_from_character(ch);
            transform.GetComponent<MeshFilter>().mesh = mesh;
        }
        else
            Debug.Assert(rigToTransform.Length == nbones);

        current_controller = controllers[0];
        current_controller.motion_controller.Setup(this);

        current_db = current_controller.motion_controller.getDB();

        if (render_mesh)
            Debug.Assert(ch.nbones() == current_db.nbones());
        else
            Debug.Assert(current_db.nbones() == rigToTransform.Length);

        feature_curr = new float[current_db.nfeatures()];
        feature_proj = new float[current_db.nfeatures()];

        float[][] latents = current_controller.motion_controller.getLatents();

        latent_curr = new float[latents[0].Length];
        latent_proj = new float[latents[0].Length];

        player_input.SwitchCurrentActionMap(action_maps[current_controller.behaviour]);

        if (set_vcam)
        {
            vcam.Follow = camera_follow;
            vcam.LookAt = camera_lookAt;
        }

        if(current_controller.behaviour == Behaviour.locomotion)
        {
            (current_controller.motion_controller as LocomotionController).whatIsTerrain = whatIsTerrain;
        }
    }

    private void FixedUpdate()
    {
        if (lock60Fps && !_sync60Fps.isSyncFrame)
            return;

        camera_distance = current_controller.motion_controller.camera_distance;
        camera_azimuth = current_controller.motion_controller.camera_azimuth;
        camera_altitude = current_controller.motion_controller.camera_altitude;

        Vector3 gamepad_stickleft = input_handler.StickLeft;
        Vector3 gamepad_stickright = input_handler.StickRight;

        if (current_controller.behaviour == Behaviour.locomotion)
        {
            camera_azimuth = current_controller.motion_controller.camera_azimuth;
            camera_altitude = current_controller.motion_controller.camera_altitude;
            camera_distance = current_controller.motion_controller.camera_distance;

            (global_pose, feature_curr, latent_curr) = current_controller.motion_controller.perform_cycle(gamepad_stickleft, gamepad_stickright,
                input_handler.RightShoulder, input_handler.LeftTrigger);
        }

        if (render_mesh)
            deform_character_mesh();
        else
            display_frame_pose();
    }
    public void SetVcam(Vector3 eye, Vector3 target)
    {
        camera_follow.position = eye;
        camera_lookAt.position = target;
    }
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
        //Debug.Log("display_pose");
        transform.position = new Vector3(global_pose.root_position.x, global_pose.root_position.y, global_pose.root_position.z);
        transform.rotation = new Quaternion(global_pose.root_rotation.y, global_pose.root_rotation.z, global_pose.root_rotation.w, global_pose.root_rotation.x);

        Matrix4x4 mirrorMatrix = Matrix4x4.Scale(new Vector3(-1, 1, -1));

        for (int i = 1; i < nbones; i++)
        {
            Transform joint = rigToTransform[i];
            JointMotionData jdata = global_pose.joints[i - 1];

            joint.position = mirrorMatrix.MultiplyPoint3x4(new Vector3(-jdata.position.x, jdata.position.y, -jdata.position.z));
            Quaternion q = new Quaternion(-jdata.rotation.y, jdata.rotation.z, -jdata.rotation.w, jdata.rotation.x);
            joint.rotation = mirrorMatrix.rotation * q;
        }
    }

    private void OnDrawGizmosSelected()
    {
        if (Application.isPlaying && gizmos)
            switch (current_controller.behaviour)
            {
                case Behaviour.locomotion:
                    try
                    {
                        (Vector3[] traj_pos, Vector3[][] terrain_toe_pos) = (current_controller.motion_controller as LocomotionController).Gizmos();
                        foreach (Vector3 v in traj_pos)
                        {
                            Gizmos.DrawSphere(v, .15f);
                        }
                        foreach (Vector3[] vec in terrain_toe_pos)
                        {
                            foreach (Vector3 v in vec)
                            {
                                Gizmos.DrawCube(v, new Vector3(.2f, .2f, .2f));
                            }
                        }
                    }
                    catch { }
                    break;
                default:
                    break;
            }
    }
    private void OnDestroy()
    {
        current_controller.motion_controller.CleanUp();
    }

}
public enum Behaviour { 
    locomotion,
    fight,
    climb,
}
[System.Serializable]
public class Controller { 
    public MotionController motion_controller;
    public Behaviour behaviour;

}
