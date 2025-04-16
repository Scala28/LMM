using Cinemachine;
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
public class ControllerOrchestrator : MonoBehaviour
{
    public Transform[] rigToTransform;

    [Header("Behaviours")]
    public List<Controller> controllers;
    private Controller current_controller;
    private DataManager.database current_db;

    #region LMM
    private float[] feature_curr;
    private float[] latent_curr;

    #endregion

    private SyncFPS _sync60Fps;
    private const float dt = 1 / 60f;

    public InputHandler input_handler {  get; private set; }
    private PlayerInput player_input;
    private Dictionary<Behaviour, string> action_maps = new Dictionary<Behaviour, string>()
    {
        {Behaviour.plane, "locomotion" },
        {Behaviour.terrain, "locomotion" },
        {Behaviour.fight, "fight" },
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
        Bone_LeftShoulder = 13,
        Bone_LeftArm = 14,
        Bone_LeftForeArm = 15,
        Bone_LeftHand = 16,
        Bone_Neck = 17,
        Bone_Head = 18,
        Bone_RightShoulder = 19,
        Bone_RightArm = 20,
        Bone_RightForeArm = 21,
        Bone_RightHand = 22
    };

    private Pose global_pose;

    [Header("Camera")]
    public CinemachineVirtualCamera vcam;
    [SerializeField] private Transform camera_follow;
    [SerializeField] private Transform camera_lookAt;
    private float camera_azimuth = 0.0f;
    private float camera_altitude = .4f;
    private float camera_distance = 4.0f;

    [Header("Others")]
    public bool lock60Fps = true;
    public bool gizmos = false;
    public bool set_vcam = true;
    public bool rigged = true;
    [SerializeField] private string ch_filename;
    private DataManager.character ch;
    public bool read_database = false;
    private Mesh mesh;

    int nbones = Enum.GetValues(typeof(character)).Length;
    private void Awake()
    {
        Application.targetFrameRate = 60;
        _sync60Fps = SyncFPS.Instance;

        input_handler = GetComponent<InputHandler>();
        player_input = GetComponent<PlayerInput>();

        if (!rigged)
        {
            ch = DataManager.load_character("Data/" + ch_filename);
            Debug.Assert(ch.nbones() == nbones);

            mesh = DataManager.gen_mesh_from_character(ch);
            transform.GetComponent<MeshFilter>().mesh = mesh;
        }
        else
            Debug.Assert(rigToTransform.Length + 1 == nbones);


        foreach (Controller controller in controllers) {
            controller.motion_controller.Setup(this);
        }

        current_controller = controllers[0];
        current_db = current_controller.motion_controller.getDB();
        Debug.Assert(current_db.nbones() == nbones);

        feature_curr = new float[current_db.nfeatures()];
        latent_curr = new float[current_controller.motion_controller.getLatents()[0].Length];

        player_input.SwitchCurrentActionMap(action_maps[current_controller.behaviour]);

        if (set_vcam)
        {
            vcam.Follow = camera_follow;
            vcam.LookAt = camera_lookAt;
        }
    }
    private void switchCurrentController(Controller controller)
    {
        Debug.Assert(controller.motion_controller.getDB().nbones() == nbones);

        float[] x_pass = new float[controller.motion_controller.getDB().nfeatures()];
        float[] z_pass = new float[controller.motion_controller.getLatents()[0].Length];

        Array.Copy(latent_curr, z_pass, latent_curr.Length); 

        //TODO: set x_pass based on current behaviour and new behaviour
        // First 27 values (plane locomotion) are in common
        switch (current_controller.behaviour)
        {
            case Behaviour.plane: // Add terrain/fight features
                break;
            case Behaviour.terrain: // Remove terrain features; add fight features if controller.behav is Fight
                break;
            case Behaviour.fight: // Remove fight features; add terrain features if controller.behav is Terrain
                break;
            default: break;
        }
        current_controller = controller;
        current_db = current_controller.motion_controller.getDB();
        current_controller.motion_controller.SetFeatureCurr(x_pass);
        current_controller.motion_controller.SetLatentCurr(z_pass);
        player_input.SwitchCurrentActionMap(action_maps[current_controller.behaviour]);
    }
    private void FixedUpdate()
    {
        if (lock60Fps && !_sync60Fps.isSyncFrame)
            return;

        if (read_database)
        {
            global_pose = current_controller.motion_controller.GetNextFrame();
            display_frame_pose();
            return;
        }

        camera_azimuth = current_controller.motion_controller.camera_azimuth;
        camera_altitude = current_controller.motion_controller.camera_altitude;
        camera_distance = current_controller.motion_controller.camera_distance;

        (global_pose, feature_curr, latent_curr) = current_controller.motion_controller.perform_cycle();

        if (!rigged)
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

        transform.position = new Vector3(global_pose.root_position.x, global_pose.root_position.y, global_pose.root_position.z);
        transform.rotation = new Quaternion(global_pose.root_rotation.y, global_pose.root_rotation.z, global_pose.root_rotation.w, global_pose.root_rotation.x);

        for (int i = 0; i < global_pose.joints.Length; i++)
        {
            Transform joint = rigToTransform[i];
            JointMotionData jdata = global_pose.joints[i];

            joint.position = new Vector3(jdata.position.x, jdata.position.y, jdata.position.z);
            joint.rotation = new Quaternion(jdata.rotation.y, jdata.rotation.z, jdata.rotation.w, jdata.rotation.x);
        }
    }

    private void OnDrawGizmosSelected()
    {
        if (Application.isPlaying && gizmos)
            switch (current_controller.behaviour)
            {
                case Behaviour.plane:
                    try
                    {
                        (Vector3[] traj_pos, Vector4[] traj_rot) = (current_controller.motion_controller as PlaneController).Gizmos();
                        foreach (Vector3 v in traj_pos)
                        {
                            Gizmos.DrawSphere(v, .15f);
                        }
                    }
                    catch { }
                    break;
                case Behaviour.terrain:
                    try
                    {
                        (Vector3[] traj_pos, Vector4[] traj_rot, Vector3[][] terrain_toe_pos) = (current_controller.motion_controller as TerrainController).Gizmos();
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
                case Behaviour.fight:
                    try
                    {
                        (Vector3[] traj_pos, Vector4[] traj_rot) = (current_controller.motion_controller as FightController).Gizmos();
                        foreach (Vector3 v in traj_pos)
                        {
                            Gizmos.DrawSphere(v, .15f);
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
    plane,
    terrain,
    fight,
    climb,
}
[System.Serializable]
public class Controller { 
    public MotionController motion_controller;
    public Behaviour behaviour;

}
