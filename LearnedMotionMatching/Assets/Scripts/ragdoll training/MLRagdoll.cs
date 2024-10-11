using System;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using static MotionMatcher.character;
using Unity.VisualScripting;

public struct CharInfo
{
    public GameObject charObj;
    public Transform transform;
    public Transform[] boneToTransform;
    public GameObject[] boneToCollider;
    public ArticulationBody root;
    public ArticulationBody[] boneToArt;
    public Vector3 cm;
    public Vector3 cmVel;
    public Vector3[] boneWorldPos;
    public Vector3[][] surfacePts;
    public Vector3[][] surfacePtsWorld;
    public Vector3[][] surfaceVels;
    public float[] boneState;
    public MotionMatcher MMScript;

    public CharInfo(int nbodies, int numStateBones) : this()
    {
        boneToCollider= new GameObject[nbodies];
        boneToArt= new ArticulationBody[nbodies];

        surfacePts = new Vector3[nbodies][];
        surfacePtsWorld = new Vector3[nbodies][];
        surfaceVels = new Vector3[nbodies][];

        boneWorldPos = new Vector3[numStateBones];
        // { LeftToe, RightToe, Spine, Head, LeftForeArm, RightForeArm },
        // we compute positions and velocities then concatenate these
        boneState = new float[36];
    }

}

public class MLRagdoll : Agent
{
    private ConfigManager _config;
    private DataManager.database db;

    CharInfo kinChar, simChar;

    private GameObject kinematicCharObj;
    private GameObject simulatedCharObj;

    private MotionMatcher MMScript;
    private SimCharacterController simController;

    public GameObject kin_char_prefab;
    public GameObject sim_char_prefab;

    private int nbodies;

    private int curFixedUpdate = -1;

    float[] prevActionOutput;
    float[] smoothedActions;
    int numActions;
    int numObservations;

    public bool updateVelOnTeleport = true;
    private int lastSimCharTeleportFixedUpdate = -1;

    [HideInInspector]
    public MotionMatcher.character[] stateBones = new MotionMatcher.character[]
    { Bone_LeftToe, Bone_RightToe, Bone_Spine, Bone_Head, Bone_LeftForeArm, Bone_RightForeArm };

    private const float dt = 1 / 60f;

    private Unity.MLAgents.Policies.BehaviorParameters behaviorParam;

    private Vector3 feetBoxSize;
    private Vector3 leftFootColliderCenter;
    private Vector3 rightFootColliderCenter;
    private float toeColliderRadius;
    private void init()
    {
        MMScript = kinematicCharObj.GetComponent<MotionMatcher>();
        Debug.Assert(MMScript.Is_initialized);

        db = MMScript.DataBase;
        nbodies = db.nbones();

        kinChar = new CharInfo(nbodies, stateBones.Length);
        kinChar.charObj = kinematicCharObj;
        kinChar.transform = kinematicCharObj.transform;
        kinChar.boneToTransform = MMScript.rigToTransform;
        kinChar.MMScript = MMScript;

        simController = simulatedCharObj.GetComponent<SimCharacterController>();
        Debug.Assert(simController.Is_initialized);

        simChar = new CharInfo(nbodies, stateBones.Length);
        simChar.charObj = simulatedCharObj;
        simChar.transform = simulatedCharObj.transform;
        simChar.boneToTransform = simController.boneToTransform;
        simChar.root = simController.boneToArtBody[(int)Bone_Entity];
        simChar.boneToArt = simController.boneToArtBody;

        foreach (var body in simulatedCharObj.GetComponentsInChildren<ArticulationBody>())
        {
            body.solverIterations = _config.Training_data.solverIterations;
            body.solverVelocityIterations = _config.Training_data.solverIterations;
        }

        for(int i=0; i<nbodies; i++)
        {
            if(i == (int)Bone_LeftFoot || i == (int)Bone_RightFoot)
            {
                kinChar.boneToCollider[i] = UnityObjUtils.getChildBoxCollider(kinChar.boneToTransform[i].gameObject);
                simChar.boneToCollider[i] = UnityObjUtils.getChildBoxCollider(simChar.boneToTransform[i].gameObject);
                if (i == (int)Bone_LeftFoot)
                    leftFootColliderCenter = simChar.boneToCollider[i].GetComponent<BoxCollider>().center;
                else if (i == (int)Bone_RightFoot)
                    rightFootColliderCenter = simChar.boneToCollider[i].GetComponent<BoxCollider>().center;
            }
            else
            {
                kinChar.boneToCollider[i] = UnityObjUtils.getChildCapsuleCollider(kinChar.boneToTransform[i].gameObject);
                simChar.boneToCollider[i] = UnityObjUtils.getChildCapsuleCollider(simChar.boneToTransform[i].gameObject);
            }
        }
        toeColliderRadius = simChar.boneToCollider[(int)Bone_LeftToe].GetComponent<CapsuleCollider>().radius;

        behaviorParam = GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
        numObservations = behaviorParam.BrainParameters.VectorObservationSize;
        numActions = behaviorParam.BrainParameters.ActionSpec.NumContinuousActions;

        curFixedUpdate = _config.Training_data.EVALUATE_EVERY_K - 1;
        resetData();
    }
    private void resetData()
    {
        for (int i = 0; i < nbodies; i++)
        {
            kinChar.surfacePts[i] = new Vector3[6];
            kinChar.surfacePtsWorld[i] = new Vector3[6];
            kinChar.surfaceVels[i] = new Vector3[6];
            simChar.surfacePts[i] = new Vector3[6];
            simChar.surfacePtsWorld[i] = new Vector3[6];
            simChar.surfaceVels[i] = new Vector3[6];
        }
        prevActionOutput = new float[numActions];
        smoothedActions = new float[numActions];
        kinChar.boneState = new float[36];
        simChar.boneState = new float[36];

        UpdateKinCmData(false, dt);
        UpdateSimCmData(false, dt);
        UpdateBoneState(false, dt, true);
        UpdateBoneSurfacePts(false, dt);
    }
    private void Awake()
    {
        _config = ConfigManager.Instance;

        kinematicCharObj = Instantiate(kin_char_prefab, Vector3.zero, Quaternion.identity);
        simulatedCharObj = Instantiate(sim_char_prefab, Vector3.zero, Quaternion.identity);

        if(Academy.Instance.IsCommunicatorOn)
        {
            int numStepPerSecond = (int)Mathf.Ceil(1f / dt);
            MaxStep = numStepPerSecond * _config.Training_data.MAX_EPISODE_LENGTH_SECONDS;
        }
        init();
    }
    private int lastEpisodeEndingFrame = 0;
    public override void OnEpisodeBegin()
    {
        lastEpisodeEndingFrame = curFixedUpdate;
        SimCharacterController.teleportSimChar(simChar, kinChar, .01f, updateVelOnTeleport);
        lastSimCharTeleportFixedUpdate = curFixedUpdate;
        Physics.Simulate(.00001f);
        resetData();
        kinChar.cmVel = Vector3.zero;
        simChar.cmVel = Vector3.zero;
    }
    private void UpdateKinCmData(bool updateVelocity, float dt)
    {
        Vector3 newKinCM = getCM(kinChar.boneToTransform);
        kinChar.cmVel = updateVelocity ? (newKinCM - kinChar.cm) / dt : kinChar.cmVel;
        kinChar.cm = newKinCM;
    }
    private void UpdateSimCmData(bool updateVelocity, float dt)
    {
        Vector3 newSimCm = getCM(simChar.boneToTransform);
        simChar.cmVel = updateVelocity ? (newSimCm - simChar.cm) / dt : simChar.cmVel;
        simChar.cm = newSimCm;
    }
    private void UpdateBoneState(bool updateVelocity, float dt, bool zeroVelocity = false, bool updateKinOnly = false)
    {
        foreach (bool isKinChar in new bool[] { true, false })
        {
            if (updateKinOnly && !isKinChar)
                continue;
            CharInfo curInfo = isKinChar ? kinChar : simChar;
            float[] copyInto = curInfo.boneState;
            int copyIdx = 0;
            for (int j = 0; j < stateBones.Length; j++)
            {
                MotionMatcher.character bone = stateBones[j];
                Vector3 boneWorldPos = curInfo.boneToTransform[(int)bone].position;
                Vector3 boneLocalPos = isKinChar ? resolvePosInKinematicRefFrame(boneWorldPos) : resolvePosInSimRefFrame(boneWorldPos);
                Vector3 prevBonePos = curInfo.boneWorldPos[j];
                Vector3 boneVel = (boneWorldPos - prevBonePos) / dt;
                boneVel = zeroVelocity ? Vector3.zero : isKinChar ? resolveVelInKinematicRefFrame(boneVel) : resolveVelInSimRefFrame(boneVel);
                ArrayUtils.copyVecIntoArray(ref copyInto, ref copyIdx, boneLocalPos);

                if (updateVelocity || zeroVelocity)
                    ArrayUtils.copyVecIntoArray(ref copyInto, ref copyIdx, boneVel);
                else
                    copyIdx += 3;
                curInfo.boneWorldPos[j] = boneWorldPos;
            }
        }
    }
    private void UpdateBoneSurfacePts(bool updateVelocity, float dt)
    {
        foreach (bool isKinChar in new bool[] { true, false })
            for (int i = 1; i < 23; i++)
            {
                var charInfo = isKinChar ? kinChar : simChar;
                Vector3[] newSurfacePts = new Vector3[6];
                Vector3[] newWorldSurfacePts = new Vector3[6];
                UnityObjUtils.getSixPointsOnCollider(charInfo.boneToCollider[i], ref newWorldSurfacePts, (MotionMatcher.character)i);
                Vector3[] prevWorldSurfacePts = charInfo.surfacePtsWorld[i];

                for (int j = 0; j < 6; j++)
                {
                    newSurfacePts[j] = isKinChar ? resolvePosInKinematicRefFrame(newWorldSurfacePts[j]) : resolvePosInSimRefFrame(newWorldSurfacePts[j]);
                    if (updateVelocity)
                    {
                        Vector3 surfaceVel = (newWorldSurfacePts[j] - prevWorldSurfacePts[j]) / dt;
                        charInfo.surfaceVels[i][j] = isKinChar ? resolveVelInKinematicRefFrame(surfaceVel) : resolveVelInSimRefFrame(surfaceVel);
                    }
                }
                charInfo.surfacePtsWorld[i] = newWorldSurfacePts;
                charInfo.surfacePts[i] = newSurfacePts;
            }
    }

    public static Vector3 getCM(Transform[] boneToTransform, Vector3[] globalBonePositions = null)
    {
        // We start at 1 because 0 is the root bone with no colliders
        float totalMass = 0f;
        Vector3 CoM = Vector3.zero;
        for (int i = 1; i < boneToTransform.Length; i++)
        {
            Transform t = boneToTransform[i];
            var ab = t.GetComponent<ArticulationBody>();
            float mass = t.GetComponent<ArticulationBody>().mass;
            Vector3 childCenter = globalBonePositions == null ? UnityObjUtils.getChildColliderCenter(t.gameObject) : globalBonePositions[i];
            CoM += mass * childCenter;
            totalMass += ab.mass;

        }
        return CoM / totalMass;
    }
    Vector3 resolveVelInKinematicRefFrame(Vector3 vel)
    {
        return MathUtils.quat_inv_mul_vec3(kinChar.transform.rotation, vel);
    }
    Vector3 resolveVelInSimRefFrame(Vector3 vel)
    {
        return MathUtils.quat_inv_mul_vec3(_config.Training_data.resolveSimReferenceFrameWithSimRotation ? 
            simChar.transform.rotation : kinChar.transform.rotation, vel);
    }
    Vector3 resolvePosInKinematicRefFrame(Vector3 pos)
    {
        return MathUtils.quat_inv_mul_vec3(kinChar.transform.rotation, pos - kinChar.cm);
    }
    Vector3 resolvePosInSimRefFrame(Vector3 pos)
    {
        return MathUtils.quat_inv_mul_vec3(_config.Training_data.resolveSimReferenceFrameWithSimRotation ? 
            simChar.transform.rotation : kinChar.transform.rotation, pos - simChar.cm);
    }
}