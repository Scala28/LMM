using System;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using static MotionMatcher.character;
using Cinemachine;
using Unity.VisualScripting;
using System.Linq;

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
        boneToCollider = new GameObject[nbodies];
        boneToArt = new ArticulationBody[nbodies];

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
    private SyncFPS _sync60Fps;
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

    private Vector3 lastKinRootPos = Vector3.zero;

    public bool updateVelOnTeleport = true;
    private int lastSimCharTeleportFixedUpdate = -1;

    [Header("Camera")]
    public CinemachineVirtualCamera vcam;

    public static MotionMatcher.character[] stateBones = new MotionMatcher.character[]
    { Bone_LeftToe, Bone_RightToe, Bone_Spine, Bone_Head, Bone_LeftForeArm, Bone_RightForeArm };

    public static MotionMatcher.character[] fullDOFBones = new MotionMatcher.character[]
    { Bone_LeftUpLeg, Bone_RightUpLeg, Bone_LeftFoot, Bone_RightFoot, Bone_LeftArm, Bone_RightArm, Bone_Hips };

    public static MotionMatcher.character[] extendedfullDOFBones = new MotionMatcher.character[]
    { Bone_LeftUpLeg, Bone_RightUpLeg, Bone_LeftFoot, Bone_RightFoot, Bone_LeftArm, Bone_RightArm, Bone_Hips, Bone_Spine,  Bone_Spine1, Bone_Spine2, Bone_LeftShoulder, Bone_RightShoulder};

    public static MotionMatcher.character[] limitedDOFBones = new MotionMatcher.character[]
    { Bone_LeftLeg, Bone_RightLeg };

    public static MotionMatcher.character[] extendedLimitedDOFBones = new MotionMatcher.character[]
    { Bone_LeftLeg, Bone_RightLeg, Bone_LeftForeArm, Bone_RightForeArm};

    public static MotionMatcher.character[] openloopBones = new MotionMatcher.character[]
    { Bone_Hips, Bone_Spine1, Bone_Spine2, Bone_Neck, Bone_Head, Bone_LeftForeArm, Bone_LeftHand, Bone_RightForeArm, Bone_RightHand, Bone_LeftShoulder, Bone_RightShoulder};

    public static MotionMatcher.character[] alwaysOpenloopBones = new MotionMatcher.character[]
    { Bone_Neck, Bone_Head, Bone_LeftHand, Bone_RightHand, Bone_LeftToe, Bone_RightToe};

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

        for (int i = 0; i < nbodies; i++)
        {
            if (i == (int)Bone_LeftFoot || i == (int)Bone_RightFoot)
            {
                kinChar.boneToCollider[i] = UnityObjUtils.getChildBoxCollider(kinChar.boneToTransform[i].gameObject);
                simChar.boneToCollider[i] = UnityObjUtils.getChildBoxCollider(simChar.boneToTransform[i].gameObject);
                feetBoxSize = simChar.boneToCollider[i].GetComponent<BoxCollider>().size;
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

        foreach (var ab in simChar.transform.GetComponentsInChildren<ArticulationBody>())
        {
            ab.gameObject.AddComponent<CollisionReporter>().agent = this;
            ab.collisionDetectionMode = CollisionDetectionMode.Continuous;
        }

        bool isInference = behaviorParam.BehaviorType == Unity.MLAgents.Policies.BehaviorType.InferenceOnly;
        if (isInference) {
            kinChar.MMScript.setVCam(vcam);
            kinChar.MMScript.training = false;
            kinChar.MMScript.gen_inputs = false;
        }

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
        _sync60Fps = SyncFPS.Instance;

        if(vcam == null)
            vcam = GameObject.FindGameObjectWithTag("camera").GetComponent<CinemachineVirtualCamera>();

        kinematicCharObj = Instantiate(kin_char_prefab, Vector3.zero, Quaternion.identity);
        simulatedCharObj = Instantiate(sim_char_prefab, Vector3.zero, Quaternion.identity);

        if (Academy.Instance.IsCommunicatorOn)
        {
            int numStepPerSecond = (int)Mathf.Ceil(1f / dt);
            MaxStep = numStepPerSecond * _config.Training_data.MAX_EPISODE_LENGTH_SECONDS;
        }
        init();
    }
    private int lastEpisodeEndingFrame = 0;
    public override void OnEpisodeBegin()
    {
        Debug.Log("Begin episode");
        lastEpisodeEndingFrame = curFixedUpdate;
        float vOffset = getVerticalOffset();
        SimCharacterController.teleportSimChar(simChar, kinChar, vOffset + .15f, updateVelOnTeleport);
        lastSimCharTeleportFixedUpdate = curFixedUpdate;
        Physics.Simulate(.0001f);
        resetData();
        kinChar.cmVel = Vector3.zero;
        simChar.cmVel = Vector3.zero;
    }
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        Quaternion[] curRotations = MMScript.local_pose.getRotations_quat();
        for(int i=0; i<fullDOFBones.Length; i++)
        {
            int bone_idx = (int)fullDOFBones[i];
            if (fullDOFBones[i] != Bone_Hips)
                simChar.boneToArt[bone_idx].SetDriveRotation(curRotations[bone_idx]);
            else
                simChar.boneToArt[bone_idx].SetDriveRotation(new Quaternion(-curRotations[bone_idx].x, -curRotations[bone_idx].y, curRotations[bone_idx].z, curRotations[bone_idx].w));
        }
        for(int i=0; i<limitedDOFBones.Length; i++)
        {
            MotionMatcher.character bone = limitedDOFBones[i];
            ArticulationBody ab = simChar.boneToArt[(int)bone];
            Vector3 target = ab.ToTargetRotationInReducedSpace(curRotations[(int)bone], true);
            ArticulationDrive drive = ab.zDrive;
            drive.target = target.z;
            ab.zDrive = drive;
        }
        for(int i=0; i<openloopBones.Length; i++)
        {
            int bone_idx = (int)openloopBones[i];
            if (openloopBones[i] != Bone_Hips)
                simChar.boneToArt[bone_idx].SetDriveRotation(curRotations[bone_idx]);
            else
                simChar.boneToArt[bone_idx].SetDriveRotation(new Quaternion(-curRotations[bone_idx].x, -curRotations[bone_idx].y, curRotations[bone_idx].z, curRotations[bone_idx].w));
        }
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        float[] state = getState();
        for (int i = 0; i < state.Length; i++)
        {
            if (float.IsNaN(state[i]) || float.IsInfinity(state[i]))
            {
                Debug.LogError($"NaN or Infinity detected in state[{i}]: {state[i]}");
            }
        }
        sensor.AddObservation(state);
    }
    public override void OnActionReceived(ActionBuffers actions)
    {
        prevActionOutput = actions.ContinuousActions.Array;
        applyActions(true);
    }
    bool updateVelocity;
    public void FixedUpdate()
    {
        if (MMScript.teleportedThisFixedUpdate)
        {
            Vector3 preTeleportSimCharOffset = lastKinRootPos - simChar.transform.position;
            SimCharacterController.teleportSimCharRoot(simChar, MMScript.origin, Vector3.zero);
            //SimCharacterController.teleportSimCharRoot(simChar, MMScript.origin, preTeleportSimCharOffset);
            applyActions(false);
            lastSimCharTeleportFixedUpdate = curFixedUpdate;
        }
        if (!_sync60Fps.isSyncFrame)
            return;

        curFixedUpdate++;
        updateVelocity = lastSimCharTeleportFixedUpdate + 1 < curFixedUpdate;
        UpdateKinCmData(updateVelocity, dt);
        UpdateBoneState(updateVelocity, dt);

        if (curFixedUpdate % _config.Training_data.EVALUATE_EVERY_K == 0)
            RequestDecision();
        else
            applyActions(_config.Training_data.applyActionOverMultipleTimeSteps);

        lastKinRootPos = kinChar.transform.position;
    }
    private float[] getState()
    {
        Vector3 cmDistance = resolvePosInKinematicRefFrame(simChar.cm);
        Vector3 kinCMVelInKinRefFrame = resolveVelInKinematicRefFrame(kinChar.cmVel);
        Vector3 simCMVelInKinRefFrame = resolveVelInKinematicRefFrame(simChar.cmVel);
        Vector3 desiredVel = resolveVelInKinematicRefFrame(MMScript.Desired_velocity);
        Vector3 velDiff = simCMVelInKinRefFrame - desiredVel;

        float[] state = new float[numObservations];
        int state_idx = 0;
        ArrayUtils.copyVecIntoArray(ref state, ref state_idx, cmDistance);
        ArrayUtils.copyVecIntoArray(ref state, ref state_idx, kinCMVelInKinRefFrame);
        ArrayUtils.copyVecIntoArray(ref state, ref state_idx, simCMVelInKinRefFrame);
        ArrayUtils.copyVecIntoArray(ref state, ref state_idx, simCMVelInKinRefFrame - kinCMVelInKinRefFrame);
        ArrayUtils.copyVecIntoArray(ref state, ref state_idx, new Vector2(desiredVel.x, desiredVel.z));
        ArrayUtils.copyVecIntoArray(ref state, ref state_idx, new Vector2(velDiff.x, velDiff.z));

        if (_config.Training_data.addOrientationDataToState)
        {
            float yawDiff = (Quaternion.Inverse(kinChar.transform.rotation) * simulatedCharObj.transform.rotation).GetYAngle();
            float yawDiffDesired = (Quaternion.Inverse(MMScript.Desired_rotation) * simulatedCharObj.transform.rotation).GetYAngle();
            ArrayUtils.copyVecIntoArray(ref state, ref state_idx, MathUtils.getContinuousRepOf2DAngle(yawDiff));
            ArrayUtils.copyVecIntoArray(ref state, ref state_idx, MathUtils.getContinuousRepOf2DAngle(yawDiffDesired));
        }

        for (int i = 0; i < 36; i++)
            state[state_idx++] = simChar.boneState[i];
        for (int i = 0; i < 36; i++)
            state[state_idx++] = simChar.boneState[i] - kinChar.boneState[i];
        for (int i = 0; i < numActions; i++)
            state[state_idx++] = smoothedActions[i];

        Debug.Assert(state_idx == numObservations);

        if (state.Contains(float.NaN))
            Debug.Log("Nan values in observations");

        return state;
    }
    private void applyActions(bool applyLastAction)
    {
        if (applyLastAction)
            for (int i = 0; i < numActions; i++)
                smoothedActions[i] = (1 - _config.Training_data.ACTION_STIFFNESS_HYPERPARAM) * smoothedActions[i] +
                    _config.Training_data.ACTION_STIFFNESS_HYPERPARAM * prevActionOutput[i];

        Quaternion[] curRotations = MMScript.local_pose.getRotations_quat();

        int actionIdx = 0;

        MotionMatcher.character[] fullDof_toUse = _config.Training_data.networkControlsAllJoints ? extendedfullDOFBones : fullDOFBones;
        applyActionsAsEulerRotations(smoothedActions, curRotations, fullDof_toUse, ref actionIdx);

        MotionMatcher.character[] limitedDOF_toUse = _config.Training_data.networkControlsAllJoints ? extendedLimitedDOFBones : limitedDOFBones;
        for (int i = 0; i < limitedDOF_toUse.Length; i++)
        {
            int boneIdx = (int)limitedDOF_toUse[i];
            ArticulationBody ab = simChar.boneToArt[boneIdx];

            float output = smoothedActions[actionIdx];
            actionIdx++;

            float target;
            var zDrive = ab.zDrive;
            float range = zDrive.upperLimit - zDrive.lowerLimit;
            if (_config.Training_data.setRotsDirectly)
            {
                var midpoint = zDrive.lowerLimit + (range / 2);
                target = (output * (range / 2)) + midpoint;
            }
            else
            {
                float angle = output * range;
                Vector3 targetRotationInJointSpace = ab.ToTargetRotationInReducedSpace(curRotations[boneIdx], true);
                target = targetRotationInJointSpace.z + angle;
            }
            zDrive.target = target;
            ab.zDrive = zDrive;
        }

        MotionMatcher.character[] openLoop_toUse = _config.Training_data.networkControlsAllJoints ? alwaysOpenloopBones : openloopBones;
        for (int i = 0; i < openLoop_toUse.Length; i++)
        {
            int boneIdx = (int)openLoop_toUse[i];
            Quaternion final = openLoop_toUse[i] != Bone_Hips ? curRotations[boneIdx] :
                new Quaternion(-curRotations[boneIdx].x, -curRotations[boneIdx].y, curRotations[boneIdx].z, curRotations[boneIdx].w);
            ArticulationBody ab = simChar.boneToArt[boneIdx];
            ab.SetDriveRotation(final);
        }
        if (_config.Training_data.setDriveTargetVelocities)
        {
            for (int i = 1; i < nbodies; i++)
                simChar.boneToArt[i].SetDriveTargetVelocity(MMScript.local_pose.joints[i - 1].angular_velocity, curRotations[i]);
        }
    }
    private void applyActionsAsEulerRotations(float[] finalActions, Quaternion[] curRotations, MotionMatcher.character[] fullDOFBonesToUse, ref int actionIdx)
    {
        for (int i = 0; i < fullDOFBonesToUse.Length; i++)
        {
            int boneIdx = (int)fullDOFBonesToUse[i];
            ArticulationBody ab = simChar.boneToArt[boneIdx];
            Vector3 output = new Vector3(finalActions[actionIdx], finalActions[actionIdx + 1], finalActions[actionIdx + 2]);
            actionIdx += 3;
            Vector3 targetRotationInJointSpace = ab.ToTargetRotationInReducedSpace(fullDOFBonesToUse[i] != Bone_Hips ? curRotations[boneIdx] :
                                                                new Quaternion(-curRotations[boneIdx].x, -curRotations[boneIdx].y, curRotations[boneIdx].z, curRotations[boneIdx].w), true);
            float scale, midpoint;

            var xdrive = ab.xDrive;
            scale = (xdrive.upperLimit - xdrive.lowerLimit) / 2f;
            midpoint = xdrive.lowerLimit + scale;
            float outputX = _config.Training_data.setRotsDirectly ? (output.x * scale) + midpoint : output.x * scale * 2;
            if (_config.Training_data.fullRangeEulerOutputs)
            {
                outputX = output.x * 180f;
            }
            xdrive.target = _config.Training_data.setRotsDirectly ? outputX : targetRotationInJointSpace.x + outputX;
            ab.xDrive = xdrive;

            var ydrive = ab.yDrive;
            scale = (ydrive.upperLimit - ydrive.lowerLimit) / 2f;
            midpoint = ydrive.lowerLimit + scale;
            float outputY = _config.Training_data.setRotsDirectly ? (output.y * scale) + midpoint : output.y * scale * 2;
            if (_config.Training_data.fullRangeEulerOutputs)
            {
                outputY = output.y * 180f;
            }
            ydrive.target = _config.Training_data.setRotsDirectly ? outputY : targetRotationInJointSpace.y + outputY;
            ab.yDrive = ydrive;

            var zdrive = ab.zDrive;
            scale = (zdrive.upperLimit - zdrive.lowerLimit) / 2f;
            midpoint = zdrive.lowerLimit + scale;
            float outputZ = _config.Training_data.setRotsDirectly ? (output.z * scale) + midpoint : output.z * scale * 2;
            if (_config.Training_data.fullRangeEulerOutputs)
            {
                outputZ = output.z * 180f;
            }
            zdrive.target = _config.Training_data.setRotsDirectly ? outputZ : targetRotationInJointSpace.z + outputZ;
            ab.zDrive = zdrive;
        }
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
            for (int i = 1; i < nbodies; i++)
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

    internal float finalReward = 0f;
    private bool endThisFrame = false;

    public void LateFixedUpdate() {

        calculateReward();

        bool isInference = behaviorParam.BehaviorType == Unity.MLAgents.Policies.BehaviorType.InferenceOnly;
        if (!isInference)
            return;

        if (_config.Training_data.clampKinCharToSim)
        {
            kinChar.MMScript.clamp_kinChar(simChar.cm);
        }
        UpdateKinCmData(false, dt);
        UpdateBoneState(false, dt, false, true);
        return;
    }
    private void calculateReward() {
        bool headApart;
        double posReward, velReward, local_posReward, cmVelReward, fallFactor;
        fallFactorReward(out fallFactor, out headApart);

        if ((headApart && curFixedUpdate > lastSimCharTeleportFixedUpdate + 1 && !_config.Training_data.clampKinCharToSim) ||
            (_config.Training_data.clampKinCharToSim && endThisFrame)) {

            finalReward = _config.Training_data.EPISODE_END_REWARD;
            SetReward(_config.Training_data.EPISODE_END_REWARD);
            //Debug.Log($"{Time.frameCount}: Calling end episode on: {curFixedUpdate}, lasted {curFixedUpdate - lastEpisodeEndingFrame} frames ({(curFixedUpdate - lastEpisodeEndingFrame) / 60f} sec)");
            endThisFrame = false;
            EndEpisode();
            return;
        }

        UpdateSimCmData(updateVelocity, dt);
        UpdateBoneSurfacePts(updateVelocity, dt);
        posAndVelReward(out posReward, out velReward);
        localPosReward(out local_posReward);
        CmVelReward(out cmVelReward);

        if (curFixedUpdate - _config.Training_data.N_FRAMES_TO_NOT_COUNT_REWARD_AFTER_TELEPORT < lastEpisodeEndingFrame)
            finalReward = 0;
        else
            finalReward = (float)(fallFactor * (posReward + velReward + local_posReward + cmVelReward));
        AddReward(finalReward);

        return;
    }
    private void posAndVelReward(out double posReward, out double velReward) {
        double posDiffsSum = 0;
        double velDiffsSum = 0;
        for (int i = 1; i < nbodies; i++) {
            for (int j = 0; j < 6; j++) {
                posDiffsSum += (kinChar.surfacePts[i][j] - simChar.surfacePts[i][j]).magnitude;
                velDiffsSum += (kinChar.surfaceVels[i][j] - simChar.surfaceVels[i][j]).magnitude;
            }
        }
        posReward = Math.Exp(-10f / nbodies * posDiffsSum);
        velReward = Math.Exp(-1f / nbodies * velDiffsSum);
    }
    private void localPosReward(out double posReward) {
        double totLoss = 0;
        for (int i = 0; i < nbodies; i++) {
            Transform kinBone = kinChar.boneToTransform[i];
            Transform simBone = simChar.boneToTransform[i];
            // diff * q1 = q2  --->  diff = q2 * inverse(q1)
            Quaternion diff = simBone.localRotation * Quaternion.Inverse(kinBone.localRotation);
            Vector3 diff_vec = new Vector3(diff.x, diff.y, diff.z);
            double angle = 2 * Math.Atan2(diff_vec.magnitude, diff.w);
            angle = Math.Abs(GeoUtils.wrap_radians((float)angle));
            totLoss += (float)angle;
        }
        posReward = Math.Exp(-10f / nbodies * totLoss);
    }
    private void CmVelReward(out double cmVelReward) {
        cmVelReward = Math.Exp(-1d * (resolveVelInKinematicRefFrame(kinChar.cmVel) - resolveVelInSimRefFrame(simChar.cmVel)).magnitude);
    }
    private void fallFactorReward(out double fallFactor, out bool headApart) {
        Vector3 kinHeadPos = kinChar.boneToTransform[(int)Bone_Head].position;
        Vector3 simHeadPos = simChar.boneToTransform[(int)Bone_Head].position;
        float squareHDistance = (kinHeadPos - simHeadPos).sqrMagnitude;
        headApart = squareHDistance > 1f;
        fallFactor = Math.Clamp(1.3 - 1.4 * Mathf.Sqrt(squareHDistance), 0d, 1d);
    }
    public void processCollision(Collision collision) {
        if (!_config.Training_data.clampKinCharToSim)
            return;
        foreach (ContactPoint contact in collision.contacts) {
            string colliderName = contact.thisCollider.gameObject.name;
            if (!colliderName.ToLower().Contains("toe") && !colliderName.ToLower().Contains("foot") && !colliderName.ToLower().Contains("leg_") && contact.otherCollider.CompareTag("Ground"))
            {
                //Debug.Log($"Collider name: {colliderName} other collider name: {contact.otherCollider.gameObject.name}");
                endThisFrame = true;
            }
        }
    }
    public void AssignLayer(int layer)
    {
        simulatedCharObj.layer = layer;
        foreach (var child in simulatedCharObj.GetComponentsInChildren<Transform>())
            child.gameObject.layer = layer;
    }

    private float getVerticalOffset()
    {
        // ClearGizmos();
        Transform leftFoot = kinChar.boneToCollider[(int)Bone_LeftFoot].transform;
        Transform rightFoot = kinChar.boneToCollider[(int)Bone_RightFoot].transform;
        float minPointOnFoot = Mathf.Min(getBottomMostPointOnFoot(leftFoot, leftFootColliderCenter), getBottomMostPointOnFoot(rightFoot, rightFootColliderCenter));
        Transform leftToe = kinChar.boneToTransform[(int)Bone_LeftToe];
        Transform rightToe = kinChar.boneToTransform[(int)Bone_RightToe];
        float minToeY = Mathf.Min(leftToe.position.y, rightToe.position.y) - toeColliderRadius;
        float maxGroundPenetration = Mathf.Max(0f, 0f - Mathf.Min(minPointOnFoot, minToeY));
        return maxGroundPenetration;
    }
    private float getBottomMostPointOnFoot(Transform foot, Vector3 center)
    {
        float x = feetBoxSize.x / 2;
        float y = feetBoxSize.y / 2;
        float z = feetBoxSize.z / 2;
        Vector3 topLeft = foot.TransformPoint(center + new Vector3(x, -y, z));
        Vector3 topRight = foot.TransformPoint(center + new Vector3(x, -y, -z));
        Vector3 bottomLeft = foot.TransformPoint(center + new Vector3(-x, -y, z));
        Vector3 bottomRight = foot.TransformPoint(center + new Vector3(-x, -y, -z));
        return Mathf.Min(topLeft.y, topRight.y, bottomLeft.y, bottomRight.y);
    }

}
