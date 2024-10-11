using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using static MotionMatcher.character;


public class SimCharacterController : MonoBehaviour
{
    private ConfigManager _config;
    private DataManager.database db;

    public Transform[] boneToTransform;
    public Collider[] boneToCollider;
    public ArticulationBody[] boneToArtBody;

    #region Control
    [HideInInspector]
    public bool Is_initialized { get { return _initialized; } }
    private bool _initialized = false;
    #endregion

    private void Awake()
    {
        _config = ConfigManager.Instance;

        getColliders();
        getArticulationBodies();
        initArticulationDrives();
        setupIgnoreCollisions();

        _initialized = true;
    }

    private void getColliders()
    {
        boneToCollider = new Collider[boneToTransform.Length];
        for (int i = 1; i < boneToTransform.Length; i++)
        {
            Transform trans = boneToTransform[i];
            if(i == (int)Bone_LeftFoot || i == (int)Bone_RightFoot)
                boneToCollider[i] = UnityObjUtils.getChildBoxCollider(trans.gameObject).GetComponent<Collider>();
            else
                boneToCollider[i] = UnityObjUtils.getChildCapsuleCollider(trans.gameObject).GetComponent<Collider>();
            if (boneToCollider[i] == null)
                Debug.Log($"Could not find collider for {(MotionMatcher.character)i}");
        }
    }
    private void getArticulationBodies()
    {
        boneToArtBody = new ArticulationBody[boneToTransform.Length];
        for(int i=0; i<boneToTransform.Length; i++)
            boneToArtBody[i] = boneToTransform[i].GetComponent<ArticulationBody>();
    }
    private void initArticulationDrives()
    {
        for(int i=1; i<boneToTransform.Length; i++)
        {
            bool musclePowerExists = _config.Training_data.MusclePowers.Any(x => x.Bone == (MotionMatcher.character)i);
            Vector3 musclePower = musclePowerExists ? _config.Training_data.MusclePowers.First(x => x.Bone == (MotionMatcher.character)i).PowerVector : Vector3.zero;
            if (musclePowerExists)
                boneToArtBody[i].SetAllDriveStiffness(musclePower);
            else
                boneToArtBody[i].SetAllDriveStiffness(_config.Training_data.boneToStiffness[i]);

            Vector3 damping;
            if (_config.Training_data.dampingScalesWithStiffness)
                damping = musclePowerExists ? musclePower * .1f : Vector3.one * _config.Training_data.boneToStiffness[i] * .1f;
            else
                damping = Vector3.one * _config.Training_data.damping;
            boneToArtBody[i].SetAllDriveDamping(damping);
            boneToArtBody[i].SetAllForceLimit(_config.Training_data.forceLimit);
        }
    }
    private void setupIgnoreCollisions()
    {
        if (!_config.Training_data.selfCollision)
            return;
        Physics.IgnoreCollision(boneToCollider[(int)Bone_LeftArm], boneToCollider[(int)Bone_Spine2]);
        Physics.IgnoreCollision(boneToCollider[(int)Bone_RightArm], boneToCollider[(int)Bone_Spine2]);

        int[] torsoColliders = new int[] { (int)Bone_Neck, (int)Bone_LeftShoulder, (int)Bone_RightShoulder, (int) Bone_Spine2,
                                            (int) Bone_Spine1, (int) Bone_Spine, (int) Bone_Hips, (int) Bone_Head};

        int[] feetColliders = new int[] { (int)Bone_LeftUpLeg, (int)Bone_RightUpLeg, (int)Bone_LeftLeg, (int)Bone_RightLeg, (int)Bone_LeftFoot, (int)Bone_LeftToe, (int)Bone_RightFoot, (int)Bone_RightToe };

        for (int i = 0; i < torsoColliders.Length; i++)
            for (int j = i + 1; j < torsoColliders.Length; j++)
                Physics.IgnoreCollision(boneToCollider[torsoColliders[i]], boneToCollider[torsoColliders[j]]);

        for (int i = 0; i < feetColliders.Length; i++)
            for (int j = i + 1; j < feetColliders.Length; j++)
                Physics.IgnoreCollision(boneToCollider[feetColliders[i]], boneToCollider[feetColliders[j]]);

        for (int i = 2; i < boneToTransform.Length; i++) // start at 2 because hip has no parent collider
        {
            int parent = db.bone_parents[i];
            Physics.IgnoreCollision(boneToCollider[i], boneToCollider[parent]);
            Physics.IgnoreCollision(boneToCollider[i], boneToCollider[(int)Bone_Head]);
            Physics.IgnoreCollision(boneToCollider[i], boneToCollider[(int)Bone_Neck]);
            if (i == (int)Bone_LeftUpLeg || i == (int)Bone_RightUpLeg)
                Physics.IgnoreCollision(boneToCollider[i], boneToCollider[(int)Bone_Spine]);
        }
    }

    public static void teleportSimChar(CharInfo sim_char, CharInfo kin_char, float verticalOffset = .01f, bool setVelocities = false)
    {
        sim_char.transform.rotation = kin_char.transform.rotation;
        Transform kin_root = kin_char.boneToTransform[(int)Bone_Entity];
        Transform kinHips = kin_char.boneToTransform[(int)Bone_Hips];
        Transform simHips = sim_char.boneToTransform[(int)Bone_Hips];
        // Adding this to root transform position will give hip transform position
        Vector3 simHipPositionOffset = sim_char.transform.position - simHips.position;
        // we need to set: 
        // simRootPosition + simHipPositionOffset = kinHipPosition 
        // simRootPosition = kinHipPosition - simHipPositionOffset

        // We teleport the sim char a little higher to prevent it from clipping into the ground and bouncing off
        sim_char.root.TeleportRoot(kinHips.position + simHipPositionOffset + Vector3.up * verticalOffset, kin_char.transform.rotation);
        sim_char.root.resetJointPhysics();
        if (setVelocities)
        {
            sim_char.root.velocity = kin_char.MMScript.local_pose.root_velocity;
        }
        for (int i = 1; i < 23; i++)
        {
            MotionMatcher.character bone = (MotionMatcher.character)i;
            ArticulationBody body = sim_char.boneToArt[i];
            if (body.jointType != ArticulationJointType.SphericalJoint)
            {
                body.resetJointPhysics();
                continue;
            }
            Quaternion targetLocalRot = kin_char.boneToTransform[i].localRotation;
            bool isFootBone = bone == Bone_LeftFoot || bone == Bone_RightFoot;
            setArtBodyDrivesToRotationAndReset(body, new Quaternion(-targetLocalRot.x, targetLocalRot.y, -targetLocalRot.z, targetLocalRot.w), true, isFootBone);
        }
    }
    private static void setArtBodyDrivesToRotationAndReset(ArticulationBody body, Quaternion targetRot, bool resetEverything, bool doNotSetZRot = false)
    {
        Vector3 TargetRotationInJointSpace = body.ToTargetRotationInReducedSpace(targetRot, false);
        if (body.dofCount == 3)
        {
            body.resetJointPosition(doNotSetZRot ? new Vector3(TargetRotationInJointSpace.x, TargetRotationInJointSpace.y, 0f) : TargetRotationInJointSpace, resetEverything);
            TargetRotationInJointSpace *= Mathf.Rad2Deg;
            var drive = body.xDrive;
            drive.target = TargetRotationInJointSpace.x;
            body.xDrive = drive;

            drive = body.yDrive;
            drive.target = TargetRotationInJointSpace.y;
            body.yDrive = drive;

            drive = body.zDrive;
            drive.target = TargetRotationInJointSpace.z;
            body.zDrive = drive;
        }
        else if (body.dofCount == 1)
        {
            float new_target = 0f;
            if (body.twistLock != ArticulationDofLock.LockedMotion)
                new_target = TargetRotationInJointSpace.x;
            else if (body.swingYLock != ArticulationDofLock.LockedMotion)
                new_target = TargetRotationInJointSpace.y;
            else if (body.swingZLock != ArticulationDofLock.LockedMotion)
                new_target = TargetRotationInJointSpace.z;
            body.resetJointPosition(new_target, resetEverything);
            TargetRotationInJointSpace *= Mathf.Rad2Deg;
            var drive = body.zDrive;
            drive.target = TargetRotationInJointSpace.z;
            body.zDrive = drive;
        }
    }
}
 