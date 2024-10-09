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

    private void Awake()
    {
        _config = ConfigManager.Instance;
        db = DataManager.load_database("Assets/Resources/terrain_db.bin");

        initColliders();
        initArticulationBodies();
        initArticulationDrives();
        setupIgnoreCollisions();
    }

    private void initColliders()
    {
        boneToCollider = new Collider[db.nbones()];
        for (int i = 1; i < db.nbones(); i++)
        {
            Transform trans = boneToTransform[i];
            if (i == (int)Bone_LeftFoot || i == (int)Bone_RightFoot)
                boneToCollider[i] = UnityObjUtils.getChildBoxCollider(trans.gameObject).GetComponent<Collider>();
            else
                boneToCollider[i] = UnityObjUtils.getChildCapsuleCollider(trans.gameObject).GetComponent<Collider>();
            if (boneToCollider[i] == null)
                Debug.Log($"Could not find collider for {(MotionMatcher.character)i}");
        }
    }
    private void initArticulationBodies()
    {
        for(int i=0; i<db.nbones(); i++)
            boneToArtBody[i] = boneToTransform[i].GetComponent<ArticulationBody>();
    }
    private void initArticulationDrives()
    {
        for(int i=1; i<db.nbones(); i++)
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

        for (int i = 2; i < db.nbones(); i++) // start at 2 because hip has no parent collider
        {
            int parent = db.bone_parents[i];
            Physics.IgnoreCollision(boneToCollider[i], boneToCollider[parent]);
            Physics.IgnoreCollision(boneToCollider[i], boneToCollider[(int)Bone_Head]);
            Physics.IgnoreCollision(boneToCollider[i], boneToCollider[(int)Bone_Neck]);
            if (i == (int)Bone_LeftUpLeg || i == (int)Bone_RightUpLeg)
                Physics.IgnoreCollision(boneToCollider[i], boneToCollider[(int)Bone_Spine]);

        }
    }
}
