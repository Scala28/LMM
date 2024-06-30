using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEditor;
using UnityEngine;

public class Pose
{
    public List<JointMotionData> joints;
    public Vector3 rootPosition;
    public Quaternion rootRotation;
    public Vector3 rootVelocity;
    public Vector3 rootAngularVelocity;

    public Pose(Transform root, Vector3 root_vel, Vector3 root_ang)
    {
        rootPosition = root.position;
        rootRotation = root.rotation;
        rootVelocity = root_vel;
        rootAngularVelocity = root_ang;
    }

    public Pose(Tensor pos, Tensor rot, Tensor vel, Tensor ang, Vector3 root_vel, Vector3 root_ang)
    {
        rootPosition = new Vector3(pos[0, 0, 0, 0], pos[0, 1, 0, 0], pos[0, 2, 0, 0]);

        rootRotation = Quaternion.Euler(rot[0, 0, 0, 0],
                                        rot[0, 1, 0, 0],
                                        rot[0, 2, 0, 0]);
        rootVelocity = root_vel;
        rootAngularVelocity = root_ang;

        joints = new List<JointMotionData>();

        for(int i=1; i < pos.batch; i++)
        {
            JointMotionData j = new JointMotionData();
            j.localPosition = new Vector3(pos[i, 0, 0, 0], pos[i, 1, 0, 0], pos[i, 2, 0, 0]);
            j.localRotation = Quaternion.Euler(rot[i, 0, 0, 0], rot[i, 1, 0, 0], rot[i, 2, 0, 0]);
            j.velocity = new Vector3(vel[i-1, 0, 0, 0], vel[i-1, 1, 0, 0], vel[i-1, 2, 0, 0]);
            j.angularVelocity = new Vector3(ang[i-1, 0, 0, 0], ang[i-1, 1, 0, 0], ang[i-1, 2, 0, 0]);

            joints.Add(j);
        }
    }

}
