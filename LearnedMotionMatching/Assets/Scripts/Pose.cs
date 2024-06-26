using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;

public class Pose
{
    public List<JointMotionData> joints;
    public Vector3 rootPosition;
    public Vector4 rootRotation;
    public Vector3 rootVelocity;
    public Vector3 rootAngularVelocity;

    public Pose(Transform root, Vector3 root_vel, Vector3 root_ang)
    {
        rootPosition = root.position;
        Quaternion q = root.rotation;
        rootRotation = new Vector4(q.x, q.y, q.z, q.w);
        rootVelocity = root_vel;
        rootAngularVelocity = root_ang;
    }

    public Pose(Tensor pos, Tensor rot, Tensor vel, Tensor ang,
        Vector3 root_pos, Vector4 root_rot, Vector3 root_vel, Vector3 root_ang)
    {
        rootPosition = root_pos;
        rootRotation = root_rot;
        rootVelocity = root_vel;
        rootAngularVelocity = root_ang;

        joints = new List<JointMotionData>();

        for(int i=0; i < pos.batch; i++)
        {
            JointMotionData j = new JointMotionData();
            j.localPosition = new Vector3(pos[i, 0, 0, 0], pos[i, 1, 0, 0], pos[i, 2, 0, 0]);
            j.localRotation = new Quaternion(rot[i, 0, 0, 0], rot[i, 1, 0, 0], rot[i, 2, 0, 0], rot[i, 3, 0, 0]);
            j.velocity = new Vector3(vel[i, 0, 0, 0], vel[i, 1, 0, 0], vel[i, 2, 0, 0]);
            j.angularVelocity = new Vector3(ang[i, 0, 0, 0], ang[i, 1, 0, 0], ang[i, 2, 0, 0]); 

            joints.Add(j);
        }
    }
}
