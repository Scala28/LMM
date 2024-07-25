using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.Barracuda;
using UnityEditor;
using UnityEngine;
using UnityEngine.XR;

public class Pose
{
    public JointMotionData[] joints;
    public Vector3 root_position;
    public Vector4 root_rotation;
    public Vector3 root_velocity;
    public Vector3 root_angular_velocity;

    public Pose(Tensor pos, Tensor rot, Tensor vel, Tensor ang, Vector3 root_vel, Vector3 root_ang)
    {
        root_position = new Vector3(pos[0, 0, 0, 0], pos[0, 1, 0, 0], pos[0, 2, 0, 0]);

        root_rotation = new Vector4(rot[0, 0, 0, 0],
                                   rot[0, 1, 0, 0],
                                   rot[0, 2, 0, 0],
                                   rot[0, 3, 0, 0]);
        root_velocity = root_vel;
        root_angular_velocity = root_ang;

        joints = new JointMotionData[pos.batch - 1];

        for (int i=1; i < pos.batch; i++)
        {
            JointMotionData j = new JointMotionData();
            j.position = new Vector3(pos[i, 0, 0, 0], pos[i, 1, 0, 0], pos[i, 2, 0, 0]);
            j.rotation = new Vector4(rot[i, 0, 0, 0], rot[i, 1, 0, 0], rot[i, 2, 0, 0], rot[i, 3, 0, 0]);
            j.velocity = new Vector3(vel[i-1, 0, 0, 0], vel[i-1, 1, 0, 0], vel[i-1, 2, 0, 0]);
            j.angular_velocity = new Vector3(ang[i-1, 0, 0, 0], ang[i-1, 1, 0, 0], ang[i-1, 2, 0, 0]);

            joints[i - 1] = j;
        }
    }
    public Pose(int nbones)
    {
        joints = new JointMotionData[nbones-1];
        root_position = Vector3.zero;
        root_rotation = new Vector4(1.0f, .0f, .0f, .0f);
        root_velocity = Vector3.zero;
        root_angular_velocity = Vector3.zero;

        for(int i=0; i<nbones-1; i++)
        {
            joints[i] = new JointMotionData();
        }
    }
    public Pose() { }
    public Pose DeepClone()
    {
        Pose clone = new Pose()
        {
            root_position = this.root_position,
            root_rotation = this.root_rotation,
            root_velocity = this.root_velocity,
            root_angular_velocity = this.root_angular_velocity,
            joints = new JointMotionData[this.joints.Length]
        };
        for(int i=0; i<joints.Length; i++)
        {
            clone.joints[i] = new JointMotionData()
            {
                position = this.joints[i].position,
                rotation = this.joints[i].rotation,
                velocity = this.joints[i].velocity,
                angular_velocity = this.joints[i].angular_velocity
            };
        }
        return clone;
    }

}
