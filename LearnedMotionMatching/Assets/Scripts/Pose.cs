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

    public Pose(Tensor pos, Tensor rot, Tensor vel, Tensor ang, Vector3 root_pos, Vector4 root_rot, Vector3 root_vel, Vector3 root_ang)
    {
        root_position = root_pos;
        root_rotation = root_rot;
        root_velocity = root_vel;
        root_angular_velocity = root_ang;

        joints = new JointMotionData[pos.batch];

        for (int i=0; i < pos.batch; i++)
        {
            JointMotionData j = new JointMotionData();
            j.position = new Vector3(pos[i, 0, 0, 0], pos[i, 1, 0, 0], pos[i, 2, 0, 0]);
            j.rotation = new Vector4(rot[i, 0, 0, 0], rot[i, 1, 0, 0], rot[i, 2, 0, 0], rot[i, 3, 0, 0]);
            j.velocity = new Vector3(vel[i, 0, 0, 0], vel[i, 1, 0, 0], vel[i, 2, 0, 0]);
            j.angular_velocity = new Vector3(ang[i, 0, 0, 0], ang[i, 1, 0, 0], ang[i, 2, 0, 0]);

            joints[i] = j;
        }
        pos.Dispose();
        rot.Dispose();
        vel.Dispose();
        ang.Dispose();
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

    public Vector3[] getPositions()
    {
        Vector3[] _out = new Vector3[joints.Length + 1];
        _out[0] = root_position;
        int count = 1;
        foreach(JointMotionData jdata in joints)
        {
            _out[count] = new Vector3(jdata.position.x,
                                      jdata.position.y,
                                      jdata.position.z); 
            count++;
        }
        return _out;
    }
    public Vector4[] getRotations()
    {
        Vector4[] _out = new Vector4[joints.Length + 1];
        _out[0] = root_rotation;
        int count = 1;
        foreach(JointMotionData jdata in joints)
        {
            _out[count] = new Vector4(jdata.rotation.x,
                                      jdata.rotation.y,
                                      jdata.rotation.z,
                                      jdata.rotation.w);
            count++;
        }
        return _out;
    }

}
