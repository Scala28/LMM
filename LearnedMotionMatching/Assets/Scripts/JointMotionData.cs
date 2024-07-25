using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class JointMotionData
{
    public Vector3 position;
    public Vector4 rotation;
    public Vector3 velocity;
    public Vector3 angular_velocity;
    
    public JointMotionData()
    {
        position = Vector3.zero;
        rotation = new Vector4(1.0f, .0f, .0f, .0f);
        velocity = Vector3.zero;
        angular_velocity = Vector3.zero;
    }
}
