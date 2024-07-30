using JetBrains.Annotations;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Security.Cryptography;
using Unity.Barracuda;
using Unity.VisualScripting.FullSerializer;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UIElements;

public static  class Parser 
{
    private const float dt = 1 / 60f;
    public static Pose parse_decompressor_out(Tensor decompressor_out, Pose currentPose, int nbones)
    {
        Tensor pos = SliceAndReshape(decompressor_out, 0 * (nbones - 1), 3 * (nbones - 1), new TensorShape(nbones - 1, 3, 1, 1));
        Tensor txy = SliceAndReshape(decompressor_out, 3 * (nbones - 1), 9 * (nbones - 1), new TensorShape(nbones - 1, 3, 2, 1));
        Tensor vel = SliceAndReshape(decompressor_out, 9 * (nbones - 1), 12 * (nbones - 1), new TensorShape(nbones - 1, 3, 1, 1));
        Tensor ang = SliceAndReshape(decompressor_out, 12 * (nbones - 1), 15 * (nbones - 1), new TensorShape(nbones - 1, 3, 1, 1));
        Tensor rVel = SliceAndReshape(decompressor_out, 15 * (nbones - 1), 15 * (nbones - 1) + 3, new TensorShape(3, 1, 1, 1));
        Tensor rAng = SliceAndReshape(decompressor_out, 15 * (nbones - 1) + 3, 15 * (nbones - 1) + 6, new TensorShape(3, 1, 1, 1));

        //Convert to quat: (nbones-1, 4, 1, 1)
        Tensor quat = Quat.quat_from_xfm_xy(txy);

        Vector3 root_vel = new Vector3(rVel[0], rVel[1], rVel[2]);
        Vector3 root_ang = new Vector3(rAng[0], rAng[1], rAng[2]);

        Vector3 world_rVel = Quat.quat_mul_vec(currentPose.root_rotation, root_vel);
        Vector3 world_rAng = Quat.quat_mul_vec(currentPose.root_rotation, root_ang);

        //Find new root pos/rot and velocities
        Vector3 root_pos = dt * world_rVel + currentPose.root_position;
        Vector4 root_rot = Quat.quat_mul(Quat.quat_from_scaled_angle_axis(world_rAng * dt), currentPose.root_rotation);

        //Convert quat to angle axis
        //Tensor euler_rotations = Quat.quat_toEuler(quat_rotations);

        // Construct pose for next frame
        Pose pose = new Pose(pos, quat, vel, ang, root_pos, root_rot, root_vel, root_ang);

        return pose;
    }
    private static Tensor SliceAndReshape(Tensor input, int sliceStart, int sliceEnd, TensorShape newShape)
    {
        Tensor sliced = new Tensor(newShape);

        int dataCount = sliceEnd - sliceStart;
        float[] data = new float[dataCount];
        for (int i = 0; i < dataCount; i++)
            data[i] = input[0, 0, 0, i + sliceStart];

        for (int i = 0; i < newShape.batch; i++)
        {
            for (int h = 0; h < newShape.height; h++)
            {
                for (int w = 0; w < newShape.width; w++)
                {
                    for (int c = 0; c < newShape.channels; c++)
                    {
                        sliced[i, h, w, c] = data[index(i, h, w, c, newShape)];
                    }
                }
            }
        }

        return sliced;
    }
    
    private static int index(int bone, int vector, int component, int subcomponent, TensorShape shape)
    {
        return bone * shape.height * shape.width * shape.channels +
               vector * shape.width * shape.channels +
               component * shape.channels +
               subcomponent;
    }
}
