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
        Tensor root_vel = SliceAndReshape(decompressor_out, 15 * (nbones - 1), 15 * (nbones - 1) + 3, new TensorShape(3, 1, 1, 1));
        Tensor root_ang = SliceAndReshape(decompressor_out, 15 * (nbones - 1) + 3, 15 * (nbones - 1) + 6, new TensorShape(3, 1, 1, 1));

        //Convert to quat: (nbones-1, 4, 1, 1)
        Tensor quat = Quat.quat_from_xfm_xy(txy);

        //Put root velocities in world space
        Vector3 world_rVel = Quat.quat_mul_vec(currentPose.root_rotation, new Vector3(root_vel[0],
                                                                                     root_vel[1],
                                                                                     root_vel[2]));
        Vector3 world_rAng = Quat.quat_mul_vec(currentPose.root_rotation, new Vector3(root_ang[0],
                                                                                     root_ang[1],
                                                                                     root_ang[2]));

        //Find new root pos/rot and velocities
        Vector3 root_pos = dt * world_rVel + currentPose.root_position;
        Vector4 root_rot = Quat.quat_mul(Quat.quat_from_scaled_angle_axis(world_rAng * dt), currentPose.root_rotation);


        Tensor positions = new Tensor(nbones, 3, 1, 1);
        positions[0, 0, 0, 0] = root_pos.x;
        positions[0, 1, 0, 0] = root_pos.y;
        positions[0, 2, 0, 0] = root_pos.z;
        for (int i = 1; i < nbones; i++)
        {
            positions[index(i, 0, 0, 0, positions.shape)] = pos[index(i - 1, 0, 0, 0, pos.shape)];
            positions[index(i, 1, 0, 0, positions.shape)] = pos[index(i - 1, 1, 0, 0, pos.shape)];
            positions[index(i, 2, 0, 0, positions.shape)] = pos[index(i - 1, 2, 0, 0, pos.shape)];
        }

        Tensor quat_rotations = new Tensor(nbones, 4, 1, 1);
        quat_rotations[0, 0, 0, 0] = root_rot.x;
        quat_rotations[0, 1, 0, 0] = root_rot.y;
        quat_rotations[0, 2, 0, 0] = root_rot.z;
        quat_rotations[0, 3, 0, 0] = root_rot.w;
        for (int i = 1; i < nbones; i++)
        {

            quat_rotations[index(i, 0, 0, 0, quat_rotations.shape)] = quat[index(i - 1, 0, 0, 0, quat.shape)];
            quat_rotations[index(i, 1, 0, 0, quat_rotations.shape)] = quat[index(i - 1, 1, 0, 0, quat.shape)];
            quat_rotations[index(i, 2, 0, 0, quat_rotations.shape)] = quat[index(i - 1, 2, 0, 0, quat.shape)];
            quat_rotations[index(i, 3, 0, 0, quat_rotations.shape)] = quat[index(i - 1, 3, 0, 0, quat.shape)];
        }


        //Convert quat to angle axis
        //Tensor euler_rotations = Quat.quat_toEuler(quat_rotations);

        // Construct pose for next frame
        Pose pose = new Pose(positions, quat_rotations, vel, ang,
            new Vector3(root_vel[0, 0, 0, 0], root_vel[1, 0, 0, 0], root_vel[2, 0, 0, 0]),  //root vel
            new Vector3(root_ang[0, 0, 0, 0], root_ang[1, 0, 0, 0], root_ang[2, 0, 0, 0]));  // root ang

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
