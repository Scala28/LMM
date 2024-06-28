using JetBrains.Annotations;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Security.Cryptography;
using Unity.Barracuda;
using Unity.VisualScripting.FullSerializer;
using UnityEngine;

public static  class DataParser 
{
    private const float dt = 1 / 60f;
    public static Pose ParseDecompressorOutput(Tensor decompressor_out, Pose currentPose, int nbones)
    {
        Tensor pos = SliceAndReshape(decompressor_out, 0 * (nbones - 1), 3 * (nbones - 1), new TensorShape(nbones - 1, 3, 1, 1));
        Tensor txy = SliceAndReshape(decompressor_out, 3 * (nbones - 1), 9 * (nbones - 1), new TensorShape(nbones - 1, 3, 2, 1));
        Tensor vel = SliceAndReshape(decompressor_out, 9 * (nbones - 1), 12 * (nbones - 1), new TensorShape(nbones - 1, 3, 1, 1));
        Tensor ang = SliceAndReshape(decompressor_out, 12 * (nbones - 1), 15 * (nbones - 1), new TensorShape(nbones - 1, 3, 1, 1));
        Tensor root_vel = SliceAndReshape(decompressor_out, 15 * (nbones - 1), 15 * (nbones - 1) + 3, new TensorShape(3, 1, 1, 1));
        Tensor root_ang = SliceAndReshape(decompressor_out, 15 * (nbones - 1) + 3, 15 * (nbones - 1) + 6, new TensorShape(3, 1, 1, 1));

        //Convert to quat: (nbones-1, 4, 1, 1)
        Tensor quat = quat_from_xfm_xy(txy);

        //Integrate root displacement
        Vector3 prev_rootpos = currentPose.rootPosition;
        Vector4 prev_rootrot = new Vector4(currentPose.rootRotation.x,
                                        currentPose.rootRotation.y,
                                        currentPose.rootRotation.z,
                                        currentPose.rootRotation.w);

        Vector3 root_pos = prev_rootpos + quat_mul_vec(prev_rootrot, new Vector3(root_vel[0, 0, 0, 0],
                                                                                root_vel[1, 0, 0, 0],
                                                                                root_vel[2, 0, 0, 0])) * dt;

        Vector4 root_rot = quat_mul(prev_rootrot, quat_from_scaled_axis_angle(
            quat_mul_vec(prev_rootrot, new Vector3(root_ang[0, 0, 0, 0],
                                                    root_ang[1, 0, 0, 0],
                                                    root_ang[2, 0, 0, 0])) * dt));

        Debug.Log(root_rot.x + ", " + root_rot.y + ", " + root_rot.z + ", " + root_rot.w);

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

        string poss = "";
        for (int i = 0; i < 9; i++)
            poss += positions[i, 0, 0, 0] + ", " + positions[i, 1, 0, 0] + ", " + positions[i, 2, 0, 0] + "\n";
        Debug.Log(poss);

        Debug.Log(quat_rotations[0, 3, 0, 0]);
        string rots = "";
        for (int i=0; i < 9; i++)
        {
            rots += quat_rotations[i, 0, 0, 0] + ", " + quat_rotations[i, 1, 0, 0] + ", "
                + quat_rotations[i, 2, 0, 0] + ", " + quat_rotations[index(i, 3, 0, 0, quat_rotations.shape)] + "\n";
        }
        Debug.Log(rots);

        //Convert quat to angle axis
        Tensor rot = quat_toEuler(quat);

        // Construct pose for next frame
        Pose pose = new Pose(pos, rot, vel, ang,
            root_pos,
            convert_ToEuler(new Quaternion(root_rot.x, root_rot.y, root_rot.z, root_rot.w)),
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
    #region QuatFunctions
    private static Tensor quat_from_xfm_xy(Tensor x)
    {
        TensorShape shape = x.shape;

        int nbones = shape.batch;
        int vectorsPerBone = shape.height;  // 3
        int componentsPerVector = shape.width;  // 2
        int subcomponentsPerComponent = shape.channels;  //1

        Tensor c0 = new Tensor(nbones, 3, 1, 1);
        Tensor c1 = new Tensor(nbones, 3, 1, 1);
        Tensor c2 = new Tensor(nbones, 3, 1, 1);

        for (int i = 0; i < nbones; i++)
        {
            // Extract vectors from the tensor
            Vector3 x0 = new Vector3(
                x[i, 0, 0, 0],
                x[i, 1, 0, 0],
                x[i, 2, 0, 0]
            );

            Vector3 x1 = new Vector3(
                x[i, 0, 1, 0],
                x[i, 1, 1, 0],
                x[i, 2, 1, 0]
            );


            // Calculate c2
            Vector3 c2Vec = _cross(x0, x1);
            c2Vec = c2Vec / Mathf.Sqrt(c2Vec.x*c2Vec.x + c2Vec.y*c2Vec.y + c2Vec.z*c2Vec.z);

            // Calculate c1
            Vector3 c1Vec = _cross(c2Vec, x0);
            c1Vec = c1Vec / Mathf.Sqrt(c1Vec.x * c1Vec.x + c1Vec.y * c1Vec.y + c1Vec.z * c1Vec.z);

            // c0 is x0
            Vector3 c0Vec = x0;

            // Assign results back to tensors
            c0[index(i, 0, 0, 0, c0.shape)] = c0Vec.x; c0[index(i, 1, 0, 0, c0.shape)] = c0Vec.y; c0[index(i, 2, 0, 0, c0.shape)] = c0Vec.z;
            c1[index(i, 0, 0, 0, c0.shape)] = c1Vec.x; c1[index(i, 1, 0, 0, c0.shape)] = c1Vec.y; c1[index(i, 2, 0, 0, c0.shape)] = c1Vec.z;
            c2[index(i, 0, 0, 0, c0.shape)] = c2Vec.x; c2[index(i, 1, 0, 0, c0.shape)] = c2Vec.y; c2[index(i, 2, 0, 0, c0.shape)] = c2Vec.z;
        }

        // Concatenate c0, c1, c2 along the third dimension
        Tensor xfm = new Tensor(nbones, 3, 3, 1);
        for (int i = 0; i < nbones; i++)
        {
            for (int j = 0; j < 3; j++)
            {

                xfm[i, j, 0, 0] = c0[i, j, 0, 0];
                xfm[i, j, 1, 0] = c1[i, j, 0, 0];
                xfm[i, j, 2, 0] = c2[i, j, 0, 0];
            }
        }
        c0.Dispose();
        c1.Dispose();
        c2.Dispose();

        Tensor ris = new Tensor(nbones, 4, 1, 1);
        for (int i = 0; i < nbones; i++)
        {
            Matrix4x4 mat = new Matrix4x4();
            mat.m00 = xfm[i, 0, 0, 0];
            mat.m01 = xfm[i, 0, 1, 0];
            mat.m02 = xfm[i, 0, 2, 0];
            mat.m10 = xfm[i, 1, 0, 0];
            mat.m11 = xfm[i, 1, 1, 0];
            mat.m12 = xfm[i, 1, 2, 0];
            mat.m20 = xfm[i, 2, 0, 0];
            mat.m21 = xfm[i, 2, 1, 0];
            mat.m22 = xfm[i, 2, 2, 0];

            Vector4 quat = quat_from_xform(mat);

            ris[i, 0, 0, 0] = quat.x;
            ris[i, 1, 0, 0] = quat.y;
            ris[i, 2, 0, 0] = quat.z;
            ris[i, 3, 0, 0] = quat.w;
        }
        return ris;
    }

    //Matrix 3x3
    private static Vector4 quat_from_xform(Matrix4x4 xfm)
    {
        Vector4 q;
        float t;

        if (xfm.m22 < 0)
        {
            if (xfm.m00 > xfm.m11)
            {
                t = 1f + xfm.m00 - xfm.m11 - xfm.m22;
                q = new Vector4(xfm.m21 - xfm.m12,
                    t,
                    xfm.m10 + xfm.m01,
                    xfm.m02 + xfm.m20);
            }
            else
            {
                t = 1f - xfm.m00 + xfm.m11 - xfm.m22;
                q = new Vector4(xfm.m02 - xfm.m20,
                    xfm.m10 + xfm.m01,
                    t,
                    xfm.m21 + xfm.m12);
            }
        }
        else
        {
            if (xfm.m00 < -xfm.m11)
            {
                t = 1 - xfm.m00 - xfm.m11 + xfm.m22;
                q = new Vector4(xfm.m10 - xfm.m01,
                    xfm.m02 + xfm.m20,
                    xfm.m21 + xfm.m12,
                    t);
            }
            else
            {
                t = 1 + xfm.m00 + xfm.m11 + xfm.m22;
                q = new Vector4(t,
                    xfm.m21 - xfm.m12, 
                    xfm.m02 - xfm.m20,
                    xfm.m10 - xfm.m01);
            }
        }

        return quat_normalize(q);
    }

    private static Vector4 quat_normalize(Vector4 q, float eps = 1e-8f)
    {
        float norm = Mathf.Sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
        return q / (norm + eps);
    }
    private static Vector3 quat_mul_vec(Vector4 q, Vector3 vec)
    {
        Vector3 q_vector = new Vector3(q.y, q.z, q.w);
        float q_scalar = q.x;

        return vec + 2f * q_scalar * _cross(q_vector, vec) +
            _cross(q_vector, 2f * _cross(q_vector, vec));
    }
    private static Vector4 quat_mul(Vector4 a, Vector4 b)
    {
        float w = a.x * b.x - a.y * a.y - a.z * b.z - a.w * b.w;
        float x = a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z;
        float y = a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y;
        float z = a.z * b.w + a.y * b.z - a.z * b.y + a.w * b.x;

        return new Vector4(w, x, y, z);
    }
    private static Vector4 quat_from_scaled_axis_angle(Vector3 ang, float eps = 1e-5f)
    {
        Vector3 x = ang / 2f;
        float halfAngle = Mathf.Sqrt(x.x * x.x + x.y * x.y + x.z * x.z);
        float c, s;
        if (halfAngle < eps)
        {
            c = 1f;
            s = 1f;
        }
        else
        {
            c = Mathf.Cos(halfAngle);
            s = Mathf.Sin(halfAngle) / halfAngle;
        }
        Vector3 q_vec = x * s;

        return new Vector4(c, q_vec.x, q_vec.y, q_vec.z);
    }

    private static Tensor quat_toEuler(Tensor quat, string order="xyz")
    {
        Tensor ris = new Tensor(quat.batch, 3, 1, 1);

        for(int i=0; i<quat.batch;  i++)
        {
            float x = quat[i, 0, 0, 0];
            float y = quat[i, 1, 0, 0];
            float z = quat[i, 2, 0, 0];
            float w = quat[i, 3, 0, 0];

            Quaternion q = new Quaternion(x, y, z, w);

            Vector3 angle = convert_ToEuler(q, order); // .eulerAngles
            ris[i, 0, 0, 0] = angle.x;
            ris[i, 1, 0, 0] = angle.y;
            ris[i, 2, 0, 0] = angle.z;
        }
        return ris;
    }
    private static Vector3 convert_ToEuler(Quaternion q, string order="xyz")
    {
        float q0 = q.x;
        float q1 = q.y;
        float q2 = q.z;
        float q3 = q.w;

        float min = -1;
        float max = 1;

        if (order == "xyz")
        {
            return new Vector3(Mathf.Atan2(2f * (q0 * q1 + q2 * q3), 1f - 2f * (q1 * q1 + q2 * q2)),
                Mathf.Asin(Mathf.Min(Mathf.Max(2f * (q0 * q2 - q3 * q1), min), max)),
                Mathf.Atan2(2f * (q0 * q3 + q1 * q2), 1f - 2f * (q2 * q2 + q3 * q3)));
        }
        else
            //TODO: order zyx
            return Vector3.zero;
    }
    public static Quaternion quat_from_euler(Vector3 angle, string order="xyz")
    {
        float radX = angle.x * Mathf.Rad2Deg;
        float radY = angle.y * Mathf.Rad2Deg;
        float radZ = angle.z * Mathf.Rad2Deg;

        // Calculate the quaternion components
        Quaternion qX = Quaternion.AngleAxis(angle.x, Vector3.right);   // Rotation around X axis
        Quaternion qY = Quaternion.AngleAxis(angle.y, Vector3.up);      // Rotation around Y axis
        Quaternion qZ = Quaternion.AngleAxis(angle.z, Vector3.forward); // Rotation around Z axis

        if (order == "xyz")
            return qZ * qY * qX;
        else
            //TODO: order zyx
            return Quaternion.identity;
    }
    #endregion
    private static Vector3 _cross(Vector3 a, Vector3 b)
    {
        float x = a.y * b.z - a.z * b.y;
        float y = a.z * b.x - a.x * b.z;
        float z = a.x * b.y - a.y * b.x;

        return new Vector3 (x, y, z);
    }
    private static int index(int bone, int vector, int component, int subcomponent, TensorShape shape)
    {
        return bone * shape.height * shape.width * shape.channels +
               vector * shape.width * shape.channels +
               component * shape.channels +
               subcomponent;
    }
}
