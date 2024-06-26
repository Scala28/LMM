using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
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
        Vector4 prev_rootrot = currentPose.rootRotation;

        Vector3 prev_rootvel = currentPose.rootVelocity;
        Vector3 prev_rootang = currentPose.rootAngularVelocity;


        Vector3 root_pos = prev_rootpos + quat_mul_vec(prev_rootrot, new Vector3(root_vel[0, 0, 0, 0],
                                                                                root_vel[1, 0, 0, 0],
                                                                                root_vel[2, 0, 0, 0])) * dt;

        Vector4 root_rot = quat_mul(prev_rootrot, quat_from_scaled_axis_angle(
            quat_mul_vec(prev_rootrot, new Vector3(root_ang[0, 0, 0, 0],
                                                    root_ang[1, 0, 0, 0],
                                                    root_ang[2, 0, 0, 0])) * dt));

        Tensor positions = new Tensor(nbones, 3, 1, 1);
        positions[0, 0, 0, 0] = root_pos.x;
        positions[0, 1, 0, 0] = root_pos.y;
        positions[0, 2, 0, 0] = root_pos.z;
        for (int i = 1; i < nbones; i++)
        {
            positions[i, 0, 0, 0] = pos[i-1, 0, 0, 0];
            positions[i, 1, 0, 0] = pos[i-1, 1, 0, 0];
            positions[i, 2, 0, 0] = pos[1-1, 2, 0, 0];
        }

        Tensor rotations = new Tensor(nbones, 4, 1, 1);
        rotations[0, 0, 0, 0] = root_rot.x;
        rotations[0, 1, 0, 0] = root_rot.y;
        rotations[0, 2, 0, 0] = root_rot.z;
        rotations[0, 3, 0, 0] = root_rot.w;
        for (int i = 1; i < nbones; i++)
        {

            rotations[i, 0, 0, 0] = quat[i-1, 0, 0, 0];
            rotations[i, 1, 0, 0] = quat[i-1, 1, 0, 0];
            rotations[i, 2, 0, 0] = quat[i-1, 2, 0, 0];
            rotations[i, 3, 0, 0] = quat[i-1, 3, 0, 0];
        }

        // Construct pose for next frame
        Pose pose = new Pose(pos, quat, vel, ang,
            root_pos,
            root_rot,
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
            Vector3 c2Vec = crossProduct(x0, x1).normalized;

            // Calculate c1
            Vector3 c1Vec = crossProduct(c2Vec, x0).normalized;

            // c0 is x0
            Vector3 c0Vec = x0;

            // Assign results back to tensors
            c0[i, 0, 0, 0] = c0Vec.x; c0[i, 1, 0, 0] = c0Vec.y; c0[i, 2, 0, 0] = c0Vec.z;
            c1[i, 0, 0, 0] = c1Vec.x; c1[i, 1, 0, 0] = c1Vec.y; c1[i, 2, 0, 0] = c1Vec.z;
            c2[i, 0, 0, 0] = c2Vec.x; c2[i, 1, 0, 0] = c2Vec.y; c2[i, 2, 0, 0] = c2Vec.z;
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
                t = 1 + xfm.m00 - xfm.m11 - xfm.m22;
                q = new Vector4(xfm.m21 - xfm.m12,
                    t,
                    xfm.m10 + xfm.m01,
                    xfm.m02 + xfm.m20);
            }
            else
            {
                t = 1 - xfm.m00 + xfm.m11 - xfm.m22;
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
        float norm = Mathf.Sqrt(Vector4.Dot(q, q));
        return q / (norm + eps);
    }
    private static Vector3 quat_mul_vec(Vector4 q, Vector3 vec)
    {
        Vector3 q_vector = new Vector3(q.x, q.y, q.z);
        float q_scalar = q.w;

        return vec + 2f * q_scalar * crossProduct(q_vector, vec) +
            crossProduct(q_vector, 2f * crossProduct(q_vector, vec));
    }
    private static Vector4 quat_mul(Vector4 a, Vector4 b)
    {
        Quaternion qa = new Quaternion(a.x, a.y, a.z, a.w);
        Quaternion qb = new Quaternion(b.x, b.y, b.z, b.w);

        Quaternion result = qa * qb;

        return new Vector4(result.x, result.y, result.z, result.w);
    }
    private static Vector4 quat_from_scaled_axis_angle(Vector3 ang, float eps = 1e-5f)
    {
        float halfAngle = Mathf.Sqrt(Vector3.Dot(ang, ang));
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
        Vector3 q_vec = ang * s;

        return new Vector4(c, q_vec.x, q_vec.y, q_vec.z);
    }
    #endregion
    private static Vector3 crossProduct(Vector3 a, Vector3 b)
    {
        return Vector3.Cross(a, b);
    }
    private static int index(int bone, int vector, int component, int subcomponent, TensorShape shape)
    {
        return bone * shape.height * shape.width * shape.channels +
               vector * shape.width * shape.channels +
               component * shape.channels +
               subcomponent;
    }
}
