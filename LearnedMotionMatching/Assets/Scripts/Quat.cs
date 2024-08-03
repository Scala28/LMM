using System.Collections;
using System.Collections.Generic;
using System.ComponentModel.Design;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.UIElements;

public static class Quat
{
    #region QuatFunctions
    public static Tensor quat_from_xfm_xy(Tensor x)
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
            Vector3 c2Vec = vec_normalize(_cross(x0, x1));

            // Calculate c1
            Vector3 c1Vec = vec_normalize(_cross(c2Vec, x0));

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

    public static Vector4 quat_normalize(Vector4 q, float eps = 1e-8f)
    {
        float norm = Mathf.Sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        return q / (norm + eps);
    }
    public static Vector4 quat_inv(Vector4 q)
    {
        return new Vector4(-q.x, q.y, q.z, q.w);
    }
    public static Vector4 quat_abs(Vector4 q)
    {
        if (q.x < 0.0f)
            return -q;
        return q;
    }
    public static Vector4 quat_exp(Vector3 x, float eps = 1e-8f)
    {
        float halfAngle = Mathf.Sqrt(x.x * x.x + x.y * x.y + x.z * x.z);
        if (halfAngle < eps)
        {
            return quat_normalize(new Vector4(1.0f, x.x, x.y, x.z));
        }
        else
        {
            float c = Mathf.Cos(halfAngle);
            float s = Mathf.Sin(halfAngle) / halfAngle;

            return new Vector4(c, s * x.x, s * x.y, s * x.z);
        }
    }
    public static Vector3 quat_log(Vector4 q, float eps = 1e-8f)
    {
        float length = Mathf.Sqrt(q.x * q.x + q.y * q.y + q.z * q.z);

        if (length < eps)
        {
            return new Vector3(q.x, q.y, q.z);
        }
        else
        {
            float halfangle = Mathf.Acos(clampf(q.w, -1.0f, 1.0f));
            return halfangle * (new Vector3(q.x, q.y, q.z) / length);
        }
    }
    public static Vector3 quat_mul_vec(Vector4 q, Vector3 vec)
    {
        Vector3 q_vector = new Vector3(q.y, q.z, q.w);
        float q_scalar = q.x;

        return vec + 2f * q_scalar * _cross(q_vector, vec) +
            _cross(q_vector, 2f * _cross(q_vector, vec));
    }
    public static Vector3 quat_inv_mul_vec(Vector4 q, Vector3 vec)
    {
        return quat_mul_vec(quat_inv(q), vec);
    }
    public static Vector4 quat_mul(Vector4 a, Vector4 b)
    {
        float w = a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w;
        float x = a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z;
        float y = a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y;
        float z = a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x;

        return new Vector4(w, x, y, z);
    }
    public static Vector4 quat_inv_mul(Vector4 a, Vector4 b)
    {
        return quat_mul(quat_inv(a), b);
    }
    public static Vector4 quat_mul_inv(Vector4 a, Vector4 b)
    {
        return quat_mul(a, quat_inv(b));
    }
    public static Vector4 quat_from_scaled_angle_axis(Vector3 ang, float eps = 1e-8f)
    {
        Vector3 x = ang / 2f;
        return quat_exp(x, eps);
    }
    public static Vector3 quat_to_scaled_angle_axis(Vector4 q, float eps = 1e-8f)
    {
        return 2.0f * quat_log(q, eps);
    }
    public static Vector4 quat_from_angle_axis(float angle, Vector3 axis)
    {
        float c = Mathf.Cos(angle / 2.0f);
        float s = Mathf.Sin(angle / 2.0f);
        return new Vector4(c, s * axis.x, s * axis.y, s * axis.z);
    }
    private static Tensor quat_toEuler(Tensor quat, string order = "xyz")
    {
        Tensor ris = new Tensor(quat.batch, 3, 1, 1);

        for (int i = 0; i < quat.batch; i++)
        {
            Vector4 q = new Vector4(quat[index(i, 0, 0, 0, quat.shape)],
                                    quat[i, 1, 0, 0],
                                    quat[i, 2, 0, 0],
                                    quat[i, 3, 0, 0]);
            Vector3 angle = convert_ToEuler(q, order);

            //from radiants to degrees
            angle = angle * Mathf.Rad2Deg;

            ris[index(i, 0, 0, 0, ris.shape)] = angle.x;
            ris[index(i, 1, 0, 0, ris.shape)] = angle.y;
            ris[index(i, 2, 0, 0, ris.shape)] = angle.z;
        }
        return ris;
    }
    public static Vector3 convert_ToEuler(Vector4 q, string order = "xyz")
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
    #endregion
    public static Vector3 vec_normalize(Vector3 vec, float eps = 1e-8f)
    {
        float norm = Mathf.Sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
        return vec / (norm + eps);
    }
    public static Vector3 _cross(Vector3 a, Vector3 b)
    {
        float x = a.y * b.z - a.z * b.y;
        float y = a.z * b.x - a.x * b.z;
        float z = a.x * b.y - a.y * b.x;

        return new Vector3(x, y, z);
    }
    private static float clampf(float x, float min, float max)
    {
        return x > max ? max : x < min ? min : x;
    }
    private static int index(int bone, int vector, int component, int subcomponent, TensorShape shape)
    {
        return bone * shape.height * shape.width * shape.channels +
               vector * shape.width * shape.channels +
               component * shape.channels +
               subcomponent;
    }
}
