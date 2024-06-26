using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;
using UnityEditor;

public class MotionMatcher : MonoBehaviour
{

    [SerializeField]
    private NNModel stepper;

    [SerializeField]
    private NNModel decompressor;

    private IWorker stepper_inference;
    private IWorker decompressor_inference;

    private const float dt = 1 / 60f;

    private Pose currentPose;

    // Start is called before the first frame update
    void Start()
    {
        Initialize_models();
        using (Tensor stepper_input = GetFirstFrameInputTensor())
        {
            if (stepper_input == null)
                return;

            Debug.Log("Input shape: " + stepper_input.shape);

            stepper_inference.Execute(stepper_input);

            using (Tensor stepper_out = stepper_inference.PeekOutput())
            {
                Debug.Log("Stepper output shape: " + stepper_out.shape);

                decompressor_inference.Execute(stepper_out);

                using (Tensor decompressor_out = decompressor_inference.PeekOutput())
                {
                    Debug.Log("Decompressor output shape: " + decompressor_out.shape);
                    ParseDecompressorOutput(decompressor_out);
                }
            }
        }
    }

    private void Initialize_models() {

        stepper_inference = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled,
            ModelLoader.Load(stepper));
        decompressor_inference = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled,
            ModelLoader.Load(decompressor));
    }

    private void OnDestroy()
    {
        stepper_inference.Dispose();
        decompressor_inference.Dispose();
    }
    private Tensor GetFirstFrameInputTensor()
    {
        (int nframes1, int nfeatures, float[] features) = DataManager.LoadBin_fromResources("features");
        (int nframes2, int nlatent, float[] latent) = DataManager.LoadBin_fromResources("latent");

        if (features == null || latent == null)
            return null;

        if (nframes1 != nframes2)
        {
            Debug.LogError("Mismatch in the number of frames between the two datasets.");
            return null;
        }

        float[] x_frame1 = new float[nfeatures];
        float[] z_frame1 = new float[nlatent];
        Array.Copy(features, 0, x_frame1, 0, nfeatures);
        Array.Copy(latent, 0, z_frame1, 0, nlatent);

        // Concatenate the first frame from both datasets
        float[] xz_frame1 = new float[nfeatures + nlatent];
        Array.Copy(x_frame1, 0, xz_frame1, 0, x_frame1.Length);
        Array.Copy(z_frame1, 0, xz_frame1, x_frame1.Length, z_frame1.Length);

        // Convert data to Tensor and return
        return new Tensor(new int[] { 1, 1, 1, nfeatures + nlatent }, xz_frame1);
    }

    // Update is called once per frame
    void Update()
    {

    }

    private Pose ParseDecompressorOutput(Tensor decompressor_out)
    {
        int nbones = 22;
        Tensor pos = SliceAndReshape(decompressor_out, 0 * (nbones - 1), 3 * (nbones - 1), new TensorShape(nbones - 1, 3, 1, 1));
        Tensor txy = SliceAndReshape(decompressor_out, 3 * (nbones - 1), 9 * (nbones - 1), new TensorShape(nbones - 1, 3, 2, 1));
        Tensor vel = SliceAndReshape(decompressor_out, 9 * (nbones - 1), 12 * (nbones - 1), new TensorShape(nbones - 1, 3, 1, 1));
        Tensor ang = SliceAndReshape(decompressor_out, 12 * (nbones - 1), 15 * (nbones - 1), new TensorShape(nbones - 1, 3, 1, 1));
        Tensor root_vel = SliceAndReshape(decompressor_out, 15 * (nbones - 1), 15 * (nbones - 1) + 3, new TensorShape(3, 1, 1, 1));
        Tensor root_ang = SliceAndReshape(decompressor_out, 15 * (nbones - 1) + 3, 15 * (nbones - 1) + 6, new TensorShape(3, 1, 1, 1));

        //Convert to quat: (nbones-1, 4, 1, 1)
        Tensor quat;
        quat_from_xfm_xy(txy, out quat);

        //Integrate root displacement
        Vector3 prev_rootpos = currentPose.rootPosition;
        Vector4 prev_rootrot = currentPose.rootRotation;

        Vector3 prev_rootvel = currentPose.rootVelocity;
        Vector3 prev_rootang = currentPose.rootAngularVelocity;


        Vector3 root_pos = prev_rootpos + quat_mul_vec(prev_rootrot, prev_rootvel) * dt;

        Vector4 root_rot = quat_mul(prev_rootrot, quat_from_scaled_axis_angle(
            quat_mul_vec(prev_rootrot, prev_rootang) * dt));

        Tensor positions = new Tensor(nbones, 3, 1, 1);
        positions[0, 0, 0, 0] = root_pos.x;
        positions[0, 1, 0, 0] = root_pos.y;
        positions[0, 2, 0, 0] = root_pos.z;
        for (int i=1; i < nbones; i++)
        {
            positions[i, 0, 0, 0] = pos[i, 0, 0, 0];
            positions[i, 1, 0, 0] = pos[i, 1, 0, 0];
            positions[i, 2, 0, 0] = pos[i, 2, 0, 0];
        }

        Tensor rotations = new Tensor(nbones, 4, 1, 1);
        rotations[0, 0, 0, 0] = root_rot.x;
        rotations[0, 1, 0, 0] = root_rot.y;
        rotations[0, 2, 0, 0] = root_rot.z;
        rotations[0, 3, 0, 0] = root_rot.w;
        for(int i=1; i < nbones; i++)
        {

            rotations[i, 0, 0, 0] = quat[i, 0, 0, 0];
            rotations[i, 1, 0, 0] = quat[i, 1, 0, 0];
            rotations[i, 2, 0, 0] = quat[i, 2, 0, 0];
            rotations[i, 3, 0, 0] = quat[i, 3, 0, 0];
        }

        // Construct pose for next frame
        Pose pose = new Pose(pos, quat, vel, ang, 
            root_pos, 
            root_rot,
            new Vector3(root_vel[0, 0, 0, 0], root_vel[1, 0, 0, 0], root_vel[2, 0, 0, 0]),  //root vel
            new Vector3(root_ang[0, 0, 0, 0], root_ang[1, 0, 0, 0], root_ang[2, 0, 0, 0]));  // root ang
        return pose;
    }
    private Tensor SliceAndReshape(Tensor input, int sliceStart, int sliceEnd, TensorShape newShape)
    {
        Tensor sliced = new Tensor(newShape);

        int channels = sliceEnd - sliceStart;

        for (int i = 0; i < newShape.batch; i++)
        {
            for (int h = 0; h < newShape.height; h++)
            {
                for (int w = 0; w < newShape.width; w++)
                {
                    for (int c = 0; c < channels; c++)
                        sliced[i, h, w, c] = input[i, h, w, c + sliceStart];
                }
            }
        }
        return sliced;
    }

    #region QuatFunctions
    private void quat_from_xfm_xy(Tensor x, out  Tensor result)
    {
        TensorShape shape = x.shape;
        int nbones = shape.batch;
        int vectorsPerBone = shape.channels;  // 3
        int componentsPerVector = shape.height;  // 2
        int subcomponentsPerComponent = shape.width;  //1

        Tensor c0 = new Tensor(nbones, 3, 1, 1);
        Tensor c1 = new Tensor(nbones, 3, 1, 1);
        Tensor c2 = new Tensor(nbones, 3, 1, 1);

        for (int i = 0; i < nbones; i++)
        {
            // Extract vectors from the tensor
            Vector3 x0 = new Vector3(
                x[index(i, 0, 0, 0, shape)],
                x[index(i, 1, 0, 0, shape)],
                x[index(i, 2, 0, 0, shape)]
            );

            Vector3 x1 = new Vector3(
                x[index(i, 0, 1, 0, shape)],
                x[index(i, 1, 1, 0, shape)],
                x[index(i, 2, 1, 0, shape)]
            );


            // Calculate c2
            Vector3 c2Vec = crossProduct(x0, x1).normalized;

            // Calculate c1
            Vector3 c1Vec = crossProduct(c2Vec, x0).normalized;

            // c0 is x0
            Vector3 c0Vec = x0;

            // Assign results back to tensors
            c0[index(i, 0, 0, 0, c0.shape)] = c0Vec.x; c0[index(i, 1, 0, 0, c0.shape)] = c0Vec.y; c0[index(i, 2, 0, 0, c0.shape)] = c0Vec.z;
            c1[index(i, 0, 0, 0, c1.shape)] = c1Vec.x; c1[index(i, 0, 1, 0, c1.shape)] = c1Vec.y; c1[index(i, 0, 2, 0, c1.shape)] = c1Vec.z;
            c2[index(i, 0, 0, 0, c2.shape)] = c2Vec.x; c2[index(i, 0, 1, 0, c2.shape)] = c2Vec.y; c2[index(i, 0, 2, 0, c2.shape)] = c2Vec.z;
        }

        // Concatenate c0, c1, c2 along the third dimension
        Tensor xfm = new Tensor(nbones, 3, 3, 1);
        for(int i=0; i<nbones; i++)
        {
            for(int j=0; j<3; j++)
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

        for (int i=0; i<nbones; i++)
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

        result = ris;
    }

    //Matrix 3x3
    private Vector4 quat_from_xform(Matrix4x4 xfm)
    {
        Vector4 q;
        float t;

        if(xfm.m22 < 0)
        {
            if(xfm.m00 > xfm.m11)
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
                q = new Vector4(t, xfm.m21 - xfm.m12, xfm.m02 - xfm.m20, xfm.m10 - xfm.m01);
            }
        }

        return quat_normalize(q);
    }
    
    private Vector4 quat_normalize(Vector4 q, float eps=1e-8f) {
        float norm = Mathf.Sqrt(Vector4.Dot(q, q));
        return q / (norm + eps);
    }
    private Vector3 quat_mul_vec(Vector4 q, Vector3 vec)
    {
        Vector3 q_vector = new Vector3(q.x, q.y, q.z);
        float q_scalar = q.w;

        return vec + 2f * q_scalar * crossProduct(q_vector, vec) +
            crossProduct(q_vector, 2f * crossProduct(q_vector, vec));
    }
    private Vector4 quat_mul(Vector4 a, Vector4 b)
    {
        Quaternion qa = new Quaternion(a.x, a.y, a.z, a.w);
        Quaternion qb = new Quaternion(b.x, b.y, b.z, b.w);

        Quaternion result = qa * qb;

        return new Vector4(result.x, result.y, result.z, result.w);
    }
    private Vector4 quat_from_scaled_axis_angle(Vector3 ang, float eps=1e-5f)
    {
        float halfAngle = Mathf.Sqrt(Vector3.Dot(ang, ang));
        float c, s;
        if(halfAngle < eps)
        {
            c = 1f;
            s = 1f;
        }
        else { 
            c = Mathf.Cos(halfAngle);
            s = Mathf.Sin(halfAngle) / halfAngle;
        }
        Vector3 q_vec = ang * s;

        return new Vector4(c, q_vec.x, q_vec.y, q_vec.z);
    }
    #endregion
    private Vector3 crossProduct(Vector3 a, Vector3 b)
    {
        return Vector3.Cross(a, b);
    }
    private int index(int bone, int vector, int component, int subcomponent, TensorShape shape)
    {
        return bone * shape.channels * shape.height * shape.width +
               vector * shape.height * shape.width +
               component * shape.width +
               subcomponent;
    }


}
