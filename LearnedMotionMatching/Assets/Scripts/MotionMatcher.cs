using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;
using UnityEditor;
using UnityEngine.XR;

public class MotionMatcher : MonoBehaviour
{
    private Rigidbody root_rb;

    #region NN Inference

    [SerializeField]
    private NNModel stepper;

    [SerializeField]
    private NNModel decompressor;

    private IWorker stepper_inference;
    private IWorker decompressor_inference;

    #endregion

    private const float dt = 1 / 60f;

    private Pose currentPose;

    private List<Transform> bones = new List<Transform>();

    // Start is called before the first frame update
    void Start()
    {
        root_rb = GetComponent<Rigidbody>();
        initialize_models();

        initialize_skeleton(this.transform);
        initialize_pose();

        Debug.Log(bones.Count);
        
        using (Tensor input = GetFrameInputTensor(1))
        {
            if (input == null)
                return;

            decompressor_inference.Execute(input);

            using(Tensor decompressor_out = decompressor_inference.PeekOutput())
            {
                Debug.Log(decompressor_out.shape);
                currentPose = DataParser.ParseDecompressorOutput(decompressor_out, currentPose, bones.Count);

                display_frame_pose();
            }
        }
    }

    private void initialize_models() {

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
    private void initialize_skeleton(Transform bone)
    {
        bones.Add(bone);
        foreach(Transform child in bone)
            if(child.CompareTag("joint"))
                initialize_skeleton(child);
    }
    private void initialize_pose()
    {
        Vector3 vel = root_rb.velocity;
        Vector3 ang = root_rb.angularVelocity;
        currentPose = new Pose(this.transform, vel, ang);
    }
    private Tensor GetFrameInputTensor(int frame_index)
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
        Array.Copy(features, frame_index * nfeatures, x_frame1, 0, nfeatures);
        Array.Copy(latent, frame_index * nlatent, z_frame1, 0, nlatent);

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
        // TODO: Play anim,
        // TODO: build input tensor from currentPose - feed stepper, decompressor
        // TODO: parse next currentPose from decompressor_out
    }

    private void display_frame_pose()
    {
        transform.position = currentPose.rootPosition; ;
        Vector4 q = currentPose.rootRotation;
        transform.rotation = new Quaternion(q.x, q.y, q.z, q.w);

        for(int i=1; i<bones.Count; i++)
        {
            JointMotionData jdata = currentPose.joints[i - 1];
            Transform joint = bones[i];

            joint.localPosition = jdata.localPosition;
            joint.localRotation = jdata.localRotation;
        }
    }

}
