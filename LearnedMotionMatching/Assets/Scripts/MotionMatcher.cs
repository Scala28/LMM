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

    private List<Vector3> tPose_offsets = new List<Vector3>();

    // Start is called before the first frame update
    void Start()
    {
        root_rb = GetComponent<Rigidbody>();
        initialize_models();

        initialize_skeleton(this.transform);
        initialize_pose();

        Debug.Log("Bones " + bones.Count);
        
        using (Tensor input = GetFrameInputTensor(562))
        {
            if (input == null)
                return;

            decompressor_inference.Execute(input);

            using(Tensor decompressor_out = decompressor_inference.PeekOutput())
            {
                Debug.Log("decompressor_out:");
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
        tPose_offsets.Add(bone != this.transform ?
            bone.localPosition : Vector3.zero);
        bones.Add(bone.transform);
        foreach (Transform child in bone)
            if (child.CompareTag("joint"))
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
        //transform.position = currentPose.rootPosition;
        // transform.rotation = currentPose.rootRotation;

        for (int i = 2; i < bones.Count; i++)
        {
            JointMotionData jdata = currentPose.joints[i - 1];
            Transform joint = bones[i];

            //joint.localPosition = jdata.localPosition * 100f;
            joint.localRotation = jdata.localRotation * joint.localRotation;
        }
        //List<Quaternion> qs = new List<Quaternion>();
        //Quaternion q2 = Quaternion.Euler(171.571228f, 5.297809f, 179.759064f);
        //qs.Add(q2);
        //Quaternion q3 = Quaternion.Euler(0.736388f, 0.525740f, -24.190214f);
        //qs.Add(q3);
        //Quaternion q4 = Quaternion.Euler(-0.492687f, -0.012149f, 93.471802f);
        //qs.Add(q4);
        //Quaternion q5 = Quaternion.Euler(-0.000002f, 0.003090f, 21.454554f);
        //qs.Add(q5);

        //for(int i=2; i<6; i++)
        //{
        //    Transform joint = bones[i];

        //    joint.localRotation = qs[i - 2] * joint.localRotation;
        //}
    }

}
