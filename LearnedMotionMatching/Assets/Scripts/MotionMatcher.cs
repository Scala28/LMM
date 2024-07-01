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

    private Model decompressor_parameters;

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
        decompressor_parameters = DataManager.Load_net_fromParameters("Assets/NNModels/decompressor.bin");

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

                Tensor out_ = add_decompressor_parameters(decompressor_out);

                currentPose = DataParser.ParseDecompressorOutput(out_, currentPose, bones.Count);
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
    private Tensor add_decompressor_parameters(Tensor decompressor_out)
    {
        Tensor ris = new Tensor(decompressor_out.shape);
        for(int i=0; i<decompressor_parameters.Mean_out.Length; i++)
        {
            ris[i] = decompressor_out[i] * decompressor_parameters.Std_out[i] + decompressor_parameters.Mean_out[i];
        }

        return ris;
    }
    private Tensor GetFrameInputTensor(int frame_index)
    {
        (int nframes1, int nfeatures, float[] features) = DataManager.Load_database_fromResources("features");
        (int nframes2, int nlatent, float[] latent) = DataManager.Load_database_fromResources("latent");

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
    }

}
