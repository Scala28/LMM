using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;
using UnityEditor;
using UnityEngine.XR;
using UnityEditor.Experimental.GraphView;
using System.ComponentModel;

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
    private Model stepper_parameters;

    private Tensor currentXZ;

    #endregion

    #region Animation
    private float frame_time;
    private const float dt = 1 / 60f;

    private Pose currentPose;

    private List<Transform> bones = new List<Transform>();

    private List<Vector3> tPose_offsets = new List<Vector3>();

    private int frameNumber;
    #endregion

    // Start is called before the first frame update
    void Start()
    {
        root_rb = GetComponent<Rigidbody>();
        initialize_models();

        decompressor_parameters = DataManager.Load_net_fromParameters("Assets/NNModels/decompressor.bin");
        stepper_parameters = DataManager.Load_net_fromParameters("Assets/NNModels/stepper.bin");

        initialize_skeleton(this.transform);
        initialize_pose();

        frameNumber = 200;
        frame_time = 0f;

        currentXZ = GetFrameInputTensor(frameNumber);
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
    private Tensor add_model_parameters(Tensor _out, Model param)
    {
        Tensor ris = new Tensor(_out.shape);
        for(int i=0; i<param.Mean_out.Length; i++)
        {
            ris[i] = _out[i] * param.Std_out[i] + param.Mean_out[i];
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
        frame_time += Time.deltaTime;
        if (frame_time < dt)
            return;

        Believe();
    }
    private void Believe()
    {
        stepper_inference.Execute(currentXZ);
        using (Tensor stepper_out = add_model_parameters(stepper_inference.PeekOutput(), stepper_parameters))
        {
            delta(stepper_out);
            currentXZ = add(currentXZ, stepper_out);
            decompressor_inference.Execute(currentXZ);
            using (Tensor decompressor_out = add_model_parameters(decompressor_inference.PeekOutput(), decompressor_parameters))
            {
                currentPose = DataParser.ParseDecompressorOutput(decompressor_out, currentPose, bones.Count, this.transform);
                display_frame_pose();
            }
        }
    }
    private void delta(Tensor stepper_out)
    { 
        for (int i = 0; i < stepper_out.length; i++)
        {
            stepper_out[i] *= dt;
        }

    }
    private Tensor add(Tensor a, Tensor b)
    {
        Tensor ris = new Tensor(a.shape);
        for(int i=0; i<a.length; i++)
        {
            ris[i] = a[i] + b[i];
        }
        return ris;
    }
    private void display_frame_pose()
    {
        for (int i=1; i<bones.Count; i++)
        {
            Transform joint = bones[i];
            JointMotionData jdata = currentPose.joints[i-1];

            joint.localRotation = Quaternion.Euler(0f, 0f, -jdata.localRotation.z) *
                Quaternion.Euler(0f, -jdata.localRotation.y, 0f) * Quaternion.Euler(jdata.localRotation.x, 0f, 0f);
        }
    }

}
