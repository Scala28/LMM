using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;

public class MotionMatcher : MonoBehaviour
{

    [SerializeField]
    private NNModel stepper;

    [SerializeField]
    private NNModel decompressor;

    private IWorker stepper_inference;
    private IWorker decompressor_inference;

    // Start is called before the first frame update
    void Start()
    {
        Initialize_models();
        using (Tensor stepper_input = GetFirstFrameInputTensor())
        {
            if (stepper_input == null)
                return;

            stepper_inference.Execute(stepper_input);
        }
    }

    private void Initialize_models() {

        stepper_inference = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled,
            ModelLoader.Load(stepper));
        decompressor_inference = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled,
            ModelLoader.Load(decompressor));
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
        return new Tensor(1, nfeatures + nlatent, xz_frame1);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
