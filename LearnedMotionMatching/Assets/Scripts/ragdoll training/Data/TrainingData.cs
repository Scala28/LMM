using System.Collections;
using System.Collections.Generic;
using System.IO.Abstractions;
using UnityEngine;
using System.IO;

[CreateAssetMenu(fileName ="newTrainingData", menuName ="Data/ML Agents/Training Data")]
public class TrainingData : ScriptableObject
{
    public string filename = "";
    public string writeFileToPath =  "../config";
    public string loadFileFromPath = "../config";

    [Header("ArticulationBody settings")]
    public int solverIterations = 32;
    public List<ConfigManager.MusclePower> MusclePowers;
    public float[] boneToStiffness = new float[23];
    public float forceLimit;
    public float damping;
    public bool dampingScalesWithStiffness;
    public bool selfCollision;

    [Header("Physical character settings")]
    public int fixedUpdateFrequency = 256;
    public bool resolveSimReferenceFrameWithSimRotation = false;
    public bool networkControlsAllJoints = false;
    public bool setRotsDirectly = false;
    public bool fullRangeEulerOutputs = false;
    public bool setDriveTargetVelocities = true;
    public bool applyActionOverMultipleTimeSteps = false;

    [Header("Kinematic character settings")]
    public float max_wandering_radius = 50f;
    public float prob_to_change_inputs = 0.005f;
    public float input_generator_halflife = .5f;
    public bool canRun = true;

    [Header("Training settings")]
    public bool addOrientationDataToState = false;

    [Header("Inference settings")]
    public bool clampKinCharToSim = true;
    public float clampingMaxDistance = 0.666f;

    [Header("Training hyperparameters")]
    public int MAX_EPISODE_LENGTH_SECONDS = 20;
    public int EVALUATE_EVERY_K = 2;
    public float ACTION_STIFFNESS_HYPERPARAM = .2f;
    public float EPISODE_END_REWARD = -.5f;
    public int N_FRAMES_TO_NOT_COUNT_REWARD_AFTER_TELEPORT = 2;

    [ContextMenu("Write out config file to current config name ")]
    public void writeCurrentConfig()
    {
        string folderpath = Application.dataPath + @"/" + writeFileToPath;
        Debug.Log($"folderpath: {folderpath}");
        string filepath = folderpath + @"/" + filename;
        // Check if the folder exists
        if (!Directory.Exists(folderpath))
        {
            // Create the folder
            Directory.CreateDirectory(folderpath);
            Debug.Log("Folder created.");
        }
        string json = JsonUtility.ToJson(this);
        File.WriteAllText(filepath, json);
    }

    [ContextMenu("Load config file")]
    public void loadConfig()
    {
        string filepath = Application.dataPath + @"/" + loadFileFromPath + @"/" + filename;
        string json = File.ReadAllText(filepath);
        JsonUtility.FromJsonOverwrite(json, this);
    }
}
public enum ActionRotationType
{
    Euler,
    AxisAngle,
    SixD,
    Exp,
}
