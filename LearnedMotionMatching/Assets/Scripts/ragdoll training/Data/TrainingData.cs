using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu(fileName ="newTrainingData", menuName ="Data/ML Agents/Training Data")]
public class TrainingData : ScriptableObject
{
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

    [Header("Training hyperparameters")]
    public int MAX_EPISODE_LENGTH_SECONDS = 20;
    public int EVALUATE_EVERY_K = 2;
    public float ACTION_STIFFNESS_HYPERPARAM = .2f;
}
public enum ActionRotationType
{
    Euler,
    AxisAngle,
    SixD,
    Exp,
}
