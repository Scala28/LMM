using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

[CreateAssetMenu(fileName ="newTrainingData", menuName ="Data/ML Agents/Training Data")]
public class TrainingData : ScriptableObject
{
    [Header("ArticulationBody settings")]
    public List<ConfigManager.MusclePower> MusclePowers;
    public float[] boneToStiffness = new float[23];
    public float forceLimit;
    public float damping;
    public bool dampingScalesWithStiffness;
    public bool selfCollision;
}
public enum ActionRotationType
{
    Euler,
    AxisAngle,
    SixD,
    Exp,
}
