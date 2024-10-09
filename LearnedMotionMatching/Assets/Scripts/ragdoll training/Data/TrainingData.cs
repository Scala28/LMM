using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

[CreateAssetMenu(fileName ="newTrainingData", menuName ="Data/ML Agents/Training Data")]
public class TrainingData : ScriptableObject
{
    
}
public enum ActionRotationType
{
    Euler,
    AxisAngle,
    SixD,
    Exp,
}
