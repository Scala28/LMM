using System;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using static MotionMatcher.character;
using Unity.VisualScripting;

public struct CharInfo
{
    public Transform[] BoneToTransform;
    public GameObject[] BoneToCollider;
    public MotionMatcher MMScript;
    public GameObject charObj;
    public Transform transform;
    public ArticulationBody root;
    public Vector3 cm;
    public Vector3 cmVel;
    public Vector3[] BoneWorldPos;
    public Vector3[][] BoneSurfacePts;
    public Vector3[][] BoneSurfacePtsWorld;
    public Vector3[][] BoneSurfaceVels;
    public ArticulationBody[] BoneToArt;
    public float[] boneState;
    public int nbodies;

    public CharInfo(int nbodies, int numStateBones) : this()
    {
        BoneSurfacePts = new Vector3[nbodies][];
        BoneSurfacePtsWorld = new Vector3[nbodies][];
        BoneSurfaceVels = new Vector3[nbodies][];
        BoneWorldPos = new Vector3[numStateBones];
        BoneToCollider = new GameObject[nbodies];
        // { LeftToe, RightToe, Spine, Head, LeftForeArm, RightForeArm }
        // we compute positions and velocities then concatenate these
        boneState = new float[36];
        this.nbodies = nbodies;
    }
}

public class MLRagdoll : Agent
{

}