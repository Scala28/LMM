using UnityEngine;
using UnityEditor;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Linq;

public class AnimationDataExporter : EditorWindow
{
    GameObject modelPrefab;
    AnimationClip animationClip;
    float sampleRate = 60f;

    [MenuItem("Tools/Export Animation Data")]
    static void Init()
    {
        AnimationDataExporter window = (AnimationDataExporter)EditorWindow.GetWindow(typeof(AnimationDataExporter));
        window.titleContent = new GUIContent("Export Animation Data");
        window.Show();
    }

    void OnGUI()
    {
        GUILayout.Label("Export FBX Animation Data (Tagged 'joint')", EditorStyles.boldLabel);

        modelPrefab = (GameObject)EditorGUILayout.ObjectField("Model Prefab", modelPrefab, typeof(GameObject), false);
        animationClip = (AnimationClip)EditorGUILayout.ObjectField("Animation Clip", animationClip, typeof(AnimationClip), false);
        sampleRate = EditorGUILayout.FloatField("Sample Rate (fps)", sampleRate);

        if (GUILayout.Button("Export to CSV"))
        {
            if (modelPrefab == null || animationClip == null)
            {
                EditorUtility.DisplayDialog("Error", "Assign both model prefab and animation clip.", "OK");
                return;
            }

            ExportData();
        }
    }

    void ExportData()
    {
        GameObject instance = Instantiate(modelPrefab);
        instance.hideFlags = HideFlags.HideAndDontSave;

        // Only keep transforms with tag "joint" (lowercase)
        Transform[] allChildren = instance.GetComponentsInChildren<Transform>();
        List<Transform> jointTransforms = new List<Transform>();
        foreach (Transform t in allChildren)
        {
            // Match lowercase tag manually
            if (t.tag.ToLower() == "joint")
                jointTransforms.Add(t);
        }

        if (jointTransforms.Count == 0)
        {
            EditorUtility.DisplayDialog("No Joints Found", "No child objects with tag 'joint' were found.", "OK");
            DestroyImmediate(instance);
            return;
        }

        string filePath = EditorUtility.SaveFilePanel("Save Animation Data", "", animationClip.name + "_joint_data.csv", "csv");
        if (string.IsNullOrEmpty(filePath)) return;

        int frameCount = Mathf.CeilToInt(sampleRate * animationClip.length);
        StringBuilder csv = new StringBuilder();

        csv.AppendLine("HIERARCHY");
        add_joint(ref csv, jointTransforms[0], jointTransforms, "");

        // Header
        csv.AppendLine("HEADER");
        csv.Append("Time");
        foreach (Transform bone in jointTransforms)
        {
            string name = bone.name;
            csv.Append($",\"{name}_PosX\",\"{name}_PosY\",\"{name}_PosZ\"");
            csv.Append($",\"{name}_RotX\",\"{name}_RotY\",\"{name}_RotZ\",\"{name}_RotW\"");
        }
        csv.AppendLine();
        csv.AppendLine("FrameCount: " + frameCount);

        // Frame data
        float time_elapsed = 0f;
        int i = 0;
        while (time_elapsed < animationClip.length)
        {
            time_elapsed = i / sampleRate;
            animationClip.SampleAnimation(instance, time_elapsed);
            csv.Append($"{time_elapsed:0.000}");

            foreach (Transform bone in jointTransforms)
            {
                Vector3 pos = bone.localPosition;
                Quaternion rot = bone.localRotation;

                csv.AppendFormat(",{0},{1},{2}", pos.x, pos.y, pos.z);
                csv.AppendFormat(",{0},{1},{2},{3}", rot.x, rot.y, rot.z, rot.w);
            }
            csv.AppendLine();
            i++;
        }

        File.WriteAllText(filePath, csv.ToString());
        DestroyImmediate(instance);

        EditorUtility.DisplayDialog("Export Complete", $"Exported {frameCount + 1} frames for {jointTransforms.Count} joints.", "OK");
    }

    void add_joint(ref StringBuilder csv,  Transform bone, List<Transform> jointTransforms, string indentation)
    {
        csv.Append(indentation);
        csv.AppendLine($"\"{bone.name}\"[");
        csv.Append(indentation);
        foreach(Transform joint in bone)
        {
            if(jointTransforms.Contains(joint))
                add_joint(ref csv, joint, jointTransforms, indentation + "\t");
        }
        csv.AppendLine(indentation + "]");
    }
}
