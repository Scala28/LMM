using UnityEngine;

public class StickmanDrawer : MonoBehaviour
{
    void OnDrawGizmos()
    {
        DrawBones(transform.GetChild(0));
    }

    void DrawBones(Transform t)
    {
        foreach (Transform child in t)
        {
            Gizmos.color = Color.green;
            Gizmos.DrawLine(t.position, child.position);

            // Optional: disegna una piccola sfera su ogni joint
            Gizmos.color = Color.red;
            Gizmos.DrawSphere(t.position, 0.01f);

            DrawBones(child);
        }
    }
}
