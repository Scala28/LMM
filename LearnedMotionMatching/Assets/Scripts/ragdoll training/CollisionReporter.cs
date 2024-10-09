using Unity.MLAgents;
using UnityEngine;

public class CollisionReporter : MonoBehaviour
{
    public MLRagdoll agent;
    private void OnCollisionEnter(Collision collision)
    {
        // agent.processCollision(collision);
    }
}