using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;

public class CollisionReporter : MonoBehaviour
{
    public Agent agent;
    private void OnCollisionEnter(Collision collision)
    {
        if(agent is MLRagdoll)
            (agent  as MLRagdoll).processCollision(collision);
    }
}
