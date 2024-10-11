using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PhysicsManager : MonoBehaviour
{
    void Awake()
    {
        Physics.simulationMode = SimulationMode.Script;
    }
    private void FixedUpdate()
    {
        Physics.Simulate(Time.fixedDeltaTime);
    }
}
