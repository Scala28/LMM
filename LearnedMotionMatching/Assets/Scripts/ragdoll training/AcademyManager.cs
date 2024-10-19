using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class AcademyManager : MonoBehaviour
{
    private SyncFPS _sync60Fps;

    void Awake()
    {
        Academy.Instance.AutomaticSteppingEnabled = false;
        _sync60Fps = SyncFPS.Instance;
    }

    void FixedUpdate()
    {
        if (_sync60Fps.isSyncFrame)
            Academy.Instance.EnvironmentStep();
    }
}
