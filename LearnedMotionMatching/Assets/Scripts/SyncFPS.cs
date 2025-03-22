using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SyncFPS : MonoBehaviour
{
    private static SyncFPS _instance;
    float timeSinceLastUpdate = 0f;
    public bool isSyncFrame = false;
    public float period = 1f / 60f;

    public static SyncFPS Instance { get { return _instance; } }
    private void Awake()
    {
        if (_instance != null && _instance != this)
        {
            Destroy(this.gameObject);
        }
        else
        {
            _instance = this;
        }
    }

    void FixedUpdate()
    {
        isSyncFrame = false;
        timeSinceLastUpdate += Time.fixedDeltaTime;
        if (timeSinceLastUpdate < period)
            return;
        isSyncFrame = true;
        timeSinceLastUpdate = 0;
    }
    public float getPeriod() => period;
}
