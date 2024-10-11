using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ConfigManager : MonoBehaviour
{
    public TrainingData Training_data;

    [System.Serializable]
    public class MusclePower
    {
        public MotionMatcher.character Bone;
        public Vector3 PowerVector;
    }

    private static ConfigManager _instance;
    public static ConfigManager Instance { get { return _instance; } }

    private void Awake()
    {
        if(_instance != null && _instance != this)
        {
            Destroy(this.gameObject);
        }
        else
        {
            Application.targetFrameRate = 60;
            Time.fixedDeltaTime = 1f / 60;
            _instance = this;
        }
    }
}
