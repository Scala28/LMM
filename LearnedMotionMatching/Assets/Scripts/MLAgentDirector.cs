using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MLAgentDirector : MonoBehaviour
{
    private ConfigManager _config;

    public int numAgents = 5;
    public GameObject ragdollAgent;

    private MLRagdoll[] agents;
    public int reportMeanRewardEveryNSteps = 10000;
    private int curStep = 0;
    private float meanReward;

    private void Awake()
    {
        if (ragdollAgent == null)
            return;
        _config = ConfigManager.Instance;

        Physics.defaultSolverIterations = _config.Training_data.solverIterations;
        Physics.defaultSolverVelocityIterations = _config.Training_data.solverIterations;

        agents = new MLRagdoll[numAgents];
        for (int i = 0; i < numAgents; i++) {
            agents[i] = CreateMLRagdoll();
            if (_config.Training_data.selfCollision)
                agents[i].AssignLayer(LayerMask.NameToLayer($"model_{i + 1}"));
            else
                agents[i].AssignLayer(LayerMask.NameToLayer($"test_model"));
        }
    }
    private MLRagdoll CreateMLRagdoll() {
        GameObject obj = Instantiate(ragdollAgent);
        MLRagdoll agent = obj.GetComponent<MLRagdoll>();
        obj.SetActive(true);
        return agent;
    }

    private void FixedUpdate()
    {
        float curStepReward = 0f;
        foreach (var agent in agents)
        {
            agent.LateFixedUpdate();
            curStepReward += agent.finalReward / numAgents;
        }
        meanReward += curStepReward / reportMeanRewardEveryNSteps;
        Debug.Log(meanReward);
        curStep++;
        if (curStep % reportMeanRewardEveryNSteps == 0) {
            Debug.Log($"Step {curStep} mean reward last {reportMeanRewardEveryNSteps} is: {meanReward}");
            meanReward = 0f;
        }
    }

}
