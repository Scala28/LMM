using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

public class InputHandler : MonoBehaviour
{
    private PlayerInput _playerInput;
    private InputActionMap _actionMap;

    #region Inputs
    private Vector2 Raw_stickLeft;
    public Vector3 StickLeft { get; private set; }
    private Vector2 Raw_stickRight;
    public Vector3 StickRight { get; private set; }

    public bool RightShoulder { get; set; }
    public bool LeftTrigger { get; private set; }

    public Dictionary<int, bool> actions { get; private set; }
    #endregion

    #region Options
    [Header("Input options")]
    public float deadzone = .2f;
    public float input_buffering_time = .1f;
    #endregion

    void Start()
    {
        _playerInput = GetComponent<PlayerInput>();
        actions = new Dictionary<int, bool>
        {
            { 1, false },
            { 2, false },
            { 3, false },
            { 4, false }
        };
    }
    private void Update()
    {
        _actionMap = _playerInput.currentActionMap;
    }

    #region Input event callbacks

    public void OnLeftStick(InputAction.CallbackContext context)
    {
        Raw_stickLeft = context.ReadValue<Vector2>();
        float movenorm = Mathf.Sqrt(Raw_stickLeft.x * Raw_stickLeft.x + Raw_stickLeft.y * Raw_stickLeft.y);
        float movex;
        float movey;
        if(movenorm > deadzone)
        {
            float dirX = Raw_stickLeft.x / movenorm;
            float dirY = Raw_stickLeft.y / movenorm;
            float clippedNorm = movenorm > 1.0f ? 1.0f : movenorm * movenorm;
            movex = dirX * clippedNorm;
            movey = dirY * clippedNorm;
        }
        else
        {
            movex = 0.0f;
            movey = 0.0f;
        }
        StickLeft = new Vector3(movex, 0.0f, movey);
    }
    public void OnRightStick(InputAction.CallbackContext context)
    {
        Raw_stickRight = context.ReadValue<Vector2>();
        float looknorm = Mathf.Sqrt(Raw_stickRight.x * Raw_stickRight.x + Raw_stickRight.y * Raw_stickRight.y);
        float lookx;
        float looky;
        if (looknorm > deadzone)
        {
            float dirX = Raw_stickRight.x / looknorm;
            float dirY = Raw_stickRight.y / looknorm;
            float clippedNorm = looknorm > 1.0f ? 1.0f : looknorm * looknorm;
            lookx = dirX * clippedNorm;
            looky = dirY * clippedNorm;
        }
        else
        {
            lookx = 0.0f;
            looky = 0.0f;
        }
        StickRight = new Vector3(lookx, 0.0f, looky);
    }
    public void OnRightShoulder(InputAction.CallbackContext context)
    {
        if (context.started)
            RightShoulder = !RightShoulder;
        switch (_actionMap.name)
        {
            case "plane":
                if (context.started)
                    RightShoulder = !RightShoulder;
                break;
            case "terrain":
                if (context.started)
                    RightShoulder = !RightShoulder;
                break;
            case "fight":
                if (context.started)
                    RightShoulder = true;
                break;
            default:
                break;

        }
    }
    public void OnLeftTrigger(InputAction.CallbackContext context)
    {
        if (context.started)
            LeftTrigger = true;
        if (context.performed)
            LeftTrigger = true;
        if (context.canceled)
            LeftTrigger = false;
    }

    public void OnActionCalled(InputAction.CallbackContext context)
    {
        if (context.started)
        {
            int action_tag = int.Parse(context.action.name);
            actions[action_tag] = true;
        }else if(context.canceled)
        {
            int action_tag = int.Parse(context.action.name);
            StartCoroutine(ResetBoolAfterDelay(val => actions[action_tag] = val, false, input_buffering_time));
        }
    }
    #endregion
    public void SetButton(Action<bool> setter, bool value = false) => setter(value);
    private IEnumerator ResetBoolAfterDelay(System.Action<bool> setter, bool value, float delay)
    {
        yield return new WaitForSeconds(delay);
        setter(value);
    }
}
