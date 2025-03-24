using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

public class InputHandler : MonoBehaviour
{
    private PlayerInput _playerInput;

    #region Inputs
    public Vector2 Raw_stickLeft;
    public Vector3 StickLeft;
    public Vector2 Raw_stickRight;
    public Vector3 StickRight;

    public bool Button1;
    public bool Button2;
    #endregion

    #region Smooth movement input
    [Header("Input options")]
    public float deadzone = .2f;
    #endregion

    void Start()
    {
        _playerInput = GetComponent<PlayerInput>();
    }

    #region Input event callbacks
    public void OnMoveInput(InputAction.CallbackContext context)
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
    public void OnLookInput(InputAction.CallbackContext context)
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
    public void OnGaitInput(InputAction.CallbackContext context)
    {
        if (context.started)
            Button1 = !Button1;
    }
    public void OnStrafeInput(InputAction.CallbackContext context)
    {
        if (context.started)
            Button2 = true;
        if (context.performed)
            Button2 = true;
        if (context.canceled)
            Button2 = false;
    }
    #endregion
}
