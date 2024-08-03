using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

public class InputHandler : MonoBehaviour
{
    #region Action readers
    private PlayerInput _playerInput;
    private InputAction moveAction;
    private InputAction lookAction;
    #endregion 

    #region Inputs
    public Vector2 RawMoveInput;
    public Vector3 MoveInput;
    public Vector2 RawLookInput;
    public Vector3 LookInput;

    public bool GaitInput;
    public bool StrafeInput;
    #endregion

    #region Smooth movement input
    [Header("Input options")]
    public float deadzone = .2f;
    public float smoothMoveInputSpeed = .2f;
    public float smoothLookInputSpeed = .8f;
    #endregion

    // Start is called before the first frame update
    void Start()
    {
        _playerInput = GetComponent<PlayerInput>();
        moveAction = _playerInput.actions["move"];
        lookAction = _playerInput.actions["look"];
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    #region Input event callbacks
    public void OnMoveInput(InputAction.CallbackContext context)
    {
        RawMoveInput = context.ReadValue<Vector2>();
        float movenorm = Mathf.Sqrt(RawMoveInput.x * RawMoveInput.x + RawMoveInput.y * RawMoveInput .y);
        float movex;
        float movey;
        if(movenorm > deadzone)
        {
            float dirX = RawMoveInput.x / movenorm;
            float dirY = RawMoveInput.y / movenorm;
            float clippedNorm = movenorm > 1.0f ? 1.0f : movenorm * movenorm;
            movex = dirX * clippedNorm;
            movey = dirY * clippedNorm;
        }
        else
        {
            movex = 0.0f;
            movey = 0.0f;
        }
        MoveInput = new Vector3(movex, 0.0f, movey);
    }
    public void OnLookInput(InputAction.CallbackContext context)
    {
        RawLookInput = context.ReadValue<Vector2>();
        float looknorm = Mathf.Sqrt(RawLookInput.x * RawLookInput.x + RawLookInput.y * RawLookInput.y);
        float lookx;
        float looky;
        if (looknorm > deadzone)
        {
            float dirX = RawLookInput.x / looknorm;
            float dirY = RawLookInput.y / looknorm;
            float clippedNorm = looknorm > 1.0f ? 1.0f : looknorm * looknorm;
            lookx = dirX * clippedNorm;
            looky = dirY * clippedNorm;
        }
        else
        {
            lookx = 0.0f;
            looky = 0.0f;
        }
        LookInput = new Vector3(lookx, 0.0f, looky);
    }
    public void OnGaitInput(InputAction.CallbackContext context)
    {
        if (context.started)
            GaitInput = !GaitInput;
    }
    public void OnStrafeInput(InputAction.CallbackContext context)
    {
        if (context.started)
            StrafeInput = true;
        if (context.performed)
            StrafeInput = true;
        if (context.canceled)
            StrafeInput = false;
    }
    #endregion
}
