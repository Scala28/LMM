using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

public class InputHandler : MonoBehaviour
{
    #region Action readers
    private PlayerInput _playerInput;
    private InputAction moveAction;
    #endregion 

    #region Inputs
    private Vector2 RawMovementInput;
    public float Input_localZ;
    public float Input_localX;
    #endregion

    #region Smooth movement input
    [Header("Input options")]
    public float smoothInputSpeed = .2f;
    private Vector2 currentMovementInput;
    private Vector2 smoothInputVelocity;
    #endregion

    // Start is called before the first frame update
    void Start()
    {
        _playerInput = GetComponent<PlayerInput>();
        moveAction = _playerInput.actions["move"];

        currentMovementInput = Vector2.zero;
    }

    // Update is called once per frame
    void Update()
    {
        //currentMovementInput = Vector2.SmoothDamp(currentMovementInput,
        //    RawMovementInput, ref smoothInputVelocity, smoothInputSpeed);
        
        //Input_localZ = currentMovementInput.y;
        //Input_localX = currentMovementInput.x;

        Input_localZ = RawMovementInput.y;
        Input_localX = RawMovementInput.x;
    }
    #region Input event callbacks
    public void OnMoveInput(InputAction.CallbackContext context)
    {
        RawMovementInput = context.ReadValue<Vector2>();
    }
    #endregion
}
