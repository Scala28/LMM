using UnityEngine;

public class InputGenerator : MonoBehaviour
{
    public Vector3 gamepad_left { get { return new Vector3(current_left.x, 0f, current_left.y); } }
    public Vector3 gamepad_right { get { return new Vector3(current_right.x, 0f, current_right.y); } }

    public bool Input_1 { get { return input_1; } }
    public bool Input_2 { get { return input_2; } }

    Vector2 target_left;
    Vector2 target_right;
    Vector2 current_left;
    Vector2 current_right;
    Vector2 velocity_left;
    Vector2 velocity_right;

    bool input_1;
    bool input_2;

    public float inputChangeHalflife = .5f;
    void Start()
    {
        target_left = Vector2.zero;
        target_right = Vector2.zero;

        input_1 = false;
        input_2 = false;
    }

    void FixedUpdate()
    {
        // tick spring update
        SpringUtils.spring_character_update(current_left, velocity_left, target_left, inputChangeHalflife, Time.fixedDeltaTime, out current_left, out velocity_left);
        SpringUtils.spring_character_update(current_right, velocity_right, target_right, inputChangeHalflife, Time.fixedDeltaTime, out current_right, out velocity_right);
    }

    public void changeDirection()
    {
        target_left = Random.value < .01f ? Vector2.zero : Random.insideUnitCircle;
        target_right = Random.value < .01f ? Vector2.zero : Random.insideUnitCircle;

        input_1 = Random.value <= .5f;
        input_2 = Random.value <= .3f;
    }
}
