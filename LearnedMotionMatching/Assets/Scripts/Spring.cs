using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

public static class Spring
{
    private const float LN2f = 0.69314718056f;
    public static void inertialize_update(
        ref Vector3 out_x,
        ref Vector3 out_v,
        ref Vector3 off_x,
        ref Vector3 off_v,
        Vector3 in_x,
        Vector3 in_v,
        float halflife,
        float dt)
    {
        decay_spring_damper_exact(ref off_x, ref off_v, halflife, dt);
        out_x = in_x + off_x;
        out_v = in_v + off_v;
    }
    public static void inertialize_update(
        ref Vector4 out_x,
        ref Vector3 out_v,
        ref Vector4 off_x,
        ref Vector3 off_v,
        Vector3 in_x,
        Vector3 in_v,
        float halflife,
        float dt)
    {
        decay_spring_damper_exact(ref off_x, ref off_v, halflife, dt);
        out_x = Quat.quat_mul(off_x, in_x);
        out_v = off_v + Quat.quat_mul_vec(off_x, in_v);
    }
    public static void inertialize_transition(
        ref Vector3 off_x,
        ref Vector3 off_v,
        Vector3 src_x,
        Vector3 src_v,
        Vector3 dst_x,
        Vector3 dst_v)
    {
        off_x = (src_x + off_x) - dst_x;
        off_v = (src_v + off_v) - dst_v;
    }
    public static void inertialize_transition(
        ref Vector4 off_x,
        ref Vector3 off_v,
        Vector4 src_x,
        Vector3 src_v,
        Vector4 dst_x,
        Vector3 dst_v)
    {
        off_x = Quat.quat_abs(Quat.quat_mul(Quat.quat_mul(off_x, src_x), Quat.quat_inv(dst_x)));
        off_v = (off_v + src_v) - dst_v;
    }
    private static void decay_spring_damper_exact(
        ref Vector3 x,
        ref Vector3 v,
        float halflife,
        float dt)
    {
        float y = halflife_to_damping(halflife) / 2.0f;
        Vector3 j1 = v + x * y;
        float eydt = fast_negexpf(y * dt);

        x = eydt * (x + j1 * dt);
        v = eydt * (v - j1 * y * dt);
    }
    private static void decay_spring_damper_exact(
        ref Vector4 x,
        ref Vector3 v,
        float halflife,
        float dt)
    {
        float y = halflife_to_damping(halflife) / 2.0f;
        Vector3 j0 = Quat.quat_to_scaled_angle_axis(x);
        Vector3 j1 = v + j0 * y;

        float eydt = fast_negexpf(y * dt);

        x = Quat.quat_from_scaled_angle_axis(eydt * (j0 + j1 * dt));
        v = eydt * (v - j1 * y * dt);
    }
    public static void simple_spring_damper_exact(
        ref float x,
        ref float v, 
        float x_goal,
        float halflife,
        float dt)
    {
        float y = halflife_to_damping(halflife) / 2.0f;
        float j0 = x - x_goal;
        float j1 = v + j0 * y;
        float eydt = fast_negexpf(y * dt);

        x = eydt * (j0 + j1 * dt) + x_goal;
        v = eydt * (v - j1 * y * dt);
    }
    public static void simple_spring_damper_exact(
        ref Vector4 x,
        ref Vector3 v,
        Vector4 x_goal,
        float halflife,
        float dt)
    {
        float y = halflife_to_damping(halflife) / 2.0f;
        Vector3 j0 = Quat.quat_to_scaled_angle_axis(Quat.quat_abs(Quat.quat_mul(x, Quat.quat_inv(x_goal))));
        Vector3 j1 = v + j0 * y;

        float eydt = fast_negexpf(y * dt);

        x = Quat.quat_mul(Quat.quat_from_scaled_angle_axis(eydt * (j0 + j1 * dt)), x_goal);
        v = eydt * (v - j1 * y * dt);
    }
    public static float halflife_to_damping(float halflife, float eps = 1e-5f)
    {
        return (4.0f * LN2f) / (halflife + eps);
    }
    public static float fast_negexpf(float x)
    {
        return 1.0f / (1.0f + x + 0.48f * x * x + 0.235f * x * x * x);
    }
}
