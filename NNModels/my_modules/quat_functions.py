import math
import torch
import numpy as np


def fk(lpos, lrot, lvel, lang, parents):
    gpos, grot, gvel, gang = [lpos[..., :1, :]], [lrot[..., :1, :]], [lvel[..., :1, :]], [lang[..., :1, :]]
    for i in range(1, len(parents)):
        gpos.append(mul_vec(grot[parents[i]], lpos[..., i:i + 1, :]) + gpos[parents[i]])
        grot.append(mul(grot[parents[i]], lrot[..., i:i + 1, :]))
        gvel.append(gvel[parents[i]] + mul_vec(grot[parents[i]], lvel[..., i:i + 1, :]) +
                    _cross(gang[parents[i]], mul_vec(grot[parents[i]], lpos[..., i:i + 1, :])))
        gang.append(mul_vec(grot[parents[i]], lang[..., i:i + 1, :]) + gang[parents[i]])

    return (
        torch.cat(gpos, dim=-2),
        torch.cat(grot, dim=-2),
        torch.cat(gvel, dim=-2),
        torch.cat(gang, dim=-2)
    )


def mul_vec(q, x):
    q_scalar = q[..., 0][..., np.newaxis]
    q_vector = q[..., 1:]

    return x + q_scalar * 2.0 * _cross(q_vector, x) + _cross(q_vector, 2.0 * _cross(q_vector, x))


def mul(a, b):
    aw, ax, ay, az = a[..., 0:1], a[..., 1:2], a[..., 2:3], a[..., 3:4]
    bw, bx, by, bz = b[..., 0:1], b[..., 1:2], b[..., 2:3], b[..., 3:4]

    return torch.cat([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw
    ], dim=-1)


def _cross(a, b):
    return torch.cat([
        a[..., 1:2] * b[..., 2:3] - a[..., 2:3] * b[..., 1:2],
        a[..., 2:3] * b[..., 0:1] - a[..., 0:1] * b[..., 2:3],
        a[..., 0:1] * b[..., 1:2] - a[..., 1:2] * b[..., 0:1]
    ], dim=-1)


def _inv(q):
    return torch.tensor([1, -1, -1, -1], dtype=torch.float) * q


def inv_mul(a, b):
    return mul(_inv(a), b)


def mul_inv(a, b):
    return mul(a, _inv(b))


def inv_mul_vec(q, x):
    return mul_vec(_inv(q), x)


def to_xform(q):
    qw, qx, qy, qz = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]

    return torch.cat((
        torch.cat((1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)), dim=-1)[...,
        np.newaxis, :],
        torch.cat((2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)), dim=-1)[...,
        np.newaxis, :],
        torch.cat((2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx - qy * qy)), dim=-1)[...,
        np.newaxis, :]
    ), dim=-2)


def _length(x):
    return torch.sqrt(torch.sum(x * x, dim=-1))


def _normalize(x, eps=1e-8):
    return x / (_length(x)[..., np.newaxis] + eps)


def from_xform(ts):
    return _normalize(
        torch.where((ts[..., 2, 2] < 0.0)[..., np.newaxis],
                    torch.where((ts[..., 0, 0] > ts[..., 1, 1])[..., np.newaxis],
                                torch.cat([
                                    (ts[..., 2, 1] - ts[..., 1, 2])[..., np.newaxis],
                                    (1.0 + ts[..., 0, 0] - ts[..., 1, 1] - ts[..., 2, 2])[..., np.newaxis],
                                    (ts[..., 1, 0] + ts[..., 0, 1])[..., np.newaxis],
                                    (ts[..., 0, 2] + ts[..., 2, 0])[..., np.newaxis]], dim=-1),
                                torch.cat([
                                    (ts[..., 0, 2] - ts[..., 2, 0])[..., np.newaxis],
                                    (ts[..., 1, 0] + ts[..., 0, 1])[..., np.newaxis],
                                    (1.0 - ts[..., 0, 0] + ts[..., 1, 1] - ts[..., 2, 2])[..., np.newaxis],
                                    (ts[..., 2, 1] + ts[..., 1, 2])[..., np.newaxis]], dim=-1)),
                    torch.where((ts[..., 0, 0] < -ts[..., 1, 1])[..., np.newaxis],
                                torch.cat([
                                    (ts[..., 1, 0] - ts[..., 0, 1])[..., np.newaxis],
                                    (ts[..., 0, 2] + ts[..., 2, 0])[..., np.newaxis],
                                    (ts[..., 2, 1] + ts[..., 1, 2])[..., np.newaxis],
                                    (1.0 - ts[..., 0, 0] - ts[..., 1, 1] + ts[..., 2, 2])[..., np.newaxis]], dim=-1),
                                torch.cat([
                                    (1.0 + ts[..., 0, 0] + ts[..., 1, 1] + ts[..., 2, 2])[..., np.newaxis],
                                    (ts[..., 2, 1] - ts[..., 1, 2])[..., np.newaxis],
                                    (ts[..., 0, 2] - ts[..., 2, 0])[..., np.newaxis],
                                    (ts[..., 1, 0] - ts[..., 0, 1])[..., np.newaxis]], dim=-1))))


def to_xform_xy(q):
    qw, qx, qy, qz = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]

    return torch.cat((
        torch.cat((1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz)), dim=-1)[..., np.newaxis, :],
        torch.cat((2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz)), dim=-1)[..., np.newaxis, :],
        torch.cat((2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx)), dim=-1)[..., np.newaxis, :]
    ), dim=-2)


def from_xfm_xy(x):
    c2 = _cross(x[..., 0], x[..., 1])
    c2 = c2 / torch.sqrt(torch.sum(torch.square(c2), dim=-1))[..., np.newaxis]
    c1 = _cross(c2, x[..., 0])
    c1 = c1 / torch.sqrt(torch.sum(torch.square(c1), dim=-1))[..., np.newaxis]
    c0 = x[..., 0]

    xfm = torch.cat([
        c0[..., np.newaxis],
        c1[..., np.newaxis],
        c2[..., np.newaxis]
    ], dim=-1)
    return from_xform(xfm)


def from_scaled_axis_angle(x, eps=1e-5):
    return _exp(x / 2.0, eps)


def _exp(x, eps=1e-5):
    halfangle = torch.sqrt(torch.sum(torch.square(x), dim=-1))[..., np.newaxis]
    c = torch.where(halfangle < eps, torch.ones_like(halfangle), torch.cos(halfangle))
    s = torch.where(halfangle < eps, torch.ones_like(halfangle), torch.sinc(halfangle / torch.pi))
    return torch.cat([c, s*x], dim=-1)


def to_euler(x, order='xyz'):
    q0 = x[..., 0:1]
    q1 = x[..., 1:2]
    q2 = x[..., 2:3]
    q3 = x[..., 3:4]

    if order == 'xyz':

        return np.concatenate([
            np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2)),
            np.arcsin((2 * (q0 * q2 - q3 * q1)).clip(-1, 1)),
            np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))], axis=-1)

    elif order == 'yzx':

        return np.concatenate([
            np.arctan2(2 * (q1 * q0 - q2 * q3), -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0),
            np.arctan2(2 * (q2 * q0 - q1 * q3), q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0),
            np.arcsin((2 * (q1 * q2 + q3 * q0)).clip(-1, 1))], axis=-1)

    else:
        raise NotImplementedError('Cannot convert from ordering %s' % order)
