import math
import torch
import numpy as np


def LoadData(filename):
    data = []
    # contains clips indices
    indices = [0]

    with open("data/" + filename + ".txt") as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(' ')]

            if inner_list == ['']:
                indices.append(len(data))
                continue

            converted = []
            for item in inner_list:
                converted.append(float(item))

            data.append(converted)

    data = np.array(data)
    indices = np.array(indices)

    return {
        'data': data,
        'indices': indices
    }


def _cross(a, b):
    return torch.cat([
        a[..., 1:2] * b[..., 2:3] - a[..., 2:3] * b[..., 1:2],
        a[..., 2:3] * b[..., 0:1] - a[..., 0:1] * b[..., 2:3],
        a[..., 0:1] * b[..., 1:2] - a[..., 1:2] * b[..., 0:1]
    ], dim=-1)


def quat_mul_vec(q, x):
    q_vector = q[..., 1:]
    q_scalar = q[..., 0][..., np.newaxis]

    return (x + _cross(2.0 * q_vector, _cross(q_vector, x))
            + 2.0 * q_scalar * _cross(q_vector, x))


def quat_mul(a, b):
    aw, ax, ay, az = a[..., 0:1], a[..., 1:2], a[..., 2:3], a[..., 3:4]
    bw, bx, by, bz = b[..., 0:1], b[..., 1:2], b[..., 2:3], b[..., 3:4]

    return torch.cat([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw
    ], dim=-1)


def quat_forwardkinematics(Y, hierarchy):
    Q = Y[..., :13]
    bone = 1

    for i in range(13, Y.size(-1), 13):
        # parent of bone i
        p = hierarchy[bone] * 13
        pos = ((Q[..., p + 0:p + 3] if p != 0 else 0) +
               quat_mul_vec(Q[..., p + 3:p + 7], Y[..., i + 0:i + 3]))
        rot = quat_mul(Q[..., p + 3:p + 7], Y[..., i + 3:i + 7])
        vel = Q[..., p + 7:p + 10] + quat_mul_vec(Q[..., p + 3:p + 7], Y[..., i + 7:i + 10]) + torch.cross(
            Q[..., p + 10:p + 13],
            quat_mul_vec(Q[..., p + 3:p + 7], Y[..., i + 0:i + 3]),
            dim=-1)
        ang = Q[..., p + 10:p + 13] + quat_mul_vec(Q[..., p + 3:p + 7], Y[..., i + 10:i + 13])

        Q = torch.cat((Q, pos, rot, vel, ang), dim=-1)
        bone += 1

    return Q


def quat_to_9drm(q):
    q = quat_normalize(q)
    qw, qx, qy, qz = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]

    return torch.cat((
        1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy),
        2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx),
        2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx - qy * qy)
    ), dim=-1)


# TODO: adapt quat_from_9drm to tensor of any dimension
def quat_from_9drm(rm):
    m11, m12, m13 = rm[..., 0:1], rm[..., 1:2], rm[..., 2:3]
    m21, m22, m23 = rm[..., 3:4], rm[..., 4:5], rm[..., 5:6]
    m31, m32, m33 = rm[..., 6:7], rm[..., 7:8], rm[..., 8:9]

    t = m11 + m22 + m33 + 1.0

    # ris = torch.empty(*(rm.size()[:-1]), 0)
    # supposing rm.shape = torch.Size((n, 9))
    ris = []

    for i in range(rm.size(0)):
        if t[i] != 0:
            ris.append(torch.cat((t[i], m32[i]-m23[i], m13[i]-m31[i], m21[i]-m12[i]), dim=0))
        else:
            c2 = 1 if m21[i] + m12[i] > 0 else (-1 if m21[i] + m12[i] < 0 else math.pow(math.copysign(1.0, m32[i]), 2))
            c3 = 1 if m31[i] + m13[i] > 0 else (-1 if m31[i] + m13[i] < 0 else math.pow(math.copysign(1.0, m32[i]), 3))
            ris.append(torch.cat((torch.zeros(1, 1), torch.sqrt(m11[i]+1.0), c2 * torch.sqrt(m22[i]+1.0), c3 * torch.sqrt(m33[i]+1.0)), dim=0))

        # resulting quaternion normalization
        q = ris[-1]
        q = quat_normalize(q)
        ris[-1] = q

    ris = torch.stack(ris, dim=0)
    return ris


def quat_to_6dr(q):
    rm = quat_to_9drm(q)
    return rm[..., 0:6]


def rm_from_6dr(r):
    a1, a2 = r[..., 0:3], r[..., 3:6]

    b1 = vec_normalize(a1)
    b2 = vec_normalize(a2 - torch.sum(b1 * a2) * b1)
    b3 = _cross(b1, b2)

    return torch.cat((
        b1, b2, b3
    ), dim=-1)


def quat_normalize(q):
    norms = torch.sqrt(torch.sum(torch.square(q), dim=-1))
    return q / norms


def vec_normalize(v):
    norms = torch.sqrt(torch.sum(torch.square(v), dim=-1))
    return v / norms

