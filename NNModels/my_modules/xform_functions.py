import math
import torch
import numpy as np


def _cross(a, b):
    return torch.cat([
        a[..., 1:2] * b[..., 2:3] - a[..., 2:3] * b[..., 1:2],
        a[..., 2:3] * b[..., 0:1] - a[..., 0:1] * b[..., 2:3],
        a[..., 0:1] * b[..., 1:2] - a[..., 1:2] * b[..., 0:1]], dim=-1)


def from_xy(x):
    c2 = _cross(x[..., 0], x[..., 1])
    c2 = c2 / torch.sqrt(torch.sum(torch.square(c2), dim=-1))[..., None]
    c1 = _cross(c2, x[..., 0])
    c1 = c1 / torch.sqrt(torch.sum(torch.square(c1), dim=-1))[..., None]
    c0 = x[..., 0]

    return torch.cat([
        c0[..., None],
        c1[..., None],
        c2[..., None]
    ], dim=-1)


def mul(x, y):
    return torch.matmul(x, y)


def mul_vec(q, x):
    return torch.matmul(q, x[..., None])[..., 0]


def inv_mul(x, y):
    return torch.matmul(x.transpose(-1, -2), y)


def inv_mul_vec(q, x):
    return torch.matmul(q.transpose(-1, -2), x[..., None])[..., 0]


def fk(lpos, lrot, lvel, lang, parents):
    gpos, grot, gvel, gang = [lpos[..., :1, :]], [lrot[..., :1, :]], [lvel[..., :1, :]], [lang[..., :1, :]]
    for i in range(1, len(parents)):
        gpos.append(mul_vec(grot[parents[i]], lpos[..., i:i + 1, :]) + gpos[parents[i]])
        grot.append(mul(grot[parents[i]], lrot[..., i:i + 1, :, :]))
        gvel.append(gvel[parents[i]] + mul_vec(grot[parents[i]], lvel[..., i:i + 1, :]) +
                    torch.cross(gang[parents[i]], mul_vec(grot[parents[i]], lpos[..., i:i + 1, :])))
        gang.append(mul_vec(grot[parents[i]], lang[..., i:i + 1, :]) + gang[parents[i]])

    return (
        torch.cat(gpos, dim=-2),
        torch.cat(grot, dim=-3),
        torch.cat(gvel, dim=-2),
        torch.cat(gang, dim=-2)
    )
