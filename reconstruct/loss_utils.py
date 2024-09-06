#
# This file is part of https://github.com/JingwenWang95/DSP-SLAM
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

import time
import torch
import numpy as np


def get_rays(sampled_pixels, invK):
    """
    This function computes the ray directions given sampled pixel
    and camera intrinsics
    :param sampled_pixels: (N, 2), order is [u, v]
    :param invK: (3, 3)
    :return: ray directions (N, 3) under camera frame
    """
    n = sampled_pixels.shape[0]
    # (n, 3) = (n, 2) (n, 1)
    u_hom = np.concatenate([sampled_pixels, np.ones((n, 1))], axis=-1)
    # (n, 3) = (n, 1, 3) * (3, 3)
    directions = (u_hom[:, None, :] * invK).sum(-1)

    return directions.astype(np.float32)


def sdf_to_occupancy(sdf_tensor, th=0.015):
    """
    :param sdf_tensor: torch tensor
    :param th: cut-off threshold, o(x>th) = 0, o(x<-th) = 1
    :return: occ_tensor: torch tensor
    """
    occ_tensor = torch.clamp(sdf_tensor, min=-th, max=th)
    occ_tensor = 0.5 - occ_tensor / (2 * th)
    return occ_tensor


def decode_sdf(decoder, lat_vec, x, max_batch=64**3):
    """
    :param decoder: DeepSDF Decoder
    :param lat_vec: torch.Tensor (code_len,), latent code
    :param x: torch.Tensor (N, 3), query positions
    :return: batched outputs (N, )
    :param max_batch: max batch size
    :return:
    """

    num_samples = x.shape[0]

    head = 0

    # get sdf values given query points
    sdf_values_chunks = []
    with torch.no_grad():
        while head < num_samples:
            x_subset = x[head : min(head + max_batch, num_samples), 0:3].cuda()

            latent_repeat = lat_vec.expand(x_subset.shape[0], -1)
            fp_inputs = torch.cat([latent_repeat, x_subset], dim=-1)
            sdf_values = decoder(fp_inputs).squeeze()

            sdf_values_chunks.append(sdf_values)
            head += max_batch

    sdf_values = torch.cat(sdf_values_chunks, 0).cuda()
    return sdf_values


def get_batch_sdf_jacobian(decoder, lat_vec, x, out_dim=1):
    """
    :param decoder: DeepSDF Decoder
    :param lat_vec: torch.Tensor (code_len,), latent code
    :param x: torch.Tensor (N, 3), query position
    :param out_dim: int, output dimension of a single input
    :return: batched Jacobian (N, out_dim, code_len + 3)
    """
    # 定义变量 n，它是查询位置张量 x 的批次大小（即张量的第一个维度大小）。
    n = x.shape[0]
    # 克隆并分离查询位置张量 x，将结果赋值给 input_x。这样做是为了创建一个没有梯度的张量副本，以便后续操作不会影响原始张量的梯度。
    input_x = x.clone().detach()
    # 使用潜在向量 lat_vec 扩展为形状为 (n, code_len) 的张量，并将结果赋值给 latent_repeat。这样做是为了将潜在向量复制为与查询位置张量 x 相同大小的张量，以便与其进行拼接
    latent_repeat = lat_vec.expand(n, -1)
    # input是 lanten_vector和點坐標（xyz）拼接後的一維向量
    # 在输入张量 inputs 的第二个维度前插入一个新的维度，将其形状变为 (n, 1, code_len + 3)。这样做是为了在后续操作中适应输出维度的要求。
    inputs = torch.cat([latent_repeat, input_x], 1)

    # 【？？？】在输入张量 inputs 的第二个维度前插入一个新的维度，将其形状变为 (n, 1, code_len + 3)。这样做是为了在后续操作中适应输出维度的要求。
    inputs = inputs.unsqueeze(1)  # (n, 1, in_dim)
    # 将输入张量 inputs 在第二个维度（列）上重复 out_dim 次，得到形状为 (n, out_dim, code_len + 3) 的张量。这样做是为了在计算批次中多个输入时，适应输出维度的要求
    inputs = inputs.repeat(1, out_dim, 1)  # (n, out_dim, in_dim)
    # 将输入张量 inputs 设置为可求导状态，以便后续计算梯度。
    inputs.requires_grad = True
    # 将输入张量 inputs 传递给解码器 decoder，得到输出张量 y。这个输出张量的形状为 (n, out_dim, out_dim)。
    y = decoder(inputs)  # (n, out_dim, out_dim)
    #  构造一个形状为 (1, out_dim, out_dim) 的单位矩阵，并在第一个维度上重复 n 次，得到形状为 (n, out_dim, out_dim) 的张量 w。这个张量将在后续的反向传播中用作权重矩阵。
    w = torch.eye(out_dim).view(1, out_dim, out_dim).repeat(n, 1, 1).cuda()  # (n, out_dim, out_dim)
    # 使用反向传播计算输出张量 y 相对于输入张量 inputs 的梯度，并使用权重矩阵 w
    y.backward(w, retain_graph=False)

    return y.detach(), inputs.grad.data.detach()


# Note that SE3 is ordered as (translation, rotation)
def get_points_to_pose_jacobian_se3(points):
    """
    :param points: Transformed points y = Tx = Rx + t, T in SE(3)
    :return: batched Jacobian of transformed points y wrt pose T using Lie Algebra (left perturbation)
    """
    n = points.shape[0]
    eye = torch.eye(3).view(1, 3, 3)
    batch_eye = eye.repeat(n, 1, 1).cuda()
    zero = torch.zeros(n).cuda()
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    negate_hat = torch.stack(
        [torch.stack([zero, -z, y], dim=-1),
         torch.stack([z, zero, -x], dim=-1),
         torch.stack([-y, x, zero], dim=-1)],
        dim=-1
    )

    return torch.cat((batch_eye, negate_hat), dim=-1)


def exp_se3(x):
    """
    :param x: Cartesian vector of Lie Algebra se(3)
    :return: exponential map of x
    """
    v = x[:3]  # translation
    w = x[3:6]  # rotation
    w_hat = torch.tensor([[0., -w[2], w[1]],
                          [w[2], 0., -w[0]],
                          [-w[1], w[0], 0.]])
    w_hat_second = torch.mm(w_hat, w_hat)

    theta = torch.norm(w)
    theta_2 = theta ** 2
    theta_3 = theta ** 3
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    eye_3 = torch.eye(3)

    eps = 1e-8

    if theta <= eps:
        e_w = eye_3
        j = eye_3
    else:
        e_w = eye_3 + w_hat * sin_theta / theta + w_hat_second * (1. - cos_theta) / theta_2
        k1 = (1 - cos_theta) / theta_2
        k2 = (theta - sin_theta) / theta_3
        j = eye_3 + k1 * w_hat + k2 * w_hat_second

    rst = torch.eye(4)
    rst[:3, :3] = e_w
    rst[:3, 3] = torch.mv(j, v)

    return rst


def get_points_to_pose_jacobian_sim3(points):
    """
    :param points: Transformed points x = Ty = Ry + t, T in Sim(3)
    :return: batched Jacobian of transformed points wrt pose T
    """
    n = points.shape[0]
    eye = torch.eye(3).view(1, 3, 3)
    batch_eye = eye.repeat(n, 1, 1).cuda()
    zero = torch.zeros(n).cuda()
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    negate_hat = torch.stack(
        [torch.stack([zero, -z, y], dim=-1),
         torch.stack([z, zero, -x], dim=-1),
         torch.stack([-y, x, zero], dim=-1)],
        dim=-1
    )

    return torch.cat((batch_eye, negate_hat, points[..., None]), dim=-1)


def exp_sim3(x):
    """
    :param x: Cartesian vector of Lie Algebra se(3)
    :return: exponential map of x
    """
    v = x[:3]  # translation
    w = x[3:6]  # rotation
    s = x[6]  # scale

    w_hat = torch.tensor([[0., -w[2], w[1]],
                          [w[2], 0., -w[0]],
                          [-w[1], w[0], 0.]])
    w_hat_second = torch.mm(w_hat, w_hat)

    theta = torch.norm(w)
    theta_2 = theta ** 2
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    e_s = torch.exp(s)
    s_2 = s ** 2
    eye_3 = torch.eye(3)

    eps = 1e-8
    if theta <= 1e-8:
        if s == 0:
            e_w = eye_3
            j = eye_3
        else:
            e_w = eye_3
            c = (e_s - 1.) / s
            j = c * eye_3
    else:
        e_w = eye_3 + w_hat * sin_theta / theta + w_hat_second * (1. - cos_theta) / theta_2
        a = e_s * sin_theta
        b = e_s * cos_theta
        c = 0. if s <= eps else (e_s - 1.) / s
        k_0 = c * eye_3
        k_1 = (a * s + (1 - b) * theta) / (s_2 + theta_2)
        k_2 = c - ((b - 1) * s + a * theta) / (s_2 + theta_2)
        j = k_0 + k_1 * w_hat / theta + k_2 * w_hat_second / theta_2

    rst = torch.eye(4)
    rst[:3, :3] = e_s * e_w
    rst[:3, 3] = torch.mv(j, v)

    return rst


def huber_norm_weights(x, b=0.02):
    """
    :param x: norm of residuals, torch.Tensor (N,)
    :param b: threshold
    :return: weight vector torch.Tensor (N, )
    """
    # x is residual norm
    res_norm = torch.zeros_like(x)
    res_norm[x <= b] = x[x <= b] ** 2
    res_norm[x > b] = 2 * b * x[x > b] - b ** 2
    x[x == 0] = 1.
    return torch.sqrt(res_norm) / x


def get_robust_res(res, b):
    """
    :param res: residual vectors
    :param b: threshold
    :return: residuals after applying huber norm
    """
    # print(res.shape[0])
    res = res.view(-1, 1, 1)
    res_norm = torch.abs(res)
    # print(res.shape[0])
    w = huber_norm_weights(res_norm, b=b)
    # print(w.shape[0])
    robust_res = w * res
    loss = torch.mean(robust_res ** 2)

    return robust_res, loss, w


def get_time():
    """
    :return: get timing statistics
    """
    torch.cuda.synchronize()
    return time.time()
