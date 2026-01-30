import numpy as np

# ==================== 工具函数 ====================

def affine_transform(src, dst):
    assert src.shape == dst.shape
    N, D = src.shape
    assert N >= D
    src_hom = np.hstack([src, np.ones((N, 1))])
    params = np.linalg.lstsq(src_hom, dst, rcond=None)[0]
    A = params[:D].T
    t = params[D]
    transformed_src = src @ A.T + t
    return A, t, transformed_src


def umeyama(src, dst, with_scale=True):
    assert src.shape == dst.shape
    N, D = src.shape
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = dst_centered.T @ src_centered / N
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    if with_scale:
        var_src = np.var(src_centered, axis=0, ddof=0).sum()
        s = np.trace(np.diag(S)) / var_src
    else:
        s = 1.0
    t = dst_mean - s * R @ src_mean
    return R, t, s
