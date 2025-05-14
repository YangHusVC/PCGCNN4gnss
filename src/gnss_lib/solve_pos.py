########################################################################
# Author(s):    Shubh Gupta
# Date:         21 September 2021
# Desc:         Code to find Newton Raphson PNT solution
########################################################################
import numpy as np
from . import constants

def newton_raphson(f, df, x0, W=None, e=1e-3, lam=1.):
    delta_x = np.ones_like(x0)
    if W is None:
        W = np.eye(len(f(x0)))

    while np.sum(np.abs(delta_x))>e:
        delta_x = lam*(np.linalg.pinv(W@df(x0)) @ (W@f(x0)))
        x0 = x0 - delta_x
    return x0, np.linalg.norm(f(x0))

def solve_pos(prange, X, Y, Z, B,cn0, e=1e-3):
  if len(prange)<4:
    return np.empty(4)
  x, y, z, cdt = -2694570., -4296490., 3854814., 4.#100., 100., 100., 0.
  #[-2694536.4616140425, -4296508.6247927, 3854823.440095416, 4.6562162820876365]
  #[-2694570.053971853, -4296490.154348378, 3854814.5274612415, 4.703347154380883] T
  if cn0 is not None:
    sigma = 10 ** (-cn0/30)  # C/N0转换为噪声标准差
    W = np.diag(1.0/sigma)   # 权重矩阵
  else:
    W = np.eye(len(prange))

  def f(vars):
    x, y, z, cdt = list(vars)
    tilde_prange = np.sqrt((x - X)**2 + (y - Y)**2 + (z - Z)**2)
    _prange = tilde_prange + cdt - B
    delta_prange = prange-_prange
    return delta_prange

  def df(vars):
    x, y, z, cdt = list(vars)
    tilde_prange = np.sqrt((x - X)**2 + (y - Y)**2 + (z - Z)**2)
    _prange = tilde_prange + cdt - B
    delta_prange = prange-_prange
    derivatives = np.zeros((len(prange), 4))
    derivatives[:, 0] = -(x - X)/tilde_prange
    derivatives[:, 1] = -(y - Y)/tilde_prange
    derivatives[:, 2] = -(z - Z)/tilde_prange
    derivatives[:, 3] = -1
    return derivatives
  
  x0 = np.array([x, y, z, cdt])
  x_fix, res_err = newton_raphson(f, df, x0, W, e=e)
  #x_fix[-1] = x_fix[-1]*1e6/constants.LIGHTSPEED

  return x_fix

