import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat


def grad_a(y):
    sigma = np.array([0.1, 0.1])[None]
    mu = np.array([-15, -5])[None]
    y = y[:, None]
    grad = -(y - mu) / (sigma**2)
    prob = np.exp(-0.5 * ((y - mu) / sigma) ** 2) / np.sqrt(2 * np.pi) / sigma
    prob = np.sum(prob, -1)[:, None]
    grad = np.mean(grad * prob, -1)
    return grad


def grad_b(y):
    sigma = np.array([0.1])[None]
    mu = np.array([0.25])[None]
    y = y[:, None]
    grad = -(y - mu) / (sigma**2)
    grad = np.mean(grad, -1)
    return grad


def grad_x(y, x):
    ga = grad_a(y)
    gb = grad_b(y)
    ab = np.array([-1, 1])[None]
    x = x[:, None]
    w = np.abs(x - ab) #distance from -1 to 1 fractional percentage
    w = w / np.sum(w, -1)[:, None]
    gx = ga * w[:, 1] + gb * w[:, 0] #weighted
    return gx


step_size = 0.005
nx = 100
ny = 50
noise_scale = 1
x = np.linspace(-1, 1, nx)
x = repeat(x, "nx -> nx ny", ny=ny).flatten()
y = np.random.randn(nx, ny).flatten()
for t in range(10000):
    gy = grad_x(y, x) + np.sqrt(2) * noise_scale * np.random.randn(len(y))
    y = y + step_size * gy
plt.scatter(x, y)
plt.savefig("temp.png")
# plt.show()