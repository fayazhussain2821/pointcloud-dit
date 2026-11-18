"""Flow matching for point clouds: the same method as for the images, on point coordinates.

Conventions: t = 0 is pure noise, t = 1 is the real cloud. A cloud is a (num_points, 2) tensor.

    x_t = (1 - t) * noise + t * cloud        velocity v = cloud - noise

    prediction | the model outputs | converted to a velocity
    v          | the velocity      | output
    x          | the clean cloud   | (output - x_t) / (1 - t)
    eps        | the noise         | (x_t - output) / t

The loss is always measured on the velocity, so x, eps and v are directly comparable.
"""

import torch
import torch.nn.functional as F


def per_sample(t, like):
    """Reshape t (B,) so it multiplies a batch shaped like `like` (B, ...)."""
    return t.view(-1, *([1] * (like.dim() - 1)))


def add_noise(clean, noise, t):
    t = per_sample(t, clean)
    return (1 - t) * noise + t * clean


def to_velocity(output, x_t, t, prediction, clip=0.05):
    """Convert the model's output into a velocity. `clip` stops division by zero near t=0 and t=1."""
    t = per_sample(t, x_t)
    if prediction == "v":
        return output
    if prediction == "x":
        return (output - x_t) / (1 - t).clamp(min=clip)
    if prediction == "eps":
        return (x_t - output) / t.clamp(min=clip)
    raise ValueError(f"Unknown prediction: {prediction}")


def flow_matching_loss(model, clean, prediction, clip=0.05):
    noise = torch.randn_like(clean)
    t = torch.rand(len(clean), device=clean.device)  # uniform t in [0, 1]
    x_t = add_noise(clean, noise, t)
    velocity = to_velocity(model(x_t, t), x_t, t, prediction, clip)
    return F.mse_loss(velocity, clean - noise)


@torch.no_grad()
def generate(model, noise, prediction, steps=50, method="euler", clip=0.05):
    """Start from noise at t=0 and follow the predicted velocity to t=1.

    euler: one model call per step. heun: two calls per step (start and end of the step, averaged),
    more accurate; its last step is plain Euler because x-prediction divides by (1 - t) = 0 at t=1.
    """
    if method not in ("euler", "heun"):
        raise ValueError(f"Unknown sampler: {method}")
    x = noise
    times = torch.linspace(0, 1, steps + 1, device=noise.device)
    for i in range(steps):
        t = times[i].expand(len(x))
        dt = times[i + 1] - times[i]
        v = to_velocity(model(x, t), x, t, prediction, clip)
        if method == "heun" and i < steps - 1:
            t_next = times[i + 1].expand(len(x))
            x_next = x + dt * v
            v = (v + to_velocity(model(x_next, t_next), x_next, t_next, prediction, clip)) / 2
        x = x + dt * v
    return x
