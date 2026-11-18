"""Quick checks for ptv3.py. Run this before training PTv3.

    python check_ptv3.py
"""

import torch

from flow_matching import flow_matching_loss
from ptv3 import PointTransformerV3, hilbert, serial_orders, undo, take, z_order
from shapes import make_dataset


def check(name, passed, detail=""):
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}", flush=True)
    if not passed:
        raise SystemExit(f"Check failed: {name}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"Device: {device}")

    check("Z-order code of cell (3, 5) is 39", z_order(torch.tensor(3), torch.tensor(5)).item() == 39)

    side = 16
    ys, xs = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    xs, ys = xs.flatten(), ys.flatten()
    codes = hilbert(xs, ys, bits=4)
    path = codes.argsort()
    steps = (xs[path].diff().abs() + ys[path].diff().abs())
    check("Hilbert curve visits every cell once", codes.sort().values.equal(torch.arange(side * side)))
    check("Hilbert curve only moves to neighbouring cells", (steps == 1).all().item())

    clouds = make_dataset(8, seed=0)[0].to(device)
    orders = serial_orders(clouds)
    valid = all(order.sort(dim=1).values.equal(torch.arange(256, device=device).expand(8, -1)) for order in orders)
    check("Each of the 4 curves sorts every point exactly once", len(orders) == 4 and valid)
    check("undo reverses take", undo(take(clouds, orders[1]), orders[1]).equal(clouds))

    model = PointTransformerV3().to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    t = torch.rand(8, device=device)
    output = model(clouds, t)
    check("Output has one 2D vector per point", output.shape == clouds.shape, str(tuple(output.shape)))
    check("Output starts at zero (adaLN-Zero)", output.abs().max().item() == 0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(6):
        optimizer.zero_grad(set_to_none=True)
        loss = flow_matching_loss(model, clouds, "v")
        loss.backward()
        if step < 5:
            optimizer.step()
    missing = [name for name, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    check("Every parameter receives a gradient", not missing, ", ".join(missing[:5]))

    model.eval()
    with torch.no_grad():
        noisy = torch.randn_like(clouds)
        output = model(noisy, t)
        shuffle = torch.randperm(256, device=device)
        shuffled = model(noisy[:, shuffle], t)
        error = (shuffled - output[:, shuffle]).abs().max().item()
        check("Shuffling the input points shuffles the output", error < 1e-3, f"largest difference {error:.6f}")
        changed = (model(noisy, 1 - t) - output).abs().max().item()
        check("The time t changes the output", changed > 1e-4, f"largest change {changed:.6f}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
