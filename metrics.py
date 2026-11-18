"""Chamfer distance and the scores built on it.

Chamfer distance (CD) between two clouds A and B: for every point of A take the squared distance to
the closest point of B and average; do the same from B to A; add the two. It is 0 when the clouds
are the same set of points, whatever their order.

Scores for a set of generated clouds G against a set of real reference clouds R (same size):
  MMD    for every real cloud, the CD to its closest generated cloud, averaged.
         Lower = the real clouds are well matched (quality).
  COV    share of real clouds that are the closest real cloud of at least one generated cloud.
         Higher = more variety, 1.0 is best.
  1-NNA  put G and R together; for each cloud, is its closest other cloud from its own set?
         The share of yes answers. 0.5 is best (the two sets cannot be told apart), 1.0 is worst.
  nearest_cd  for every generated cloud, the CD to its closest real cloud, averaged.
  shapes      for every generated cloud, the shape of its closest real cloud, counted.

For overfitting (G against the few training clouds): nearest_cd, the worst one, and how many of the
training clouds are the closest match of at least one generated cloud.
"""

import torch


def pairwise_chamfer(A, B, chunk=32):
    """(len(A), len(B)) matrix of Chamfer distances between clouds A (a, N, 2) and B (b, M, 2)."""
    out = torch.empty(len(A), len(B), device=A.device)
    for i in range(0, len(A), chunk):
        a = A[i:i + chunk]
        for j in range(0, len(B), chunk):
            b = B[j:j + chunk]
            a_rep = a[:, None].expand(-1, len(b), -1, -1).reshape(-1, *a.shape[1:])
            b_rep = b[None].expand(len(a), -1, -1, -1).reshape(-1, *b.shape[1:])
            d = torch.cdist(a_rep, b_rep) ** 2  # (pairs, N, M) squared distances
            cd = d.min(dim=2).values.mean(dim=1) + d.min(dim=1).values.mean(dim=1)
            out[i:i + len(a), j:j + len(b)] = cd.view(len(a), len(b))
    return out


def chamfer(a, b):
    return pairwise_chamfer(a[None], b[None])[0, 0]


def generation_scores(generated, reference, reference_labels=None, shape_names=None):
    gr = pairwise_chamfer(generated, reference)
    gg = pairwise_chamfer(generated, generated).fill_diagonal_(float("inf"))
    rr = pairwise_chamfer(reference, reference).fill_diagonal_(float("inf"))

    generated_right = gg.min(dim=1).values < gr.min(dim=1).values
    reference_right = rr.min(dim=1).values < gr.min(dim=0).values
    scores = {
        "mmd": gr.min(dim=0).values.mean().item(),
        "cov": gr.argmin(dim=1).unique().numel() / len(reference),
        "nna": (generated_right.sum() + reference_right.sum()).item() / (len(generated) + len(reference)),
        "nearest_cd": gr.min(dim=1).values.mean().item(),
    }
    if reference_labels is not None:
        closest = reference_labels.to(gr.device)[gr.argmin(dim=1)]
        names = shape_names or [str(k) for k in range(int(reference_labels.max()) + 1)]
        scores["shapes"] = {name: int((closest == k).sum()) for k, name in enumerate(names)}
    return scores


def overfit_scores(generated, train_clouds):
    gt = pairwise_chamfer(generated, train_clouds)
    nearest, closest = gt.min(dim=1)
    return {
        "nearest_cd": nearest.mean().item(),
        "worst_nearest_cd": nearest.max().item(),
        "covered": closest.unique().numel(),
    }
