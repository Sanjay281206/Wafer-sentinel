"""Grad-CAM for the last conv block -> heatmap overlay (explainability for the demo)."""
import numpy as np
import torch
import torch.nn.functional as F


def gradcam(model, x, class_idx):
    model.eval()
    feats = {}
    h = model.features[-1].register_forward_hook(lambda m, i, o: feats.setdefault("a", o))
    x = x.clone().requires_grad_(True)
    logits = model(x)
    h.remove()
    a = feats["a"]
    g = torch.autograd.grad(logits[0, class_idx], a)[0]
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().numpy()
