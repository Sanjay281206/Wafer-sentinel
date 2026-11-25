"""Integer-only (hardware-faithful) version of the WaferCNN, in pure numpy.

Every step here is exactly what the FPGA accelerator and the C firmware do, so results must match bit for bit:
  input: one-hot die state (outside, pass, fail) as 0/1       (activation scale 1)
  conv:  int8 weights (per output channel) x uint8 activations -> int32 accumulator + int32 bias
  requantise: q = clamp((acc*m + 2^(s-1)) >> s, 0, 255)  with per-channel uint16 m and shift s (ReLU = the clamp at 0)
  maxpool 2x2 after the first three convs; global average pool = (sum of 64 + 32) >> 6; final linear layer -> int32 logits
BatchNorm is folded into the convolution before quantising.
"""
import os
import numpy as np

P = 32          # FPGA lanes (output channels computed in parallel)
LAYERS = [(3, 16, 64), (16, 32, 32), (32, 64, 16), (64, 64, 8)]   # (in ch, out ch, input size) per conv layer


# ---------------------------------------------------------------- float reference (BN folded)
def fold_bn(w, gamma, beta, mean, var, eps=1e-5):
    k = gamma / np.sqrt(var + eps)
    return (w * k[:, None, None, None]).astype(np.float32), (beta - mean * k).astype(np.float32)


def onehot(map_u8):
    m = np.asarray(map_u8)
    return np.stack([(m == c) for c in (0, 1, 2)]).astype(np.int32)


def im2col(x):
    """x (C,H,W) -> (H*W, C*9), zero padding 1, column order [c][ky][kx] (same as weight.reshape(oc,-1))."""
    C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(1, 2))      # (C,H,W,3,3)
    return win.transpose(1, 2, 0, 3, 4).reshape(H * W, C * 9)


def pool2(a):
    c, h, w = a.shape
    return a.reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


def float_forward(map_u8, folded, fc):
    a = onehot(map_u8).astype(np.float32)
    acts = []
    for i, (wf, bf) in enumerate(folded):
        oc = wf.shape[0]; h = a.shape[1]
        out = np.maximum(im2col(a) @ wf.reshape(oc, -1).T + bf, 0).reshape(h, h, oc).transpose(2, 0, 1)
        acts.append(out)
        a = pool2(out) if i < 3 else out
    gap = a.reshape(a.shape[0], -1).mean(1)
    return gap @ fc["w"].T + fc["b"], acts


# ---------------------------------------------------------------- quantisation
def calibrate(maps, folded, fc, pct=99.99):
    vals = [[] for _ in folded]
    for m in maps:
        _, acts = float_forward(m, folded, fc)
        for i, a in enumerate(acts):
            v = a[a > 0]
            if v.size:
                vals[i].append(np.percentile(v, pct))
    return [max(float(np.mean(v)) if v else 1.0, 1e-6) / 255.0 for v in vals]      # activation scale per layer


def mult_shift(M):
    """M (float, >=0) -> (m uint16, s) with M ~= m / 2^s, m in [32768, 65535]."""
    if M <= 0:
        return 0, 1
    s = int(np.floor(np.log2(65535.0 / M)))
    s = max(1, min(s, 47))
    m = int(min(round(M * 2.0 ** s), 65535))
    return m, s


def quantize(folded, fc, s_out):
    convs, s_in = [], 1.0
    for (wf, bf), so in zip(folded, s_out):
        oc = wf.shape[0]
        sw = np.maximum(np.abs(wf.reshape(oc, -1)).max(1) / 127.0, 1e-12)
        wq = np.clip(np.round(wf / sw[:, None, None, None]), -127, 127).astype(np.int8)
        bq = np.clip(np.round(bf / (s_in * sw)), -2 ** 31, 2 ** 31 - 1).astype(np.int64)
        ms = [mult_shift(s_in * s / so) for s in sw]
        convs.append({"wq": wq, "bq": bq, "m": np.array([a for a, _ in ms], np.int64),
                      "s": np.array([b for _, b in ms], np.int64)})
        s_in = so
    swf = np.maximum(np.abs(fc["w"]).max(1) / 127.0, 1e-12)
    return {"convs": convs,
            "fc_w": np.clip(np.round(fc["w"] / swf[:, None]), -127, 127).astype(np.int8),
            "fc_b": np.round(fc["b"] / (s_in * swf)).astype(np.int64),
            "logit_scale": (s_in * swf).tolist()}


# ---------------------------------------------------------------- integer forward (bit-exact reference)
def int_forward(map_u8, qp, return_fc_in=False):
    a = onehot(map_u8)
    for i, L in enumerate(qp["convs"]):
        oc = L["wq"].shape[0]; h = a.shape[1]
        acc = im2col(a) @ L["wq"].reshape(oc, -1).T.astype(np.int32)                  # int32
        t = acc.astype(np.int64) + L["bq"][None, :]
        v = (t * L["m"][None, :] + (np.int64(1) << (L["s"][None, :] - 1))) >> L["s"][None, :]
        q = np.clip(v, 0, 255).astype(np.int32).reshape(h, h, oc).transpose(2, 0, 1)
        a = pool2(q) if i < 3 else q
    gap = (a.reshape(a.shape[0], -1).sum(1) + 32) >> 6                                # 64 values -> /64 with rounding
    logits = gap.astype(np.int64) @ qp["fc_w"].T.astype(np.int64) + qp["fc_b"]
    return (logits.astype(np.int32), gap.astype(np.uint8)) if return_fc_in else logits.astype(np.int32)


# ---------------------------------------------------------------- exports for RTL and C
def _w32(x):
    return f"{int(x) & 0xFFFFFFFF:08x}"


def export_rtl(qp, outdir):
    """weights.hex: one 256-bit word per line (lane 0 = least significant byte); bias/mult/shift: one entry per lane."""
    os.makedirs(outdir, exist_ok=True)
    groups = []                                           # (weights [K, lanes], bias, mult, shift) per group of P lanes
    for L in qp["convs"]:
        oc = L["wq"].shape[0]
        for g in range(max(1, oc // P)):
            lo, hi = g * P, min((g + 1) * P, oc)
            W = np.zeros((L["wq"][0].size, P), np.int64); W[:, :hi - lo] = L["wq"][lo:hi].reshape(hi - lo, -1).T
            pad = lambda a, fill: np.concatenate([a[lo:hi], np.full(P - (hi - lo), fill, np.int64)])
            groups.append((W, pad(L["bq"], 0), pad(L["m"], 0), pad(L["s"], 1)))
    W = np.zeros((64, P), np.int64); W[:, :8] = qp["fc_w"].T.astype(np.int64)
    groups.append((W, np.concatenate([qp["fc_b"], np.zeros(P - 8, np.int64)]), np.zeros(P, np.int64), np.ones(P, np.int64)))
    with open(os.path.join(outdir, "weights.hex"), "w") as f:
        for Wg, *_ in groups:
            for row in Wg:
                f.write("".join(f"{int(v) & 0xFF:02x}" for v in row[::-1]) + "\n")
    for name, idx, fmt in [("bias.hex", 1, _w32), ("mult.hex", 2, lambda x: f"{int(x) & 0xFFFF:04x}"),
                           ("shift.hex", 3, lambda x: f"{int(x) & 0xFF:02x}")]:
        with open(os.path.join(outdir, name), "w") as f:
            for gr in groups:
                f.writelines(fmt(v) + "\n" for v in gr[idx])
    return {"words": sum(g[0].shape[0] for g in groups), "groups": len(groups)}


def export_map_hex(map_u8, path):
    """64x64 map (values 0..2) -> 1024 words of 4 pixels (pixel i at byte i%4 of word i//4)."""
    b = np.asarray(map_u8, np.uint8).reshape(-1).reshape(1024, 4).astype(np.uint32)
    words = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24)
    with open(path, "w") as f:
        f.writelines(f"{int(w):08x}\n" for w in words)


def export_c_header(qp, path):
    def arr(ctype, name, a):
        a = np.asarray(a).reshape(-1)
        return f"static const {ctype} {name}[{a.size}] = {{" + ",".join(str(int(v)) for v in a) + "};\n"
    with open(path, "w") as f:
        f.write("/* generated by make_int8_reference.py - do not edit */\n#include <stdint.h>\n")
        for i, L in enumerate(qp["convs"]):
            f.write(arr("int8_t", f"W{i}", L["wq"]) + arr("int32_t", f"B{i}", L["bq"]) +
                    arr("uint16_t", f"M{i}", L["m"]) + arr("uint8_t", f"S{i}", L["s"]))
        f.write(arr("int8_t", "WFC", qp["fc_w"]) + arr("int32_t", "BFC", qp["fc_b"]))
