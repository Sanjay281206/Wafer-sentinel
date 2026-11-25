"""End-to-end: wafer map -> CNN -> patterns -> die stats -> yield impact -> root-cause report."""
import numpy as np
from .classes import PATTERNS
from .data import to_tensor_input
from .yield_calc import die_stats, estimate_yield_impact
from .llm import RootCauseLLM


class Sentinel:
    def __init__(self, ckpt=None, gguf=None, onnx_path=None, thresholds=None):
        self.session = self.model = None
        self.ckpt = ckpt
        self.onnx_path = onnx_path
        self.model_format = "onnx" if onnx_path else ("pytorch" if ckpt else "heuristic")
        self.model_loaded = False
        self.model_name = "WaferCNN" if ckpt or onnx_path else "Spatial heuristic"
        self.input_size = None
        self.thr = thresholds or [0.5] * len(PATTERNS)
        if onnx_path:
            import onnxruntime as ort
            self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self.model_loaded = True
            self.input_size = tuple(int(x) for x in self.session.get_inputs()[0].shape if isinstance(x, int))
        elif ckpt:
            import torch
            from .model import WaferCNN
            ck = torch.load(ckpt, map_location="cpu")
            self.model = WaferCNN(len(PATTERNS))
            self.model.load_state_dict(ck["state"])
            self.model.eval()
            self.thr = ck.get("thresholds", self.thr)
            self.input_size = int(ck.get("size", 64))
            self.model_loaded = True
        self.llm = RootCauseLLM(gguf)

    def predict_probs(self, m):
        x = to_tensor_input(m[None])
        if self.session is not None:
            logit = self.session.run(None, {self.session.get_inputs()[0].name: x})[0][0]
        elif self.model is not None:
            import torch
            with torch.no_grad(): logit = self.model(torch.tensor(x))[0].numpy()
        else:  # no model: heuristic stand-in so the demo runs end-to-end
            logit = self._heuristic(m)
        return 1 / (1 + np.exp(-logit))

    @staticmethod
    def _heuristic(m):
        from .yield_calc import die_stats
        z = die_stats(m)["zone_fail_rate"]
        l = np.full(len(PATTERNS), -4.0)
        if z["edge"] > .3 and z["center"] < .2: l[PATTERNS.index("Edge-Ring")] = 3
        if z["center"] > .25 and z["edge"] < .2: l[PATTERNS.index("Center")] = 3
        if z["mid"] > .22 and z["center"] < .2 and z["edge"] < .2: l[PATTERNS.index("Donut")] = 3
        return l

    def analyse(self, m, baseline_yield=0.95):
        import time
        t0 = time.perf_counter()
        p = self.predict_probs(m)
        inference_ms = (time.perf_counter() - t0) * 1000
        pats = [PATTERNS[i] for i in range(len(PATTERNS)) if p[i] > self.thr[i]]
        confs = {PATTERNS[i]: round(float(p[i]), 3) for i in range(len(PATTERNS)) if PATTERNS[i] in pats}
        stats = die_stats(m)
        impact = estimate_yield_impact(stats, baseline_yield)
        return {"patterns": pats, "confidence": confs, "probs": {k: round(float(v), 3) for k, v in zip(PATTERNS, p)},
                "stats": stats, "impact": impact, "inference_ms": round(inference_ms, 2),
                "model": {"name": self.model_name, "format": self.model_format,
                          "loaded": self.model_loaded, "input_size": self.input_size,
                          "inference_device": "ONNX Runtime CPU" if self.session else ("PyTorch CPU" if self.model else "Heuristic CPU")},
                "report": self.llm.explain(pats, confs, stats, impact)}
