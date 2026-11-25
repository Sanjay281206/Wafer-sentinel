"""Defect taxonomy. WM-811K has 8 single patterns + 'none'.
MixedWM38 = 8 basic patterns + mixtures + normal; we model it as multi-label
over the 8 basic patterns, so mixtures come for free."""

PATTERNS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Scratch", "Random"]
P2I = {p: i for i, p in enumerate(PATTERNS)}
NUM_CLASSES = len(PATTERNS)  # "none" = all labels zero

WM811K_MAP = {p: p for p in PATTERNS}
WM811K_MAP["none"] = None
