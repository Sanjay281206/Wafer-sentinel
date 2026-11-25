"""Hand-written pattern -> process-cause knowledge base. This is the part YOU own;
the LLM only phrases and combines it. Edit freely, cite your sources in the report."""

KB = {
    "Center": {
        "causes": ["CMP over-polish or non-uniform slurry at wafer centre",
                   "Thermal / gas-flow non-uniformity at centre in CVD or etch",
                   "Spin-coat resist thickness issue at centre"],
        "action": "Check CMP pad/slurry profile and chamber centre-zone temperature; inspect resist coat recipe.",
        "severity": "high",
        "indication": "A centre-concentrated defect population suggests a radial process gradient, rather than a random contamination event.",
        "processes": ["CMP", "CVD or etch", "Spin coating / lithography", "Chuck temperature control"]},
    "Donut": {
        "causes": ["Ring-shaped gas distribution or showerhead issue in etch/deposition",
                   "Radial temperature profile on the chuck (zone heater mismatch)",
                   "CMP zone-pressure imbalance"],
        "action": "Review chuck zone temperatures and showerhead flow; compare against last PM log.",
        "severity": "high",
        "indication": "A mid-radius ring indicates a radial or annular process non-uniformity around the wafer, often described as a donut-shaped defect.",
        "processes": ["Etch or deposition", "Chuck-zone thermal control", "Showerhead / gas distribution", "CMP pressure control"]},
    "Edge-Ring": {
        "causes": ["Etch or deposition non-uniformity at the wafer edge",
                   "Edge bead removal (EBR) / resist edge issue",
                   "Edge clamp-ring or focus-ring wear"],
        "action": "Inspect focus/clamp ring wear and EBR settings; check edge exclusion and chamber edge gas flow.",
        "severity": "medium",
        "indication": "A strong edge-ring pattern indicates wafer-boundary process variation, where edge exclusion, tool hardware and local gas or temperature gradients can dominate die loss.",
        "processes": ["Etch or deposition", "Edge bead removal", "Lithography edge process", "Focus-ring / clamp-ring maintenance"]},
    "Edge-Loc": {
        "causes": ["Localised edge handling damage or clamp contact",
                   "Local etch loading effect near the edge",
                   "Wafer-edge particle contamination"],
        "action": "Check wafer handler / clamp at that azimuth; run edge particle scan.",
        "severity": "medium",
        "indication": "Failures concentrated at one edge location suggest a local azimuth-dependent event rather than uniform wafer-boundary variation.",
        "processes": ["Wafer handling", "Clamp or end-effector contact", "Local edge etch loading", "Particle inspection"]},
    "Loc": {
        "causes": ["Localised particle contamination or photomask defect (repeating reticle field)",
                   "Local lithography defocus / hot-spot",
                   "Localised implant or contamination event"],
        "action": "Check if the cluster repeats at the same die position across wafers (reticle defect); run particle inspection.",
        "severity": "medium",
        "indication": "One spatial cluster suggests a local defect source, such as a particle, reticle field, focus excursion or contamination hotspot.",
        "processes": ["Particle control", "Lithography / photomask", "Implant", "Local contamination inspection"]},
    "Near-full": {
        "causes": ["Catastrophic process excursion (wrong recipe, gas or chemical issue)",
                   "Probe card / tester setup error (contact failure)",
                   "Wrong wafer or lot processed"],
        "action": "Hold the lot; verify tester/probe-card setup first, then audit the recipe and tool log.",
        "severity": "critical",
        "indication": "A near-full failure population may indicate a catastrophic process excursion or measurement/setup failure, not a normal spatial defect signature.",
        "processes": ["Recipe and chamber audit", "Electrical test / probe card", "Metrology verification", "Lot traceability"]},
    "Scratch": {
        "causes": ["Mechanical scratch from handling, CMP pad debris or cassette contact",
                   "Robot-blade or end-effector damage"],
        "action": "Inspect CMP consumables and wafer transfer path; check cassette slots for the lot.",
        "severity": "high",
        "indication": "A linear scratch suggests a mechanical transfer or process-consumption trail, rather than a radial process gradient.",
        "processes": ["Wafer handling", "CMP pad and consumables", "Cassette / FOUP", "Robot end-effector"]},
    "Random": {
        "causes": ["Random particle defects / elevated background defect density",
                   "Cleanroom contamination or chemical purity issue"],
        "action": "Check defect density trend (Poisson/negative-binomial yield model) and cleanroom particle counts.",
        "severity": "low",
        "indication": "Random, spatially dispersed failures are consistent with particle or cleanroom contamination, although a spatially broad process issue can also appear random.",
        "processes": ["Cleanroom particle control", "Chemical purity", "Metrology cleanliness", "Process-defect trend monitoring"]},
}
NONE_ENTRY = {"causes": ["No systematic pattern detected"], "action": "Continue monitoring; no action.", "severity": "none"}


def lookup(patterns):
    if not patterns:
        return {"none": NONE_ENTRY}
    return {p: KB[p] for p in patterns}
