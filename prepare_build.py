"""Check the export and prepare a self-contained Vivado build folder for the ZCU104.

  python scripts/prepare_build.py --export fpga_export --out build

What it does
  1. Checks weights.hex / bias.hex / mult.hex / shift.hex have exactly the sizes the RTL expects (1963 x 256-bit weight
     words, 7 x 32 lane entries). A wrong size means the model is not the width-16 WaferCNN the accelerator is built for.
  2. Copies the RTL and the .hex files into <out>/rtl and <out>/hex and rewrites the RTL's default file names to ABSOLUTE
     paths of those copies. Vivado synthesis resolves $readmemh() relative to its own run folders; with absolute paths the
     ROM contents can never silently end up empty (the classic 'accelerator outputs garbage' bug).
  3. Prints the Vivado command to run next.
"""
import argparse, os, re, shutil, sys

EXPECT = {"weights.hex": (1963, 64), "bias.hex": (224, 8), "mult.hex": (224, 4), "shift.hex": (224, 2)}   # lines, hex digits per line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="fpga_export"); ap.add_argument("--out", default="build")
    ap.add_argument("--rtl", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rtl"))
    ap.add_argument("--clock_mhz", type=int, default=100)
    a = ap.parse_args()
    export, out, rtl = os.path.abspath(a.export), os.path.abspath(a.out), os.path.abspath(a.rtl)

    for name, (lines, width) in EXPECT.items():
        p = os.path.join(export, name)
        if not os.path.exists(p):
            sys.exit(f"missing {p}: run  python -m wafersentinel.make_int8_reference --ckpt runs/combined.pt")
        rows = [l.strip() for l in open(p) if l.strip()]
        if len(rows) != lines or any(len(r) != width or not re.fullmatch(r"[0-9a-fA-F]+", r) for r in rows):
            sys.exit(f"{name}: expected {lines} lines of {width} hex digits, found {len(rows)} lines "
                     f"(first line length {len(rows[0]) if rows else 0}). Is this the width-16 WaferCNN?")
        print(f"ok  {name}: {len(rows)} lines x {width} hex digits")
    if not os.path.isdir(os.path.join(export, "golden")):
        print("warning: no golden/ folder (needed for the simulation checks)")

    os.makedirs(os.path.join(out, "rtl"), exist_ok=True); os.makedirs(os.path.join(out, "hex"), exist_ok=True)
    paths = {}
    for name in EXPECT:
        shutil.copy(os.path.join(export, name), os.path.join(out, "hex", name))
        paths[name] = os.path.join(out, "hex", name).replace("\\", "/")
    for f in ("wafer_cnn_accel.v", "wafer_cnn_axil.v"):
        s = open(os.path.join(rtl, f)).read()
        for key, name in (("WFILE", "weights.hex"), ("BFILE", "bias.hex"), ("MFILE", "mult.hex"), ("SFILE", "shift.hex")):
            s = s.replace(f'{key} = "{name}"', f'{key} = "{paths[name]}"')
        open(os.path.join(out, "rtl", f), "w").write(s)
        n = sum(1 for k in paths.values() if k in s)
        print(f"ok  rtl/{f}: {n} absolute ROM paths set" if n else f"ok  rtl/{f}: copied (the core gets its paths from the wrapper)")
    print(f"\nBuild folder ready: {out}\nNext (Vivado command prompt, in the package root):\n"
          f"  vivado -mode batch -source vivado/build_zcu104.tcl -tclargs \"{out.replace(chr(92), '/')}\" {a.clock_mhz} 8")


if __name__ == "__main__":
    main()
