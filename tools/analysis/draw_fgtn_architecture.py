"""Publication architecture diagram: Hybrid FGTN-Mamba neck + Video Swin-T on AVA.

Faithful to projects/fgtn_ava/fgtn_neck.py and the ava22 config:
(a) full detection pipeline, (b) zoomed HybridFGTNBlock with every sub-op.

Usage:
    python tools/analysis/draw_fgtn_architecture.py --out figures/
"""

import argparse
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

# palette: one hue per subsystem (light fill, saturated border)
C_BACK = dict(fc="#E8F1FA", ec="#2B6CB0")
C_NECK = dict(fc="#FFF8E7", ec="#B7791F")
C_MAMBA = dict(fc="#E9F7EC", ec="#2F855A")
C_ATTN = dict(fc="#FFF0E5", ec="#C05621")
C_GATE = dict(fc="#F1EEFB", ec="#6B46C1")
C_OP = dict(fc="#F7F7F7", ec="#4A5568")
C_HEAD = dict(fc="#FBE9F1", ec="#B83280")
ARROW = "#333333"

SHADOW = [pe.SimplePatchShadow(offset=(1.6, -1.6), alpha=0.18, rho=0.4), pe.Normal()]


def box(ax, cx, cy, w, h, text, colors, fs=10, lw=1.6, weight="normal", shadow=True):
    """Rounded box centered at (cx, cy)."""
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.35",
                       fc=colors["fc"], ec=colors["ec"], lw=lw, zorder=4)
    if shadow:
        p.set_path_effects(SHADOW)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, weight=weight, zorder=5)
    return dict(cx=cx, cy=cy, w=w, h=h)


def container(ax, x0, y0, x1, y1, colors, title=None, fs=11):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.5",
                                fc=colors["fc"], ec=colors["ec"], lw=1.6, linestyle=(0, (5, 3)), zorder=1))
    if title:
        ax.text((x0 + x1) / 2, y1 - 2.6, title, ha="center", va="center",
                fontsize=fs, weight="bold", color=colors["ec"], zorder=5)


def arrow(ax, p0, p1, color=ARROW, lw=1.6, ls="-", z=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw, linestyle=ls, zorder=z))


def varrow(ax, a, b, **kw):  # bottom-up between boxes
    arrow(ax, (a["cx"], a["cy"] + a["h"] / 2 + 0.4), (b["cx"], b["cy"] - b["h"] / 2 - 0.45), **kw)


def harrow(ax, a, b, **kw):  # left-to-right between boxes
    arrow(ax, (a["cx"] + a["w"] / 2 + 0.4, a["cy"]), (b["cx"] - b["w"] / 2 - 0.45, b["cy"]), **kw)


def opnode(ax, x, y, sym, r=1.9, fs=13):
    c = Circle((x, y), r, fc="white", ec=ARROW, lw=1.6, zorder=6)
    c.set_path_effects(SHADOW)
    ax.add_patch(c)
    ax.text(x, y, sym, ha="center", va="center", fontsize=fs, zorder=7)
    return dict(cx=x, cy=y, w=2 * r, h=2 * r)


def draw(out_dir, title):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    fig, ax = plt.subplots(figsize=(12.5, 14))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 141)
    ax.axis("off")
    ax.text(60, 139.5, title, ha="center", va="center", fontsize=17, weight="bold")

    # ================= (a) pipeline =================
    ax.text(2, 135.5, "(a)  Detection pipeline on AVA v2.2", fontsize=15, weight="bold")
    ya = 122  # row center

    b_in = box(ax, 8, ya, 14.5, 11, "Input clip\n$3{\\times}32{\\times}256^2$\n32×2×1 sampling", C_OP, fs=8.6)

    container(ax, 16.5, ya - 9.5, 64.5, ya + 9.5, dict(fc="#F4F9FE", ec=C_BACK["ec"]),
              title="Video Swin-T  (Kinetics-400 pre-trained)", fs=10.5)
    st = []
    labels = ["Patch\nEmbed 3D\n2×4×4", "Stage 1\n×2\nC=96", "Stage 2\n×2\nC=192",
              "Stage 3\n×6\nC=384", "Stage 4\n×2\nC=768"]
    for i, lb in enumerate(labels):
        st.append(box(ax, 22.5 + i * 9.2, ya - 0.6, 8.2, 9.5, lb, C_BACK, fs=8.4))
    for a, b in zip(st, st[1:]):
        harrow(ax, a, b, lw=1.3)
    ax.text(40.5, ya - 6.6, "3D W-MSA / SW-MSA,  window (8, 7, 7)", fontsize=8.5,
            ha="center", style="italic", color=C_BACK["ec"])
    harrow(ax, b_in, dict(cx=18.8, cy=ya - 0.6, w=0, h=0))

    b_neck = box(ax, 76, ya, 16, 12, "FGTN Neck\n2 × Hybrid\nFGTN-Mamba\nblock", C_NECK, fs=10.5,
                 lw=2.2, weight="bold")
    arrow(ax, (65.6, ya - 0.6), (67.3, ya - 0.6))
    ax.text(70.5, ya + 2.6, "$F_4$", fontsize=11, ha="center")
    ax.text(70.5, ya - 3.4, "768×16\n×8×8", fontsize=8, ha="center")

    b_roi = box(ax, 95, ya, 13, 11, "RoIAlign3D\ntemporal pool\noutput 8×8", C_OP, fs=9)
    b_head = box(ax, 111.5, ya, 13, 11, "BBoxHead\n81 classes\nsigmoid\nmulti-label", C_HEAD, fs=9)
    harrow(ax, b_neck, b_roi)
    harrow(ax, b_roi, b_head)
    b_prop = box(ax, 95, ya + 10.5, 15, 5.5, "person proposals\nFAIR, recall 93.9%", C_OP, fs=8, shadow=False)
    varrow(ax, dict(cx=95, cy=ya + 2, w=0, h=7), b_prop)  # placeholder, replaced below
    # (arrow direction: proposals -> RoIAlign)
    ax.patches[-1].remove()
    arrow(ax, (95, ya + 7.4), (95, ya + 6))

    # zoom guides: neck -> panel (b)
    ax.plot([68, 2.5], [ya - 6.3, 110.4], color=C_NECK["ec"], lw=1.1, ls=":", zorder=0)
    ax.plot([84, 117.5], [ya - 6.3, 110.4], color=C_NECK["ec"], lw=1.1, ls=":", zorder=0)

    # ================= (b) block detail =================
    ax.add_patch(FancyBboxPatch((2.5, 2), 115, 108, boxstyle="round,pad=0.6",
                                fc="#FFFDF7", ec=C_NECK["ec"], lw=2, zorder=0))
    ax.text(6, 107.3, "(b)  Hybrid FGTN-Mamba block", fontsize=15, weight="bold")
    ax.text(6, 104.2, "pre-norm residual;  reshape $(B,C,T,H,W)\\rightarrow(B{\\cdot}H{\\cdot}W,\\,C,\\,T)$ — "
                      "one temporal sequence per spatial location", fontsize=9.5, style="italic")

    xm, xg, xa = 31, 60, 89  # column centers

    b_x = box(ax, xg, 8, 40, 5.5, "$x \\in \\mathbb{R}^{(B\\cdot HW)\\times C\\times T}$     $C{=}768$,  $T{=}16$",
              C_OP, fs=10.5)
    b_ln = box(ax, xg, 17.5, 20, 5, "LayerNorm", C_OP, fs=10.5)
    varrow(ax, b_x, b_ln)

    y_rail = 23.5
    ax.plot([xm, xa], [y_rail, y_rail], color=ARROW, lw=1.6, zorder=3)
    ax.plot([xg, xg], [b_ln["cy"] + 2.5, y_rail], color=ARROW, lw=1.6, zorder=3)

    # ---- Bi-Mamba branch (left) ----
    container(ax, 13, 25.5, 49, 81, dict(fc="#F4FBF5", ec=C_MAMBA["ec"]),
              title="Bi-Mamba  (bimamba v2) — global, $O(T)$", fs=9.5)
    ax.text(xm, 74.4, "$h_t = e^{\\Delta A}h_{t-1} + \\Delta B\\,x_t$,   $y_t = C\\,h_t + D\\,x_t$",
            fontsize=9.5, ha="center", color=C_MAMBA["ec"])
    ax.text(xm, 71.4, "$\\Delta{=}\\mathrm{softplus}(\\mathrm{dt\\_proj})$;   $B, C = \\mathrm{x\\_proj}(x)$ "
                      "(input-dependent)", fontsize=8, ha="center", color=C_MAMBA["ec"])

    b_inp = box(ax, xm, 30.5, 30, 5, "in_proj:  Linear $C \\rightarrow 2C'$,  split $(x, z)$", C_MAMBA, fs=9)
    arrow(ax, (xm, y_rail), (xm, b_inp["cy"] - b_inp["h"] / 2 - 0.45))

    xf, xb = xm - 8.5, xm + 8.5
    b_cf = box(ax, xf, 39, 14.5, 6.5, "causal DWConv1d\n$k{=}4$  +  SiLU", C_MAMBA, fs=8.5)
    b_cb = box(ax, xb, 39, 14.5, 6.5, "causal DWConv1d\n$k{=}4$  +  SiLU", C_MAMBA, fs=8.5)
    b_sf = box(ax, xf, 49, 14.5, 7, "selective scan\nforward\n$t: 1 \\rightarrow T$", C_MAMBA, fs=8.5)
    b_sb = box(ax, xb, 49, 14.5, 7, "selective scan\nbackward\n$t: T \\rightarrow 1$ (flip)", C_MAMBA, fs=8.5)
    arrow(ax, (xf, b_inp["cy"] + 2.9), (xf, b_cf["cy"] - 3.7))
    arrow(ax, (xb, b_inp["cy"] + 2.9), (xb, b_cb["cy"] - 3.7))
    varrow(ax, b_cf, b_sf)
    varrow(ax, b_cb, b_sb)

    n_comb = opnode(ax, xm, 57.5, "+")
    arrow(ax, (xf, b_sf["cy"] + 3.9), (xm - 1.5, 56.4))
    arrow(ax, (xb, b_sb["cy"] + 3.9), (xm + 1.5, 56.4))

    n_zg = opnode(ax, xm, 62.5, "×")
    arrow(ax, (xm, 59.4), (xm, 60.2))
    ax.text(xm + 2.8, 62.5, "SiLU$(z)$", fontsize=9, va="center")
    b_op = box(ax, xm, 67.8, 22, 4.6, "out_proj:  Linear $C' \\rightarrow C$", C_MAMBA, fs=9)
    arrow(ax, (xm, 64.4), (xm, 65.1))

    # ---- Windowed MHSA branch (right) ----
    container(ax, 71, 25.5, 107, 81, dict(fc="#FFF7F0", ec=C_ATTN["ec"]),
              title="Windowed MHSA — local, $W{=}5$", fs=10)
    ax.text(xa, 73, "attend  iff  $|i-j| \\leq 2$", fontsize=9.5, ha="center", color=C_ATTN["ec"])

    b_qkv = box(ax, xa, 31.5, 30, 6, "QKV:  Linear $C \\rightarrow 3C$\n4 heads,  $d{=}192$", C_ATTN, fs=9)
    arrow(ax, (xa, y_rail), (xa, b_qkv["cy"] - b_qkv["h"] / 2 - 0.45))
    b_msk = box(ax, xa - 3.5, 42, 23, 7.5, "band mask $M$\nwindow $W{=}5$\ncentered per frame", C_ATTN, fs=9)
    varrow(ax, b_qkv, b_msk)
    # band-matrix glyph next to the mask box
    gx, gy, cell = 99.5, 38.6, 1.15
    for i in range(6):
        for j in range(6):
            on = abs(i - j) <= 2
            ax.add_patch(Rectangle((gx + j * cell, gy + (5 - i) * cell), cell * 0.88, cell * 0.88,
                                   fc=C_ATTN["ec"] if on else "#E8E8E8", ec="none", zorder=5))
    b_sdp = box(ax, xa, 55.5, 30, 6.5, "$\\mathrm{softmax}\\!\\left(QK^{\\top}/\\sqrt{d} + M\\right)V$", C_ATTN, fs=10)
    varrow(ax, b_msk, dict(cx=b_msk["cx"], cy=b_sdp["cy"], w=b_sdp["w"], h=b_sdp["h"]))
    b_prj = box(ax, xa, 67.8, 22, 4.6, "proj:  Linear $C \\rightarrow C$", C_ATTN, fs=9)
    varrow(ax, b_sdp, b_prj)

    # ---- gate (center) ----
    b_g1 = box(ax, xg, 33, 17, 5, "Conv1d 1×1\n$C \\rightarrow C$", C_GATE, fs=9)
    arrow(ax, (xg, y_rail), (xg, b_g1["cy"] - b_g1["h"] / 2 - 0.45))
    b_sg = box(ax, xg, 42, 13, 4.6, "sigmoid $\\sigma$", C_GATE, fs=9.5)
    varrow(ax, b_g1, b_sg)
    ax.text(62, 50, "$g \\in (0,1)^{C\\times T}$\nper-channel router", fontsize=9, ha="left",
            color=C_GATE["ec"], weight="bold")

    # ---- fusion ----
    y_mul = 86
    n_mm = opnode(ax, xm, y_mul, "×")
    n_ma = opnode(ax, xa, y_mul, "×")
    arrow(ax, (xm, b_op["cy"] + 2.7), (xm, y_mul - 2.3))
    arrow(ax, (xa, b_prj["cy"] + 2.7), (xa, y_mul - 2.3))
    # gate routing to the two multipliers
    ax.plot([xg, xg], [44.6, 53.5], color=C_GATE["ec"], lw=1.5, zorder=3)
    ax.plot([xg, xm + 3.6], [53.5, y_mul], color=C_GATE["ec"], lw=1.5, ls=(0, (4, 2.5)), zorder=3)
    ax.plot([xg, xa - 3.6], [53.5, y_mul], color=C_GATE["ec"], lw=1.5, ls=(0, (4, 2.5)), zorder=3)
    arrow(ax, (xm + 3.6, y_mul), (xm + 2.3, y_mul), color=C_GATE["ec"], lw=1.5)
    arrow(ax, (xa - 3.6, y_mul), (xa - 2.3, y_mul), color=C_GATE["ec"], lw=1.5)
    ax.text(xm + 6.2, y_mul + 2, "$g$", fontsize=11, color=C_GATE["ec"], style="italic")
    ax.text(xa - 8.7, y_mul + 2, "$1-g$", fontsize=11, color=C_GATE["ec"], style="italic")

    n_sum = opnode(ax, xg, 95, "+")
    arrow(ax, (xm, y_mul + 2.3), (xg - 1.6, 94))
    arrow(ax, (xa, y_mul + 2.3), (xg + 1.6, 94))
    ax.text(xg, 99.8, "$\\mathrm{out} = g \\odot x_{\\mathrm{mamba}} + (1-g) \\odot x_{\\mathrm{attn}}$",
            fontsize=11.5, ha="center", weight="bold")

    b_dp = box(ax, xg + 22, 95, 16, 4.6, "DropPath $p{=}0.1$", C_OP, fs=9)
    arrow(ax, (xg + 2.3, 95), (xg + 13.4, 95))
    n_res = opnode(ax, 103, 95, "+")
    arrow(ax, (xg + 30.7, 95), (100.6, 95))

    # residual skip: x -> right side -> res add
    ax.plot([b_x["cx"] + 20.4, 112.5, 112.5, 105.4], [8, 8, 95, 95], color=ARROW, lw=1.5, zorder=3)
    arrow(ax, (106.5, 95), (105.4, 95))
    ax.text(113.8, 50, "residual", fontsize=9.5, rotation=90, va="center")
    arrow(ax, (103, 97.4), (103, 99.6))
    ax.text(103, 101.6, "block output\n$(B{\\cdot}HW,\\,C,\\,T)$", fontsize=9.5, ha="center")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fgtn_architecture.{ext}")
    print(f"wrote {out}/fgtn_architecture.png + .pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="figures")
    parser.add_argument("--title", default="Hybrid FGTN-Mamba with Video Swin-T")
    args = parser.parse_args()
    draw(args.out, args.title)
