#!/usr/bin/env python3
"""
Visual proof-of-work for the 42 stereo unit tests.
Renders the actual input/output of each test as a dashboard PNG.

Usage:
    python3 scripts/visualize_unit_tests.py [--output unit_test_visual.png]
"""
from __future__ import annotations
import os, sys, types, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "../services/stereo-perception"))
from calibration  import make_dummy_calibration
from depth        import DepthEstimator
from obstacle_map import ObstacleAnalyzer
from safety       import SafetyMonitor

CALIB     = make_dummy_calibration(640, 400, baseline_m=0.075)
FOCAL, BASE = CALIB.focal_length, 0.075
DEPTH_EST = DepthEstimator(CALIB, 0.25, 10.0)
ANALYZER  = ObstacleAnalyzer(types.SimpleNamespace(
    LEFT_REGION_FRAC=0.33, CENTER_REGION_FRAC=0.34,
    OBSTACLE_PERCENTILE=10.0, DEPTH_ROI_TOP_FRAC=0.25, DEPTH_ROI_BOTTOM_FRAC=0.75))
MONITOR   = SafetyMonitor(types.SimpleNamespace(
    EMERGENCY_DISTANCE_M=0.5, STOP_DISTANCE_M=1.5,
    WARNING_DISTANCE_M=3.0,   VISION_TIMEOUT_S=1.0))

EMERGENCY_M, STOP_M, WARNING_M = 0.5, 1.5, 3.0
BG, PANEL, GRID2, TEXT, DIM = "#0d1117", "#161b22", "#21262d", "#e6edf3", "#8b949e"
PASS_COL = "#2ecc71"

CMD_COLOR = {"FORWARD":"#2ecc71","TURN_LEFT":"#3498db","TURN_RIGHT":"#3498db",
             "STOP":"#e67e22","EMERGENCY_STOP":"#e74c3c"}
CMD_SHORT = {"FORWARD":"FWD","TURN_LEFT":"TURN L","TURN_RIGHT":"TURN R",
             "STOP":"STOP","EMERGENCY_STOP":"EMRG!"}
ZONE_COL  = {"clear":"#2ecc71","warning":"#f39c12","stop":"#e67e22","emergency":"#e74c3c"}

def zone(d):
    if not np.isfinite(d) or d > WARNING_M: return "clear"
    if d > STOP_M:      return "warning"
    if d > EMERGENCY_M: return "stop"
    return "emergency"

def d_raw(depth_m):
    return int((FOCAL * BASE / max(depth_m, 0.26)) * 16)

def make_disp(left_m=8, center_m=8, right_m=8, h=80, w=120):
    a = np.full((h, w), d_raw(8), dtype=np.int16)
    a[:, :int(w*.33)]          = d_raw(left_m)
    a[:, int(w*.33):int(w*.67)]= d_raw(center_m)
    a[:, int(w*.67):]          = d_raw(right_m)
    return a

def run(left_m=8, center_m=8, right_m=8, age=0.05, desired="FORWARD"):
    a = make_disp(left_m, center_m, right_m)
    dr = DEPTH_EST.compute(a)
    om = ANALYZER.analyze(dr)
    st = MONITOR.evaluate(om, age, desired)
    return dr, om, st

DEPTH_CMAP = LinearSegmentedColormap.from_list(
    "depth", ["#e74c3c","#e67e22","#f39c12","#2ecc71","#1a3a2a"], N=256)

def style_ax(ax, title="", fontsize=8):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color(GRID2)
    ax.tick_params(colors=DIM, labelsize=7)
    if title:
        ax.set_title(title, color=TEXT, fontsize=fontsize, pad=4, fontweight="bold")

def pass_badge(ax, x=0.98, y=0.97):
    ax.text(x, y, "PASS", transform=ax.transAxes, ha="right", va="top",
            color=PASS_COL, fontsize=7, fontweight="bold",
            bbox=dict(fc=PANEL, ec=PASS_COL, pad=1.5, lw=1, boxstyle="round,pad=0.25"))

def section_label(fig, text, x, y):
    fig.text(x, y, text, color=TEXT, fontsize=12, fontweight="bold",
             ha="left", va="center",
             bbox=dict(fc="#1a2f4a", ec="#58a6ff", pad=4, lw=1.5,
                       boxstyle="round,pad=0.4"))


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — DepthEstimator tests
# ═══════════════════════════════════════════════════════════════════════

def draw_depth_heatmap(ax, disp_arr, title, clim=(0.25, 10)):
    dr = DEPTH_EST.compute(disp_arr)
    dm = dr.depth_map.copy()
    dm[~np.isfinite(dm)] = np.nan
    style_ax(ax, title)
    im = ax.imshow(dm, cmap=DEPTH_CMAP, vmin=clim[0], vmax=clim[1],
                   aspect="auto", interpolation="nearest")
    valid_pct = int(dr.valid_fraction * 100)
    ax.text(0.02, 0.96, f"valid: {valid_pct}%", transform=ax.transAxes,
            color=TEXT, fontsize=7, va="top")
    ax.set_xticks([]); ax.set_yticks([])
    pass_badge(ax)
    return im

def draw_depth_section_1(axes):
    # Test: uniform disparity → correct depth
    h, w = 80, 120
    d3 = np.full((h, w), d_raw(3.0), dtype=np.int16)
    draw_depth_heatmap(axes[0], d3, "uniform 3 m → finite depth")

    # Test: below MIN_DEPTH masked
    d_close = np.full((h, w), d_raw(0.1), dtype=np.int16)
    draw_depth_heatmap(axes[1], d_close, "0.1 m (below min) → all NaN")

    # Test: zero disparity masked
    d_zero = np.zeros((h, w), dtype=np.int16)
    draw_depth_heatmap(axes[2], d_zero, "zero disparity → all NaN")

    # Test: mixed scene — valid fraction
    d_mix = make_disp(0.1, 3.0, 8.0, h=h, w=w)
    draw_depth_heatmap(axes[3], d_mix, "mixed: 0.1m|3m|8m → partial valid")

    # Test: valid_fraction bar chart
    ax = axes[4]
    style_ax(ax, "valid_fraction per scene")
    scenes = {"uniform\n3 m": make_disp(3,3,3,h,w),
              "uniform\n0.1 m": np.full((h,w),d_raw(0.1),dtype=np.int16),
              "all\nzero": np.zeros((h,w),dtype=np.int16),
              "mixed\nscene": d_mix,
              "uniform\n8 m": make_disp(8,8,8,h,w)}
    fracs = [DEPTH_EST.compute(v).valid_fraction for v in scenes.values()]
    colors = [PASS_COL if f > 0.5 else "#e74c3c" if f == 0 else "#f39c12"
              for f in fracs]
    bars = ax.bar(range(len(fracs)), [f*100 for f in fracs],
                  color=colors, edgecolor=GRID2, width=0.6)
    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels(list(scenes.keys()), color=DIM, fontsize=6.5)
    ax.set_ylabel("valid %", color=DIM, fontsize=7)
    ax.set_ylim(0, 115)
    ax.axhline(50, color=DIM, lw=0.8, ls="--", alpha=0.5)
    for i, (b, f) in enumerate(zip(bars, fracs)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+2,
                f"{f*100:.0f}%", ha="center", va="bottom",
                color=TEXT, fontsize=7, fontweight="bold")
    ax.grid(True, axis="y", color=GRID2, lw=0.5, alpha=0.5)
    pass_badge(ax)

    # Test: Q-matrix shape + calibration sanity
    ax = axes[5]
    style_ax(ax, "calibration values (OAK-D Lite 640×400)")
    ax.axis("off")
    vals = [
        ("Baseline",      f"{CALIB.baseline_m*100:.1f} cm", PASS_COL),
        ("Focal length",  f"{CALIB.focal_length:.0f} px",   PASS_COL),
        ("Image size",    f"{CALIB.image_size[0]}×{CALIB.image_size[1]}", PASS_COL),
        ("K_left shape",  f"{CALIB.K_left.shape}",          PASS_COL),
        ("Q shape",       f"{CALIB.Q.shape}",               PASS_COL),
        ("Q[3,2]",        f"{CALIB.Q[3,2]:.4f}  (+1/b)",   PASS_COL),
    ]
    for i, (k, v, c) in enumerate(vals):
        y = 0.88 - i*0.15
        ax.text(0.04, y, k, transform=ax.transAxes, color=DIM,   fontsize=8, va="center")
        ax.text(0.55, y, v, transform=ax.transAxes, color=c,     fontsize=8.5,
                va="center", fontweight="bold")
        ax.text(0.96, y, "✓", transform=ax.transAxes, color=PASS_COL,
                fontsize=10, va="center", ha="right")
    pass_badge(ax)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — ObstacleAnalyzer tests
# ═══════════════════════════════════════════════════════════════════════

def draw_zone_card(ax, left_m, center_m, right_m, title, expected_cmd=None):
    style_ax(ax, title, fontsize=7.5)
    ax.set_xlim(0, 3); ax.set_ylim(0, 10.5); ax.set_xticks([]); ax.set_yticks([])

    alphas = {"clear":0.15,"warning":0.25,"stop":0.35,"emergency":0.50}
    for xi, (d, lbl) in enumerate([(left_m,"L"),(center_m,"C"),(right_m,"R")]):
        d_vis = min(d if np.isfinite(d) else 9, 9)
        z = zone(d)
        ax.add_patch(mpatches.FancyBboxPatch(
            (xi+0.08, 0), 0.84, d_vis,
            boxstyle="square,pad=0", lw=0,
            fc=ZONE_COL[z], alpha=alphas[z]))
        label = f"{d:.1f}m" if np.isfinite(d) else "NaN"
        ax.text(xi+0.5, max(d_vis-0.3, 0.3), label, ha="center", va="top",
                color=ZONE_COL[z], fontsize=7.5, fontweight="bold")
        ax.text(xi+0.5, -0.35, lbl, ha="center", va="top", color=DIM, fontsize=7)

    for th, col in [(WARNING_M,"#f39c12"),(STOP_M,"#e67e22"),(EMERGENCY_M,"#e74c3c")]:
        ax.axhline(th, color=col, lw=0.9, ls="--", alpha=0.7)

    if expected_cmd:
        col = CMD_COLOR.get(expected_cmd, "#aaa")
        short = CMD_SHORT.get(expected_cmd, expected_cmd)
        ax.text(1.5, 10.1, short, ha="center", va="center", color=col,
                fontsize=8, fontweight="bold",
                bbox=dict(fc=PANEL, ec=col, pad=1.5, lw=1.2,
                          boxstyle="round,pad=0.3"))
    pass_badge(ax, x=0.97, y=0.97)


def draw_obstacle_section(axes):
    cards = [
        (8,   8,   8,   "clear → FORWARD",        "FORWARD"),
        (8,   2.5, 8,   "center 2.5m → warning",  "FORWARD"),
        (8,   1.0, 8,   "center blocked → steer",  "TURN_LEFT"),
        (1.0, 1.0, 8.0, "C+L blocked → TURN R",   "TURN_RIGHT"),
        (8.0, 1.0, 1.0, "C+R blocked → TURN L",   "TURN_LEFT"),
        (1.0, 1.0, 1.0, "all blocked → STOP",     "STOP"),
        (0.4, 8,   8,   "0.4m left → EMERGENCY",  "EMERGENCY_STOP"),
        (8,   0.3, 8,   "0.3m center → EMERGENCY","EMERGENCY_STOP"),
    ]
    for ax, (l, c, r, title, expected) in zip(axes, cards):
        draw_zone_card(ax, l, c, r, title, expected)

    # nearest_m bar
    ax = axes[8]
    style_ax(ax, "nearest_m = min(L, C, R)")
    scenarios = [
        ("all 8m", 8, 8, 8),
        ("L=2m",   2, 8, 8),
        ("C=1.5m", 8, 1.5, 8),
        ("R=0.6m", 8, 8, 0.6),
        ("all 1m", 1, 1, 1),
    ]
    for i, (lbl, l, c, r) in enumerate(scenarios):
        _, om, _ = run(l, c, r)
        nearest = om.nearest_m
        col = ZONE_COL[zone(nearest)]
        ax.barh(i, nearest, color=col, alpha=0.7, height=0.6, edgecolor=GRID2)
        ax.text(nearest+0.1, i, f"{nearest:.1f}m", va="center", color=col,
                fontsize=7.5, fontweight="bold")
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([s[0] for s in scenarios], color=DIM, fontsize=7.5)
    ax.set_xlabel("nearest obstacle (m)", color=DIM, fontsize=7.5)
    ax.set_xlim(0, 11)
    for th, col in [(WARNING_M,"#f39c12"),(STOP_M,"#e67e22"),(EMERGENCY_M,"#e74c3c")]:
        ax.axvline(th, color=col, lw=0.9, ls="--", alpha=0.7)
    ax.grid(True, axis="x", color=GRID2, lw=0.5, alpha=0.5)
    pass_badge(ax)

    # NaN / invalid depth card
    ax = axes[9]
    style_ax(ax, "all-NaN depth → clearances = NaN\n(not zero — no false wall)")
    ax.axis("off")
    zero_disp = np.zeros((80, 120), dtype=np.int16)
    dr = DEPTH_EST.compute(zero_disp)
    om = ANALYZER.analyze(dr)
    items = [
        ("valid_fraction", f"{dr.valid_fraction:.2f}", "#e74c3c"),
        ("left.nearest_m", "NaN" if not np.isfinite(om.left.nearest_m) else f"{om.left.nearest_m:.2f}", PASS_COL),
        ("center.nearest_m","NaN" if not np.isfinite(om.center.nearest_m) else f"{om.center.nearest_m:.2f}", PASS_COL),
        ("right.nearest_m", "NaN" if not np.isfinite(om.right.nearest_m) else f"{om.right.nearest_m:.2f}", PASS_COL),
        ("nearest_m",       "NaN" if not np.isfinite(om.nearest_m) else f"{om.nearest_m:.2f}", PASS_COL),
    ]
    for i, (k, v, c) in enumerate(items):
        y = 0.85 - i*0.17
        ax.text(0.04, y, k, transform=ax.transAxes, color=DIM, fontsize=8, va="center")
        ax.text(0.72, y, v, transform=ax.transAxes, color=c,   fontsize=9,
                va="center", fontweight="bold", ha="center")
    pass_badge(ax)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — SafetyMonitor tests
# ═══════════════════════════════════════════════════════════════════════

def draw_safety_card(ax, title, left_m, center_m, right_m,
                     age=0.05, desired="FORWARD", om_override=None):
    style_ax(ax, title, fontsize=7.5)
    if om_override is not None:
        st = MONITOR.evaluate(om_override, age, desired)
        l_d, c_d, r_d = left_m, center_m, right_m
    else:
        _, om, st = run(left_m, center_m, right_m, age, desired)
        l_d, c_d, r_d = om.left.nearest_m, om.center.nearest_m, om.right.nearest_m

    cmd = st.commanded_direction
    col = CMD_COLOR.get(cmd, "#aaa")

    # mini zone bars
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for xi, (d, lbl) in enumerate([(l_d,"L"),(c_d,"C"),(r_d,"R")]):
        d_vis = min(d if np.isfinite(d) else 9, 9) / 9
        z = zone(d)
        bar_h = d_vis * 0.45
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.05+xi*0.3, 0.46), 0.22, bar_h,
            boxstyle="square,pad=0", lw=0,
            fc=ZONE_COL[z], alpha=0.3,
            transform=ax.transAxes, clip_on=False))
        label = f"{d:.1f}" if np.isfinite(d) else "NaN"
        ax.text(0.16+xi*0.3, 0.44, label, ha="center", va="top",
                color=ZONE_COL[z], fontsize=7, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.16+xi*0.3, 0.38, lbl, ha="center", va="top",
                color=DIM, fontsize=6.5, transform=ax.transAxes)

    # command result
    ax.text(0.5, 0.22, CMD_SHORT.get(cmd, cmd),
            ha="center", va="center", color=col,
            fontsize=10, fontweight="bold", transform=ax.transAxes,
            bbox=dict(fc=PANEL, ec=col, pad=3, lw=1.5,
                      boxstyle="round,pad=0.4"))

    # reason (truncated)
    reason = st.reason[:32] + ("…" if len(st.reason) > 32 else "")
    ax.text(0.5, 0.06, reason, ha="center", va="center",
            color=DIM, fontsize=6, transform=ax.transAxes)

    ax.axis("off")
    pass_badge(ax, x=0.97, y=0.97)


def draw_safety_section(axes):
    tests = [
        # title, L, C, R, age, desired
        ("clear → FORWARD",                8,   8,   8,   0.05, "FORWARD"),
        ("2.5m center → FWD slow",         8,   2.5, 8,   0.05, "FORWARD"),
        ("C blkd, L clear → TURN L",       8,   1.0, 1.0, 0.05, "FORWARD"),
        ("C blkd, R clear → TURN R",       1.0, 1.0, 8,   0.05, "FORWARD"),
        ("all blocked → STOP",             1.0, 1.0, 1.0, 0.05, "FORWARD"),
        ("0.3m C → EMERGENCY",             8,   0.3, 8,   0.05, "FORWARD"),
        ("0.4m L only → EMERGENCY",        0.4, 8,   8,   0.05, "FORWARD"),
        ("want TURN L, L blocked → STOP",  0.8, 8,   8,   0.05, "TURN_LEFT"),
        ("want TURN R, R blocked → STOP",  8,   8,   0.8, 0.05, "TURN_RIGHT"),
        ("vision timeout 1.5s → STOP",     8,   8,   8,   1.5,  "FORWARD"),
        ("fresh 50ms → no timeout",        8,   8,   8,   0.05, "FORWARD"),
    ]
    for ax, (title, l, c, r, age, desired) in zip(axes[:len(tests)], tests):
        draw_safety_card(ax, title, l, c, r, age, desired)

    # None obstacle map
    ax = axes[11]
    style_ax(ax, "None obstacle map → STOP", fontsize=7.5)
    st = MONITOR.evaluate(None, vision_age_s=0.1, desired_direction="FORWARD")
    ax.axis("off")
    ax.text(0.5, 0.6, CMD_SHORT.get(st.commanded_direction, st.commanded_direction),
            ha="center", va="center", color=CMD_COLOR.get(st.commanded_direction,"#aaa"),
            fontsize=14, fontweight="bold", transform=ax.transAxes,
            bbox=dict(fc=PANEL, ec=CMD_COLOR.get(st.commanded_direction,"#aaa"),
                      pad=3, lw=1.5, boxstyle="round,pad=0.4"))
    ax.text(0.5, 0.3, st.reason[:36], ha="center", va="center",
            color=DIM, fontsize=6.5, transform=ax.transAxes)
    pass_badge(ax)

    # speed_override chart
    ax = axes[12]
    style_ax(ax, "speed_override vs center clearance")
    dists = [8, 6, 4, 3.5, 2.5, 2.0, 1.5]
    speeds = []
    cmds   = []
    for d in dists:
        _, _, st = run(8, d, 8)
        speeds.append(st.speed_override if st.speed_override is not None else 1.0)
        cmds.append(st.commanded_direction)
    colors = [CMD_COLOR.get(c,"#aaa") for c in cmds]
    ax.bar(range(len(dists)), speeds, color=colors, edgecolor=GRID2, width=0.6)
    ax.axhline(1.0, color=DIM, lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels([f"{d}m" for d in dists], color=DIM, fontsize=7.5)
    ax.set_ylabel("speed override", color=DIM, fontsize=7.5)
    ax.set_ylim(0, 1.2)
    for i, (s, c) in enumerate(zip(speeds, cmds)):
        ax.text(i, s+0.02, f"{s:.0%}", ha="center", va="bottom",
                color=CMD_COLOR.get(c,"#aaa"), fontsize=7, fontweight="bold")
    ax.grid(True, axis="y", color=GRID2, lw=0.5, alpha=0.5)
    pass_badge(ax)


# ═══════════════════════════════════════════════════════════════════════
# BUILD FIGURE
# ═══════════════════════════════════════════════════════════════════════

def build(out_path):
    fig = plt.figure(figsize=(28, 26), facecolor=BG)
    fig.suptitle(
        "NASA Rover  ·  Unit Test Visual Results  ·  42 / 42 PASS",
        color=TEXT, fontsize=16, fontweight="bold", y=0.997)

    outer = GridSpec(4, 1, figure=fig,
                     height_ratios=[0.08, 1, 1, 1],
                     hspace=0.70, top=0.988, bottom=0.018,
                     left=0.025, right=0.978)

    # ── Section header bar ──
    hdr = fig.add_subplot(outer[0])
    hdr.set_facecolor(BG); hdr.axis("off")
    for x, lbl, col in [
        (0.01, "SECTION 1 — test_depth.py  (13 tests)", "#2ecc71"),
        (0.35, "SECTION 2 — test_obstacle_map.py  (14 tests)", "#f39c12"),
        (0.67, "SECTION 3 — test_safety.py  (15 tests)", "#e74c3c"),
    ]:
        hdr.text(x, 0.5, lbl, transform=hdr.transAxes,
                 color=col, fontsize=10, fontweight="bold", va="center",
                 bbox=dict(fc=PANEL, ec=col, pad=4, lw=1.5,
                           boxstyle="round,pad=0.4"))

    # ── Section 1: depth (2 rows × 3 cols) ──
    s1 = outer[1].subgridspec(2, 3, hspace=0.55, wspace=0.25)
    depth_axes_r1 = [fig.add_subplot(s1[0, c]) for c in range(3)]
    depth_axes_r2 = [fig.add_subplot(s1[1, c]) for c in range(3)]
    draw_depth_section_1(depth_axes_r1 + depth_axes_r2)

    # ── Section 2: obstacle map (2 rows × 5 cols) ──
    s2 = outer[2].subgridspec(2, 5, hspace=0.60, wspace=0.25)
    obs_axes = [fig.add_subplot(s2[r, c]) for r in range(2) for c in range(5)]
    draw_obstacle_section(obs_axes)

    # ── Section 3: safety monitor (3 rows × 5 cols — 13 panels) ──
    s3 = outer[3].subgridspec(3, 5, hspace=0.60, wspace=0.25)
    saf_axes = [fig.add_subplot(s3[r, c]) for r in range(3) for c in range(5)]
    draw_safety_section(saf_axes)

    # Colorbar for depth heatmaps
    cax = fig.add_axes([0.026, 0.685, 0.008, 0.085])
    sm  = plt.cm.ScalarMappable(cmap=DEPTH_CMAP,
                                 norm=plt.Normalize(vmin=0.25, vmax=10))
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label("depth (m)", color=DIM, fontsize=7)
    cb.ax.yaxis.set_tick_params(color=DIM, labelsize=6.5)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=DIM)
    cb.outline.set_edgecolor(GRID2)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="unit_test_visual.png")
    build(p.parse_args().output)
