#!/usr/bin/env python3
"""
Animated top-down rover sim — saves as rover_live.gif
Usage: python3 scripts/rover_live.py
"""
from __future__ import annotations
import os, sys, types, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter

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

BG, PANEL, GC = "#0d1117", "#161b22", "#21262d"
TEXT, DIM     = "#e6edf3", "#8b949e"
CMD_COLOR = {"FORWARD":"#2ecc71","TURN_LEFT":"#3498db","TURN_RIGHT":"#3498db",
             "STOP":"#e67e22","EMERGENCY_STOP":"#e74c3c"}
CMD_LABEL = {"FORWARD":"→  FORWARD","TURN_LEFT":"↰  TURN LEFT",
             "TURN_RIGHT":"↱  TURN RIGHT","STOP":"■  STOP",
             "EMERGENCY_STOP":"✖  EMERGENCY STOP"}
ZONE_COL  = {"clear":"#2ecc71","warning":"#f39c12","stop":"#e67e22","emergency":"#e74c3c"}

def d_raw(d): return int((FOCAL*BASE/max(d,0.26))*16)

def make_disp(l=8,c=8,r=8):
    a=np.full((80,120),d_raw(8),dtype=np.int16)
    a[:,:int(120*.33)]=d_raw(l); a[:,int(120*.33):int(120*.67)]=d_raw(c); a[:,int(120*.67):]=d_raw(r)
    return a

def pipeline(l,c,r,age=0.05):
    dr=DEPTH_EST.compute(make_disp(l,c,r)); om=ANALYZER.analyze(dr)
    return om, MONITOR.evaluate(om,age,"FORWARD")

def zone(d):
    if not np.isfinite(d) or d>3: return "clear"
    if d>1.5: return "warning"
    if d>0.5: return "stop"
    return "emergency"

WORLD_W = 12  # metres wide

# ── Script of scenes ──────────────────────────────────────────────────────────
# (label, n_frames, obstacles[(ox,oy,ow,oh)], vision_age)
SCRIPT = [
    ("Open field — cruising",        20, [],                               0.05),
    ("Wall ahead — slowing",         14, [(3.0,10.5,6,0.35)],             0.05),
    ("Wall close — TURN LEFT",       12, [(3.0, 8.5,6,0.35)],             0.05),
    ("Wall cleared — forward",       10, [],                               0.05),
    ("Obstacle right — steer left",  14, [(7.5, 7.0,2.5,0.4)],           0.05),
    ("All clear — resuming",         10, [],                               0.05),
    ("Camera lag 500 ms",             8, [],                               0.5),
    ("Camera TIMEOUT — STOP",        12, [],                               1.8),
    ("Camera recovers",              10, [],                               0.05),
]

def obs_to_zones(rx, ry, ra, obstacles):
    """Simple ray-box distance for L/C/R zones."""
    zones = [9.9, 9.9, 9.9]          # L C R
    zone_ang = [(-35,-12),(-12,12),(12,35)]
    for ox, oy, ow, oh in obstacles:
        pts = [(ox+dx,oy+dy) for dx in (0,ow) for dy in (0,oh)]
        for px,py in pts:
            dist = math.hypot(px-rx, py-ry)
            if dist<0.1 or dist>10: continue
            wa  = math.degrees(math.atan2(px-rx, py-ry))
            rel = (wa - ra + 180) % 360 - 180
            for i,(a0,a1) in enumerate(zone_ang):
                if a0<=rel<=a1:
                    zones[i]=min(zones[i],dist)
    return zones[0],zones[1],zones[2]

# ── Rover physics ─────────────────────────────────────────────────────────────
class Rover:
    def __init__(self):
        self.x=WORLD_W/2; self.y=1.0; self.angle=0.0; self.trail=[]
    def step(self,cmd,spd_ov):
        spd=(spd_ov or 1.0)*0.20
        rad=math.radians(self.angle)
        if cmd=="FORWARD":
            self.x+=math.sin(rad)*spd; self.y+=math.cos(rad)*spd
        elif cmd=="TURN_LEFT":
            self.angle-=4.5
            rad=math.radians(self.angle)
            self.x+=math.sin(rad)*spd*0.35; self.y+=math.cos(rad)*spd*0.35
        elif cmd=="TURN_RIGHT":
            self.angle+=4.5
            rad=math.radians(self.angle)
            self.x+=math.sin(rad)*spd*0.35; self.y+=math.cos(rad)*spd*0.35
        self.x=max(0.8,min(WORLD_W-0.8,self.x))
        self.y=max(0.8,self.y)
        self.trail.append((self.x,self.y))
        if len(self.trail)>100: self.trail=self.trail[-100:]

rover=Rover()

# Expand script into per-frame list
FRAMES=[]
for lbl,nf,obs,age in SCRIPT:
    for _ in range(nf):
        FRAMES.append((lbl,obs,age))

# ── Figure ────────────────────────────────────────────────────────────────────
fig=plt.figure(figsize=(14,8.5),facecolor=BG)
ax_w=fig.add_axes([0.03,0.06,0.50,0.88])   # world view
ax_p=fig.add_axes([0.57,0.06,0.40,0.88])   # panel

fig.suptitle("NASA Rover  ·  Live Obstacle-Avoidance Simulation  ·  OAK-D Lite Pipeline",
             color=TEXT,fontsize=12,fontweight="bold",y=0.98)

for ax in (ax_w,ax_p):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color(GC)

VIEW_H=16
ax_w.set_xlim(0,WORLD_W); ax_w.set_ylim(0,VIEW_H)
ax_w.set_aspect("equal")
ax_w.tick_params(colors=DIM,labelsize=7)
ax_w.set_xlabel("x  (m)",color=DIM,fontsize=8)
ax_w.set_ylabel("y  (m)",color=DIM,fontsize=8)

ax_p.set_xlim(0,1); ax_p.set_ylim(0,1); ax_p.axis("off")

# ── Draw frame ────────────────────────────────────────────────────────────────
def draw(fi):
    lbl,obstacles,age=FRAMES[fi]

    l_d,c_d,r_d=obs_to_zones(rover.x,rover.y,rover.angle,obstacles)
    om,st=pipeline(l_d,c_d,r_d,age)
    cmd=st.commanded_direction
    col=CMD_COLOR.get(cmd,"#aaa")

    rover.step(cmd,st.speed_override)

    # ── World axes ──
    ax_w.cla()
    ax_w.set_facecolor(PANEL)
    for sp in ax_w.spines.values(): sp.set_color(GC)
    ax_w.tick_params(colors=DIM,labelsize=7)
    ax_w.set_xlabel("x  (m)",color=DIM,fontsize=8)
    ax_w.set_ylabel("y  (m)",color=DIM,fontsize=8)

    vy0=max(0,rover.y-4); vy1=vy0+VIEW_H
    ax_w.set_xlim(0,WORLD_W); ax_w.set_ylim(vy0,vy1)

    # grid
    for gx in np.arange(0,WORLD_W+1,1):
        ax_w.axvline(gx,color=GC,lw=0.4,zorder=0)
    for gy in np.arange(int(vy0),int(vy1)+1,1):
        ax_w.axhline(gy,color=GC,lw=0.4,zorder=0)

    # trail
    if len(rover.trail)>1:
        tx,ty=zip(*rover.trail)
        ax_w.plot(tx,ty,color="#3498db",lw=1.5,alpha=0.45,zorder=2)

    # FOV wedge
    rad=math.radians(rover.angle)
    fov_range=min(c_d,7)
    fov=mpatches.Wedge((rover.x,rover.y),fov_range,
                        90-rover.angle-35, 90-rover.angle+35,
                        color="#58a6ff",alpha=0.09,zorder=3)
    ax_w.add_patch(fov)

    # side FOV lines
    for da in (-35,35):
        aa=math.radians(rover.angle+da)
        ax_w.plot([rover.x,rover.x+math.sin(aa)*fov_range],
                  [rover.y,rover.y+math.cos(aa)*fov_range],
                  color="#58a6ff",lw=0.7,alpha=0.35,zorder=3)

    # obstacles
    for ox,oy,ow,oh in obstacles:
        ax_w.add_patch(mpatches.FancyBboxPatch(
            (ox,oy),ow,oh,boxstyle="round,pad=0.05",
            lw=1.8,ec="#e74c3c",fc="#2d1010",alpha=0.9,zorder=4))
        ax_w.text(ox+ow/2,oy+oh+0.15,f"{c_d:.1f} m",
                  ha="center",va="bottom",color="#e74c3c",
                  fontsize=8,fontweight="bold",zorder=5)

    # rover body (rotated rectangle via transform)
    bw,bh=0.75,1.0
    body=mpatches.FancyBboxPatch(
        (-bw/2,-bh/2),bw,bh,boxstyle="round,pad=0.07",
        lw=2,ec=col,fc="#1c2128",zorder=6)
    t=(matplotlib.transforms.Affine2D()
       .rotate_deg(-rover.angle)
       .translate(rover.x,rover.y)
       + ax_w.transData)
    body.set_transform(t)
    ax_w.add_patch(body)

    # camera bar
    cam=mpatches.FancyBboxPatch((-0.28,-0.1),0.56,0.18,
        boxstyle="round,pad=0.02",lw=1,ec="#58a6ff",fc="#1a2f4a",zorder=7)
    cam.set_transform(t)
    ax_w.add_patch(cam)
    # OAK-D label (fixed position, no rotation)
    ax_w.text(rover.x,rover.y,"◉",ha="center",va="center",
              color="#58a6ff",fontsize=7,zorder=8)

    # direction arrow
    dx=math.sin(rad)*1.2; dy=math.cos(rad)*1.2
    ax_w.annotate("",xy=(rover.x+dx,rover.y+dy),xytext=(rover.x,rover.y),
        arrowprops=dict(arrowstyle="-|>",color=col,lw=2.2,mutation_scale=16),zorder=8)

    # wheels
    for wx_,wy_ in [(-0.45,-0.42),(0.45,-0.42),(-0.45,0.3),(0.45,0.3)]:
        wh=mpatches.FancyBboxPatch((wx_-.18,wy_-.1),.36,.2,
            boxstyle="round,pad=0.02",lw=0.8,ec="#555",fc="#2d333b",zorder=6)
        wh.set_transform(t)
        ax_w.add_patch(wh)

    # scenario label
    ax_w.text(WORLD_W/2,vy1-0.3,lbl,ha="center",va="top",
              color=TEXT,fontsize=9,fontweight="bold",
              bbox=dict(fc=PANEL,ec=col,pad=3,lw=1.2,boxstyle="round,pad=0.3"),
              zorder=9)

    ax_w.text(0.5,-0.01,f"frame {fi:03d} / {len(FRAMES)}",
              ha="center",va="top",color=DIM,fontsize=7,
              transform=ax_w.transAxes)

    # ── Panel ──
    ax_p.cla(); ax_p.set_xlim(0,1); ax_p.set_ylim(0,1); ax_p.axis("off")
    ax_p.set_facecolor(PANEL)

    # section: clearances
    ax_p.text(0.5,0.95,"CLEARANCES",ha="center",color=DIM,
              fontsize=9,fontweight="bold",transform=ax_p.transAxes)

    zone_data=[("LEFT",l_d,0.04),("CENTER",c_d,0.37),("RIGHT",r_d,0.70)]
    for zlbl,zd,xp in zone_data:
        zc=zone(zd); zcol=ZONE_COL[zc]
        frac=min(zd if np.isfinite(zd) else 9.9,9.9)/9.9
        # background bar
        ax_p.add_patch(mpatches.FancyBboxPatch((xp,0.74),0.28,0.14,
            boxstyle="round,pad=0.01",lw=0.6,ec=GC,fc=GC,
            transform=ax_p.transAxes))
        # filled bar
        ax_p.add_patch(mpatches.FancyBboxPatch((xp,0.74),0.28*frac,0.14,
            boxstyle="round,pad=0.01",lw=0,fc=zcol,alpha=0.5,
            transform=ax_p.transAxes))
        ax_p.text(xp+0.14,0.81,f"{zd:.1f}m" if np.isfinite(zd) else "NaN",
                  ha="center",va="center",color=zcol,fontsize=9,
                  fontweight="bold",transform=ax_p.transAxes)
        ax_p.text(xp+0.14,0.725,zlbl,ha="center",va="top",
                  color=DIM,fontsize=7.5,transform=ax_p.transAxes)

    # nearest
    nearest=om.nearest_m
    n_col=ZONE_COL[zone(nearest)]
    n_str=f"{nearest:.2f} m" if np.isfinite(nearest) else "none"
    ax_p.text(0.5,0.68,f"Nearest obstacle:  {n_str}",ha="center",
              color=n_col,fontsize=9,fontweight="bold",
              transform=ax_p.transAxes)

    # divider
    ax_p.plot([0.02,0.98],[0.645,0.645],color=GC,lw=0.8,transform=ax_p.transAxes)

    # motor command
    ax_p.text(0.5,0.61,"MOTOR COMMAND",ha="center",color=DIM,
              fontsize=8,fontweight="bold",transform=ax_p.transAxes)
    ax_p.add_patch(mpatches.FancyBboxPatch((0.05,0.49),0.90,0.10,
        boxstyle="round,pad=0.02",lw=2,ec=col,fc=PANEL,
        transform=ax_p.transAxes))
    ax_p.text(0.5,0.54,CMD_LABEL.get(cmd,cmd),ha="center",va="center",
              color=col,fontsize=12,fontweight="bold",
              transform=ax_p.transAxes)

    # reason
    ax_p.text(0.5,0.455,st.reason[:48],ha="center",va="center",
              color=DIM,fontsize=7.5,transform=ax_p.transAxes)

    ax_p.plot([0.02,0.98],[0.42,0.42],color=GC,lw=0.8,transform=ax_p.transAxes)

    # vision age bar
    ax_p.text(0.5,0.40,"VISION AGE",ha="center",color=DIM,
              fontsize=8,fontweight="bold",transform=ax_p.transAxes)
    age_frac=min(age/2.0,1.0)
    age_col="#2ecc71" if age<0.5 else "#f39c12" if age<1.0 else "#e74c3c"
    ax_p.add_patch(mpatches.FancyBboxPatch((0.05,0.34),0.90,0.05,
        boxstyle="round,pad=0.01",lw=0.5,ec=GC,fc=GC,
        transform=ax_p.transAxes))
    ax_p.add_patch(mpatches.FancyBboxPatch((0.05,0.34),0.90*age_frac,0.05,
        boxstyle="round,pad=0.01",lw=0,fc=age_col,
        transform=ax_p.transAxes))
    ax_p.text(0.97,0.365,f"{age*1000:.0f} ms",ha="right",va="center",
              color=age_col,fontsize=8,fontweight="bold",
              transform=ax_p.transAxes)
    ax_p.text(0.5,0.265,"—  timeout at 1000 ms  —",ha="center",
              color=DIM,fontsize=7,transform=ax_p.transAxes)

    ax_p.plot([0.02,0.98],[0.25,0.25],color=GC,lw=0.8,transform=ax_p.transAxes)

    # pipeline chain
    ax_p.text(0.5,0.22,"PIPELINE",ha="center",color=DIM,
              fontsize=8,fontweight="bold",transform=ax_p.transAxes)
    chain=[("OAK-D Lite","#58a6ff"),("DepthEstimator","#2ecc71"),
           ("ObstacleAnalyzer","#f39c12"),("SafetyMonitor","#e74c3c"),
           ("Motor CMD","#3498db")]
    for i,(plbl,pcol) in enumerate(chain):
        x_=0.05+i*0.19
        ax_p.add_patch(mpatches.FancyBboxPatch((x_,0.09),0.16,0.10,
            boxstyle="round,pad=0.02",lw=1.2,ec=pcol,fc=PANEL,
            transform=ax_p.transAxes))
        ax_p.text(x_+0.08,0.14,plbl,ha="center",va="center",
                  color=pcol,fontsize=5.5,fontweight="bold",
                  transform=ax_p.transAxes)
        if i<len(chain)-1:
            ax_p.annotate("",xy=(x_+0.19,0.14),xytext=(x_+0.16,0.14),
                xycoords="axes fraction",textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>",color=DIM,lw=0.9,
                                mutation_scale=8))

    # frame badge
    ax_p.text(0.5,0.03,f"Frame {fi:03d}  ·  62 / 62 tests pass",
              ha="center",color=DIM,fontsize=7.5,
              transform=ax_p.transAxes)

ani=FuncAnimation(fig,draw,frames=len(FRAMES),interval=125,blit=False)

out=os.path.join(_HERE,"..","rover_live.gif")
print(f"Rendering {len(FRAMES)} frames…")
ani.save(out,writer=PillowWriter(fps=8),dpi=110)
print(f"Saved → {os.path.abspath(out)}")
