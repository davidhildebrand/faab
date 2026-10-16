"""Schematic: why grid-sampling a surface undersamples a dense cell field, and how
a synapse-slab enumerates every crossing cell instead. Cross-section cartoon."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.RandomState(4)
ng, nr = 40, 7
xg = np.sort(rng.uniform(1.5, 48.5, ng))
xr = np.sort(rng.uniform(4, 46, nr))
ybot, ytop = 5, 35


def sheet(x):
    return 20 + 3.0 * np.sin(x / 9.0)                       # gently curved medulla sheet


fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6.5))
for ax in (axA, axB):
    for x in xg:
        ax.plot([x, x], [ybot, ytop], color="0.72", lw=2, solid_capstyle="round")
    for x in xr:
        ax.plot([x, x], [ybot, ytop], color="firebrick", lw=2.6, solid_capstyle="round")
    xx = np.linspace(0, 50, 300)
    ax.plot(xx, sheet(xx), color="royalblue", lw=2)
    ax.set_xlim(0, 50); ax.set_ylim(0, 40); ax.set_xlabel("lateral position (µm)")
    ax.set_yticks([])

# Panel A: sparse grid samples on the surface
allx = np.concatenate([xg, xr]); isr = np.array([False] * ng + [True] * nr)
hitr = 0
for x in np.arange(2, 49, 2.5):
    y = sheet(x)
    j = int(np.argmin(np.abs(allx - x)))
    on = abs(allx[j] - x) < 0.6
    col = "firebrick" if (on and isr[j]) else ("0.35" if on else "black")
    axA.plot(x, y, marker="x", ms=9, mew=2.2, color=col)
    hitr += on and isr[j]
axA.plot([], [], "x", color="firebrick", label="sample lands on an Mi1 (rare)")
axA.plot([], [], "x", color="0.35", label="lands on some other cell")
axA.plot([], [], "x", color="black", label="lands in a gap")
axA.legend(loc="lower center", fontsize=8, ncol=3)
axA.set_title(f"A. Grid-sampling the fitted surface (2.5 µm)\n"
              f"sparse points on a DENSE field of thin cells -> most Mi1 missed "
              f"({hitr}/{nr} red hit here; in data 14/103)", fontsize=11)
axA.set_ylabel("depth through medulla (µm)")

# Panel B: synapse-slab
axB.axhspan(15, 25, color="gold", alpha=0.18)
axB.axhline(15, color="goldenrod", lw=0.8, ls="--"); axB.axhline(25, color="goldenrod", lw=0.8, ls="--")
axB.text(50.5, 20, "slab\n(sheet ± ~5 µm)", va="center", fontsize=9, color="goldenrod")
for x, red in [(x, False) for x in xg] + [(x, True) for x in xr]:
    ys = rng.uniform(15.5, 24.5, 5)
    axB.scatter([x] * 5, ys, s=14, color=("firebrick" if red else "0.45"), zorder=3, edgecolors="none")
axB.set_title("B. Synapse-slab enumeration\n"
              "synapses (dots) already sit ON every cell -> collect their parent cells\n"
              "-> EVERY cell crossing the slab is enumerated (all Mi1 included)", fontsize=11)
fig.suptitle("Why grid-sampling misses cells, and why a synapse-slab doesn't", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig("figures/explain_slab.png", dpi=130, bbox_inches="tight")
print("saved explain_slab.png")
