"""Interactive 3D view (rotatable in a browser) of full skeletons rigidly
aligned to a template. Exports a self-contained HTML via the plotly backend.

Uses the skeletons cached by template_pairwise_viz.py (no CAVE access if the
cache exists; otherwise fetches read-only).

    .venv/bin/python faab/align3d_interactive.py
"""
from faab.config import REPO
import os
import pickle

import navis
from navis.transforms.align import align_rigid

from faab.clients.cave import latest_roots, throttle, DATASTACK

OUT = os.path.join(REPO, "figures")
CACHE = os.path.join(OUT, "skel_cache.pkl")
IDS = [648518347611904256, 648518347550912741, 648518347638273763]

TEMPLATE_COLOR = (0.47, 0.47, 0.47)
PAIR_COLORS = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def short(i):
    return str(int(i))[-6:]


def load_skeletons_um():
    if os.path.exists(CACHE):
        sk, roots = pickle.load(open(CACHE, "rb"))
        print("loaded cached skeletons:", [short(r) for r in roots])
        return sk, roots
    # fallback: read-only fetch
    import caveclient
    from fafbseg import flywire
    client = caveclient.CAVEclient(DATASTACK)
    roots, t = latest_roots(IDS, client=client)
    t = throttle(t)
    sk = flywire.get_l2_skeleton(roots, dataset=DATASTACK, max_threads=1, progress=False) / 1000.0
    for s in sk:
        s.units = "um"
    pickle.dump((sk, roots), open(CACHE, "wb"))
    return sk, roots


def main():
    sk, roots = load_skeletons_um()
    labels = [short(r) for r in roots]

    tmpl = sk[0].copy()
    tmpl.name = f"template {labels[0]}"

    scene = [tmpl]
    colors = [TEMPLATE_COLOR]
    for c, i in enumerate(range(1, len(sk))):
        aligned = align_rigid(navis.NeuronList(sk[i]), target=sk[0], progress=False)
        if isinstance(aligned, tuple):
            aligned = aligned[0]
        a = aligned[0]
        a.name = labels[i]
        scene.append(a)
        colors.append(PAIR_COLORS[c % len(PAIR_COLORS)])

    nl = navis.NeuronList(scene)
    fig = navis.plot3d(nl, backend="plotly", colors=colors, inline=False)
    fig.update_layout(
        title="Aedes neurons rigidly aligned to template (drag to rotate)",
        showlegend=True,
    )

    out = os.path.join(OUT, "align3d_interactive.html")
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    print("wrote interactive 3D ->", out)


if __name__ == "__main__":
    main()
