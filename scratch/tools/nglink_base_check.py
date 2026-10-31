"""Check that faab.nglink builds on the README scene: layer list, sources of neuron layers, mesh visibility, URL size."""
from faab import nglink

spec = {"meshes": [{"kind": "OL", "segments": ["21", "22", "23", "24"], "alpha": 0.15}, {"kind": "neuropil"}],
        "neurons": [{"name": "targets", "color": "#1f77d6", "ids": ["648518347618283625"]},
                    {"name": "inputs", "colors": {"648518347498326095": "#ff7f0e"}, "ids": ["648518347498326095"]}],
        "select": "inputs"}
st = nglink.build(spec)
for L in st["layers"]:
    src = L.get("source")
    n_src = len(src) if isinstance(src, list) else 1
    live = any("segmentation_annotations" in str(s) for s in (src if isinstance(src, list) else [src]))
    print(f"{L['type']:13} {L['name']:20} visible={L.get('visible', True)!s:5} sources={n_src} seatable_live={live} "
          f"segments={L.get('segments', [])[:6]}")
print("background", st.get("projectionBackgroundColor"), "layout", st["layout"], "selected", st["selectedLayer"])
print("url chars", len(nglink.to_url(st)))
