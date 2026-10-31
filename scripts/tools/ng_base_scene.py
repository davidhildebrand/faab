"""Download the proofreading scene recommended in the FAAB README (a saved neuroglancer state) and keep it in
data/ng_readme_scene.json, the base every faab.nglink link is built on. Its layers include the live SeaTable
annotation source, so a viewer sees at once which neurons are already annotated (David, 2026-10-31). A state download
is read-only; one CAVE call. Usage: ng_base_scene.py [STATE_ID]  (default: the README scene)"""
import json
import os
import sys

from faab.clients.cave import client as cave_client
from faab.config import DATA

README_SCENE = 5971265676902400

sid = int(sys.argv[1]) if len(sys.argv) > 1 else README_SCENE
state = cave_client().state.get_state_json(sid)
state["_faab_scene_id"] = sid
path = os.path.join(DATA, "ng_readme_scene.json")
tmp = path + ".part"
json.dump(state, open(tmp, "w"), indent=1)
os.replace(tmp, path)
print(f"scene {sid} -> {path}")
print("top-level keys:", sorted(k for k in state if k != "layers"))
for L in state.get("layers", []):
    src = L.get("source")
    src = src.get("url") if isinstance(src, dict) else src if isinstance(src, str) else [
        s.get("url") if isinstance(s, dict) else s for s in src or []]
    print(f"  {L.get('type')!s:13} {L.get('name')!s:28} visible={L.get('visible', True)!s:5} "
          f"segments={len(L.get('segments', []))} keys={sorted(k for k in L if k not in ('source', 'segments'))} src={src}")
