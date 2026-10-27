"""Upload long Spelunker states (from figures/*_ng.txt files) to the CAVE neuroglancer state server and print short
links. A state upload only stores a view; it does not touch segmentation or annotation tables. <= 1 CAVE call / 5 s.
Usage: ng_short_link.py PATH_TO_ng.txt [...]"""
import json
import sys
import urllib.parse

from faab.clients.cave import client as cave_client, throttle

client = cave_client()
t = 0.0
for path in sys.argv[1:]:
    url = open(path).read().strip()
    state = json.loads(urllib.parse.unquote(url.split("#!", 1)[1]))
    t = throttle(t)
    sid = client.state.upload_state_json(state)
    short = client.state.build_neuroglancer_url(sid, ngl_url="https://spelunker.cave-explorer.org")
    print(f"{path.rsplit('/', 1)[-1]} ({len(url)} chars) -> {short}")
