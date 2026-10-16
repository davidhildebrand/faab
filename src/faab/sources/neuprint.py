"""neuPrint access helper (hemibrain + Janelia optic-lobe, etc.).

Token resolution order (read-only use only):
  1. NEUPRINT_APPLICATION_CREDENTIALS env var (the raw token)
  2. ~/.neuprint/token.json  ->  {"token": "<JWT>"}

The token is a Janelia neuPrint auth token (free, from neuprint.janelia.org) and
is SEPARATE from the CAVE/SeaTable tokens. It is stored OUTSIDE the git repo so it
can never be committed.

    from faab.sources.neuprint import client, list_datasets
    c = client("hemibrain:v1.2.1")
"""
import os
import json

import neuprint

SERVER = "https://neuprint.janelia.org"
TOKEN_FILE = os.path.expanduser("~/.neuprint/token.json")

# Dataset tags verified available to this token (via list_datasets()).
HEMIBRAIN = "hemibrain:v1.2.1"     # central brain only (no optic lobe / no Dm2)
OPTIC_LOBE = "optic-lobe:v1.1"     # one optic lobe; 726 Dm2
MALE_CNS = "male-cns:v1.0"         # whole male CNS (brain+VNC, both OLs); 1452 Dm2 — most complete


def _token():
    t = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    if t:
        return t.strip()
    if os.path.exists(TOKEN_FILE):
        return json.load(open(TOKEN_FILE))["token"].strip()
    raise RuntimeError(
        "No neuPrint token found. Get one at https://neuprint.janelia.org "
        "(log in -> account -> Auth Token) and save it to ~/.neuprint/token.json "
        'as {"token": "..."} (see setup instructions).')


def client(dataset=HEMIBRAIN, server=SERVER):
    """Return a neuprint.Client for the given dataset (read-only queries)."""
    return neuprint.Client(server, dataset=dataset, token=_token())


def list_datasets(server=SERVER):
    """All datasets visible to this token (dataset tag -> info)."""
    return client(dataset=HEMIBRAIN, server=server).fetch_datasets()
