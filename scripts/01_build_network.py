"""
Step 1 - Download Bangalore's arterial road network and build the link graph.
Requires internet (queries the OpenStreetMap Overpass API via OSMnx).

    python scripts/01_build_network.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import road_network

if __name__ == "__main__":
    links, edge_index = road_network.build_links(verbose=True)
    road_network.save(links, edge_index)
    print("\nDone. Road network ready.")
