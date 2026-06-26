"""
road_network.py
Download Bangalore's arterial road network from OpenStreetMap and turn it into
a graph the GNN can use: each ROAD SEGMENT (link) becomes a node, and two links
are connected if they meet at a junction. This is the "line graph" view, which
matches reality - congestion lives on road segments and spreads to the segments
they connect to.

Real, free data: OpenStreetMap via OSMnx.
"""

import sys
import json
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import osmnx as ox

import config as C


# --------------------------------------------------------------------------- #
# Robust parsing of messy OSM tags
# --------------------------------------------------------------------------- #
def _first(value):
    """OSM tags are sometimes lists; take the first sensible entry."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _parse_speed(maxspeed, highway):
    ms = _first(maxspeed)
    if ms is not None:
        digits = "".join(ch for ch in str(ms) if ch.isdigit())
        if digits:
            val = int(digits)
            if "mph" in str(ms).lower():
                val = int(val * 1.609)
            if 5 <= val <= 120:
                return val
    return C.DEFAULT_SPEED_KMH.get(_first(highway), 40)


def _parse_lanes(lanes):
    ln = _first(lanes)
    try:
        return max(1, min(6, int(float(ln))))
    except (TypeError, ValueError):
        return 2


def _class_ordinal(highway):
    hw = _first(highway) or "tertiary"
    hw = hw.replace("_link", "")
    if hw in C.ROAD_CLASS_ORDER:
        return C.ROAD_CLASS_ORDER.index(hw)
    return 0


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------- #
# Main builder
# --------------------------------------------------------------------------- #
def build_links(verbose=True):
    if verbose:
        print(f"Downloading OSM drive network for bbox {C.BBOX} ...")
    G = ox.graph_from_bbox(
        bbox=C.BBOX,
        network_type=C.NETWORK_TYPE,
        simplify=True,
        custom_filter=C.MAJOR_ROADS_FILTER,
    )

    # Project to metres, then merge near-duplicate intersection nodes.
    G = ox.project_graph(G)
    try:
        G = ox.consolidate_intersections(
            G, tolerance=C.CONSOLIDATE_TOLERANCE_M,
            rebuild_graph=True, dead_ends=False, reconnect_edges=True,
        )
    except Exception as exc:  # noqa: BLE001 - consolidation is best-effort
        if verbose:
            print(f"  (intersection consolidation skipped: {exc})")

    # Edges in projected CRS (for length), geometry reprojected to lat/lon for maps.
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()
    edges_ll = edges.to_crs(epsg=4326)

    n = len(edges)
    if verbose:
        print(f"Got {n} road segments after consolidation.")

    # Build per-link feature table.
    centroids = edges_ll.geometry.centroid
    rows = []
    for i in range(n):
        e = edges.iloc[i]
        cy, cx = centroids.iloc[i].y, centroids.iloc[i].x  # lat, lon
        rows.append({
            "link_id": i,
            "u": int(e["u"]), "v": int(e["v"]),
            "length_m": float(e.get("length", 50.0) or 50.0),
            "speed_kmh": _parse_speed(e.get("maxspeed"), e.get("highway")),
            "lanes": _parse_lanes(e.get("lanes")),
            "class_ord": _class_ordinal(e.get("highway")),
            "lat": cy, "lon": cx,
        })
    links = pd.DataFrame(rows)
    links["geometry"] = edges_ll.geometry.values  # lat/lon LineStrings for plotting

    # Hotspot proximity feature: nearest-hotspot influence in [0, 1].
    hot = np.zeros(n)
    for _, hlat, hlon, sev in C.HOTSPOTS:
        d = _haversine_m(links["lat"].values, links["lon"].values, hlat, hlon)
        infl = sev * np.exp(-(d / C.HOTSPOT_RADIUS_M) ** 2)
        hot = np.maximum(hot, infl)
    links["hotspot"] = hot

    # --------------------------------------------------------------------- #
    # Link adjacency: two links connected if they share a junction node.
    # --------------------------------------------------------------------- #
    node_to_links = {}
    for i, row in links.iterrows():
        for node in (row["u"], row["v"]):
            node_to_links.setdefault(node, []).append(i)

    pairs = set()
    for incident in node_to_links.values():
        for a_idx in range(len(incident)):
            for b_idx in range(a_idx + 1, len(incident)):
                a, b = incident[a_idx], incident[b_idx]
                pairs.add((a, b))
                pairs.add((b, a))  # undirected -> both directions for GCN

    if pairs:
        edge_index = np.array(sorted(pairs), dtype=np.int64).T  # [2, E]
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    if verbose:
        avg_deg = edge_index.shape[1] / max(n, 1)
        print(f"Link graph: {n} nodes, {edge_index.shape[1]} edges "
              f"(avg degree {avg_deg:.1f}).")
    return links, edge_index


def save(links, edge_index):
    import geopandas as gpd
    gdf = gpd.GeoDataFrame(links, geometry="geometry", crs="EPSG:4326")
    gdf.to_pickle(C.LINKS_PATH)
    np.save(C.EDGE_INDEX_PATH, edge_index)
    meta = {
        "n_links": int(len(links)),
        "n_graph_edges": int(edge_index.shape[1]),
        "bbox": C.BBOX,
        "hotspots": C.HOTSPOTS,
    }
    C.META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Saved links -> {C.LINKS_PATH}")
    print(f"Saved edge_index -> {C.EDGE_INDEX_PATH}")
    print(f"Saved meta -> {C.META_PATH}")


def load():
    links = pd.read_pickle(C.LINKS_PATH)
    edge_index = np.load(C.EDGE_INDEX_PATH)
    return links, edge_index
