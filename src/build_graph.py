"""
Phase 3: Spatial graph over the monitoring stations.

Edges: each station connects to its K nearest neighbours (made symmetric), weighted by a
Gaussian kernel of Haversine distance, w_ij = exp(-(d_ij / sigma)^2), sigma = std of all
pairwise distances. Self-loops have weight 1. The GAT uses log(w_ij) as an attention bias,
so nearer stations start with more influence but the model can learn to override it.
"""
import numpy as np
import pandas as pd

import config as C


def haversine_matrix(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def main():
    st = pd.read_csv(C.STATIONS_CSV)
    n = len(st)
    dist = haversine_matrix(st["lat"].values, st["lon"].values).astype(np.float32)
    off = dist[~np.eye(n, dtype=bool)]
    sigma = float(off.std())

    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        nn_idx = np.argsort(dist[i])[1:C.KNN + 1]
        adj[i, nn_idx] = np.exp(-(dist[i, nn_idx] / sigma) ** 2)
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 1.0)

    np.save(C.DIST_NPY, dist)
    np.save(C.ADJ_NPY, adj)
    edges = int((adj > 0).sum() - n) // 2
    print(f"Graph: {n} stations, {edges} undirected edges (K={C.KNN}), sigma={sigma:.2f} km")
    print(f"Distances: min={off.min():.2f} km, mean={off.mean():.2f} km, max={off.max():.2f} km")


if __name__ == "__main__":
    main()
