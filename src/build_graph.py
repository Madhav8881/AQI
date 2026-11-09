"""
Phase 3: Spatial Graph Construction for 12 Delhi Air Quality Monitoring Stations.
Computes Haversine pairwise distance matrix and builds thresholded Gaussian adjacency matrix.
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in kilometers between two points on the Earth."""
    R = 6371.0 # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def build_spatial_graph():
    print("=" * 60)
    print(" Phase 3: Constructing Spatial Adjacency Matrix ")
    print("=" * 60)
    
    stations_file = os.path.join(RAW_DIR, "selected_stations.csv")
    df = pd.read_csv(stations_file)
    n = len(df)
    print(f"Loaded {n} monitoring stations.")
    
    # 1. Compute Haversine distance matrix (N x N in km)
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = haversine_distance(
                df.loc[i, 'lat'], df.loc[i, 'lon'],
                df.loc[j, 'lat'], df.loc[j, 'lon']
            )
            
    print(f"Pairwise distances (km): Min={dist_matrix[dist_matrix > 0].min():.2f} km, "
          f"Mean={dist_matrix[dist_matrix > 0].mean():.2f} km, Max={dist_matrix.max():.2f} km")
    
    # 2. Build Thresholded Gaussian Kernel Adjacency Matrix
    # Standard formula: A_ij = exp(-(dist_ij / sigma)^2) if dist_ij <= threshold, else 0
    sigma = np.std(dist_matrix[dist_matrix > 0]) # standard deviation of pairwise distances
    dist_threshold = 25.0 # Connect stations within 25 km
    
    adj_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if dist_matrix[i, j] <= dist_threshold:
                adj_matrix[i, j] = np.exp(- (dist_matrix[i, j] / sigma) ** 2)
            else:
                adj_matrix[i, j] = 0.0
                
    # Add Self-loops (A_ii = 1.0)
    np.fill_diagonal(adj_matrix, 1.0)
    
    # Save distance and adjacency matrices
    np.save(os.path.join(PROCESSED_DIR, "distance_matrix.npy"), dist_matrix)
    np.save(os.path.join(PROCESSED_DIR, "adjacency_matrix.npy"), adj_matrix)
    print("Saved 'distance_matrix.npy' and 'adjacency_matrix.npy' to data/processed/")
    
    # Print adjacency summary
    edge_count = np.sum(adj_matrix > 0) - n # exclude self loops
    density = edge_count / (n * (n - 1))
    print(f"Graph Statistics: {n} Nodes, {edge_count // 2} Undirected Edges, Graph Density: {density:.2%}")
    
    # 3. Generate Spatial Graph Plot
    plt.figure(figsize=(9, 7))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Draw edges
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j] > 0:
                weight = adj_matrix[i, j]
                plt.plot(
                    [df.loc[i, 'lon'], df.loc[j, 'lon']],
                    [df.loc[i, 'lat'], df.loc[j, 'lat']],
                    color='#3B82F6', alpha=float(weight * 0.7), linewidth=float(weight * 3.5), zorder=1
                )
                
    # Draw station nodes
    plt.scatter(df['lon'], df['lat'], color='#EF4444', s=130, edgecolor='black', linewidth=1.5, zorder=2)
    
    # Annotate station names
    for _, row in df.iterrows():
        plt.annotate(
            row['name'],
            (row['lon'], row['lat']),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8.5,
            fontweight='bold',
            color='#1F2937'
        )
        
    plt.title("Spatial Connectivity Graph of Delhi NCR Air Quality Monitoring Stations", fontsize=12, pad=12, fontweight='bold')
    plt.xlabel("Longitude (°E)", fontsize=10)
    plt.ylabel("Latitude (°N)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_DIR, "station_spatial_graph.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved spatial network graph visualization to: {plot_path}")
    print("=" * 60)
    return adj_matrix, dist_matrix

if __name__ == "__main__":
    build_spatial_graph()
