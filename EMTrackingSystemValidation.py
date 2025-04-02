import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from mpl_toolkits.mplot3d import Axes3D


# =======================
# Data Loading & Preprocessing
# =======================

def load_csv(file_path):
    df = pd.read_csv(file_path, delimiter=",", skiprows=1)
    return df


# =======================
# Chunk Extraction
# =======================

def extract_stationary_chunks(df, distance_threshold=10.0, min_distance=40.0, window_size=250, chunk_size=200):
    all_candidates = []
    total_points = len(df)
    i = 0

    while i + window_size <= total_points:
        window = df.iloc[i:i + window_size]
        max_range = window[['x', 'y', 'z']].max() - window[['x', 'y', 'z']].min()
        if (max_range < distance_threshold).all():
            midpoint = i + window_size // 2
            middle_chunk = df.iloc[midpoint - chunk_size//2: midpoint + chunk_size//2]
            all_candidates.append(middle_chunk)
            i += window_size
        else:
            i += 10

    selected_chunks = []
    chunk_centers = []

    for chunk in all_candidates:
        avg_pos = chunk[['x', 'y', 'z']].median().values
        if not any(np.linalg.norm(avg_pos - other) < min_distance for other in chunk_centers):
            selected_chunks.append(chunk)
            chunk_centers.append(avg_pos)

    print(f"Found {len(selected_chunks)} well-separated chunks.")
    return pd.concat(selected_chunks, ignore_index=True)

# =======================
# Grid Index Assignment
# =======================

def assign_snake_grid_indices(avg_positions, shape=(5, 5, 4)):
    grid_coords = []
    idx = 0
    x_dim, y_dim, z_dim = shape

    for z in range(z_dim):
        for y in range(y_dim):
            x_range = range(x_dim) if y % 2 == 0 else reversed(range(x_dim))
            for x in x_range:
                if idx >= len(avg_positions): break
                grid_coords.append((x + 1, y + 1, z + 1))
                idx += 1

    coords_df = pd.DataFrame(avg_positions, columns=['x', 'y', 'z'])
    coords_df[['grid_x', 'grid_y', 'grid_z']] = pd.DataFrame(grid_coords[:len(coords_df)])
    return coords_df


def assign_grid_indices_no_snake(avg_positions, shape=(5, 5, 4)):
    grid_coords = []
    idx = 0
    x_dim, y_dim, z_dim = shape

    for z in range(z_dim):
        for y in range(y_dim):
            for x in range(x_dim):
                if idx >= len(avg_positions): break
                grid_coords.append((x + 1, y + 1, z + 1))
                idx += 1

    coords_df = pd.DataFrame(avg_positions, columns=['x', 'y', 'z'])
    coords_df[['grid_x', 'grid_y', 'grid_z']] = pd.DataFrame(grid_coords[:len(coords_df)])
    return coords_df


# =======================
# Analysis Functions
# =======================

def analyze_relative_distances(df, block_size=200):
    num_points = len(df)
    num_blocks = num_points // block_size

    avg_positions = []
    std_devs = []

    for i in range(num_blocks):
        block = df.iloc[i*block_size:(i+1)*block_size]
        avg = block[['x', 'y', 'z']].mean().values
        std = block[['x', 'y', 'z']].std().values
        avg_positions.append(avg)
        std_devs.append(std)

    avg_positions = np.array(avg_positions)
    std_devs = np.array(std_devs)

    dist_mat = distance_matrix(avg_positions, avg_positions)
    np.fill_diagonal(dist_mat, np.inf)
    nearest_distances = np.min(dist_mat, axis=1)

    print("\n--- Relative Distance Analysis ---")
    print(f"95% Confidence Interval of samples @ each point: ±{np.mean(std_devs)*2:.4f} mm")
    print(f"Mean distance between Points: {np.mean(nearest_distances):.4f} mm")
    print(f"95% Confidence Interval: ±{np.std(nearest_distances)*2:.4f} mm")

    return avg_positions, nearest_distances, std_devs

def analyze_absolute_distances(coords_df, spacing=95.0, reference_grid=(3, 3, 1)):
    ref_point = coords_df[
        (coords_df['grid_x'] == reference_grid[0]) &
        (coords_df['grid_y'] == reference_grid[1]) &
        (coords_df['grid_z'] == reference_grid[2])
    ]

    print("reference point: ", ref_point)

    if ref_point.empty:
        print(f"Reference point {reference_grid} not found.")
        return coords_df

    shift_x, shift_y, shift_z = ref_point[['x', 'y', 'z']].values[0]
    coords_df['x'] -= shift_x
    coords_df['y'] -= shift_y
    coords_df['z'] -= shift_z

    coords_df['expected_x'] = (coords_df['grid_x'] - reference_grid[0]) * spacing
    coords_df['expected_y'] = (coords_df['grid_y'] - reference_grid[1]) * spacing
    coords_df['expected_z'] = (coords_df['grid_z'] - reference_grid[2]) * spacing

    deviation = coords_df[['x', 'y', 'z']].values - coords_df[['expected_x', 'expected_y', 'expected_z']].values
    coords_df['deviation_mag'] = np.linalg.norm(deviation, axis=1)

    # print("\n--- Absolute Distance Analysis ---")
    print(f"Mean deviation: {coords_df['deviation_mag'].mean():.4f} mm")
    print(f"95% CI: ±{coords_df['deviation_mag'].std()*2:.4f} mm")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(coords_df['x'], coords_df['y'], coords_df['z'],
                    c=coords_df['deviation_mag'], cmap='plasma', s=100)
    plt.colorbar(sc, label='Deviation (mm)')
    ax.set_title("Deviation from Expected Grid")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

    return coords_df


# =======================
# Plotting Utilities
# =======================

def plot_df(df, title="3D Plot", color='blue', label='Data', s=50):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(df['x'], df['y'], df['z'], c=color, label=label, s=s)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_distance_heatmap(avg_positions, distances):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(avg_positions[:, 0], avg_positions[:, 1], avg_positions[:, 2],
                    c=distances, cmap='viridis', s=100)
    plt.colorbar(sc, label='Distance to Nearest Neighbor (mm)')
    ax.set_title("3D Distance Heatmap")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


# =======================
# Main Script
# =======================

file_path = "/Users/rebeccaedelman/Downloads/locs_00001.csv"
df = load_csv(file_path)
df_stationary = extract_stationary_chunks(df)
plot_df(df_stationary)

# # Analyze full data
avg_positions, distances, std_devs = analyze_relative_distances(df_stationary)

print("\n--- Total Matrix ---")
coords_df = assign_snake_grid_indices(avg_positions)
coords_df = analyze_absolute_distances(coords_df, reference_grid=(3, 3, 1))

# Analyze Z slice (grid_x = 2, 3, 4 AND grid_z = 2, 3)
print("\n--- Middle Z Layers ---")
mid_z = assign_snake_grid_indices(avg_positions)
mid_z = mid_z[
    mid_z['grid_z'].isin([2, 3])
].copy()
mid_z = analyze_absolute_distances(mid_z, reference_grid=(3, 3, 2))

# Analyze middle X and Z slice (grid_x = 2, 3, 4 AND grid_z = 2, 3)
print("\n--- Middle Z & X Layers ---")
mid_xz = assign_snake_grid_indices(avg_positions)
mid_xz = mid_xz[
    mid_xz['grid_x'].isin([2, 3, 4]) &
    mid_xz['grid_z'].isin([2, 3])
].copy()
mid_xz = analyze_absolute_distances(mid_xz, reference_grid=(3, 3, 2))


#### USE TO CHECK IF ROWS / COL ASSIGNMENTS ARE OK (they got funky when we introduced the metal at the top layer) ####
# # Plot final filtered region
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(coords_df['x'], coords_df['y'], coords_df['z'], c=coords_df['grid_x'], cmap='coolwarm', s=100)
# plt.colorbar(sc, label='Grid X Index')
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.show()

# # Plot final filtered region
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(coords_df['x'], coords_df['y'], coords_df['z'], c=coords_df['grid_y'], cmap='coolwarm', s=100)
# plt.colorbar(sc, label='Grid Y Index')
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.show()
