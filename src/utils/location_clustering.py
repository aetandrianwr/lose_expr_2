"""
Location clustering for hierarchical embeddings
"""
import pickle
import numpy as np
import torch
from sklearn.cluster import KMeans
from collections import defaultdict, Counter


def compute_location_cooccurrence(data_path):
    """
    Compute location co-occurrence matrix for clustering
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # Build co-occurrence matrix
    cooccur = defaultdict(lambda: defaultdict(int))

    for sample in data:
        locs = sample['X']
        # Count co-occurrences in sliding windows
        for i in range(len(locs)):
            for j in range(max(0, i-5), min(len(locs), i+6)):  # Window of 5
                if i != j:
                    cooccur[locs[i]][locs[j]] += 1

    return cooccur


def cluster_locations(train_path, num_locations, num_clusters=50):
    """
    Cluster locations based on co-occurrence patterns

    Returns:
        loc_to_cluster: torch.LongTensor mapping location ID to cluster ID
        cluster_centers: cluster information
    """
    print("Computing location co-occurrence...")
    cooccur = compute_location_cooccurrence(train_path)

    # Build feature matrix: for each location, its co-occurrence vector
    print(f"Building feature matrix for {num_locations} locations...")
    feature_matrix = np.zeros((num_locations, num_locations))

    for loc1 in range(num_locations):
        if loc1 in cooccur:
            for loc2, count in cooccur[loc1].items():
                feature_matrix[loc1, loc2] = count

    # Normalize rows
    row_sums = feature_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    feature_matrix = feature_matrix / row_sums

    # Apply KMeans clustering
    print(f"Clustering into {num_clusters} clusters...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(feature_matrix)

    # Convert to torch tensor
    loc_to_cluster = torch.LongTensor(cluster_labels)

    print(f"Clustering complete!")
    cluster_sizes = Counter(cluster_labels)
    print(f"Cluster sizes: min={min(cluster_sizes.values())}, max={max(cluster_sizes.values())}, mean={np.mean(list(cluster_sizes.values())):.1f}")

    return loc_to_cluster, kmeans.cluster_centers_


def compute_location_frequency_buckets(train_path, num_locations, num_buckets=10):
    """
    Compute frequency buckets for each location (log-scaled)

    Returns:
        loc_freq_bucket: torch.LongTensor mapping location ID to frequency bucket
        location_freq: torch.FloatTensor with normalized frequency for each location
    """
    with open(train_path, 'rb') as f:
        data = pickle.load(f)

    # Count target frequencies
    target_counts = Counter()
    for sample in data:
        target_counts[sample['Y']] += 1

    # Also count in sequences
    seq_counts = Counter()
    for sample in data:
        seq_counts.update(sample['X'])

    # Combine both (target is more important)
    combined_counts = Counter()
    for loc in range(num_locations):
        combined_counts[loc] = target_counts.get(loc, 0) * 2 + seq_counts.get(loc, 0)

    # Convert to array
    freq_array = np.array([combined_counts[i] for i in range(num_locations)])

    # Normalize for probability distribution
    freq_probs = freq_array / freq_array.sum()

    # Compute log-frequency buckets
    log_freq = np.log1p(freq_array)  # log(1 + freq)
    min_log = log_freq[log_freq > 0].min() if (log_freq > 0).any() else 0
    max_log = log_freq.max()

    # Create buckets
    freq_buckets = np.zeros(num_locations, dtype=np.int64)
    if max_log > min_log:
        freq_buckets = ((log_freq - min_log) / (max_log - min_log) * (num_buckets - 1)).astype(np.int64)
        freq_buckets = np.clip(freq_buckets, 0, num_buckets - 1)

    print(f"Frequency bucketing complete!")
    print(f"Bucket distribution: {Counter(freq_buckets)}")

    return torch.LongTensor(freq_buckets), torch.FloatTensor(freq_probs)
