"""
Deep data analysis to understand characteristics and distribution shift.
"""
import pickle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def analyze_dataset(path, name):
    print(f"\n{'='*80}")
    print(f"Analyzing {name}")
    print(f"{'='*80}")

    with open(path, 'rb') as f:
        data = pickle.load(f)

    print(f"Total samples: {len(data)}")

    # Analyze locations
    all_locs = []
    all_targets = []
    seq_lens = []

    for sample in data:
        all_locs.extend(sample['X'])
        all_targets.append(sample['Y'])
        seq_lens.append(len(sample['X']))

    loc_counter = Counter(all_locs)
    target_counter = Counter(all_targets)

    print(f"\nSequence Statistics:")
    print(f"  Min length: {min(seq_lens)}")
    print(f"  Max length: {max(seq_lens)}")
    print(f"  Mean length: {np.mean(seq_lens):.2f}")
    print(f"  Median length: {np.median(seq_lens):.2f}")

    print(f"\nLocation Statistics:")
    print(f"  Unique locations in sequences: {len(loc_counter)}")
    print(f"  Unique target locations: {len(target_counter)}")
    print(f"  Most common location: {loc_counter.most_common(1)[0]}")
    print(f"  Most common target: {target_counter.most_common(1)[0]}")

    # Location frequency distribution
    freq_values = list(loc_counter.values())
    print(f"\nLocation Frequency Distribution:")
    print(f"  Locations appearing once: {sum(1 for v in freq_values if v == 1)}")
    print(f"  Locations appearing 2-5 times: {sum(1 for v in freq_values if 2 <= v <= 5)}")
    print(f"  Locations appearing 6-10 times: {sum(1 for v in freq_values if 6 <= v <= 10)}")
    print(f"  Locations appearing >10 times: {sum(1 for v in freq_values if v > 10)}")

    # Target frequency distribution
    target_freq_values = list(target_counter.values())
    print(f"\nTarget Frequency Distribution:")
    print(f"  Targets appearing once: {sum(1 for v in target_freq_values if v == 1)}")
    print(f"  Targets appearing 2-5 times: {sum(1 for v in target_freq_values if 2 <= v <= 5)}")
    print(f"  Targets appearing >5 times: {sum(1 for v in target_freq_values if v > 5)}")
    print(f"  Top 10 most common targets: {target_counter.most_common(10)}")

    # Temporal features analysis
    weekdays = [sample['weekday_X'][-1] for sample in data]
    start_mins = [sample['start_min_X'][-1] for sample in data]

    print(f"\nTemporal Features:")
    weekday_dist = Counter(weekdays)
    print(f"  Weekday distribution: {dict(sorted(weekday_dist.items()))}")
    print(f"  Hour distribution (last visit):")
    hours = [int(sm / 60) for sm in start_mins]
    hour_dist = Counter(hours)
    for h in range(0, 24, 4):
        count = sum(hour_dist[i] for i in range(h, min(h+4, 24)))
        print(f"    {h:02d}:00-{min(h+4, 24):02d}:00: {count}")

    return loc_counter, target_counter, seq_lens

# Analyze all three splits
train_locs, train_targets, train_lens = analyze_dataset(
    '/content/lose_expr_2/data/geolife/geolife_transformer_7_train.pk', 'TRAIN'
)
val_locs, val_targets, val_lens = analyze_dataset(
    '/content/lose_expr_2/data/geolife/geolife_transformer_7_validation.pk', 'VALIDATION'
)
test_locs, test_targets, test_lens = analyze_dataset(
    '/content/lose_expr_2/data/geolife/geolife_transformer_7_test.pk', 'TEST'
)

# Analyze distribution shift
print(f"\n{'='*80}")
print("DISTRIBUTION SHIFT ANALYSIS")
print(f"{'='*80}")

# Compare target distributions
train_target_set = set(train_targets.keys())
val_target_set = set(val_targets.keys())
test_target_set = set(test_targets.keys())

print(f"\nTarget Location Coverage:")
print(f"  Train unique targets: {len(train_target_set)}")
print(f"  Val unique targets: {len(val_target_set)}")
print(f"  Test unique targets: {len(test_target_set)}")
print(f"  Val targets not in train: {len(val_target_set - train_target_set)}")
print(f"  Test targets not in train: {len(test_target_set - train_target_set)}")
print(f"  Test targets not in train or val: {len(test_target_set - train_target_set - val_target_set)}")

# Analyze sequence context locations
all_train_locs = set(train_locs.keys())
all_val_locs = set(val_locs.keys())
all_test_locs = set(test_locs.keys())

print(f"\nSequence Location Coverage:")
print(f"  Train unique locations: {len(all_train_locs)}")
print(f"  Val unique locations: {len(all_val_locs)}")
print(f"  Test unique locations: {len(all_test_locs)}")
print(f"  Val locations not in train: {len(all_val_locs - all_train_locs)}")
print(f"  Test locations not in train: {len(all_test_locs - all_train_locs)}")

# Top-K accuracy potential
print(f"\nTop-K Target Coverage Analysis:")
train_top_targets = [loc for loc, _ in train_targets.most_common(100)]
for k in [1, 5, 10, 20, 50, 100]:
    top_k = set(train_top_targets[:k])
    val_coverage = sum(val_targets[t] for t in top_k if t in val_targets) / len(val_lens)
    test_coverage = sum(test_targets[t] for t in top_k if t in test_targets) / len(test_lens)
    print(f"  Top-{k:3d} coverage: Val={val_coverage*100:.2f}%, Test={test_coverage*100:.2f}%")

print(f"\n{'='*80}")
print("KEY INSIGHTS")
print(f"{'='*80}")
print("1. Check for targets in test that never appear in train")
print("2. Analyze if val/test have different temporal patterns")
print("3. Look for location co-occurrence patterns")
print("4. Understand user-specific behavior patterns")
