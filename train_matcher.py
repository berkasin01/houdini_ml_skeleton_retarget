import os
import glob
import json
import argparse
import re
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

FINGER_KEYWORDS = ['thumb', 'index', 'middle', 'ring', 'pinky']

BODY_KEYWORDS = ['spine', 'head', 'neck', 'shoulder', 'clavicle',
                 'arm', 'forearm', 'elbow', 'wrist', 'hand',
                 'hip', 'pelvis', 'thigh', 'leg', 'knee', 'ankle', 'foot', 'toe',
                 'jaw', 'eye', 'eyelid', 'eyebrow', 'brow', 'mouth', 'lip',
                 'twist', 'calf', 'shin', 'upperarm', 'lowerarm', 'upperleg', 'lowerleg',
                 'upleg', 'lowleg', 'root', 'ball']

SIDE_KEYWORDS = ['left', 'right', 'l', 'r']

SYNONYMS = {
    'hips': 'pelvis',
    'hip': 'pelvis',
    'clavicle': 'shoulder',
    'upperarm': 'arm',
    'upper_arm': 'arm',
    'lowerarm': 'forearm',
    'lower_arm': 'forearm',
    'thigh': 'upperleg',
    'upleg': 'upperleg',
    'up_leg': 'upperleg',
    'calf': 'lowerleg',
    'shin': 'lowerleg',
    'lowleg': 'lowerleg',
    'low_leg': 'lowerleg',
    'brow': 'eyebrow',
}


def normalize_token(token):
    return SYNONYMS.get(token, token)


def extract_name_tokens(name):
    name_lower = name.lower()
    tokens = set()

    parts = re.split(r'[_\-.:]+', name_lower)

    camel_parts = []
    for part in parts:
        camel_split = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', part)
        camel_parts.extend([p.lower() for p in camel_split])

    parts = parts + camel_parts

    for part in parts:
        part = part.strip()
        if not part:
            continue

        part_normalized = normalize_token(part)

        for kw in FINGER_KEYWORDS:
            if kw in part_normalized:
                tokens.add(kw)

        for kw in BODY_KEYWORDS:
            if kw in part_normalized:
                tokens.add(normalize_token(kw))

        if part_normalized in BODY_KEYWORDS or part_normalized in SYNONYMS.values():
            tokens.add(part_normalized)

        if part in ['l', 'left']:
            tokens.add('left')
        elif part in ['r', 'right']:
            tokens.add('right')
        elif part.startswith('l_') or part.endswith('_l'):
            tokens.add('left')
        elif part.startswith('r_') or part.endswith('_r'):
            tokens.add('right')

        numbers = re.findall(r'\d+', part)
        for num in numbers:
            tokens.add(str(int(num)))

    if name_lower.startswith('l_') or name_lower.startswith('l.') or name_lower.endswith('_l') or name_lower.endswith('.l'):
        tokens.add('left')
    if name_lower.startswith('r_') or name_lower.startswith('r.') or name_lower.endswith('_r') or name_lower.endswith('.r'):
        tokens.add('right')

    if 'twist' in name_lower:
        tokens.add('twist')

    for kw in BODY_KEYWORDS:
        if kw in name_lower:
            tokens.add(normalize_token(kw))

    for syn, canonical in SYNONYMS.items():
        if syn in name_lower:
            tokens.add(canonical)

    return tokens


def compute_name_similarity_features(name_a, name_b):
    tokens_a = extract_name_tokens(name_a)
    tokens_b = extract_name_tokens(name_b)

    features = []

    if len(tokens_a) == 0 and len(tokens_b) == 0:
        jaccard = 0.5
    else:
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        jaccard = intersection / union if union > 0 else 0.0
    features.append(jaccard)

    finger_a = tokens_a & set(FINGER_KEYWORDS)
    finger_b = tokens_b & set(FINGER_KEYWORDS)

    if finger_a and finger_b:
        finger_match = 1.0 if finger_a == finger_b else 0.0
    elif finger_a or finger_b:
        finger_match = 0.0
    else:
        finger_match = 0.5
    features.append(finger_match)

    nums_a = set(t for t in tokens_a if t.isdigit())
    nums_b = set(t for t in tokens_b if t.isdigit())

    if nums_a and nums_b:
        num_match = 1.0 if nums_a & nums_b else 0.0
    elif nums_a or nums_b:
        num_match = 0.0
    else:
        num_match = 0.5
    features.append(num_match)

    side_a = 'left' if 'left' in tokens_a else ('right' if 'right' in tokens_a else 'center')
    side_b = 'left' if 'left' in tokens_b else ('right' if 'right' in tokens_b else 'center')

    if side_a == 'center' or side_b == 'center':
        side_match = 0.5
    else:
        side_match = 1.0 if side_a == side_b else 0.0
    features.append(side_match)

    body_keywords_normalized = set(normalize_token(kw) for kw in BODY_KEYWORDS)
    body_a = tokens_a & body_keywords_normalized
    body_b = tokens_b & body_keywords_normalized
    body_overlap = len(body_a & body_b)
    features.append(float(body_overlap))

    is_finger = 1.0 if (finger_a or finger_b) else 0.0
    features.append(is_finger)

    return np.array(features, dtype=np.float32)


def load_rig(path):
    df = pd.read_csv(path)
    n = len(df)

    names = df["name"].astype(str).to_list()
    parent = df["parent_idx"].to_numpy(dtype=np.int32)
    pos = df[["pos_root_x", "pos_root_y", "pos_root_z"]].to_numpy(dtype=np.float32)
    vecp = df[["vec_parent_x", "vec_parent_y", "vec_parent_z"]].to_numpy(dtype=np.float32)
    blen = df["bone_len"].to_numpy(dtype=np.float32)

    if "depth" in df.columns:
        depth = df["depth"].to_numpy(dtype=np.int32)
    else:
        root = int(np.where(parent < 0)[0][0]) if np.any(parent < 0) else 0
        depth = compute_depth(parent, root, n)

    children = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if 0 <= p < n:
            children[p].append(i)

    child_count = np.array([len(c) for c in children], dtype=np.int32)
    is_leaf = (child_count == 0).astype(np.int32)

    return {
        "path": path,
        "filename": os.path.basename(path),
        "n": n,
        "name": names,
        "name_to_idx": {name: i for i, name in enumerate(names)},
        "parent": parent,
        "children": children,
        "pos": pos,
        "vecp": vecp,
        "blen": blen,
        "depth": depth,
        "child_count": child_count,
        "is_leaf": is_leaf,
        "root": int(np.where(parent < 0)[0][0]) if np.any(parent < 0) else 0,
    }


def compute_depth(parent, root, n):
    children = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if 0 <= p < n:
            children[p].append(i)

    depth = np.full(n, 0, dtype=np.int32)
    q = [root]
    for u in q:
        for v in children[u]:
            depth[v] = depth[u] + 1
            q.append(v)
    return depth


def median_scale(blen):
    x = blen[blen > 1e-8]
    return float(np.median(x)) if len(x) else 1.0


def safe_normalize(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return v / norms


def extract_joint_features(rig):
    pos = rig["pos"]
    vecp = rig["vecp"]
    blen = rig["blen"]
    depth = rig["depth"]
    child_count = rig["child_count"]
    is_leaf = rig["is_leaf"]

    n = len(pos)
    scale = median_scale(blen)

    pos_norm = pos / scale
    dir_parent = safe_normalize(vecp)
    blen_norm = (blen / scale).reshape(-1, 1)

    max_depth = max(depth.max(), 1)
    depth_norm = (depth / max_depth).astype(np.float32).reshape(-1, 1)

    max_cc = max(child_count.max(), 1)
    cc_norm = (child_count / max_cc).astype(np.float32).reshape(-1, 1)

    leaf = is_leaf.astype(np.float32).reshape(-1, 1)

    features = np.hstack([
        pos_norm,
        dir_parent,
        blen_norm,
        depth_norm,
        cc_norm,
        leaf,
    ]).astype(np.float32)

    return features


def make_pair_features(f_a, f_b, name_a, name_b):
    diff = np.abs(f_a - f_b)
    prod = f_a * f_b
    name_feat = compute_name_similarity_features(name_a, name_b)
    return np.concatenate([f_a, f_b, diff, prod, name_feat]).astype(np.float32)


def augment_rig(rig):
    n = rig["n"]

    pos = rig["pos"].copy()
    vecp = rig["vecp"].copy()
    blen = rig["blen"].copy()
    parent = rig["parent"].copy()
    names = rig["name"].copy()

    scale = median_scale(blen)

    pos += rng.normal(0, 0.01 * scale, size=pos.shape).astype(np.float32)

    blen_factor = rng.normal(1.0, 0.05, size=n).astype(np.float32)
    blen = blen * np.clip(blen_factor, 0.9, 1.1)

    vecp_dir = safe_normalize(vecp)
    vecp = vecp_dir * blen.reshape(-1, 1)

    global_scale = rng.uniform(0.9, 1.1)
    pos *= global_scale
    blen *= global_scale
    vecp *= global_scale

    mapping = np.arange(n, dtype=np.int32)

    if rng.random() < 0.2:
        n_junk = rng.integers(1, 4)
        for junk_i in range(n_junk):
            p = rng.integers(0, n)
            direction = rng.normal(0, 1, size=3).astype(np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            length = rng.uniform(0.1, 0.5) * scale * global_scale

            new_pos = pos[p] + direction * length
            new_vecp = direction * length

            pos = np.vstack([pos, new_pos.reshape(1, 3)])
            vecp = np.vstack([vecp, new_vecp.reshape(1, 3)])
            blen = np.append(blen, length)
            parent = np.append(parent, p)
            mapping = np.append(mapping, -1)
            names = names + [f"junk_{junk_i}"]

    n_aug = len(parent)
    children = [[] for _ in range(n_aug)]
    for i, p in enumerate(parent):
        if 0 <= p < n_aug:
            children[p].append(i)

    child_count = np.array([len(c) for c in children], dtype=np.int32)
    is_leaf = (child_count == 0).astype(np.int32)

    root = int(np.where(parent < 0)[0][0]) if np.any(parent < 0) else 0
    depth = compute_depth(parent, root, n_aug)

    return {
        "n": n_aug,
        "name": names,
        "parent": parent,
        "children": children,
        "pos": pos,
        "vecp": vecp,
        "blen": blen,
        "depth": depth,
        "child_count": child_count,
        "is_leaf": is_leaf,
    }, mapping


def build_dataset(rigs, labels_data, aug_per_rig=100, cross_rig_repeats=50, neg_per_pos=4):
    X = []
    y = []

    rig_by_name = {r["filename"]: r for r in rigs}

    print(f"\n1. Processing labeled cross-rig pairs...")
    cross_rig_samples = 0

    for pair in labels_data.get("rig_pairs", []):
        rig_a_name = pair["rig_A"]
        rig_b_name = pair["rig_B"]
        matches = pair.get("matches", {})

        if rig_a_name not in rig_by_name:
            print(f"   Warning: {rig_a_name} not found")
            continue
        if rig_b_name not in rig_by_name:
            print(f"   Warning: {rig_b_name} not found")
            continue

        rig_A = rig_by_name[rig_a_name]
        rig_B = rig_by_name[rig_b_name]

        for repeat_i in range(cross_rig_repeats):
            if repeat_i == 0:
                aug_A, aug_B = rig_A, rig_B
                feat_A = extract_joint_features(aug_A)
                feat_B = extract_joint_features(aug_B)
                map_A = {i: i for i in range(rig_A["n"])}
                map_B = {i: i for i in range(rig_B["n"])}
            else:
                aug_A, map_A_arr = augment_rig(rig_A)
                aug_B, map_B_arr = augment_rig(rig_B)
                feat_A = extract_joint_features(aug_A)
                feat_B = extract_joint_features(aug_B)
                map_A = {int(map_A_arr[i]): i for i in range(len(map_A_arr)) if map_A_arr[i] >= 0}
                map_B = {int(map_B_arr[i]): i for i in range(len(map_B_arr)) if map_B_arr[i] >= 0}

            n_A = len(feat_A)
            n_B = len(feat_B)

            for a_name, b_name in matches.items():
                if a_name not in rig_A["name_to_idx"]:
                    continue
                if b_name not in rig_B["name_to_idx"]:
                    continue

                orig_a_idx = rig_A["name_to_idx"][a_name]
                orig_b_idx = rig_B["name_to_idx"][b_name]

                if repeat_i == 0:
                    a_idx, b_idx = orig_a_idx, orig_b_idx
                else:
                    if orig_a_idx not in map_A or orig_b_idx not in map_B:
                        continue
                    a_idx = map_A[orig_a_idx]
                    b_idx = map_B[orig_b_idx]

                name_a = aug_A["name"][a_idx]
                name_b = aug_B["name"][b_idx]

                X.append(make_pair_features(feat_A[a_idx], feat_B[b_idx], name_a, name_b))
                y.append(1.0)

                neg_indices = [i for i in range(n_B) if i != b_idx]
                if len(neg_indices) > neg_per_pos:
                    neg_indices = list(rng.choice(neg_indices, size=neg_per_pos, replace=False))

                for neg_idx in neg_indices:
                    neg_name = aug_B["name"][neg_idx]
                    X.append(make_pair_features(feat_A[a_idx], feat_B[neg_idx], name_a, neg_name))
                    y.append(0.0)

        valid_matches = len(matches)
        cross_rig_samples += valid_matches * cross_rig_repeats * (1 + neg_per_pos)
        print(f"   {rig_a_name} <-> {rig_b_name}: {valid_matches} matches")

    print(f"   Total cross-rig samples: {cross_rig_samples}")

    print(f"\n2. Processing same-rig augmentations ({aug_per_rig} per rig)...")

    for rig_idx, rig in enumerate(rigs):
        feat_orig = extract_joint_features(rig)
        n_orig = rig["n"]
        rig_samples = 0

        for aug_i in range(aug_per_rig):
            aug_rig, mapping = augment_rig(rig)
            feat_aug = extract_joint_features(aug_rig)
            n_aug = aug_rig["n"]

            for aug_idx in range(n_aug):
                orig_idx = mapping[aug_idx]

                if orig_idx >= 0:
                    name_aug = aug_rig["name"][aug_idx]
                    name_orig = rig["name"][orig_idx]

                    X.append(make_pair_features(feat_aug[aug_idx], feat_orig[orig_idx], name_aug, name_orig))
                    y.append(1.0)
                    rig_samples += 1

                    neg_indices = [i for i in range(n_orig) if i != orig_idx]
                    if len(neg_indices) > neg_per_pos:
                        neg_indices = rng.choice(neg_indices, size=neg_per_pos, replace=False)

                    for neg_idx in neg_indices:
                        neg_name = rig["name"][neg_idx]
                        X.append(make_pair_features(feat_aug[aug_idx], feat_orig[neg_idx], name_aug, neg_name))
                        y.append(0.0)
                        rig_samples += 1
                else:
                    name_junk = aug_rig["name"][aug_idx]
                    neg_indices = rng.choice(n_orig, size=min(neg_per_pos, n_orig), replace=False)
                    for neg_idx in neg_indices:
                        neg_name = rig["name"][neg_idx]
                        X.append(make_pair_features(feat_aug[aug_idx], feat_orig[neg_idx], name_junk, neg_name))
                        y.append(0.0)
                        rig_samples += 1

        print(f"   [{rig_idx + 1}/{len(rigs)}] {rig['filename']}: {rig_samples} samples")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y


def train_model(X, y, epochs=20, batch_size=512, lr=1e-3):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    mu = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    sd = X_train.std(axis=0, keepdims=True).astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd)

    X_train = (X_train - mu) / sd
    X_val = (X_val - mu) / sd

    print(f"\nTraining on {len(X_train)} samples, validating on {len(X_val)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Positive ratio: {y_train.mean():.3f}")

    model = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_auc', mode='max', patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc', mode='max', factor=0.5, patience=2, min_lr=1e-6
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    return model, mu, sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Path to ground truth labels JSON")
    parser.add_argument("--rigs", required=True, help="Directory containing rig CSVs")
    parser.add_argument("--out", default="joint_matcher_v3", help="Output prefix")
    parser.add_argument("--aug", type=int, default=100, help="Augmentations per rig")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    args = parser.parse_args()

    rig_files = sorted(glob.glob(os.path.join(args.rigs, "*.csv")))
    if not rig_files:
        print(f"No CSV files found in {args.rigs}")
        return

    print(f"Loading {len(rig_files)} rigs...")
    rigs = [load_rig(f) for f in rig_files]
    for rig in rigs:
        print(f"  {rig['filename']}: {rig['n']} joints")

    print(f"\nLoading labels from {args.labels}...")
    with open(args.labels, 'r') as f:
        labels_data = json.load(f)

    n_pairs = len(labels_data.get("rig_pairs", []))
    n_labels = sum(len(p.get("matches", {})) for p in labels_data.get("rig_pairs", []))
    print(f"  Found {n_pairs} rig pairs with {n_labels} total labels")

    X, y = build_dataset(rigs, labels_data, aug_per_rig=args.aug, cross_rig_repeats=50)

    print(f"\n=== Dataset Summary ===")
    print(f"Total samples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Positive: {int(y.sum())} ({100 * y.mean():.1f}%)")
    print(f"Negative: {int(len(y) - y.sum())} ({100 * (1 - y.mean()):.1f}%)")

    model, mu, sd = train_model(X, y, epochs=args.epochs)

    model.save(f"{args.out}.keras")
    np.savez(f"{args.out}_norm.npz", mu=mu, sd=sd)

    print(f"\nSaved: {args.out}.keras")
    print(f"Saved: {args.out}_norm.npz")


if __name__ == "__main__":
    main()