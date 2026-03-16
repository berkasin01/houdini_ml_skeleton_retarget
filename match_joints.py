import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import linear_sum_assignment


def load_rig(path):
    df = pd.read_csv(path)
    n = len(df)

    names = df["name"].astype(str).to_list()
    ptnum = df["ptnum"].to_numpy(dtype=np.int32) if "ptnum" in df.columns else np.arange(n, dtype=np.int32)
    parent = df["parent_idx"].to_numpy(dtype=np.int32)
    pos = df[["pos_root_x", "pos_root_y", "pos_root_z"]].to_numpy(dtype=np.float32)
    vecp = df[["vec_parent_x", "vec_parent_y", "vec_parent_z"]].to_numpy(dtype=np.float32)
    blen = df["bone_len"].to_numpy(dtype=np.float32)

    # Get or compute depth
    if "depth" in df.columns:
        depth = df["depth"].to_numpy(dtype=np.int32)
    else:
        root = int(np.where(parent < 0)[0][0]) if np.any(parent < 0) else 0
        depth = compute_depth(parent, root, n)

    # Compute children
    children = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if 0 <= p < n:
            children[p].append(i)

    child_count = np.array([len(c) for c in children], dtype=np.int32)
    is_leaf = (child_count == 0).astype(np.int32)

    return {
        "path": path,
        "n": n,
        "name": names,
        "ptnum": ptnum,
        "parent": parent,
        "children": children,
        "pos": pos,
        "vecp": vecp,
        "blen": blen,
        "depth": depth,
        "child_count": child_count,
        "is_leaf": is_leaf,
    }


def compute_depth(parent, root, n):
    """Compute depth from root"""
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
    """Extract features for each joint (must match training exactly)"""
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
        pos_norm,  # 3
        dir_parent,  # 3
        blen_norm,  # 1
        depth_norm,  # 1
        cc_norm,  # 1
        leaf,  # 1
    ]).astype(np.float32)

    return features


def compute_all_scores(model, mu, sd, feat_A, feat_B):
    """Compute match scores for all pairs of joints"""
    n_A = feat_A.shape[0]
    n_B = feat_B.shape[0]

    # Build all pairs
    pairs = []
    for b in range(n_B):
        for a in range(n_A):
            diff = np.abs(feat_A[a] - feat_B[b])
            prod = feat_A[a] * feat_B[b]
            pair = np.concatenate([feat_A[a], feat_B[b], diff, prod])
            pairs.append(pair)

    pairs = np.array(pairs, dtype=np.float32)

    # Normalize
    pairs = (pairs - mu) / sd

    # Predict
    scores = model.predict(pairs, verbose=0).flatten()

    # Reshape to (n_B, n_A)
    scores = scores.reshape(n_B, n_A)

    return scores


def greedy_match(scores, threshold=0.5, pos_A=None, pos_B=None):
    """
    Greedy one-to-one matching.
    Optionally filter by side (L/R) based on x-position.
    """
    n_B, n_A = scores.shape

    # Get candidates above threshold
    candidates = []
    for b in range(n_B):
        for a in range(n_A):
            if scores[b, a] >= threshold:
                # Optional: filter by side
                if pos_A is not None and pos_B is not None:
                    side_A = "L" if pos_A[a, 0] < -0.01 else ("R" if pos_A[a, 0] > 0.01 else "C")
                    side_B = "L" if pos_B[b, 0] < -0.01 else ("R" if pos_B[b, 0] > 0.01 else "C")
                    if side_A in ("L", "R") and side_B in ("L", "R") and side_A != side_B:
                        continue

                candidates.append((scores[b, a], b, a))

    # Sort by score descending
    candidates.sort(reverse=True)

    # Greedy assignment
    used_A = set()
    used_B = set()
    matches = {}

    for score, b, a in candidates:
        if b not in used_B and a not in used_A:
            matches[b] = (a, score)
            used_A.add(a)
            used_B.add(b)

    return matches


def full_match(scores, pos_A=None, pos_B=None):
    """
    Full matching using Hungarian algorithm.
    Guarantees every joint in B gets matched to exactly one joint in A.
    Uses side-awareness to prevent L/R cross-matching.
    """
    n_B, n_A = scores.shape

    # Create cost matrix (we want to maximize score, so use negative)
    # Add side penalty to prevent L/R cross-matching
    cost = -scores.copy()

    if pos_A is not None and pos_B is not None:
        for b in range(n_B):
            for a in range(n_A):
                side_A = "L" if pos_A[a, 0] < -0.01 else ("R" if pos_A[a, 0] > 0.01 else "C")
                side_B = "L" if pos_B[b, 0] < -0.01 else ("R" if pos_B[b, 0] > 0.01 else "C")
                # Heavy penalty for cross-side matching
                if side_A in ("L", "R") and side_B in ("L", "R") and side_A != side_B:
                    cost[b, a] += 1000  # Large penalty

    # Handle case where B has more joints than A
    if n_B > n_A:
        # Pad cost matrix with high-cost dummy columns
        padding = np.full((n_B, n_B - n_A), 1000)
        cost = np.hstack([cost, padding])

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost)

    # Build matches dict
    matches = {}
    for b, a in zip(row_ind, col_ind):
        if a < n_A:  # Not a dummy column
            matches[b] = (a, float(scores[b, a]))
        else:
            # No valid match (more B joints than A joints)
            # Find the best available A joint anyway
            best_a = int(np.argmax(scores[b, :]))
            matches[b] = (best_a, float(scores[b, best_a]))

    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--A", required=True, help="Source rig CSV (the one with more joints)")
    parser.add_argument("--B", required=True, help="Target rig CSV (the one you want to match)")
    parser.add_argument("--model", default="joint_matcher.keras", help="Trained model")
    parser.add_argument("--norm", default="joint_matcher_norm.npz", help="Normalization params")
    parser.add_argument("--threshold", type=float, default=0.5, help="Match threshold (ignored if --full)")
    parser.add_argument("--full", action="store_true", help="Force full matching - every B joint gets a match")
    parser.add_argument("--out", default="mapping.json", help="Output file")
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)
    norm = np.load(args.norm)
    mu = norm["mu"]
    sd = norm["sd"]

    # Load rigs
    print("Loading rigs...")
    rig_A = load_rig(args.A)
    rig_B = load_rig(args.B)
    print(f"  A: {rig_A['n']} joints")
    print(f"  B: {rig_B['n']} joints")

    # Extract features
    print("Extracting features...")
    feat_A = extract_joint_features(rig_A)
    feat_B = extract_joint_features(rig_B)
    print(f"  Feature dim: {feat_A.shape[1]}")

    # Compute scores
    print("Computing match scores...")
    scores = compute_all_scores(model, mu, sd, feat_A, feat_B)

    print(f"\n  Score stats: min={scores.min():.3f}, max={scores.max():.3f}, mean={scores.mean():.3f}")
    print(f"  Scores > 0.9: {(scores > 0.9).sum()}")
    print(f"  Scores > 0.7: {(scores > 0.7).sum()}")
    print(f"  Scores > 0.5: {(scores > 0.5).sum()}")

    # Match
    if args.full:
        print(f"\nUsing FULL matching mode (Hungarian algorithm)")
        matches = full_match(scores, pos_A=rig_A["pos"], pos_B=rig_B["pos"])
    else:
        matches = greedy_match(scores, threshold=args.threshold, pos_A=rig_A["pos"], pos_B=rig_B["pos"])

    # Prepare output
    matched_B = set(matches.keys())
    unmatched_B = [int(rig_B["ptnum"][i]) for i in range(rig_B["n"]) if i not in matched_B]

    output = {
        "A_csv": args.A,
        "B_csv": args.B,
        "threshold": args.threshold,
        "matches": [
            {
                "B_idx": int(b),
                "B_ptnum": int(rig_B["ptnum"][b]),
                "B_name": rig_B["name"][b],
                "A_idx": int(a),
                "A_ptnum": int(rig_A["ptnum"][a]),
                "A_name": rig_A["name"][a],
                "score": float(score),
            }
            for b, (a, score) in sorted(matches.items())
        ],
        "delete_B_ptnums": unmatched_B,
    }

    # Save
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)

    # Print results
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Matched: {len(matches)}, Unmatched: {len(unmatched_B)}")

    if matches:
        avg_score = np.mean([s for _, s in matches.values()])
        print(f"Average score: {avg_score:.3f}")

    print(f"\nMatches:")
    for m in output["matches"]:
        conf = "HIGH" if m["score"] >= 0.9 else ("MED" if m["score"] >= 0.7 else "LOW")
        print(f"  {m['B_name']:30} -> {m['A_name']:30} ({conf}, {m['score']:.3f})")

    print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()