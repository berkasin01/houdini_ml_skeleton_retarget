import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf


def load_rig_csv(path):
    df = pd.read_csv(path)
    n = len(df)

    ptnum = df["ptnum"].to_numpy(dtype=np.int32) if "ptnum" in df.columns else np.arange(n, dtype=np.int32)
    names = df["name"].astype(str).to_numpy() if "name" in df.columns else np.array([f"j{i}" for i in range(n)])

    parent = df["parent_idx"].to_numpy(dtype=np.int32)
    pos = df[["pos_root_x", "pos_root_y", "pos_root_z"]].to_numpy(dtype=np.float32)
    vecp = df[["vec_parent_x", "vec_parent_y", "vec_parent_z"]].to_numpy(dtype=np.float32)
    blen = df["bone_len"].to_numpy(dtype=np.float32)

    depth = df["depth"].to_numpy(dtype=np.int32) if "depth" in df.columns else None
    cc = df["child_count"].to_numpy(dtype=np.int32) if "child_count" in df.columns else None
    leaf = df["is_leaf"].to_numpy(dtype=np.int32) if "is_leaf" in df.columns else None

    root = int(np.where(parent < 0)[0][0]) if np.any(parent < 0) else 0

    if depth is None or cc is None or leaf is None:
        depth, cc, leaf = compute_depth_child_leaf(parent, root)

    children = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if 0 <= p < n:
            children[p].append(i)

    return {
        "path": path, "df": df, "ptnum": ptnum, "name": names, "n": n,
        "root": root, "parent": parent, "children": children,
        "pos": pos, "vecp": vecp, "blen": blen,
        "depth": depth, "child_count": cc, "is_leaf": leaf,
    }


def compute_depth_child_leaf(parent, root):
    n = len(parent)
    children = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if 0 <= p < n:
            children[p].append(i)

    depth = np.full(n, -1, dtype=np.int32)
    depth[root] = 0
    q = [root]
    for u in q:
        for v in children[u]:
            depth[v] = depth[u] + 1
            q.append(v)

    cc = np.array([len(ch) for ch in children], dtype=np.int32)
    leaf = (cc == 0).astype(np.int32)
    return depth, cc, leaf


def median_scale(blen):
    x = blen[blen > 1e-8]
    return float(np.median(x)) if len(x) else 1.0


def safe_unit(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.where(n < 1e-8, 1.0, n)
    return v / n


def pathlen_from_root(parent, root, blen):
    n = len(parent)
    children = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if 0 <= p < n:
            children[p].append(i)

    pl = np.zeros(n, dtype=np.float32)
    order = [int(root)]
    for u in order:
        for v in children[u]:
            order.append(v)

    for v in order[1:]:
        p = int(parent[v])
        pl[v] = pl[p] + float(blen[v])
    return pl


# FIXED version - matches build_model_v2.py exactly
def compute_subtree_stats(parent, children, blen, pos):
    n = len(parent)
    subtree_count = np.zeros(n, dtype=np.float32)
    subtree_blen = np.zeros(n, dtype=np.float32)
    subtree_extent = np.zeros(n, dtype=np.float32)

    roots = [i for i in range(n) if parent[i] < 0]
    if not roots:
        roots = [0]

    order = []
    visited = set()

    def post_order(node):
        if node in visited or node < 0 or node >= n:
            return
        visited.add(node)
        for c in children[node]:
            post_order(c)
        order.append(node)

    for root in roots:
        post_order(root)

    for i in range(n):
        if i not in visited:
            order.append(i)

    for u in order:
        subtree_count[u] = 1
        subtree_blen[u] = blen[u] if u < len(blen) else 0
        subtree_extent[u] = 0

        for c in children[u]:
            subtree_count[u] += subtree_count[c]
            subtree_blen[u] += subtree_blen[c]
            if c < len(pos) and u < len(pos):
                child_dist = np.linalg.norm(pos[c] - pos[u])
                subtree_extent[u] = max(subtree_extent[u], child_dist + subtree_extent[c])

    return subtree_count, subtree_blen, subtree_extent


# FIXED version - matches build_model_v2.py exactly
def compute_branch_signature(parent, children, root, depth, blen):
    n = len(parent)
    branch_id = np.zeros(n, dtype=np.float32)
    branch_depth = np.zeros(n, dtype=np.float32)

    for i in range(n):
        curr = i
        max_iterations = n + 1
        iterations = 0

        while curr != root and iterations < max_iterations:
            p = parent[curr]
            if p < 0 or p >= n:
                break
            if len(children[p]) > 1:
                child_idx = children[p].index(curr) if curr in children[p] else 0
                branch_id[i] = child_idx + 1
                branch_depth[i] = depth[p] if p < len(depth) else 0
                break
            curr = p
            iterations += 1

    return branch_id, branch_depth


def compute_x_position_sign(pos):
    return np.sign(pos[:, 0]).astype(np.float32).reshape(-1, 1)


def compute_local_frame(pos, parent, children, vecp):
    n = len(parent)
    dir_to_parent = safe_unit(vecp)

    dir_to_children = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        if len(children[i]) > 0:
            child_vecs = [pos[c] - pos[i] for c in children[i]]
            avg_vec = np.mean(child_vecs, axis=0)
            norm = np.linalg.norm(avg_vec)
            if norm > 1e-8:
                dir_to_children[i] = avg_vec / norm

    cross = np.cross(dir_to_parent, dir_to_children)
    cross_norm = np.linalg.norm(cross, axis=1, keepdims=True)
    cross_norm = np.where(cross_norm < 1e-8, 1.0, cross_norm)
    cross = cross / cross_norm

    return dir_to_parent, dir_to_children, cross


def compute_neighborhood_encoding(pos, parent, children, depth, k=2):
    n = len(parent)
    ancestor_enc = np.zeros((n, k * 3), dtype=np.float32)
    descendant_enc = np.zeros((n, k * 3), dtype=np.float32)

    for i in range(n):
        ancestors = []
        curr = i
        for _ in range(k):
            p = parent[curr]
            if p < 0:
                break
            ancestors.append(p)
            curr = p

        for j, anc in enumerate(ancestors[:k]):
            ancestor_enc[i, j * 3:(j + 1) * 3] = pos[anc] - pos[i]

        descendants = []
        q = list(children[i])
        while q and len(descendants) < k:
            c = q.pop(0)
            descendants.append(c)
            q.extend(children[c])

        for j, desc in enumerate(descendants[:k]):
            descendant_enc[i, j * 3:(j + 1) * 3] = pos[desc] - pos[i]

    return ancestor_enc, descendant_enc


def joint_features_v2(rig):
    """Extract geometric features - MUST MATCH build_model_v2.py EXACTLY"""
    pos = rig["pos"]
    vecp = rig["vecp"]
    blen = rig["blen"]
    parent = rig["parent"]
    children = rig["children"]
    root = rig["root"]
    depth = rig["depth"]
    child_count = rig["child_count"]
    is_leaf = rig["is_leaf"]

    n = len(parent)
    scale = median_scale(blen)

    # === Basic normalized features ===
    pos_n = pos / scale
    bl_n = (blen / scale).reshape(-1, 1)
    vdir = safe_unit(vecp)
    dist_from_origin = np.linalg.norm(pos_n, axis=1, keepdims=True)

    # === Depth and structure ===
    dmax = float(np.max(depth)) if np.max(depth) > 0 else 1.0
    cmax = float(np.max(child_count)) if np.max(child_count) > 0 else 1.0
    depth_n = (depth.astype(np.float32) / dmax).reshape(-1, 1)
    cc_n = (child_count.astype(np.float32) / cmax).reshape(-1, 1)
    leaf_n = is_leaf.astype(np.float32).reshape(-1, 1)

    # === Path length from root ===
    pl = pathlen_from_root(parent, root, blen)
    pl_n = (pl / scale).reshape(-1, 1)

    # === Subtree statistics ===
    subtree_count, subtree_blen, subtree_extent = compute_subtree_stats(parent, children, blen, pos)
    sc_max = max(np.max(subtree_count), 1.0)
    subtree_count_n = (subtree_count / sc_max).reshape(-1, 1)
    subtree_blen_n = (subtree_blen / scale / sc_max).reshape(-1, 1)
    subtree_extent_n = (subtree_extent / scale).reshape(-1, 1)

    # === Branch signature ===
    branch_id, branch_depth = compute_branch_signature(parent, children, root, depth, blen)
    bid_max = max(np.max(branch_id), 1.0)
    branch_id_n = (branch_id / bid_max).reshape(-1, 1)
    branch_depth_n = (branch_depth / dmax).reshape(-1, 1)

    # === Left/Right indicator (purely geometric) ===
    x_sign = compute_x_position_sign(pos)

    # === Local frame ===
    dir_parent, dir_children, dir_cross = compute_local_frame(pos, parent, children, vecp)

    # === Neighborhood encoding (k=2 to match training) ===
    ancestor_enc, descendant_enc = compute_neighborhood_encoding(pos, parent, children, depth, k=2)
    ancestor_enc_n = ancestor_enc / scale
    descendant_enc_n = descendant_enc / scale

    # === Height relative to root ===
    height_n = (pos[:, 1:2] / scale)

    # === Sibling features ===
    sibling_count = np.zeros(n, dtype=np.float32)
    sibling_index = np.zeros(n, dtype=np.float32)
    for i in range(n):
        p = parent[i]
        if p >= 0:
            siblings = children[p]
            sibling_count[i] = len(siblings)
            sibling_index[i] = siblings.index(i) if i in siblings else 0
    sib_count_max = max(np.max(sibling_count), 1.0)
    sibling_count_n = (sibling_count / sib_count_max).reshape(-1, 1)
    sibling_index_n = (sibling_index / sib_count_max).reshape(-1, 1)

    # === Combine all features (MUST match build_model_v2.py order) ===
    features = np.concatenate([
        pos_n,  # 3
        vdir,  # 3
        bl_n,  # 1
        depth_n,  # 1
        cc_n,  # 1
        leaf_n,  # 1
        dist_from_origin,  # 1
        pl_n,  # 1
        subtree_count_n,  # 1
        subtree_blen_n,  # 1
        subtree_extent_n,  # 1
        branch_id_n,  # 1
        branch_depth_n,  # 1
        x_sign,  # 1
        dir_children,  # 3
        dir_cross,  # 3
        ancestor_enc_n,  # 6
        descendant_enc_n,  # 6
        height_n,  # 1
        sibling_count_n,  # 1
        sibling_index_n,  # 1  = 39 total
    ], axis=1).astype(np.float32)

    return features


def standardize_apply(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)


def score_matrix(model, mu, sd, fA, fB, batch_b=64):
    nA = fA.shape[0]
    nB = fB.shape[0]
    d = fA.shape[1]
    S = np.zeros((nB, nA), dtype=np.float32)

    fa = fA[None, :, :]
    for b0 in range(0, nB, batch_b):
        b1 = min(nB, b0 + batch_b)
        fb = fB[b0:b1, None, :]
        fb_b = np.broadcast_to(fb, (b1 - b0, nA, d))
        fa_b = np.broadcast_to(fa, (b1 - b0, nA, d))

        # Pair features: [fb, fa, |fb-fa|, fb*fa]
        X = np.concatenate([fb_b, fa_b, np.abs(fb_b - fa_b), fb_b * fa_b], axis=2)
        X = X.reshape(-1, X.shape[2]).astype(np.float32)
        X = standardize_apply(X, mu, sd)

        p = model.predict(X, verbose=0).reshape((b1 - b0, nA))
        S[b0:b1] = p.astype(np.float32)

    return S


def mirror_features_x(f):
    g = f.copy()
    g[:, 0] *= -1.0  # pos_x
    g[:, 3] *= -1.0  # vdir_x
    g[:, 13] *= -1.0  # x_sign
    g[:, 14] *= -1.0  # dir_children_x
    g[:, 17] *= -1.0  # dir_cross_x
    return g


def get_geometric_side(pos_x):
    if pos_x < -0.01:
        return "L"
    elif pos_x > 0.01:
        return "R"
    return "C"


def greedy_matching(S, threshold, topk=12, posA=None, posB=None, parentA=None, parentB=None):
    nB, nA = S.shape
    cand = []

    for b in range(nB):
        k = min(topk, nA)
        idx = np.argpartition(S[b], -k)[-k:]
        for a in idx:
            sc = float(S[b, a])
            if sc < threshold:
                continue

            if posA is not None and posB is not None:
                sideA = get_geometric_side(posA[a, 0])
                sideB = get_geometric_side(posB[b, 0])
                if sideA in ("L", "R") and sideB in ("L", "R") and sideA != sideB:
                    continue

            cand.append((sc, b, int(a)))

    cand.sort(reverse=True)

    usedA = set()
    usedB = set()
    out = {}

    for sc, b, a in cand:
        if b in usedB or a in usedA:
            continue
        usedB.add(b)
        usedA.add(a)
        out[b] = (a, sc)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", required=True, help="source rig CSV")
    ap.add_argument("--B", required=True, help="incoming rig CSV")
    ap.add_argument("--model", default="pairwise_model_v2.keras")
    ap.add_argument("--norm", default="pairwise_norm_v2.npz")
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--out", default="mapping.json")
    args = ap.parse_args()

    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)
    norm = np.load(args.norm)
    mu = norm["mu"].astype(np.float32)
    sd = norm["sd"].astype(np.float32)

    print(f"Loading rigs...")
    A = load_rig_csv(args.A)
    B = load_rig_csv(args.B)
    print(f"  A: {A['n']} joints")
    print(f"  B: {B['n']} joints")

    print("Extracting features...")
    fA = joint_features_v2(A)
    fB = joint_features_v2(B)
    print(f"  Feature dimension: {fA.shape[1]} (expect 39)")
    print(f"  Pair feature dimension: {fA.shape[1] * 4} (expect 156)")

    # DEBUG: Check raw feature ranges
    print(f"\n  DEBUG Raw feature stats:")
    print(f"    fA: min={fA.min():.4f}, max={fA.max():.4f}, mean={fA.mean():.4f}")
    print(f"    fB: min={fB.min():.4f}, max={fB.max():.4f}, mean={fB.mean():.4f}")

    # DEBUG: Check normalization params
    print(f"\n  DEBUG Normalization params:")
    print(f"    mu shape: {mu.shape}, range: [{mu.min():.4f}, {mu.max():.4f}]")
    print(f"    sd shape: {sd.shape}, range: [{sd.min():.4f}, {sd.max():.4f}]")

    # DEBUG: Test one pair manually
    test_pair = np.concatenate([fB[0], fA[0], np.abs(fB[0] - fA[0]), fB[0] * fA[0]])
    print(f"\n  DEBUG Test pair (B[0] vs A[0]):")
    print(f"    Raw pair features: min={test_pair.min():.4f}, max={test_pair.max():.4f}")
    test_norm = (test_pair - mu.flatten()) / sd.flatten()
    print(f"    After standardization: min={test_norm.min():.4f}, max={test_norm.max():.4f}")

    # Check for NaN/Inf
    print(f"    Has NaN: {np.isnan(test_norm).any()}, Has Inf: {np.isinf(test_norm).any()}")

    print("Computing scores (normal orientation)...")
    S1 = score_matrix(model, mu, sd, fA, fB, batch_b=64)

    # DEBUG: Show score statistics
    print(f"\n  DEBUG Score matrix stats:")
    print(f"    Shape: {S1.shape}")
    print(f"    Min: {S1.min():.4f}, Max: {S1.max():.4f}, Mean: {S1.mean():.4f}")
    print(f"    Scores > 0.5: {(S1 > 0.5).sum()}")
    print(f"    Scores > 0.3: {(S1 > 0.3).sum()}")
    print(f"    Scores > 0.1: {(S1 > 0.1).sum()}")

    # Show top 5 matches by score
    print(f"\n  Top 10 scoring pairs:")
    flat_idx = np.argsort(S1.flatten())[-10:][::-1]
    for idx in flat_idx:
        b_idx = idx // S1.shape[1]
        a_idx = idx % S1.shape[1]
        score = S1[b_idx, a_idx]
        print(f"    B[{b_idx}] {B['name'][b_idx]:25} -> A[{a_idx}] {A['name'][a_idx]:25} = {score:.4f}")

    m1 = greedy_matching(S1, args.threshold, topk=16, posA=A["pos"], posB=B["pos"],
                         parentA=A["parent"], parentB=B["parent"])
    s1 = sum(sc for _, sc in m1.values())

    print("Computing scores (mirrored)...")
    fB_flip = mirror_features_x(fB)
    S2 = score_matrix(model, mu, sd, fA, fB_flip, batch_b=64)
    m2 = greedy_matching(S2, args.threshold, topk=16, posA=A["pos"], posB=-B["pos"][:, :1],
                         parentA=A["parent"], parentB=B["parent"])
    s2 = sum(sc for _, sc in m2.values())

    use_mirror = (s2 > s1)
    mapping = m2 if use_mirror else m1

    mapped_B = set(mapping.keys())
    delete_B = [int(B["ptnum"][i]) for i in range(B["n"]) if i not in mapped_B]

    out = {
        "A_csv": args.A,
        "B_csv": args.B,
        "used_mirror_x": bool(use_mirror),
        "sum_score": float(s2 if use_mirror else s1),
        "threshold": args.threshold,
        "matches": [
            {
                "B_row": int(b),
                "B_ptnum": int(B["ptnum"][b]),
                "B_name": str(B["name"][b]),
                "A_row": int(a),
                "A_ptnum": int(A["ptnum"][a]),
                "A_name": str(A["name"][a]),
                "score": float(sc),
            }
            for b, (a, sc) in sorted(mapping.items(), key=lambda kv: kv[0])
        ],
        "delete_B_ptnums": delete_B,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"A joints: {A['n']}, B joints: {B['n']}")
    print(f"Matched: {len(out['matches'])}, Unmatched: {len(delete_B)}")

    if out['matches']:
        avg_conf = sum(m['score'] for m in out['matches']) / len(out['matches'])
        print(f"Average confidence: {avg_conf:.3f}")
        print(f"Used mirror: {use_mirror}")

    high_conf = [m for m in out["matches"] if m["score"] >= 0.90]
    med_conf = [m for m in out["matches"] if 0.75 <= m["score"] < 0.90]
    low_conf = [m for m in out["matches"] if m["score"] < 0.75]

    print(f"\nConfidence breakdown:")
    print(f"  High (>=0.90): {len(high_conf)}")
    print(f"  Medium (0.75-0.90): {len(med_conf)}")
    print(f"  Low (<0.75): {len(low_conf)}")

    if low_conf:
        print(f"\nLow confidence matches:")
        for m in sorted(low_conf, key=lambda x: x["score"])[:15]:
            print(f"  {m['B_name']:30} -> {m['A_name']:30} ({m['score']:.3f})")

    print(f"\nAll matches:")
    for m in out["matches"]:
        print(f"  {m['B_name']:30} -> {m['A_name']:30} ({m['score']:.3f})")

    print(f"\nOutput written to: {args.out}")


if __name__ == "__main__":
    main()