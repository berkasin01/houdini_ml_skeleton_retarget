"""
Geometry-Only Skeleton Joint Matching Model v2

Key changes from v1:
1. NO name-based supervision - cross-rig pairs use geometric alignment instead
2. Richer structural features including local neighborhood encoding
3. Contrastive learning approach for better embedding space
4. Graph-aware features that capture the skeleton topology

The core insight: instead of relying on name normalization to create training pairs,
we use the GEOMETRY itself. Two joints should match if they occupy similar positions
in the normalized skeleton space (relative to root, spine, etc.)
"""

import os
import glob
import json
import re
import numpy as np
import pandas as pd
from collections import defaultdict

# ============== CONFIGURATION ==============
TRAIN_DIR = "training_skeletons"
MODEL_OUT = "pairwise_model_v2.keras"
NORM_OUT = "pairwise_norm_v2.npz"

# Training params
AUG_PER_RIG = 150  # Augmentations of same rig
NEG_PER_POS = 6  # Negative samples per positive
JUNK_MAX = 8  # Max junk joints to add

CROSS_RIG_AUG_PER_PAIR = 60  # Cross-rig augmentations
CROSS_NEG_PER_POS = 6

# Geometric matching params for cross-rig
# RELAXED threshold - rigs may have different proportions
GEO_MATCH_THRESHOLD = 0.35  # Normalized distance threshold (was 0.18, too strict)

EPOCHS = 25
BATCH_SIZE = 512
LR = 5e-4
SEED = 42

# Progress tracking
import time

rng = np.random.default_rng(SEED)


# ============== DATA LOADING ==============

def load_rig_csv(path):
    """Load a rig from CSV with all geometric features"""
    df = pd.read_csv(path)
    n = len(df)

    parent = df["parent_idx"].to_numpy(dtype=np.int32)
    pos = df[["pos_root_x", "pos_root_y", "pos_root_z"]].to_numpy(dtype=np.float32)
    vecp = df[["vec_parent_x", "vec_parent_y", "vec_parent_z"]].to_numpy(dtype=np.float32)
    blen = df["bone_len"].to_numpy(dtype=np.float32)

    # Load or compute depth, child count, leaf status
    depth = df["depth"].to_numpy(dtype=np.int32) if "depth" in df.columns else None
    cc = df["child_count"].to_numpy(dtype=np.int32) if "child_count" in df.columns else None
    leaf = df["is_leaf"].to_numpy(dtype=np.int32) if "is_leaf" in df.columns else None

    root = int(np.where(parent < 0)[0][0]) if np.any(parent < 0) else 0

    if depth is None or cc is None or leaf is None:
        depth, cc, leaf = compute_depth_child_leaf(parent, root)

    names = df["name"].astype(str).to_list() if "name" in df.columns else [f"j{i}" for i in range(n)]

    # Compute children list
    children = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if 0 <= p < n:
            children[p].append(i)

    return {
        "path": path,
        "n": n,
        "root": root,
        "parent": parent,
        "children": children,
        "pos": pos,
        "vecp": vecp,
        "blen": blen,
        "depth": depth,
        "child_count": cc,
        "is_leaf": leaf,
        "name": names,
    }


def compute_depth_child_leaf(parent, root):
    """Compute tree structure metrics"""
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


# ============== GEOMETRIC UTILITIES ==============

def median_scale(blen):
    """Get median bone length for normalization"""
    x = blen[blen > 1e-8]
    return float(np.median(x)) if len(x) else 1.0


def safe_unit(v):
    """Normalize vectors safely"""
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.where(n < 1e-8, 1.0, n)
    return v / n


def pathlen_from_root(parent, root, blen):
    """Compute path length from root to each joint"""
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


def compute_subtree_stats(parent, children, blen, pos):
    """Compute subtree statistics for each joint"""
    n = len(parent)

    # Subtree joint count
    subtree_count = np.zeros(n, dtype=np.float32)
    # Subtree total bone length
    subtree_blen = np.zeros(n, dtype=np.float32)
    # Subtree bounding box size (max extent)
    subtree_extent = np.zeros(n, dtype=np.float32)

    # Find root(s)
    roots = [i for i in range(n) if parent[i] < 0]
    if not roots:
        roots = [0]

    # Build post-order traversal (children before parents)
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

    # Also handle any disconnected nodes
    for i in range(n):
        if i not in visited:
            order.append(i)

    # Process in post-order (leaves first, then parents)
    for u in order:
        subtree_count[u] = 1
        subtree_blen[u] = blen[u] if u < len(blen) else 0
        subtree_extent[u] = 0

        for c in children[u]:
            subtree_count[u] += subtree_count[c]
            subtree_blen[u] += subtree_blen[c]
            # Extent: max distance from this joint to any descendant
            if c < len(pos) and u < len(pos):
                child_dist = np.linalg.norm(pos[c] - pos[u])
                subtree_extent[u] = max(subtree_extent[u], child_dist + subtree_extent[c])

    return subtree_count, subtree_blen, subtree_extent


def compute_branch_signature(parent, children, root, depth, blen):
    """
    Compute a signature for each joint based on its branch in the skeleton.
    This helps distinguish left/right sides and different limbs.
    """
    n = len(parent)

    # Find the path from root to each joint
    branch_id = np.zeros(n, dtype=np.float32)
    branch_depth = np.zeros(n, dtype=np.float32)  # Depth at which branch started

    # For each joint, find where it branched off from the spine
    for i in range(n):
        # Walk up to find the first ancestor with multiple children
        curr = i
        max_iterations = n + 1  # Prevent infinite loops
        iterations = 0

        while curr != root and iterations < max_iterations:
            p = parent[curr]
            if p < 0 or p >= n:
                break
            if len(children[p]) > 1:
                # Determine which branch we're in (by child index)
                child_idx = children[p].index(curr) if curr in children[p] else 0
                branch_id[i] = child_idx + 1
                branch_depth[i] = depth[p] if p < len(depth) else 0
                break
            curr = p
            iterations += 1

    return branch_id, branch_depth


def compute_x_position_sign(pos):
    """
    Compute the sign of the x-position (left/right side indicator).
    This is a purely geometric feature - negative x = one side, positive = other.
    """
    return np.sign(pos[:, 0]).astype(np.float32).reshape(-1, 1)


def compute_local_frame(pos, parent, children, vecp):
    """
    Compute local coordinate frame features for each joint.
    This captures the local orientation without relying on names.
    """
    n = len(parent)

    # Direction to parent (normalized)
    dir_to_parent = safe_unit(vecp)

    # Average direction to children (if any)
    dir_to_children = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        if len(children[i]) > 0:
            child_vecs = []
            for c in children[i]:
                child_vecs.append(pos[c] - pos[i])
            avg_vec = np.mean(child_vecs, axis=0)
            norm = np.linalg.norm(avg_vec)
            if norm > 1e-8:
                dir_to_children[i] = avg_vec / norm

    # Cross product gives perpendicular direction (helps identify orientation)
    cross = np.cross(dir_to_parent, dir_to_children)
    cross_norm = np.linalg.norm(cross, axis=1, keepdims=True)
    cross_norm = np.where(cross_norm < 1e-8, 1.0, cross_norm)
    cross = cross / cross_norm

    return dir_to_parent, dir_to_children, cross


def compute_neighborhood_encoding(pos, parent, children, depth, k=3):
    """
    Encode the local neighborhood of each joint.
    This captures the local structure around each joint.
    """
    n = len(parent)

    # Features: relative positions of k nearest ancestors and descendants
    ancestor_encoding = np.zeros((n, k * 3), dtype=np.float32)
    descendant_encoding = np.zeros((n, k * 3), dtype=np.float32)

    for i in range(n):
        # Ancestors (up to k)
        ancestors = []
        curr = i
        for _ in range(k):
            p = parent[curr]
            if p < 0:
                break
            ancestors.append(p)
            curr = p

        for j, anc in enumerate(ancestors):
            if j < k:
                rel_pos = pos[anc] - pos[i]
                ancestor_encoding[i, j * 3:(j + 1) * 3] = rel_pos

        # Descendants (BFS, up to k)
        descendants = []
        q = list(children[i])
        while q and len(descendants) < k:
            c = q.pop(0)
            descendants.append(c)
            q.extend(children[c])

        for j, desc in enumerate(descendants[:k]):
            rel_pos = pos[desc] - pos[i]
            descendant_encoding[i, j * 3:(j + 1) * 3] = rel_pos

    return ancestor_encoding, descendant_encoding


# ============== FEATURE EXTRACTION ==============

def joint_features_v2(rig):
    """
    Extract comprehensive geometric features for each joint.
    NO NAME INFORMATION is used here.
    """
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
    subtree_count, subtree_blen, subtree_extent = compute_subtree_stats(
        parent, children, blen, pos
    )
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

    # === Neighborhood encoding ===
    ancestor_enc, descendant_enc = compute_neighborhood_encoding(pos, parent, children, depth, k=2)
    ancestor_enc_n = ancestor_enc / scale
    descendant_enc_n = descendant_enc / scale

    # === Height relative to root (Y coordinate is typically up) ===
    height_n = (pos[:, 1:2] / scale)  # Y coordinate

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

    # === Combine all features ===
    features = np.concatenate([
        pos_n,  # 3: normalized position
        vdir,  # 3: direction to parent
        bl_n,  # 1: bone length
        depth_n,  # 1: depth in tree
        cc_n,  # 1: child count
        leaf_n,  # 1: is leaf
        dist_from_origin,  # 1: distance from origin
        pl_n,  # 1: path length from root
        subtree_count_n,  # 1: subtree joint count
        subtree_blen_n,  # 1: subtree total bone length
        subtree_extent_n,  # 1: subtree extent
        branch_id_n,  # 1: branch identifier
        branch_depth_n,  # 1: branch start depth
        x_sign,  # 1: left/right indicator
        dir_children,  # 3: average direction to children
        dir_cross,  # 3: cross product (orientation)
        ancestor_enc_n,  # 6: ancestor relative positions (k=2)
        descendant_enc_n,  # 6: descendant relative positions (k=2)
        height_n,  # 1: height (Y coordinate)
        sibling_count_n,  # 1: number of siblings
        sibling_index_n,  # 1: index among siblings
    ], axis=1).astype(np.float32)

    return features


# ============== GEOMETRIC MATCHING (NO NAMES) ==============

def find_geometric_correspondences(rigA, rigB, threshold=GEO_MATCH_THRESHOLD):
    """
    Find corresponding joints between two rigs using ONLY geometry.
    No names are used here.

    Strategy:
    1. Normalize both skeletons to same scale
    2. Align them (they should already be in T-pose/A-pose)
    3. For each joint in A, find the closest joint in B
    4. Keep only mutual nearest neighbors with distance below threshold
    """
    posA = rigA["pos"].copy()
    posB = rigB["pos"].copy()

    # Normalize by median bone length
    scaleA = median_scale(rigA["blen"])
    scaleB = median_scale(rigB["blen"])

    posA_n = posA / scaleA
    posB_n = posB / scaleB

    nA = len(posA)
    nB = len(posB)

    # Compute distance matrix
    dist = np.zeros((nA, nB), dtype=np.float32)
    for i in range(nA):
        dist[i] = np.linalg.norm(posB_n - posA_n[i], axis=1)

    # Find mutual nearest neighbors
    correspondences = []

    # For each joint in A, find nearest in B
    nearest_B_for_A = np.argmin(dist, axis=1)
    nearest_A_for_B = np.argmin(dist, axis=0)

    for a in range(nA):
        b = nearest_B_for_A[a]
        if nearest_A_for_B[b] == a:  # Mutual nearest neighbor
            d = dist[a, b]
            if d < threshold:
                # Additional check: depth should be similar
                depth_diff = abs(int(rigA["depth"][a]) - int(rigB["depth"][b]))
                if depth_diff <= 2:  # Allow some depth variation
                    correspondences.append((a, b, d))

    return correspondences


def find_geometric_correspondences_hungarian(rigA, rigB, threshold=GEO_MATCH_THRESHOLD):
    """
    Use Hungarian algorithm for optimal bipartite matching based on geometry.
    This gives better results than greedy mutual nearest neighbor.
    """
    from scipy.optimize import linear_sum_assignment

    posA = rigA["pos"].copy()
    posB = rigB["pos"].copy()

    scaleA = median_scale(rigA["blen"])
    scaleB = median_scale(rigB["blen"])

    posA_n = posA / scaleA
    posB_n = posB / scaleB

    nA = len(posA)
    nB = len(posB)

    # Compute cost matrix (distance + depth penalty)
    cost = np.zeros((nA, nB), dtype=np.float32)
    for i in range(nA):
        pos_dist = np.linalg.norm(posB_n - posA_n[i], axis=1)
        depth_diff = np.abs(rigA["depth"][i] - rigB["depth"]).astype(np.float32)
        # Combine position distance with depth penalty
        cost[i] = pos_dist + 0.1 * depth_diff

    # Make square matrix if needed (pad with high cost)
    max_dim = max(nA, nB)
    cost_square = np.full((max_dim, max_dim), 1000.0, dtype=np.float32)
    cost_square[:nA, :nB] = cost

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost_square)

    correspondences = []
    for a, b in zip(row_ind, col_ind):
        if a < nA and b < nB:
            d = cost[a, b]
            if d < threshold:
                correspondences.append((a, b, d))

    return correspondences


# ============== AUGMENTATION ==============

def random_rotation_matrix(rng):
    """Generate a random rotation matrix"""
    u1, u2, u3 = rng.random(3)
    qx = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    qy = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    qz = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    qw = np.sqrt(u1) * np.cos(2 * np.pi * u3)

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=np.float32)


def rebuild_positions_from_parent(parent, root, vdir, blen):
    """Rebuild positions from bone directions and lengths"""
    n = len(parent)
    children = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if 0 <= p < n:
            children[p].append(i)

    pos = np.zeros((n, 3), dtype=np.float32)
    order = [root]
    for u in order:
        for v in children[u]:
            order.append(v)

    for i in order:
        p = parent[i]
        if p < 0 or i == root:
            continue
        pos[i] = pos[p] + vdir[i] * blen[i]

    return pos


def make_augmented_rig(A):
    """Create augmented version of a rig with geometric variations"""
    nA = A["n"]
    parent = A["parent"].copy()
    root = A["root"]

    vecp = A["vecp"].astype(np.float32).copy()
    vdir = safe_unit(vecp)
    blen = A["blen"].astype(np.float32).copy()

    # Bone length variation
    length_noise = rng.normal(0.0, 0.10, size=nA).astype(np.float32)
    length_noise[root] = 0.0
    blen2 = np.clip(blen * (1.0 + length_noise), 1e-6, None)

    # Rebuild positions
    pos2 = rebuild_positions_from_parent(parent, root, vdir, blen2)

    # Global scale variation
    s = float(rng.uniform(0.85, 1.20))
    pos2 *= s
    blen2 *= s

    # Small position noise
    pos2 += rng.normal(0.0, 0.003, size=pos2.shape).astype(np.float32)

    # Add junk joints
    junk_n = int(rng.integers(0, JUNK_MAX + 1))
    map_b_to_a = np.arange(nA, dtype=np.int32)

    if junk_n > 0:
        parent_list = parent.tolist()
        pos_list = pos2.tolist()
        bl_list = blen2.tolist()
        vdir_list = vdir.tolist()
        map_list = map_b_to_a.tolist()

        valid_parents = [i for i in range(nA) if i != root]
        for _ in range(junk_n):
            p = int(rng.choice(valid_parents))
            base_dir = np.array(vdir_list[p], dtype=np.float32)
            noise = rng.normal(0.0, 0.3, size=3).astype(np.float32)
            d = base_dir + noise
            dn = np.linalg.norm(d)
            if dn < 1e-6:
                d = np.array([1, 0, 0], dtype=np.float32)
            else:
                d = d / dn

            L = float(max(0.03, abs(rng.normal(0.25, 0.15))) * median_scale(A["blen"]) * s)
            new_pos = (np.array(pos_list[p], dtype=np.float32) + d * L).astype(np.float32)

            parent_list.append(p)
            pos_list.append([float(new_pos[0]), float(new_pos[1]), float(new_pos[2])])
            bl_list.append(L)
            vdir_list.append([float(d[0]), float(d[1]), float(d[2])])
            map_list.append(-1)  # Junk joint maps to nothing

        parent = np.array(parent_list, dtype=np.int32)
        pos2 = np.array(pos_list, dtype=np.float32)
        blen2 = np.array(bl_list, dtype=np.float32)
        vdir = np.array(vdir_list, dtype=np.float32)
        map_b_to_a = np.array(map_list, dtype=np.int32)

    # Recompute structure
    root = int(np.where(parent < 0)[0][0]) if np.any(parent < 0) else 0
    depth, cc, leaf = compute_depth_child_leaf(parent, root)

    children = [[] for _ in range(len(parent))]
    for i, p in enumerate(parent):
        if 0 <= p < len(parent):
            children[p].append(i)

    vecp2 = vdir * blen2.reshape(-1, 1)

    return {
        "parent": parent,
        "children": children,
        "root": root,
        "pos": pos2,
        "vecp": vecp2,
        "blen": blen2,
        "depth": depth,
        "child_count": cc,
        "is_leaf": leaf,
        "map_b_to_a": map_b_to_a,
        "n": len(parent),
        "name": ["aug"] * len(parent),  # Placeholder
    }


def apply_global_aug(rig):
    """Apply global augmentation (noise, scale) to a rig"""
    pos2 = rig["pos"].copy().astype(np.float32)
    vecp2 = rig["vecp"].copy().astype(np.float32)
    blen2 = rig["blen"].copy().astype(np.float32)

    # Small position noise
    pos2 += rng.normal(0.0, 0.002, size=pos2.shape).astype(np.float32)

    # Small scale variation
    s = float(rng.uniform(0.95, 1.05))
    pos2 *= s
    vecp2 *= s
    blen2 *= s

    return {
        **rig,
        "pos": pos2,
        "vecp": vecp2,
        "blen": blen2,
    }


# ============== DATASET BUILDING ==============

def pair_features(fb, fa):
    """Create pair features from two joint feature vectors"""
    return np.concatenate([
        fb,  # B features
        fa,  # A features
        np.abs(fb - fa),  # Absolute difference
        fb * fa,  # Element-wise product
    ], axis=0).astype(np.float32)


def build_dataset(rigs):
    """Build training dataset using geometric matching for cross-rig pairs"""
    X = []
    y = []

    print("Building same-rig augmentation pairs...")
    total_start = time.time()

    # Same-rig augmentation (this is always reliable)
    for rig_idx, A in enumerate(rigs):
        rig_start = time.time()
        fA = joint_features_v2(A)
        nA = fA.shape[0]

        rig_samples = 0
        for aug_idx in range(AUG_PER_RIG):
            B = make_augmented_rig(A)
            fB = joint_features_v2(B)
            map_b_to_a = B["map_b_to_a"]
            nB = fB.shape[0]

            for b in range(nB):
                a_true = int(map_b_to_a[b])
                if 0 <= a_true < nA:
                    # Positive pair
                    X.append(pair_features(fB[b], fA[a_true]))
                    y.append(1.0)
                    rig_samples += 1

                    # Negative pairs - use vectorized random
                    neg_candidates = np.delete(np.arange(nA), a_true)
                    negs = rng.choice(neg_candidates, size=min(NEG_PER_POS, len(neg_candidates)), replace=False)

                    for a in negs:
                        X.append(pair_features(fB[b], fA[a]))
                        y.append(0.0)
                        rig_samples += 1
                else:
                    # Junk joint - sample negatives
                    negs = rng.choice(nA, size=min(NEG_PER_POS, nA), replace=False)
                    for a in negs:
                        X.append(pair_features(fB[b], fA[a]))
                        y.append(0.0)
                        rig_samples += 1

        elapsed = time.time() - rig_start
        print(
            f"  [{rig_idx + 1}/{len(rigs)}] {os.path.basename(A['path'])}: {nA} joints, {rig_samples} samples, {elapsed:.1f}s")

    same_rig_time = time.time() - total_start
    print(f"\nSame-rig augmentation complete: {len(X)} samples in {same_rig_time:.1f}s")

    print("\nBuilding cross-rig pairs using geometric matching...")
    cross_start = time.time()

    # Cross-rig pairs using GEOMETRIC matching (no names!)
    cross_rig_pairs_found = 0
    total_pairs = len(rigs) * (len(rigs) - 1) // 2
    pair_count = 0

    for i in range(len(rigs)):
        for j in range(i + 1, len(rigs)):
            pair_count += 1
            A = rigs[i]
            B = rigs[j]

            # Find geometric correspondences
            try:
                correspondences = find_geometric_correspondences_hungarian(A, B)
            except ImportError:
                correspondences = find_geometric_correspondences(A, B)

            if len(correspondences) < 3:  # Reduced from 5 to 3
                continue  # Not enough correspondences

            cross_rig_pairs_found += 1
            print(
                f"    Pair {os.path.basename(A['path'])} <-> {os.path.basename(B['path'])}: {len(correspondences)} matches")

            # Pre-compute features once per pair
            fA_base = joint_features_v2(A)
            fB_base = joint_features_v2(B)
            nA = fA_base.shape[0]

            for _ in range(CROSS_RIG_AUG_PER_PAIR):
                A_aug = apply_global_aug(A)
                B_aug = apply_global_aug(B)

                fA = joint_features_v2(A_aug)
                fB = joint_features_v2(B_aug)

                for a_idx, b_idx, _ in correspondences:
                    # Positive pair
                    X.append(pair_features(fB[b_idx], fA[a_idx]))
                    y.append(1.0)

                    # Negative pairs - vectorized
                    neg_candidates = np.delete(np.arange(nA), a_idx)
                    negs = rng.choice(neg_candidates, size=min(CROSS_NEG_PER_POS, len(neg_candidates)), replace=False)

                    for aa in negs:
                        X.append(pair_features(fB[b_idx], fA[aa]))
                        y.append(0.0)

            if pair_count % 20 == 0:
                print(f"  Processed {pair_count}/{total_pairs} rig pairs, found {cross_rig_pairs_found} with matches")

    cross_time = time.time() - cross_start
    print(f"  Found {cross_rig_pairs_found} cross-rig pairs with sufficient correspondences in {cross_time:.1f}s")

    total_time = time.time() - total_start
    print(f"\nTotal dataset building time: {total_time:.1f}s")

    X = np.stack(X, axis=0).astype(np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    return X, y


# ============== MODEL TRAINING ==============

def standardize_fit(X):
    """Compute standardization parameters"""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def standardize_apply(X, mu, sd):
    """Apply standardization"""
    return ((X - mu) / sd).astype(np.float32)


def train_model(X, y):
    """Train the pairwise matching model"""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # Shuffle
    idx = rng.permutation(len(X))
    X = X[idx]
    y = y[idx]

    # Split
    split = int(len(X) * 0.9)
    Xtr, Xva = X[:split], X[split:]
    ytr, yva = y[:split], y[split:]

    # Standardize
    mu, sd = standardize_fit(Xtr)
    Xtr = standardize_apply(Xtr, mu, sd)
    Xva = standardize_apply(Xva, mu, sd)

    # Build model - slightly larger to handle more features
    inp = keras.Input(shape=(X.shape[1],), dtype=tf.float32)
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.BinaryAccuracy(name="acc"),
            keras.metrics.Precision(name="prec"),
            keras.metrics.Recall(name="rec"),
        ],
    )

    # Callbacks
    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    print(f"\nTraining model on {len(Xtr)} samples, validating on {len(Xva)} samples")
    print(f"Feature dimension: {X.shape[1]}")

    model.fit(
        Xtr,
        ytr,
        validation_data=(Xva, yva),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=cb,
    )

    model.save(MODEL_OUT)
    np.savez(NORM_OUT, mu=mu, sd=sd)

    return model, mu, sd


# ============== MAIN ==============

def main():
    csvs = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.csv")))
    if len(csvs) < 2:
        raise RuntimeError(f"Need at least 2 CSVs in {TRAIN_DIR}")

    print(f"Loading {len(csvs)} rigs...")
    rigs = [load_rig_csv(p) for p in csvs]

    print("\nLoaded rigs:")
    for r in rigs:
        print(f"  - {os.path.basename(r['path'])}: {r['n']} joints")

    X, y = build_dataset(rigs)
    pos_count = int(y.sum())
    neg_count = len(y) - pos_count
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Positive: {pos_count}, Negative: {neg_count}")
    print(f"  Ratio: 1:{neg_count / pos_count:.1f}")

    model, mu, sd = train_model(X, y)
    print(f"\nSaved model: {MODEL_OUT}")
    print(f"Saved normalization: {NORM_OUT}")


if __name__ == "__main__":
    main()



