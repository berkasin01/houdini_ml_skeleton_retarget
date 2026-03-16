"""
HDA PYTHON MODULE - AI Skeleton Matcher v7
===========================================
For use with joint_matcher_v2.keras or v3 (46 features: geometric + name)
Now includes SYNONYM support for pelvis/hips, clavicle/shoulder, etc.

Button callback: hou.phm().run_matcher(hou.pwd())
"""

import hou
import numpy as np
import re


FINGER_KEYWORDS = ['thumb', 'index', 'middle', 'ring', 'pinky']

BODY_KEYWORDS = ['spine', 'head', 'neck', 'shoulder', 'clavicle',
                 'arm', 'forearm', 'elbow', 'wrist', 'hand',
                 'hip', 'pelvis', 'thigh', 'leg', 'knee', 'ankle', 'foot', 'toe',
                 'jaw', 'eye', 'eyelid', 'eyebrow', 'brow', 'mouth', 'lip',
                 'twist', 'calf', 'shin', 'upperarm', 'lowerarm', 'upperleg', 'lowerleg',
                 'upleg', 'lowleg', 'root', 'ball']

# Synonyms - words that mean the same thing (normalized to canonical form)
SYNONYMS = {
    # Hip/Pelvis
    'hips': 'pelvis',
    'hip': 'pelvis',

    # Shoulder/Clavicle
    'clavicle': 'shoulder',

    # Arm variations
    'upperarm': 'arm',
    'upper_arm': 'arm',

    # Forearm variations
    'lowerarm': 'forearm',
    'lower_arm': 'forearm',

    # Leg variations
    'thigh': 'upperleg',
    'upleg': 'upperleg',
    'up_leg': 'upperleg',

    # Lower leg variations
    'calf': 'lowerleg',
    'shin': 'lowerleg',
    'lowleg': 'lowerleg',
    'low_leg': 'lowerleg',

    # Eyebrow
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

        # Normalize synonyms first
        part_normalized = normalize_token(part)

        for kw in FINGER_KEYWORDS:
            if kw in part_normalized:
                tokens.add(kw)

        for kw in BODY_KEYWORDS:
            if kw in part_normalized:
                tokens.add(normalize_token(kw))

        # Direct match
        if part_normalized in BODY_KEYWORDS or part_normalized in SYNONYMS.values():
            tokens.add(part_normalized)

        if part in ['l', 'left']:
            tokens.add('left')
        elif part in ['r', 'right']:
            tokens.add('right')

        numbers = re.findall(r'\d+', part)
        for num in numbers:
            tokens.add(str(int(num)))

    if name_lower.startswith('l_') or name_lower.startswith('l.') or name_lower.endswith('_l') or name_lower.endswith(
            '.l'):
        tokens.add('left')
    if name_lower.startswith('r_') or name_lower.startswith('r.') or name_lower.endswith('_r') or name_lower.endswith(
            '.r'):
        tokens.add('right')

    if 'twist' in name_lower:
        tokens.add('twist')

    # Check for specific keywords in full name
    for kw in BODY_KEYWORDS:
        if kw in name_lower:
            tokens.add(normalize_token(kw))

    # Check synonyms in full name
    for syn, canonical in SYNONYMS.items():
        if syn in name_lower:
            tokens.add(canonical)

    return tokens


def compute_name_similarity_features(name_a, name_b):
    tokens_a = extract_name_tokens(name_a)
    tokens_b = extract_name_tokens(name_b)

    features = []

    # 1. Jaccard similarity
    if len(tokens_a) == 0 and len(tokens_b) == 0:
        jaccard = 0.5
    else:
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        jaccard = intersection / union if union > 0 else 0.0
    features.append(jaccard)

    # 2. Finger keyword match
    finger_a = tokens_a & set(FINGER_KEYWORDS)
    finger_b = tokens_b & set(FINGER_KEYWORDS)

    if finger_a and finger_b:
        finger_match = 1.0 if finger_a == finger_b else 0.0
    elif finger_a or finger_b:
        finger_match = 0.0
    else:
        finger_match = 0.5
    features.append(finger_match)

    # 3. Segment number match
    nums_a = set(t for t in tokens_a if t.isdigit())
    nums_b = set(t for t in tokens_b if t.isdigit())

    if nums_a and nums_b:
        num_match = 1.0 if nums_a & nums_b else 0.0
    elif nums_a or nums_b:
        num_match = 0.0
    else:
        num_match = 0.5
    features.append(num_match)

    # 4. Side match
    side_a = 'left' if 'left' in tokens_a else ('right' if 'right' in tokens_a else 'center')
    side_b = 'left' if 'left' in tokens_b else ('right' if 'right' in tokens_b else 'center')

    if side_a == 'center' or side_b == 'center':
        side_match = 0.5
    else:
        side_match = 1.0 if side_a == side_b else 0.0
    features.append(side_match)

    # 5. Body keyword overlap (normalized)
    body_keywords_normalized = set(normalize_token(kw) for kw in BODY_KEYWORDS)
    body_a = tokens_a & body_keywords_normalized
    body_b = tokens_b & body_keywords_normalized
    body_overlap = len(body_a & body_b)
    features.append(float(body_overlap))

    # 6. Is finger joint
    is_finger = 1.0 if (finger_a or finger_b) else 0.0
    features.append(is_finger)

    return np.array(features, dtype=np.float32)

def extract_skeleton_data(geo):
    points = geo.points()
    n = len(points)
    if n == 0:
        return None

    names = []
    pos = []
    parent_idx = []

    for p in points:
        names.append(p.attribValue("name"))
        pos.append(p.position())
        parent_idx.append(int(p.attribValue("parent_idx")))

    pos = np.array([[p[0], p[1], p[2]] for p in pos], dtype=np.float32)
    parent_idx = np.array(parent_idx, dtype=np.int32)

    # Depth
    depth = np.zeros(n, dtype=np.int32)
    for i in range(n):
        cur = i
        d = 0
        seen = {cur}
        while True:
            par = parent_idx[cur]
            if par < 0 or par in seen or par >= n:
                break
            seen.add(par)
            d += 1
            cur = par
        depth[i] = d

    # Root-relative
    root_idx = int(np.where(parent_idx < 0)[0][0]) if np.any(parent_idx < 0) else 0
    pos_root = pos - pos[root_idx]

    # Bone vectors
    vec_parent = np.zeros((n, 3), dtype=np.float32)
    bone_len = np.zeros(n, dtype=np.float32)
    for i in range(n):
        par = parent_idx[i]
        if 0 <= par < n:
            v = pos[i] - pos[par]
            vec_parent[i] = v
            bone_len[i] = np.linalg.norm(v)

    # Children
    children = [[] for _ in range(n)]
    for i, par in enumerate(parent_idx):
        if 0 <= par < n:
            children[par].append(i)

    child_count = np.array([len(c) for c in children], dtype=np.int32)
    is_leaf = (child_count == 0).astype(np.int32)

    return {
        "n": n, "names": names, "pos": pos, "pos_root": pos_root,
        "vec_parent": vec_parent, "bone_len": bone_len,
        "parent_idx": parent_idx, "depth": depth,
        "child_count": child_count, "is_leaf": is_leaf,
        "children": children,
    }


def extract_geometric_features(skel):
    pos_root = skel["pos_root"]
    vec_parent = skel["vec_parent"]
    bone_len = skel["bone_len"]
    depth = skel["depth"]
    child_count = skel["child_count"]
    is_leaf = skel["is_leaf"]

    valid_bones = bone_len[bone_len > 1e-8]
    scale = float(np.median(valid_bones)) if len(valid_bones) > 0 else 1.0

    pos_norm = pos_root / scale

    norms = np.linalg.norm(vec_parent, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    dir_parent = vec_parent / norms

    blen_norm = (bone_len / scale).reshape(-1, 1)
    depth_norm = (depth / max(depth.max(), 1)).astype(np.float32).reshape(-1, 1)
    cc_norm = (child_count / max(child_count.max(), 1)).astype(np.float32).reshape(-1, 1)
    leaf = is_leaf.astype(np.float32).reshape(-1, 1)

    return np.hstack([pos_norm, dir_parent, blen_norm, depth_norm, cc_norm, leaf]).astype(np.float32)


def compute_scores(model, mu, sd, feat_target, feat_source, names_target, names_source):
    n_target = feat_target.shape[0]
    n_source = feat_source.shape[0]

    pairs = []
    for s in range(n_source):
        for t in range(n_target):
            diff = np.abs(feat_target[t] - feat_source[s])
            prod = feat_target[t] * feat_source[s]
            name_feat = compute_name_similarity_features(names_source[s], names_target[t])
            pair = np.concatenate([feat_target[t], feat_source[s], diff, prod, name_feat])
            pairs.append(pair)

    pairs = np.array(pairs, dtype=np.float32)
    pairs = (pairs - mu) / sd

    scores = model.predict(pairs, verbose=0).flatten()
    scores = scores.reshape(n_source, n_target)

    return scores


def hungarian_match(scores, pos_target, pos_source, min_score=0.0):
    from scipy.optimize import linear_sum_assignment

    n_source, n_target = scores.shape

    cost = -scores.copy()
    for s in range(n_source):
        for t in range(n_target):
            side_t = "L" if pos_target[t, 0] < -0.01 else ("R" if pos_target[t, 0] > 0.01 else "C")
            side_s = "L" if pos_source[s, 0] < -0.01 else ("R" if pos_source[s, 0] > 0.01 else "C")
            if side_t in ("L", "R") and side_s in ("L", "R") and side_t != side_s:
                cost[s, t] += 1000

    if n_source > n_target:
        # Pad with dummy columns for extra source joints
        cost = np.hstack([cost, np.full((n_source, n_source - n_target), 1000)])

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = {}
    used_targets = set()  # Track which targets are already matched

    # First pass: collect all valid matches
    candidate_matches = []
    for s, t in zip(row_ind, col_ind):
        if t < n_target:
            score = float(scores[s, t])
            if score >= min_score:
                candidate_matches.append((score, s, t))

    # Sort by score descending - highest scores first
    candidate_matches.sort(reverse=True)

    # Second pass: assign matches, each target only once (highest score wins)
    for score, s, t in candidate_matches:
        if t not in used_targets:
            matches[s] = (t, score)
            used_targets.add(t)

    return matches


def greedy_match(scores, pos_target, pos_source, threshold=0.5):
    """Greedy matching with threshold"""
    n_source, n_target = scores.shape
    candidates = []

    for s in range(n_source):
        for t in range(n_target):
            if scores[s, t] >= threshold:
                side_t = "L" if pos_target[t, 0] < -0.01 else ("R" if pos_target[t, 0] > 0.01 else "C")
                side_s = "L" if pos_source[s, 0] < -0.01 else ("R" if pos_source[s, 0] > 0.01 else "C")
                if side_t in ("L", "R") and side_s in ("L", "R") and side_t != side_s:
                    continue
                candidates.append((scores[s, t], s, t))

    candidates.sort(reverse=True)
    used_t, used_s, matches = set(), set(), {}

    for score, s, t in candidates:
        if s not in used_s and t not in used_t:
            matches[s] = (t, score)
            used_s.add(s)
            used_t.add(t)

    return matches


def run_matcher(hda):
    import tensorflow as tf

    print("=" * 60)
    print("AI SKELETON MATCHER v7")
    print("(Geometric + Name Features + Synonyms)")
    print("=" * 60)

    source_node = hda.node("SOURCE_SKEL_IN_PYTHON")
    target_node = hda.node("TARGET_SKEL_IN_PYTHON")

    print(f"Source node: {source_node}")
    print(f"Target node: {target_node}")

    if not source_node or not target_node:
        hou.ui.displayMessage("Cannot find skeleton nodes!", severity=hou.severityType.Error)
        return

    geo_source = source_node.geometry()
    geo_target = target_node.geometry()

    if not geo_source or not geo_target:
        hou.ui.displayMessage("No geometry found!", severity=hou.severityType.Error)
        return

    # Parameters
    model_path = hou.expandString(
        hda.parm("model_path").evalAsString() if hda.parm("model_path") else "$HIP/joint_matcher_v2.keras")
    norm_path = hou.expandString(
        hda.parm("norm_path").evalAsString() if hda.parm("norm_path") else "$HIP/joint_matcher_v2_norm.npz")
    min_score = hda.parm("min_score_thresh").evalAsFloat() if hda.parm("min_score_thresh") else 0.5
    do_full = (hda.parm("match_mode").evalAsInt() == 0) if hda.parm("match_mode") else True

    # Extract skeletons
    skel_source = extract_skeleton_data(geo_source)
    skel_target = extract_skeleton_data(geo_target)
    print(f"Source: {skel_source['n']} joints")
    print(f"Target: {skel_target['n']} joints")

    # Debug: Test synonym matching
    print("\n--- Synonym Test ---")
    test_pairs = [("hip", "pelvis"), ("shoulder_l", "clavicle_l"), ("upperarm_l", "arm_l"), ("thigh_l", "upperleg_l")]
    for a, b in test_pairs:
        tokens_a = extract_name_tokens(a)
        tokens_b = extract_name_tokens(b)
        feat = compute_name_similarity_features(a, b)
        print(f"  {a} <-> {b}: tokens={tokens_a & tokens_b}, jaccard={feat[0]:.2f}")
    print("-------------------\n")

    # AI Matching
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    norm = np.load(norm_path)
    mu, sd = norm["mu"], norm["sd"]

    feat_source = extract_geometric_features(skel_source)
    feat_target = extract_geometric_features(skel_target)

    print("Computing scores (with name features + synonyms)...")
    scores = compute_scores(model, mu, sd, feat_target, feat_source,
                            skel_target["names"], skel_source["names"])

    print(f"  Range: {scores.min():.3f} - {scores.max():.3f}")
    print(f"  >0.9: {(scores > 0.9).sum()}, >0.5: {(scores > 0.5).sum()}")

    try:
        from scipy.optimize import linear_sum_assignment
        scipy_ok = True
    except ImportError:
        scipy_ok = False

    if do_full and scipy_ok:
        print("\nHungarian matching...")
        matches = hungarian_match(scores, skel_target["pos"], skel_source["pos"], min_score)
    else:
        print(f"\nGreedy matching...")
        matches = greedy_match(scores, skel_target["pos"], skel_source["pos"], min_score)

    print(f"Matched: {len(matches)} / {skel_source['n']} joints")
    # Check for duplicate target matches
    print("\n🔍 CHECKING FOR DUPLICATES...")
    target_matches = {}  # target_name -> list of (source_name, score)

    for s, (t, score) in matches.items():
        src_name = skel_source["names"][s]
        tgt_name = skel_target["names"][t]

        if tgt_name not in target_matches:
            target_matches[tgt_name] = []
        target_matches[tgt_name].append((src_name, score))

    # Find duplicates
    duplicates_found = False
    for tgt_name, sources in target_matches.items():
        if len(sources) > 1:
            duplicates_found = True
            print(f"⚠️ DUPLICATE: {tgt_name} matched by:")
            for src_name, score in sorted(sources, key=lambda x: -x[1]):
                print(f"    • {src_name} ({score:.3f})")

    if not duplicates_found:
        print("✅ No duplicates - all targets matched only once!")

    print("\nPopulating overrides...")

    num_joints = skel_source['n']

    joint_order = []

    unmatched = [(i, skel_source["names"][i]) for i in range(num_joints) if i not in matches]
    unmatched.sort(key=lambda x: x[1])
    joint_order.extend(unmatched)

    matched_list = [(i, skel_source["names"][i], matches[i][1]) for i in range(num_joints) if i in matches]
    matched_list.sort(key=lambda x: x[2])
    joint_order.extend([(i, name) for i, name, _ in matched_list])

    num_parm = hda.parm("num_overrides")
    if num_parm:
        num_parm.set(num_joints)

    matched_count = 0
    unmatched_count = 0

    for idx, (i, src_name) in enumerate(joint_order):
        parm_idx = idx + 1

        src_parm = hda.parm(f"override_source{parm_idx}")
        if src_parm:
            src_parm.set(src_name)

        tgt_parm = hda.parm(f"override_target{parm_idx}")
        if tgt_parm:
            if i in matches:
                t, _ = matches[i]
                tgt_parm.set(skel_target["names"][t])
                matched_count += 1
            else:
                tgt_parm.set("")
                unmatched_count += 1

    use_parm = hda.parm("use_overrides")
    if use_parm:
        use_parm.set(1)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total:     {matched_count} / {num_joints}")
    print(f"Unmatched: {unmatched_count}")

    if unmatched_count > 0:
        print(f"\n⚠ UNMATCHED:")
        for i, name in unmatched:
            print(f"  • {name}")

    print(f"\n🖐 FINGER MATCHES:")
    for s, (t, score) in sorted(matches.items(), key=lambda x: -x[1][1]):
        src_name = skel_source["names"][s]
        tgt_name = skel_target["names"][t]
        if any(kw in src_name.lower() for kw in FINGER_KEYWORDS):
            print(f"  {score:.3f}: {src_name} -> {tgt_name}")

    print("=" * 60)

    python_node = hda.node("match_processor") or hda.node("python1")
    if python_node:
        python_node.cook(force=True)

    hou.ui.displayMessage(
        f"Matching Complete!\n\n"
        f"Total: {matched_count} / {num_joints}\n"
        f"Unmatched: {unmatched_count}\n\n"
        f"Check console for details.",
        title="AI Skeleton Matcher v7"
    )


def clear_overrides(hda):
    num_parm = hda.parm("num_overrides")
    if num_parm:
        num_parm.set(0)
    use_parm = hda.parm("use_overrides")
    if use_parm:
        use_parm.set(0)
    print("Overrides cleared!")