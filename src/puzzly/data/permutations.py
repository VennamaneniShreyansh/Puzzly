# src/puzzly/data/permutations.py
import itertools
import random
import json

def generate_permutation_set(num_patches=9, num_permutations=50, seed=42, candidate_pool_size=2000):
    """
    Generates a fixed set of `num_permutations` orderings of 0..num_patches-1,
    chosen to be maximally different from each other (max-Hamming-distance greedy selection).
    Deterministic given the same seed - same output every time, on any machine.
    """
    rng = random.Random(seed)

    candidate_pool = set()
    base = list(range(num_patches))
    while len(candidate_pool) < candidate_pool_size:
        rng.shuffle(base)
        candidate_pool.add(tuple(base))
    candidate_pool = list(candidate_pool)

    chosen = [candidate_pool.pop(rng.randrange(len(candidate_pool)))]

    for _ in range(num_permutations - 1):
        best_candidate = None
        best_min_dist = -1

        for cand in candidate_pool:
            min_dist = min(
                sum(a != b for a, b in zip(cand, existing))
                for existing in chosen
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = cand

        chosen.append(best_candidate)
        candidate_pool.remove(best_candidate)

    return chosen


def save_permutation_set(perm_set, path="permutation_set.json"):
    with open(path, "w") as f:
        json.dump(perm_set, f)


def load_permutation_set(path="permutation_set.json"):
    with open(path) as f:
        return [tuple(p) for p in json.load(f)]