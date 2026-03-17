#!/usr/bin/env python3

import argparse
import json
import math
from collections import Counter, defaultdict


def is_number(x):
    return isinstance(x, (int, float)) and not (
        isinstance(x, float) and (math.isnan(x) or math.isinf(x))
    )


def normalize_status(st):
    st = str(st).lower() if st is not None else "unknown"
    if st in ("sat", "unsat"):
        return st
    return "unknown"


def parse_baseline(entry, cap):
    if not isinstance(entry, list) or len(entry) < 2:
        return "unknown", float(cap)

    st = normalize_status(entry[0])
    t = entry[1]

    if not is_number(t):
        return "unknown", float(cap)

    t = float(t)
    if t > cap:
        return "unknown", float(cap)

    return st, t


def parse_compass(entry, cap, time_field):
    if not isinstance(entry, list) or len(entry) < 5:
        return None

    flag = str(entry[4]).lower() if entry[4] is not None else "failed"

    if time_field == "solve" and flag == "succeed" and len(entry) > 5:
        t = entry[5]
    else:
        t = entry[3] if len(entry) > 3 else None

    if not is_number(t):
        t = float(cap)
    else:
        t = float(t)

    if t > cap:
        return "unknown", float(cap), flag

    st = "sat" if flag == "succeed" else "unknown"
    return st, t, flag


def is_conclusive(st):
    return st in ("sat", "unsat")


def simulate_portfolio(b_st, b_t, c_st, c_t):
    b_con = is_conclusive(b_st)
    c_con = is_conclusive(c_st)

    if b_con and c_con:
        if b_t <= c_t:
            return b_st, b_t, "solver"
        return c_st, c_t, "compass"

    if b_con:
        return b_st, b_t, "solver"

    if c_con:
        return c_st, c_t, "compass"

    return "unknown", max(b_t, c_t), "both_unknown"


def compute_metrics(status_list, time_list):
    counts = Counter(status_list)
    time_sums = Counter()
    for st, t in zip(status_list, time_list):
        time_sums[st] += float(t)

    total_time = sum(time_sums.values())
    n = len(status_list)
    overall_avg = total_time / n if n else 0.0

    per_status_avg = {}
    for st in ("sat", "unsat", "unknown"):
        per_status_avg[st] = time_sums[st] / counts[st] if counts[st] else 0.0

    return {
        "n": n,
        "counts": counts,
        "time_sums": time_sums,
        "total_time": total_time,
        "overall_avg": overall_avg,
        "per_status_avg": per_status_avg,
    }


def fmt_counts(counts):
    return f"sat={counts.get('sat', 0)} unsat={counts.get('unsat', 0)} unknown={counts.get('unknown', 0)}"


def fmt_time_sums(time_sums):
    return f"sat_sum={time_sums.get('sat', 0.0):.3f} unsat_sum={time_sums.get('unsat', 0.0):.3f} unknown_sum={time_sums.get('unknown', 0.0):.3f}"


def fmt_avgs(per_status_avg, overall_avg):
    return (
        f"sat_avg={per_status_avg.get('sat', 0.0):.3f} "
        f"unsat_avg={per_status_avg.get('unsat', 0.0):.3f} "
        f"unknown_avg={per_status_avg.get('unknown', 0.0):.3f} "
        f"overall_avg={overall_avg:.3f}"
    )


def run_scope(name, keys, baseline, compass, cap, time_field, require_compass):
    b_statuses = []
    b_times = []

    p_statuses = []
    p_times = []

    wins = Counter()
    conversions = Counter()
    disagreements = 0

    for k in keys:
        b_st, b_t = parse_baseline(baseline.get(k), cap)

        c = parse_compass(compass.get(k), cap, time_field)
        if require_compass and c is None:
            continue

        if c is None:
            p_st, p_t, w = b_st, b_t, "no_compass"
        else:
            c_st, c_t, _ = c
            if is_conclusive(b_st) and is_conclusive(c_st) and b_st != c_st:
                disagreements += 1
            p_st, p_t, w = simulate_portfolio(b_st, b_t, c_st, c_t)

        b_statuses.append(b_st)
        b_times.append(b_t)

        p_statuses.append(p_st)
        p_times.append(p_t)

        wins[w] += 1
        conversions[(b_st, p_st)] += 1

    b_metrics = compute_metrics(b_statuses, b_times)
    p_metrics = compute_metrics(p_statuses, p_times)

    saved_total = b_metrics["total_time"] - p_metrics["total_time"]
    saved_avg = saved_total / b_metrics["n"] if b_metrics["n"] else 0.0

    faster = 0
    slower = 0
    equal = 0
    for bt, pt in zip(b_times, p_times):
        if pt < bt:
            faster += 1
        elif pt > bt:
            slower += 1
        else:
            equal += 1

    print("=" * 80)
    print(f"Scope: {name} (n={b_metrics['n']}, cap={cap}, compass_time_field={time_field})")
    print("-" * 80)
    print("Baseline:")
    print(f"  counts: {fmt_counts(b_metrics['counts'])}")
    print(f"  time_sums: {fmt_time_sums(b_metrics['time_sums'])}")
    print(f"  total_time: {b_metrics['total_time']:.3f}  overall_avg: {b_metrics['overall_avg']:.3f}")
    print(
        f"  per_status_avg: sat={b_metrics['per_status_avg']['sat']:.3f} unsat={b_metrics['per_status_avg']['unsat']:.3f} unknown={b_metrics['per_status_avg']['unknown']:.3f}"
    )
    print("Portfolio (Solver || COMPASS):")
    print(f"  counts: {fmt_counts(p_metrics['counts'])}")
    print(f"  time_sums: {fmt_time_sums(p_metrics['time_sums'])}")
    print(f"  total_time: {p_metrics['total_time']:.3f}  overall_avg: {p_metrics['overall_avg']:.3f}")
    print(
        f"  per_status_avg: sat={p_metrics['per_status_avg']['sat']:.3f} unsat={p_metrics['per_status_avg']['unsat']:.3f} unknown={p_metrics['per_status_avg']['unknown']:.3f}"
    )
    print("-" * 80)
    print(f"Wins: {dict(wins)}")
    print(f"Disagreements (both conclusive but different): {disagreements}")
    print("Conversions (baseline -> portfolio):")
    for (a, b), c in conversions.most_common(10):
        print(f"  {a} -> {b}: {c}")
    print("-" * 80)
    print(f"Time saved total: {saved_total:.3f}  avg per instance: {saved_avg:.3f}")
    print(f"Portfolio faster/slower/equal: {faster}/{slower}/{equal}")

    latex_row = (
        f"Parallel Portfolio & {p_metrics['counts'].get('sat', 0)} & {p_metrics['counts'].get('unsat', 0)} & {p_metrics['counts'].get('unknown', 0)} "
        f"& {p_metrics['per_status_avg']['sat']:.1f} & {p_metrics['overall_avg']:.1f} & {int(round(p_metrics['total_time']))} \\\\"
    )
    print("-" * 80)
    print("LaTeX row (portfolio):")
    print(latex_row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_keys",
        default="test_rl/predictor/smt_comp_NIA/QF_NIA_test.json",
    )
    parser.add_argument(
        "--baseline",
        default="test_rl/test_solve/NIA/NIA.json",
    )
    parser.add_argument(
        "--compass",
        default="test_rl/info_dict_gai_6_normal_0503_pre_llm_llama3.1:70b_1200s_QF_NIA.txt",
    )
    parser.add_argument("--cap", type=float, default=1200.0)
    parser.add_argument(
        "--compass_time_field",
        choices=("total", "solve"),
        default="total",
    )

    args = parser.parse_args()

    with open(args.test_keys, "r") as f:
        test_dict = json.load(f)
    with open(args.baseline, "r") as f:
        baseline = json.load(f)
    with open(args.compass, "r") as f:
        compass = json.load(f)

    keys_all = list(test_dict.keys())
    keys_intersection = [k for k in keys_all if k in compass]

    run_scope(
        name="ALL (missing COMPASS treated as 'no_compass')",
        keys=keys_all,
        baseline=baseline,
        compass=compass,
        cap=args.cap,
        time_field=args.compass_time_field,
        require_compass=False,
    )

    run_scope(
        name="INTERSECTION (only instances with COMPASS cache)",
        keys=keys_intersection,
        baseline=baseline,
        compass=compass,
        cap=args.cap,
        time_field=args.compass_time_field,
        require_compass=True,
    )


if __name__ == "__main__":
    main()
