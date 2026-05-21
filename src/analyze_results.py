"""
analyze_results.py
==================
Performs the statistical analysis described in the proposal Section IV.E:

  Step 1: Shapiro-Wilk normality test on differences
  Step 2: Wilcoxon signed-rank test (primary inferential test)
  Step 3: Bonferroni correction (alpha' = 0.017 for 3 outcomes)
  Step 4: Spearman rank correlation (latency vs PSS recovery time)
  Step 5: PSS validation (Pearson + Bland-Altman vs expert RULA)

Outputs:
  results/summary_stats.csv
  results/test_results.txt
  results/bland_altman.png
  results/pss_timelines.png
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

LOG_DIR = "data/sessions"
RESULTS_DIR = "results"
ALPHA = 0.05
ALPHA_CORRECTED = ALPHA / 3  # Bonferroni for 3 primary outcomes


# ============================================================
# DATA LOADING
# ============================================================
def per_participant_summary(log_dir=LOG_DIR):
    """
    For each participant x condition, compute:
      - mean PSS over the session
      - count of threshold-exceeding deviations (rising edges above PSS=0.4)
      - intervention count (from events file)
      - mean intervention latency (experimental only)
      - mean PSS recovery time after intervention (experimental only)
    """
    rows = []
    for frames_csv in glob.glob(os.path.join(log_dir, "*_frames.csv")):
        base = os.path.basename(frames_csv).replace("_frames.csv", "")
        parts = base.split("_")
        participant = parts[0]
        condition = parts[1]

        df = pd.read_csv(frames_csv)
        if len(df) == 0:
            continue

        mean_pss = df["pss_smooth"].mean()

        # Threshold-exceeding deviations: count rising edges above 0.4
        above = (df["pss_smooth"] >= 0.4).astype(int)
        deviations = int((above.diff() == 1).sum())

        # Events
        events_csv = frames_csv.replace("_frames.csv", "_events.csv")
        intervention_count = 0
        mean_latency = None
        mean_recovery = None
        if os.path.exists(events_csv):
            ev = pd.read_csv(events_csv)
            interventions = ev[ev["event_type"] == "intervention"]
            intervention_count = len(interventions)
            if intervention_count > 0:
                latencies = pd.to_numeric(interventions["latency_s"],
                                          errors="coerce").dropna()
                if len(latencies) > 0:
                    mean_latency = float(latencies.mean())
                # Recovery: time after each intervention until PSS < 0.30
                mean_recovery = compute_mean_recovery_time(df, interventions)

        rows.append({
            "participant": participant,
            "condition": condition,
            "mean_pss": mean_pss,
            "deviations": deviations,
            "interventions": intervention_count,
            "mean_latency_s": mean_latency,
            "mean_recovery_s": mean_recovery,
            "session_duration_s": df["timestamp_s"].iloc[-1],
        })

    return pd.DataFrame(rows)


def compute_mean_recovery_time(frames_df, interventions_df,
                               recovery_threshold=0.30):
    """
    For each intervention, compute time from intervention until PSS first
    drops below recovery_threshold. Return mean across all interventions.
    """
    recovery_times = []
    for _, row in interventions_df.iterrows():
        t_intervention = float(row["timestamp_s"])
        post = frames_df[frames_df["timestamp_s"] > t_intervention]
        below = post[post["pss_smooth"] < recovery_threshold]
        if len(below) > 0:
            recovery_times.append(below["timestamp_s"].iloc[0]
                                  - t_intervention)
    if not recovery_times:
        return None
    return float(np.mean(recovery_times))


# ============================================================
# WILCOXON
# ============================================================
def wilcoxon_paired(control_vals, exp_vals, label):
    """Wilcoxon signed-rank with effect size r and Shapiro-Wilk on diffs."""
    control_vals = np.asarray(control_vals, dtype=float)
    exp_vals = np.asarray(exp_vals, dtype=float)
    diffs = control_vals - exp_vals
    n = len(control_vals)

    shapiro_p = float("nan")
    if n >= 3:
        _, shapiro_p = stats.shapiro(diffs)

    try:
        w_stat, p_val = stats.wilcoxon(control_vals, exp_vals,
                                       alternative="two-sided")
    except ValueError as e:
        return {"outcome": label, "error": str(e)}

    # Effect size r = |Z| / sqrt(N)
    z = stats.norm.isf(p_val / 2.0) * np.sign(np.median(diffs))
    r = abs(z) / np.sqrt(n) if n > 0 else float("nan")

    return {
        "outcome": label,
        "n_pairs": n,
        "median_control": float(np.median(control_vals)),
        "median_experimental": float(np.median(exp_vals)),
        "wilcoxon_W": float(w_stat),
        "p_uncorrected": float(p_val),
        "p_significant_corrected": p_val < ALPHA_CORRECTED,
        "effect_size_r": float(r),
        "shapiro_p_on_diffs": float(shapiro_p),
    }


def run_primary_analysis(summary_df, tlx_df=None):
    results = []

    wide = (summary_df.pivot(index="participant", columns="condition",
                             values="mean_pss")
                       .dropna())
    if len(wide) > 0:
        results.append(wilcoxon_paired(
            wide["control"], wide["experimental"], "mean_pss"))

    wide_dev = (summary_df.pivot(index="participant", columns="condition",
                                 values="deviations")
                          .dropna())
    if len(wide_dev) > 0:
        results.append(wilcoxon_paired(
            wide_dev["control"], wide_dev["experimental"],
            "deviation_count"))

    if tlx_df is not None:
        wide_tlx = (tlx_df.pivot(index="participant", columns="condition",
                                  values="tlx_composite")
                          .dropna())
        if len(wide_tlx) > 0:
            results.append(wilcoxon_paired(
                wide_tlx["control"], wide_tlx["experimental"],
                "nasa_tlx_composite"))

    return results


# ============================================================
# SPEARMAN: latency vs recovery
# ============================================================
def latency_vs_recovery_correlation(summary_df):
    """Spearman rho for cobot latency vs PSS recovery time."""
    exp = summary_df[summary_df["condition"] == "experimental"].copy()
    exp = exp.dropna(subset=["mean_latency_s", "mean_recovery_s"])
    if len(exp) < 3:
        return None

    rho, p_val = stats.spearmanr(exp["mean_latency_s"],
                                 exp["mean_recovery_s"])
    return {
        "n": len(exp),
        "spearman_rho": float(rho),
        "p_value": float(p_val),
        "significant": p_val < ALPHA,
    }


# ============================================================
# PSS VALIDATION
# ============================================================
def pss_vs_rula_validation(pss_scores, rula_scores, out_path):
    pss_scores = np.asarray(pss_scores, dtype=float)
    rula_scores = np.asarray(rula_scores, dtype=float)

    # RULA grand score is 1-7. Normalize to [0, 1] for comparison with PSS.
    rula_norm = (rula_scores - 1) / 6.0

    pearson_r, pearson_p = stats.pearsonr(pss_scores, rula_norm)
    spearman_rho, spearman_p = stats.spearmanr(pss_scores, rula_norm)
    rmse = float(np.sqrt(np.mean((pss_scores - rula_norm) ** 2)))

    # Bland-Altman
    mean_vals = (pss_scores + rula_norm) / 2.0
    diff_vals = pss_scores - rula_norm
    bias = float(np.mean(diff_vals))
    sd = float(np.std(diff_vals, ddof=1))
    upper_loa = bias + 1.96 * sd
    lower_loa = bias - 1.96 * sd

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mean_vals, diff_vals, alpha=0.7)
    ax.axhline(bias, color="red", linestyle="--",
               label=f"Bias = {bias:.3f}")
    ax.axhline(upper_loa, color="gray", linestyle=":",
               label=f"+1.96 SD = {upper_loa:.3f}")
    ax.axhline(lower_loa, color="gray", linestyle=":",
               label=f"-1.96 SD = {lower_loa:.3f}")
    ax.set_xlabel("Mean of PSS and normalized RULA")
    ax.set_ylabel("PSS - normalized RULA")
    ax.set_title("Bland-Altman: PSS vs Expert RULA")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "rmse": rmse,
        "bland_altman_bias": bias,
        "bland_altman_upper_loa": float(upper_loa),
        "bland_altman_lower_loa": float(lower_loa),
    }


# ============================================================
# PSS TIMELINE PLOTS (per participant)
# ============================================================
def plot_pss_timelines(log_dir=LOG_DIR, out_path=None):
    out_path = out_path or os.path.join(RESULTS_DIR, "pss_timelines.png")
    files = sorted(glob.glob(os.path.join(log_dir, "*_frames.csv")))
    if not files:
        return

    # Group by participant
    by_participant = {}
    for f in files:
        base = os.path.basename(f).replace("_frames.csv", "")
        parts = base.split("_")
        pid = parts[0]
        cond = parts[1]
        by_participant.setdefault(pid, {})[cond] = f

    n = len(by_participant)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (pid, files_for_pid) in zip(axes, by_participant.items()):
        for cond, color in [("control", "tab:blue"),
                            ("experimental", "tab:orange")]:
            if cond not in files_for_pid:
                continue
            df = pd.read_csv(files_for_pid[cond])
            ax.plot(df["timestamp_s"] / 60.0, df["pss_smooth"],
                    label=cond, color=color, alpha=0.8)
        ax.axhline(0.4, color="red", linestyle=":", linewidth=1)
        ax.set_ylabel(f"{pid}\nPSS")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(0, 1)
    axes[-1].set_xlabel("Time (minutes)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"PSS timeline plot: {out_path}")


# ============================================================
# REPORT WRITER
# ============================================================
def write_report(primary, latency_corr, validation, out_path):
    with open(out_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("EMPATHETIC CONSERVATOR - STATISTICAL ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"alpha (uncorrected):        {ALPHA}\n")
        f.write(f"alpha (Bonferroni, k=3):    {ALPHA_CORRECTED:.4f}\n\n")

        f.write("PRIMARY OUTCOMES (Wilcoxon signed-rank)\n")
        f.write("-" * 60 + "\n")
        for r in primary:
            if "error" in r:
                f.write(f"\n{r['outcome']}: ERROR - {r['error']}\n")
                continue
            f.write(f"\nOutcome: {r['outcome']}\n")
            f.write(f"  N pairs:                       {r['n_pairs']}\n")
            f.write(f"  Median (control):              "
                    f"{r['median_control']:.4f}\n")
            f.write(f"  Median (experimental):         "
                    f"{r['median_experimental']:.4f}\n")
            f.write(f"  Wilcoxon W:                    {r['wilcoxon_W']:.3f}\n")
            f.write(f"  p (uncorrected):               "
                    f"{r['p_uncorrected']:.4f}\n")
            f.write(f"  Significant after Bonferroni:  "
                    f"{r['p_significant_corrected']}\n")
            f.write(f"  Effect size r:                 "
                    f"{r['effect_size_r']:.3f}\n")
            f.write(f"  Shapiro-Wilk p (on diffs):     "
                    f"{r['shapiro_p_on_diffs']:.4f}\n")

        f.write("\n\nSPEARMAN: latency vs PSS recovery time\n")
        f.write("-" * 60 + "\n")
        if latency_corr:
            f.write(f"  N:               {latency_corr['n']}\n")
            f.write(f"  Spearman rho:    {latency_corr['spearman_rho']:.3f}\n")
            f.write(f"  p:               {latency_corr['p_value']:.4f}\n")
            f.write(f"  Significant:     {latency_corr['significant']}\n")
        else:
            f.write("  Insufficient data (need >= 3 experimental sessions "
                    "with interventions).\n")

        if validation:
            f.write("\n\nPSS VALIDATION (vs expert RULA on 20 photographs)\n")
            f.write("-" * 60 + "\n")
            for k, v in validation.items():
                f.write(f"  {k}: {v:.4f}\n")

    print(f"Report: {out_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading session data...")
    summary = per_participant_summary()
    if len(summary) == 0:
        print(f"No session data found in {LOG_DIR}/")
        return
    summary.to_csv(os.path.join(RESULTS_DIR, "summary_stats.csv"),
                   index=False)
    print(f"  Loaded {len(summary)} session-level rows")

    print("\nRunning primary Wilcoxon analyses...")
    tlx_path = "data/nasa_tlx.csv"
    tlx_df = pd.read_csv(tlx_path) if os.path.exists(tlx_path) else None
    primary = run_primary_analysis(summary, tlx_df)

    print("Computing latency-recovery correlation...")
    latency_corr = latency_vs_recovery_correlation(summary)

    validation = None
    val_path = "data/pss_validation.csv"
    if os.path.exists(val_path):
        print("Running PSS validation against expert RULA...")
        val_df = pd.read_csv(val_path)
        validation = pss_vs_rula_validation(
            val_df["pss_score"], val_df["rula_score"],
            os.path.join(RESULTS_DIR, "bland_altman.png")
        )

    print("Plotting PSS timelines...")
    plot_pss_timelines()

    write_report(primary, latency_corr, validation,
                 os.path.join(RESULTS_DIR, "test_results.txt"))


if __name__ == "__main__":
    main()
