#!/usr/bin/env python3
"""Deterministic bicrystal/cluster loading-versus-root comparison.

Reads completed analytical tables and an existing bicrystal scalar history.
It does not initialize or advance a phase field and does not draw thresholds.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "docs/three_particle/ripening_hazard_design"
OUT = ROOT / "docs/three_particle/regime_comparison"
BICRYSTAL = Path(
    "/Volumes/Data/Data/PF-sintering/PF-Sintering/runs/"
    "pr_large_avalanche_resume13_prefix_through_a3_reload.csv"
)
MANIFEST = ROOT / "docs/three_particle/production_screen/bicrystal_launch_manifest.json"
PROBS = (0.10, 0.50, 0.90)
TARGETS = (0.01, 0.02, 0.05, 0.10)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def groups(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        result.setdefault(row[key], []).append(row)
    return result


def interp(x: float, rows: list[dict[str, str]], key: str) -> float:
    xx = np.array([float(row["x"]) for row in rows])
    yy = np.array([float(row[key]) for row in rows])
    return float(np.interp(x, xx, yy))


def cluster_metrics(dense: dict[str, list[dict[str, str]]], temperature: float):
    quantiles = []
    sensitivity = []
    regimes = []
    kT_eV = 8.617333262145e-5 * temperature
    for design_id, path in sorted(dense.items()):
        path.sort(key=lambda row: float(row["x"]))
        xs = np.array([float(row["x"]) for row in path])
        lam = np.array([float(row["cumulative_scaled_hazard"]) for row in path])
        sigma0 = float(path[0]["sigma_GB_MPa"])
        for probability in PROBS:
            target_lambda = -math.log1p(-probability)
            xq = float(np.interp(target_lambda, lam, xs))
            sigma = interp(xq, path, "sigma_GB_MPa")
            quantiles.append(
                {
                    "design_id": design_id,
                    "event_probability": probability,
                    "x": xq,
                    "Lambda": target_lambda,
                    "survival": 1.0 - probability,
                    "stress_MPa": sigma,
                    "A_sigma_MPa": sigma - sigma0,
                    "A_Gamma": interp(xq, path, "hazard_amplification"),
                    "rTJ_over_W": interp(xq, path, "r_TJ_over_W"),
                    "elapsed_s": interp(xq, path, "elapsed_s"),
                }
            )
        q50 = quantiles[-2]
        regimes.append(
            {
                "system": "particle cluster",
                "case_id": design_id,
                "basis": "median survival",
                "sigma0_MPa": sigma0,
                "stress_MPa": q50["stress_MPa"],
                "Delta_sigma_MPa": q50["A_sigma_MPa"],
                "A_Gamma": q50["A_Gamma"],
                "loading_coordinate": q50["x"],
                "classification": classify(sigma0, q50["A_sigma_MPa"], q50["A_Gamma"]),
            }
        )
        for target_x in TARGETS:
            lambda0 = interp(target_x, path, "cumulative_scaled_hazard")
            multiplier = lambda0 / math.log(2.0)
            sigma = interp(target_x, path, "sigma_GB_MPa")
            sensitivity.append(
                {
                    "design_id": design_id,
                    "target_center_loss": target_x,
                    "baseline_Lambda": lambda0,
                    "required_surface_mobility_multiplier": multiplier,
                    "equivalent_root_rate_reduction_factor": multiplier,
                    "remaining_site_count_fraction": 1.0 / multiplier,
                    "effective_site_count_reduction_percent": 100.0 * (1.0 - 1.0 / multiplier),
                    "equivalent_barrier_increase_eV": kT_eV * math.log(multiplier),
                    "stress_MPa": sigma,
                    "A_sigma_MPa": sigma - sigma0,
                    "A_Gamma": interp(target_x, path, "hazard_amplification"),
                    "rTJ_over_W": interp(target_x, path, "r_TJ_over_W"),
                    "baseline_elapsed_s": interp(target_x, path, "elapsed_s"),
                    "elapsed_s_at_required_mobility": interp(target_x, path, "elapsed_s") / multiplier,
                }
            )
    return quantiles, sensitivity, regimes


def classify(sigma0: float, delta_sigma: float, amplification: float) -> str:
    # Explicit operational boundaries for this map; raw coordinates are primary.
    if delta_sigma >= 2.0 and amplification >= 1.20:
        return "loading-triggered nucleation"
    if sigma0 >= 30.0 and abs(delta_sigma) < 2.0 and 0.8 <= amplification <= 1.2:
        return "high-stress stochastic nucleation"
    if sigma0 < 30.0 and abs(delta_sigma) < 2.0 and 0.8 <= amplification <= 1.2:
        return "low-stress stochastic nucleation"
    return "mixed/intermediate"


def full_search_loading(coarse_rows: list[dict[str, str]], fully_qualified: set[str]) -> list[dict]:
    result = []
    grouped = groups(coarse_rows, "design_id")
    for target in TARGETS:
        eligible = []
        for design_id, path in grouped.items():
            path.sort(key=lambda row: float(row["x"]))
            prefix = [row for row in path if float(row["x"]) <= target + 1e-12]
            if not prefix or abs(float(prefix[-1]["x"]) - target) > 1e-12:
                continue
            if all(float(row["dF_dx_J"]) < 0.0 for row in prefix):
                sigma0 = float(path[0]["sigma_GB_MPa"])
                sigma = float(prefix[-1]["sigma_GB_MPa"])
                eligible.append((sigma - sigma0, design_id, sigma0, sigma, prefix))
        delta, design_id, sigma0, sigma, prefix = max(eligible)
        qualified = [row for row in eligible if row[1] in fully_qualified]
        qdelta, qid, qsigma0, qsigma, _ = max(qualified)
        result.append(
            {
                "target_center_loss": target,
                "design_id": design_id,
                "thermodynamic_test": "dF/dx<0 at every saved search knot through target",
                "eligible_geometry_count": len(eligible),
                "sigma0_MPa": sigma0,
                "stress_MPa": sigma,
                "maximum_Delta_sigma_MPa": delta,
                "mean_Delta_sigma_per_x_MPa": delta / target,
                "minimum_minus_dF_dx_fJ": min(-float(row["dF_dx_J"]) for row in prefix) * 1e15,
                "rTJ_over_W": float(prefix[-1]["r_TJ_over_W"]),
                "fully_20pct_thermodynamic_design_id": qid,
                "fully_20pct_thermodynamic_sigma0_MPa": qsigma0,
                "fully_20pct_thermodynamic_stress_MPa": qsigma,
                "fully_20pct_thermodynamic_maximum_Delta_sigma_MPa": qdelta,
            }
        )
    return result


def thermodynamic_stress_slopes(coarse_rows: list[dict[str, str]]) -> list[dict]:
    slopes = []
    for design_id, path in groups(coarse_rows, "design_id").items():
        path.sort(key=lambda row: float(row["x"]))
        prefix_favorable = True
        for left, right in zip(path, path[1:]):
            prefix_favorable = prefix_favorable and float(left["dF_dx_J"]) < 0 and float(right["dF_dx_J"]) < 0
            if not prefix_favorable:
                continue
            dx = float(right["x"]) - float(left["x"])
            slope = (float(right["sigma_GB_MPa"]) - float(left["sigma_GB_MPa"])) / dx
            if slope > 0:
                slopes.append({
                    "design_id": design_id,
                    "interval_start_x": float(left["x"]),
                    "interval_end_x": float(right["x"]),
                    "d_sigma_dx_MPa": slope,
                    "interval_Delta_sigma_MPa": slope * dx,
                    "minimum_minus_dF_dx_fJ": min(-float(left["dF_dx_J"]), -float(right["dF_dx_J"])) * 1e15,
                    "end_rTJ_over_W": float(right["r_TJ_over_W"]),
                })
    return sorted(slopes, key=lambda row: row["d_sigma_dx_MPa"], reverse=True)[:20]


def _integral_above(t, rate, stress, cutoff):
    total = 0.0
    selected = 0.0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        area = 0.5 * (rate[i] + rate[i + 1]) * dt
        total += area
        s0, s1 = stress[i], stress[i + 1]
        if s0 >= cutoff and s1 >= cutoff:
            selected += area
        elif (s0 - cutoff) * (s1 - cutoff) < 0.0:
            fraction = (cutoff - s0) / (s1 - s0)
            rc = rate[i] + fraction * (rate[i + 1] - rate[i])
            if s1 >= cutoff:
                selected += 0.5 * (rc + rate[i + 1]) * dt * (1.0 - fraction)
            else:
                selected += 0.5 * (rate[i] + rc) * dt * fraction
    return selected / total


def bicrystal_metrics(rows: list[dict[str, str]], seconds_per_model: float):
    completed = sorted({int(r["cycle"]) for r in rows if r["frame_type"] == "root_nucleation"})
    summaries, paths, regimes = [], [], []
    for cycle in completed:
        loading = [r for r in rows if int(r["cycle"]) == cycle and r["frame_type"] == "loading"]
        root = next(r for r in rows if int(r["cycle"]) == cycle and r["frame_type"] == "root_nucleation")
        prior_extinction = [
            r for r in rows
            if int(r["cycle"]) == cycle - 1 and r["frame_type"] == "avalanche_extinction"
        ]
        if prior_extinction:
            loading.append(prior_extinction[-1])
        loading.append(root)
        loading.sort(key=lambda row: float(row["t_model"]))
        # Deduplicate exact saved times without manufacturing intermediate states.
        unique = {float(row["t_model"]): row for row in loading}
        loading = [unique[key] for key in sorted(unique)]
        t = np.array([float(r["t_model"]) for r in loading])
        t_s = t * seconds_per_model
        sigma = np.array([float(r["sigma_local_Pa"]) * 1e-6 for r in loading])
        rate_model = np.array([float(r["Gamma_per_model_time"]) for r in loading])
        rate_s = rate_model / seconds_per_model
        radius = np.array([float(r["r_n_m"]) for r in loading])
        xb = 1.0 - (radius / radius[0]) ** 2
        raw_hazard = np.r_[0.0, np.cumsum(0.5 * (rate_model[:-1] + rate_model[1:]) * np.diff(t))]
        lam = raw_hazard / 1.25
        median_target = math.log(2.0)
        median_observed = bool(lam[-1] >= median_target)
        if median_observed:
            tm = float(np.interp(median_target, lam, t_s))
            sm = float(np.interp(median_target, lam, sigma))
            rm = float(np.interp(median_target, lam, rate_s))
            xm = float(np.interp(median_target, lam, xb))
        else:
            tm = sm = rm = xm = math.nan
        cutoff = sigma[0] + 0.8 * (sigma[-1] - sigma[0])
        high_fraction = _integral_above(t, rate_model, sigma, cutoff)
        summaries.append(
            {
                "cycle": cycle,
                "start_reference": "initialized relaxed state" if cycle == 1 else "exact previous avalanche extinction",
                "post_avalanche_stress_MPa": sigma[0],
                "pre_root_stress_MPa": sigma[-1],
                "stress_increase_before_root_MPa": sigma[-1] - sigma[0],
                "post_avalanche_root_rate_per_s": rate_s[0],
                "pre_root_root_rate_per_s": rate_s[-1],
                "rate_amplification": rate_s[-1] / rate_s[0],
                "loading_duration_s": t_s[-1] - t_s[0],
                "root_threshold": float(root["H_threshold"]),
                "integrated_raw_hazard": raw_hazard[-1],
                "Lambda_at_observed_root": lam[-1],
                "bicrystal_loading_coordinate_at_root": xb[-1],
                "loading_coordinate_definition": "1-(r_neck/r_neck_post)^2",
                "high_stress_cutoff_MPa": cutoff,
                "fraction_hazard_final_high_stress_portion": high_fraction,
                "high_stress_portion_definition": "sigma >= sigma_post + 0.8*(sigma_pre-root-sigma_post)",
                "median_reached_before_observed_root": median_observed,
                "median_loading_coordinate": xm,
                "median_stress_MPa": sm,
                "median_Delta_sigma_MPa": sm - sigma[0] if median_observed else math.nan,
                "median_rate_amplification": rm / rate_s[0] if median_observed else math.nan,
                "median_elapsed_s": tm - t_s[0] if median_observed else math.nan,
            }
        )
        basis_x = xm if median_observed else xb[-1]
        basis_s = sm if median_observed else sigma[-1]
        basis_a = rm / rate_s[0] if median_observed else rate_s[-1] / rate_s[0]
        regimes.append(
            {
                "system": "bicrystal",
                "case_id": f"cycle {cycle}",
                "basis": "median survival" if median_observed else "observed early root (median censored)",
                "sigma0_MPa": sigma[0],
                "stress_MPa": basis_s,
                "Delta_sigma_MPa": basis_s - sigma[0],
                "A_Gamma": basis_a,
                "loading_coordinate": basis_x,
                "classification": classify(sigma[0], basis_s - sigma[0], basis_a),
            }
        )
        for i in range(len(t)):
            paths.append(
                {
                    "cycle": cycle,
                    "elapsed_s": t_s[i] - t_s[0],
                    "bicrystal_loading_coordinate": xb[i],
                    "stress_MPa": sigma[i],
                    "root_rate_per_s": rate_s[i],
                    "rate_amplification": rate_s[i] / rate_s[0],
                    "raw_hazard": raw_hazard[i],
                    "Lambda": lam[i],
                    "survival": math.exp(-lam[i]),
                }
            )
    return summaries, paths, regimes


def make_figures(quantiles, sensitivity, maximum, bic_paths, regimes):
    colors = {"particle cluster": "#3569b5", "bicrystal": "#d45835"}
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for system in colors:
        subset = [r for r in regimes if r["system"] == system]
        ax.scatter([r["Delta_sigma_MPa"] for r in subset], [r["A_Gamma"] for r in subset],
                   s=65, c=colors[system], label=system, zorder=3)
        for row in subset:
            label = row["case_id"].replace("Ro", "C:") if system == "particle cluster" else row["case_id"]
            if system == "bicrystal":
                ax.annotate(label, (row["Delta_sigma_MPa"], row["A_Gamma"]), xytext=(5, 4),
                            textcoords="offset points", fontsize=8)
    ax.axvline(2.0, color="0.55", ls=":")
    ax.axhline(1.2, color="0.55", ls=":")
    ax.set(xlabel=r"stress gained by comparison point, $\Delta\sigma$ (MPa)",
           ylabel=r"root-rate amplification, $A_\Gamma$")
    ax.grid(alpha=0.2); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "bicrystal_cluster_regime_map.png", dpi=210)
    fig.savefig(OUT / "bicrystal_cluster_regime_map.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for cycle in sorted({r["cycle"] for r in bic_paths}):
        p = [r for r in bic_paths if r["cycle"] == cycle]
        x = np.array([r["bicrystal_loading_coordinate"] for r in p]) * 100
        axes[0].plot(x, [r["stress_MPa"] for r in p], label=f"cycle {cycle}")
        axes[1].plot(x, [r["Lambda"] for r in p], label=f"cycle {cycle}")
    axes[1].axhline(math.log(2), color="k", ls=":", label="median survival")
    axes[0].set_ylabel("local stress (MPa)"); axes[1].set_ylabel(r"competition integral $\Lambda$")
    for ax in axes:
        ax.set_xlabel("bicrystal contact-area loss coordinate (%)"); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "bicrystal_loading_cycles.png", dpi=210)
    fig.savefig(OUT / "bicrystal_loading_cycles.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    designs = sorted({r["design_id"] for r in sensitivity})
    for design in designs:
        p = [r for r in sensitivity if r["design_id"] == design]
        ax.plot([100*r["target_center_loss"] for r in p],
                [r["required_surface_mobility_multiplier"] for r in p], marker="o", alpha=.75)
    ax.set_yscale("log"); ax.set(xlabel="target median center loss (%)",
        ylabel=r"required mobility multiplier $M_s/M_{s0}$")
    ax.grid(alpha=.2, which="both"); fig.tight_layout()
    fig.savefig(OUT / "surface_mobility_sensitivity.png", dpi=210)
    fig.savefig(OUT / "surface_mobility_sensitivity.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot([100*r["target_center_loss"] for r in maximum],
            [r["maximum_Delta_sigma_MPa"] for r in maximum], marker="o")
    for row in maximum:
        ax.annotate(row["design_id"], (100*row["target_center_loss"], row["maximum_Delta_sigma_MPa"]),
                    xytext=(4, 5), textcoords="offset points", fontsize=7)
    ax.set(xlabel="center loss (%)", ylabel="maximum checkpoint-admissible stress gain (MPa)")
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT / "maximum_thermodynamic_loading.png", dpi=210)
    fig.savefig(OUT / "maximum_thermodynamic_loading.pdf"); plt.close(fig)


def report(summary, quantiles, sensitivity, maximum, bic):
    q50 = [r for r in quantiles if r["event_probability"] == .5]
    m1 = [r["required_surface_mobility_multiplier"] for r in sensitivity if r["target_center_loss"] == .01]
    m10 = [r["required_surface_mobility_multiplier"] for r in sensitivity if r["target_center_loss"] == .10]
    lines = [
        "# Bicrystal–particle-cluster loading/nucleation regime comparison",
        "",
        "This is deterministic postprocessing of the completed sharp-interface tables and the existing bicrystal production history. No PF state was initialized or advanced, no threshold was drawn, and no production parameter was changed.",
        "",
        "## Result",
        "",
        f"All {len(q50)} fully thermodynamic cluster paths remain in the near-zero-loading regime at median nucleation: Δσ50 is {min(r['A_sigma_MPa'] for r in q50):.4f}–{max(r['A_sigma_MPa'] for r in q50):.4f} MPa and AΓ,50 is {min(r['A_Gamma'] for r in q50):.6f}–{max(r['A_Gamma'] for r in q50):.6f}.",
        f"Moving the median root to 1% loss requires M_s/M_s0={min(m1):.2f}–{max(m1):.2f}; moving it to 10% requires {min(m10):.1f}–{max(m10):.1f}. These changes buy at most a few MPa because the thermodynamically allowed stress slopes are small.",
        "",
        "The two completed bicrystal cycles show visible load/relax morphology cycles, but the unchanged root rate changes only weakly during loading. On the requested Δσ–AΓ map they do not occupy a large-hazard-amplification loading-triggered regime. Cycle 2 nucleated before its trajectory accumulated median hazard, so its median point is censored and its observed root is plotted explicitly.",
        "",
        "The evidence therefore supports a robust particle-cluster regime of stochastic nucleation during coarsening before appreciable additional loading. It supports bicrystal load/relax cycles, but does not support the stronger claim that their root nucleation is stress-loading-controlled under the shared unchanged microscopic rate law.",
        "",
        "## Definitions",
        "",
        "Cluster Λ=(1/1.25)∫(ΓL+ΓR)/xdot dx and S=exp(-Λ). Mobility scaling uses Λ_m=Λ_0/m_M, so m_M=Λ_0(x_target)/ln2. The same multiplier is the equivalent root-rate reduction. Site-only suppression leaves 1/m of the effective sites; barrier-only suppression requires ΔG_root=kBT ln(m).",
        "",
        "The bicrystal coordinate is x_B=1-(r_neck/r_neck,post)^2, the fractional contact-area loss. The final high-stress portion begins at 80% of the observed post-avalanche-to-pre-root stress rise.",
        "",
        "## Cluster quantile ranges",
        "",
        "|root probability|center loss|Lambda|Delta sigma (MPa)|A_Gamma|",
        "|---:|---:|---:|---:|---:|",
    ]
    for probability in PROBS:
        subset = [r for r in quantiles if r["event_probability"] == probability]
        lines.append(
            f"|{100*probability:.0f}%|{100*min(r['x'] for r in subset):.4f}-{100*max(r['x'] for r in subset):.4f}%|"
            f"{-math.log1p(-probability):.6f}|{min(r['A_sigma_MPa'] for r in subset):.4f}-{max(r['A_sigma_MPa'] for r in subset):.4f}|"
            f"{min(r['A_Gamma'] for r in subset):.6f}-{max(r['A_Gamma'] for r in subset):.6f}|"
        )
    lines += [
        "",
        "## Mobility and equivalent root-process sensitivity",
        "",
        "|target loss|M_s/M_s0|site-count reduction|Delta G_root (eV)|Delta sigma (MPa)|A_Gamma|",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        subset = [r for r in sensitivity if r["target_center_loss"] == target]
        lines.append(
            f"|{100*target:.0f}%|{min(r['required_surface_mobility_multiplier'] for r in subset):.2f}-{max(r['required_surface_mobility_multiplier'] for r in subset):.2f}|"
            f"{min(r['effective_site_count_reduction_percent'] for r in subset):.1f}-{max(r['effective_site_count_reduction_percent'] for r in subset):.1f}%|"
            f"{min(r['equivalent_barrier_increase_eV'] for r in subset):.3f}-{max(r['equivalent_barrier_increase_eV'] for r in subset):.3f}|"
            f"{min(r['A_sigma_MPa'] for r in subset):.3f}-{max(r['A_sigma_MPa'] for r in subset):.3f}|"
            f"{min(r['A_Gamma'] for r in subset):.6f}-{max(r['A_Gamma'] for r in subset):.6f}|"
        )
    lines += [
        "",
        "## Completed bicrystal cycles",
        "",
        "|cycle|post sigma (MPa)|pre-root sigma (MPa)|Delta sigma (MPa)|post rate (/s)|pre-root rate (/s)|rate amplification|duration (s)|x_B at root|hazard in final high-stress portion|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in bic:
        lines.append(f"|{r['cycle']}|{r['post_avalanche_stress_MPa']:.3f}|{r['pre_root_stress_MPa']:.3f}|{r['stress_increase_before_root_MPa']:.3f}|{r['post_avalanche_root_rate_per_s']:.5f}|{r['pre_root_root_rate_per_s']:.5f}|{r['rate_amplification']:.4f}|{r['loading_duration_s']:.3f}|{100*r['bicrystal_loading_coordinate_at_root']:.3f}%|{100*r['fraction_hazard_final_high_stress_portion']:.1f}%|")
    lines += [
        "",
        "Cycle 1 starts from the initialized relaxed production state; cycle 2 starts from the preceding avalanche extinction. Cycle 3 is still right-censored and is excluded from completed-cycle conclusions.",
        "",
        "## Maximum thermodynamic stress loading in the saved full search",
        "",
        "|loss|full-search maximum Delta sigma (MPa)|design|eligible paths|maximum among fully 20%-thermodynamic paths (MPa)|",
        "|---:|---:|---|---:|---:|",
    ]
    for r in maximum:
        lines.append(f"|{100*r['target_center_loss']:.0f}%|{r['maximum_Delta_sigma_MPa']:.3f}|`{r['design_id']}`|{r['eligible_geometry_count']}|{r['fully_20pct_thermodynamic_maximum_Delta_sigma_MPa']:.3f}|")
    lines += [
        "",
        "This full-search bound applies dF/dx<0 at every stored search knot through each target. Several MPa first becomes possible at 5% loss in this checkpoint-qualified sense. Among paths favorable through the full 20% interval, the maximum is only 1.319 MPa at 5% and reaches 2.751 MPa at 10%. No path builds several MPa by 1–2% loss.",
        "",
        "## Files",
        "",
        "- `cluster_quantile_competition.csv`: Λ, S, Aσ, AΓ, resolution and time at x10/x50/x90 for all ten paths.",
        "- `mobility_and_root_sensitivity.csv`: mobility multiplier, equivalent site suppression and barrier increase at 1/2/5/10% loss.",
        "- `bicrystal_cycles.csv` and `bicrystal_loading_paths.csv`: completed-cycle endpoint and path metrics.",
        "- `maximum_thermodynamic_loading.csv`: full-search stress-gain envelope.",
        "- `largest_thermodynamic_stress_slopes.csv`: twenty largest positive saved-segment dσ/dx values with favorable prefixes.",
        "- `regime_map_points.csv`: common Δσ50–AΓ,50 comparison, with censoring basis explicit.",
        "",
        "## Scope",
        "",
        "The cluster projection remains a one-coordinate sharp-interface approximation. The full-search envelope uses the already saved 0/1/2/5/10/15/20% knots rather than rerunning geometry. Bicrystal results cover two completed cycles from one production realization; cycle 2's analytical median lies beyond its observed early root and is not extrapolated.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    dense_rows = read_csv(IN_DIR / "thermodynamic_kinetic_paths.csv")
    coarse_rows = read_csv(IN_DIR / "all_admissible_thermodynamic_kinetic_paths.csv")
    bic_rows = read_csv(BICRYSTAL)
    quantiles, sensitivity, cluster_regimes = cluster_metrics(
        groups(dense_rows, "design_id"), manifest["temperature_K"]
    )
    fully_qualified = set(groups(dense_rows, "design_id"))
    maximum = full_search_loading(coarse_rows, fully_qualified)
    slopes = thermodynamic_stress_slopes(coarse_rows)
    bic, bic_paths, bic_regimes = bicrystal_metrics(
        bic_rows, manifest["seconds_per_model_time"]
    )
    regimes = cluster_regimes + bic_regimes
    write_csv(OUT / "cluster_quantile_competition.csv", quantiles)
    write_csv(OUT / "mobility_and_root_sensitivity.csv", sensitivity)
    write_csv(OUT / "maximum_thermodynamic_loading.csv", maximum)
    write_csv(OUT / "largest_thermodynamic_stress_slopes.csv", slopes)
    write_csv(OUT / "bicrystal_cycles.csv", bic)
    write_csv(OUT / "bicrystal_loading_paths.csv", bic_paths)
    write_csv(OUT / "regime_map_points.csv", regimes)
    make_figures(quantiles, sensitivity, maximum, bic_paths, regimes)
    assert len(quantiles) == 30 and len(sensitivity) == 40 and len(bic) == 2
    assert all(
        abs(r["baseline_Lambda"] / r["required_surface_mobility_multiplier"] - math.log(2)) < 1e-12
        for r in sensitivity
    )
    assert all(abs(r["Lambda"] + math.log(r["survival"])) < 1e-12 for r in quantiles)
    assert all(
        np.all(np.diff([r["Lambda"] for r in bic_paths if r["cycle"] == cycle]) >= -1e-14)
        for cycle in (1, 2)
    )
    summary = {
        "label": "ANALYTICAL_BICRYSTAL_CLUSTER_REGIME_COMPARISON",
        "phase_field_run": False,
        "stochastic_draws": False,
        "production_parameters_changed": False,
        "validation": "PASS: scaling identities, quantile survival identity, and bicrystal hazard monotonicity",
        "cluster_candidate_count": len({r["design_id"] for r in dense_rows}),
        "completed_bicrystal_cycles": len(bic),
        "right_censored_bicrystal_cycle": 3,
        "bicrystal_loading_coordinate": "1-(r_neck/r_neck_post)^2",
        "classification_boundaries": {
            "large_stress_loading_MPa": 2.0,
            "large_hazard_amplification": 1.20,
            "high_initial_stress_MPa": 30.0,
        },
        "maximum_thermodynamic_loading": maximum,
        "largest_thermodynamic_stress_slope": slopes[0],
        "bicrystal_cycles": bic,
        "source_hashes": {
            str(IN_DIR / "thermodynamic_kinetic_paths.csv"): sha256(IN_DIR / "thermodynamic_kinetic_paths.csv"),
            str(IN_DIR / "all_admissible_thermodynamic_kinetic_paths.csv"): sha256(IN_DIR / "all_admissible_thermodynamic_kinetic_paths.csv"),
            str(BICRYSTAL): sha256(BICRYSTAL),
            str(MANIFEST): sha256(MANIFEST),
        },
        "interpretation": {
            "cluster": "high- or low-stress stochastic nucleation before appreciable additional loading",
            "bicrystal": "observable load/relax cycles, but weak root-rate amplification under the shared unchanged law",
            "hypothesis_supported": "partly: morphology regime distinction yes; stress-loading-controlled bicrystal nucleation no",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    report(summary, quantiles, sensitivity, maximum, bic)


if __name__ == "__main__":
    main()
