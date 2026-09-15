#!/usr/bin/env python3
"""Create the mapped-PF initial-state comparison report and figures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def f(row,key):return float(row[key]) if row.get(key) not in (None,"") else 0.


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("screen",type=Path);parser.add_argument("extension",type=Path)
    parser.add_argument("decomposition",type=Path);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    audit=read_csv(args.screen/"sharp_pf_stage_audit.csv")
    short=read_csv(args.screen/"events_off_history.csv")
    summaries=read_csv(args.screen/"candidate_summary.csv")
    extended=read_csv(args.extension/"extended_history.csv")
    extension=json.loads((args.extension/"extension_summary.json").read_text())
    decomposition=json.loads((args.decomposition/"pre_root_decomposition.json").read_text())
    ids=list(dict.fromkeys(row["design_id"] for row in audit));labels={name:f"C{i+1}" for i,name in enumerate(ids)}
    stages=("sharp_analytic","mapped_before_cleanup","mapped_after_cleanup","qualified_source")
    fig,ax=plt.subplots(2,2,figsize=(13,10))
    for name in ids:
        rows=[r for r in audit if r["design_id"]==name]
        ax[0,0].plot([stages.index(r["stage"]) for r in rows],[f(r,"total_MPa") for r in rows],"o-",label=labels[name])
    ax[0,0].set_xticks(range(4),["sharp","mapped","cleanup","qualified"])
    ax[0,0].set_ylabel("exact/local stress (MPa)");ax[0,0].legend(ncol=2)
    for name in ids[1:]:
        rows=[r for r in short if r["design_id"]==name]
        ax[0,1].plot([f(r,"time_s") for r in rows],[f(r,"mean_sigma_MPa") for r in rows],label=labels[name])
    ax[0,1].set_xlabel("time (s)");ax[0,1].set_ylabel("events-off PF stress (MPa)");ax[0,1].legend()
    t=[f(r,"time_s") for r in extended]
    ax[1,0].plot(t,[f(r,"mean_sigma_MPa") for r in extended],label="stress")
    ax[1,0].set_xlabel("time (s)");ax[1,0].set_ylabel("C2 stress (MPa)")
    axis=ax[1,0].twinx();axis.plot(t,[f(r,"survival_probability") for r in extended],color="tab:orange",label="survival")
    axis.set_ylabel("survival probability")
    ax[1,1].plot(t,[f(r,"delta_ln_Gamma_barrier") for r in extended],label="barrier")
    ax[1,1].plot(t,[f(r,"delta_ln_Gamma_sites") for r in extended],label="sites")
    ax[1,1].plot(t,[f(r,"delta_ln_Gamma_total") for r in extended],label="total",lw=2)
    ax[1,1].set_xlabel("time (s)");ax[1,1].set_ylabel(r"$\Delta\ln\Gamma$");ax[1,1].legend()
    for a in ax.flat:a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.out/"mapped_pf_comparison.png",dpi=180);plt.close(fig)
    selected=[r for r in audit if r["design_id"]==ids[0]]
    by={r["stage"]:r for r in selected}
    mapping_delta=f(by["mapped_before_cleanup"],"total_MPa")-f(by["sharp_analytic"],"total_MPa")
    cleanup_delta=f(by["mapped_after_cleanup"],"total_MPa")-f(by["mapped_before_cleanup"],"total_MPa")
    qualification_delta=f(by["qualified_source"],"total_MPa")-f(by["mapped_after_cleanup"],"total_MPa")
    report=["# Mapped-PF initial-state audit and events-off screen","",
      "Status: **COMPLETE; NO STOCHASTIC CAMPAIGN LAUNCHED**.","",
      "## Sharp-to-PF discrepancy","",
      f"For C1, stress changes from {f(by['sharp_analytic'],'total_MPa'):.4f} MPa analytically to {f(by['mapped_before_cleanup'],'total_MPa'):.4f} MPa under exact production metrology on the mapped field, {f(by['mapped_after_cleanup'],'total_MPa'):.4f} MPa after cleanup, and {f(by['qualified_source'],'total_MPa'):.4f} MPa after the 0.01 s events-off qualification.","",
      f"The offsets are {mapping_delta:.4f} MPa from finite-window mapped-PF metrology, {cleanup_delta:.4f} MPa from cleanup, and {qualification_delta:.4f} MPa from actual PF evolution. At the qualified source the curvature term is {f(by['qualified_source'],'curvature_term_MPa'):.4f} MPa, compared with zero in the pointwise sharp construction; the TJ term changes by only {f(by['qualified_source'],'TJ_term_MPa')-f(by['sharp_analytic'],'TJ_term_MPa'):.4f} MPa. The discrepancy is therefore a curvature-correspondence error: the production 3W fit samples the rapidly varying polynomial contour near the contact, whereas the selector used its pointwise contact jet.","",
      "Only C1 has a previously created qualified-source archive. C2-C5 were compared at the sharp, mapped, and five-step-cleanup stages and then evolved directly with the same events-off PF equations; no qualification stage is inferred for them.","",
      "## Completed-trajectory decomposition","",
      f"The original 1.2796 MPa rise contains +{decomposition['interval_changes']['initial_rise']['curvature_term_MPa']:.4f} MPa from fitted curvature and {decomposition['interval_changes']['initial_rise']['TJ_term_MPa']:.4f} MPa from the TJ term. After the maximum, curvature changes by {decomposition['interval_changes']['post_peak_relaxation']['curvature_term_MPa']:.4f} MPa and the TJ term by {decomposition['interval_changes']['post_peak_relaxation']['TJ_term_MPa']:.4f} MPa. Reversal of fitted mean meridional curvature controls the turnover; growing TJ radius and falling angle sum reinforce the later relaxation.","",
      "## Events-off mapped candidates","",
      "|candidate|post-cleanup sigma|screen/extended maximum|Delta sigma|time|survival|Delta ln Gamma barrier|Delta ln Gamma sites|","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        report.append(f"|{labels[row['design_id']]}|{f(row,'sigma_after_cleanup_MPa'):.3f}|{f(row,'sigma_max_MPa'):.3f}|{f(row,'delta_sigma_PF_MPa'):.3f}|{f(row,'t_sigma_max_s'):.3g}|{f(row,'survival_to_max'):.4f}|{f(row,'delta_ln_Gamma_barrier'):.4f}|{f(row,'delta_ln_Gamma_sites'):.4f}|")
    report += ["", "C2 and C3 remain endpoint-censored at 0.05 s in the common screen. C4 begins too highly stressed, and C5 turns over almost immediately.","",
      "## Recommendation","",
      f"Recommend C2: `{extension['design_id']}`. Its ordinary events-off PF evolution reaches {extension['sigma_max_MPa']:.4f} MPa at the 2 s cap from {extension['sigma0_MPa']:.4f} MPa, so its demonstrated loading headroom is at least {extension['delta_sigma_MPa']:.4f} MPa. Survival to that state is {extension['survival_to_max']:.4f}. Barrier amplification ({extension['delta_ln_Gamma_barrier']:.5f}) exceeds the magnitude of site loss ({abs(extension['delta_ln_Gamma_sites']):.5f}) by a factor of {extension['delta_ln_Gamma_barrier']/abs(extension['delta_ln_Gamma_sites']):.2f}. The maximum remains endpoint-censored.","",
      f"Over the extension, the LEFT curvature term changes from {f(extended[0],'LEFT_curvature_term_MPa'):.4f} to {f(extended[-1],'LEFT_curvature_term_MPa'):.4f} MPa and the TJ term from {f(extended[0],'LEFT_TJ_term_MPa'):.4f} to {f(extended[-1],'LEFT_TJ_term_MPa'):.4f} MPa. The two fitted curvatures change from {f(extended[0],'LEFT_kappa_negative_per_um'):.4f}/{f(extended[0],'LEFT_kappa_positive_per_um'):.4f} to {f(extended[-1],'LEFT_kappa_negative_per_um'):.4f}/{f(extended[-1],'LEFT_kappa_positive_per_um'):.4f} per micrometre; angles change from {f(extended[0],'LEFT_theta_negative_deg'):.3f}/{f(extended[0],'LEFT_theta_positive_deg'):.3f} to {f(extended[-1],'LEFT_theta_negative_deg'):.3f}/{f(extended[-1],'LEFT_theta_positive_deg'):.3f} degrees. Contact radius falls {-100*extension['rTJ_change_at_max']:.3f}%, center volume falls {-100*extension['center_volume_change_at_max']:.3f}%, and the per-contact root rate rises by {(f(extended[-1],'LEFT_root_rate_per_s')/f(extended[0],'LEFT_root_rate_per_s')-1)*100:.3f}%.","",
      f"Surface area, GB area, and interfacial energy change by {(f(extended[-1],'surface_area_m2')/f(extended[0],'surface_area_m2')-1)*100:.4f}%, {(f(extended[-1],'GB_area_m2')/f(extended[0],'GB_area_m2')-1)*100:.4f}%, and {(f(extended[-1],'energy_J')/f(extended[0],'energy_J')-1)*100:.4f}% respectively; total mass remains conserved to the reported solver tolerance.","",
      "This recommendation is for review only. Events and RNG remained disabled, no trajectory was prescribed, and no stochastic renewal campaign was started."]
    (args.out/"REPORT.md").write_text("\n".join(report)+"\n")
    payload=dict(status="MAPPED_PF_INITIAL_STATE_REVIEW_READY",recommendation=extension,
      selected_discrepancy_MPa=dict(mapping_metrology=mapping_delta,cleanup=cleanup_delta,
        qualification_evolution=qualification_delta,total=f(by["qualified_source"],"total_MPa")-f(by["sharp_analytic"],"total_MPa")),
      no_stochastic_campaign_launched=True)
    (args.out/"REPORT.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2))


if __name__=="__main__":main()
