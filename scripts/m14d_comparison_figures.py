"""Milestone 14D Sections 9-13: three-panel comparison figure (apparent
sintering stress, contact width, near-TJ curvature, all vs. reduced time
tau) across the three conditions analyzed with scripts/m14c_stress_analysis.py:
old geometry at gamma_gb=1.4142136, new geometry at gamma_gb=1.4142136,
new geometry at gamma_gb=1.6. Run from the repository root after the
corresponding runs/m14d_stress_*.json files exist; writes
runs/m14c_figs/m14d_fig_comparison.png."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(p):
    with open(p) as fh:
        return json.load(fh)


def main():
    groups = {
        "OLD A72/480, gg=1.414": ("runs/m14d_stress_OLD_gg1414.json", ["OLD_gg1414_A", "OLD_gg1414_C"]),
        "NEW A100/320, gg=1.414": ("runs/m14d_stress_NEW_ABC.json",
                                   ["NEW_A_filling", "NEW_B_depleting", "NEW_C_depleting2"]),
        "NEW A100/320, gg=1.6": ("runs/m14d_stress_NEW_gg16.json", ["NEW_gg16_A", "NEW_gg16_C"]),
    }
    t0 = load("runs/m14d_stress_NEW_t0.json")["NEW_t0"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {"OLD A72/480, gg=1.414": "C0", "NEW A100/320, gg=1.414": "C1", "NEW A100/320, gg=1.6": "C2"}
    for label, (path, keys) in groups.items():
        d = load(path)
        tau = [d[k]["tau"] for k in keys]
        sigma = [d[k]["sigma_sint_app_curvature_form"] / 1e6 for k in keys]
        Lc = [d[k]["L_contact"] * 1e9 for k in keys]
        near = [d[k]["kasa_crosscheck_particle_windows"]["top"]["near_kappa_geom"] for k in keys]
        if "NEW" in label and "1.414" in label:
            tau = [0.0] + tau
            sigma = [t0["sigma_sint_app_curvature_form"] / 1e6] + sigma
            Lc = [t0["L_contact"] * 1e9] + Lc
            near = [t0["kasa_crosscheck_particle_windows"]["top"]["near_kappa_geom"]] + near
        axes[0].plot(tau, sigma, "o-", color=colors[label], label=label)
        axes[1].plot(tau, Lc, "o-", color=colors[label], label=label)
        axes[2].plot(tau, near, "o-", color=colors[label], label=label)

    axes[0].set_xlabel("tau (s)")
    axes[0].set_ylabel("sigma_sint_app (MPa)")
    axes[0].set_title("apparent sintering stress")
    axes[1].set_xlabel("tau (s)")
    axes[1].set_ylabel("L_contact (nm)")
    axes[1].set_title("contact width")
    axes[2].set_xlabel("tau (s)")
    axes[2].set_ylabel("near-TJ kappa (1/m)")
    axes[2].set_title("near-TJ curvature (Kasa)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Milestone 14D: old vs new geometry, gamma_gb=1.414/1.6 comparison")
    fig.tight_layout()
    fig.savefig("runs/m14c_figs/m14d_fig_comparison.png", dpi=140)
    print("wrote comparison figure")


if __name__ == "__main__":
    main()
