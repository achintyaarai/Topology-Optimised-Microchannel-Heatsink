"""
run_sweep.py — Parameter sweep over ε-constraint parameters
=============================================================
Runs main.py with different combinations of volfrac, eps_flow,
eps_tv, and w_blob to map out the Pareto front.
"""

import itertools
import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt


def run_sweep():
    # Define Parameter Space
    volfracs = [0.35, 0.45, 0.55]
    eps_flows = [1.0, 1.5, 2.0]
    eps_tvs = [1.5, 2.0, 3.0]
    w_blobs = [0.1]  # Fixed blob weight for sweep

    # Generate all combinations
    combinations = list(itertools.product(volfracs, eps_flows, eps_tvs, w_blobs))

    print(f"Total combinations scheduled: {len(combinations)}")

    for idx, (v, ef, et, wb) in enumerate(combinations):
        print(f"\n{'=' * 60}")
        print(f"--- Run {idx + 1}/{len(combinations)}: "
              f"volfrac={v}, eps_flow={ef}, eps_tv={et}, w_blob={wb} ---")
        print(f"{'=' * 60}")
        try:
            subprocess.run([
                "python", "main.py",
                "--nelx", "80",
                "--nely", "40",
                "--volfrac", str(v),
                "--eps_flow", str(ef),
                "--eps_tv", str(et),
                "--w_blob", str(wb),
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Run failed for v={v}, ef={ef}, et={et}: {e}")

    print("\n--- Parameter Sweep Completed. Generating Dashboard... ---")
    generate_plots()


def generate_plots():
    if not os.path.isfile("sweep_results.csv"):
        print("sweep_results.csv not found. Did any runs complete successfully?")
        return

    df = pd.read_csv("sweep_results.csv")

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # 1. volfrac vs Pressure Drop
    sc0 = axs[0, 0].scatter(df['volfrac'], df['pressure_drop'],
                             c=df['eps_flow'], cmap='viridis', s=80, edgecolors='k')
    axs[0, 0].set(title='Volume Fraction vs Pressure Drop',
                  xlabel='Volume Fraction', ylabel='ΔP [Pa]')
    fig.colorbar(sc0, ax=axs[0, 0], label='ε_flow')

    # 2. eps_tv vs Peak Temperature
    sc1 = axs[0, 1].scatter(df['eps_tv'], df['peak_temp'],
                             c=df['volfrac'], cmap='coolwarm', s=80, edgecolors='k')
    axs[0, 1].set(title='ε_TV vs Peak Temperature',
                  xlabel='ε_TV', ylabel='Peak T [K]')
    fig.colorbar(sc1, ax=axs[0, 1], label='volfrac')

    # 3. Pareto Front: Jf vs JQ
    sc2 = axs[1, 0].scatter(df['Jf'], df['JQ'],
                             c=df['volfrac'], cmap='plasma', s=80, edgecolors='k')
    axs[1, 0].set(title='Pareto Front: Flow vs Thermal',
                  xlabel='Fluid Dissipation Jf', ylabel='Thermal JQ')
    fig.colorbar(sc2, ax=axs[1, 0], label='volfrac')

    # 4. JTV vs Stagnation Ratio
    sc3 = axs[1, 1].scatter(df['JTV'], df['stagnation_ratio'],
                             c=df['eps_flow'], cmap='viridis', s=80, edgecolors='k')
    axs[1, 1].set(title='Temperature Variance vs Stagnation Ratio',
                  xlabel='JTV', ylabel='Stag. ratio')
    fig.colorbar(sc3, ax=axs[1, 1], label='ε_flow')

    plt.tight_layout()
    plt.savefig("parameter_sweep_metrics.png", dpi=300, bbox_inches='tight')
    print("Saved parameter_sweep_metrics.png")


if __name__ == "__main__":
    run_sweep()