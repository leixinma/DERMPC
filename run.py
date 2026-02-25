#!/usr/bin/env python3
"""DER-MPC: open-loop vs closed-loop vibration suppression comparison.

Usage:
    python run.py                   # headless, saves plots to results/
    python run.py --live            # with real-time visualization
    python run.py --config my.yaml  # custom config file
"""

import argparse
import pathlib
import sys
import os

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from der_mpc.dynamics import DERBeam2D
from der_mpc.sim import Simulator
from der_mpc.mpc import MPCController
from der_mpc.viz import LiveViz, plot_summary


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(description="DER-MPC demo")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"),
                    help="Path to config YAML")
    ap.add_argument("--live", action="store_true",
                    help="Enable real-time visualization")
    ap.add_argument("--no-mpc", action="store_true",
                    help="Run open-loop only (skip MPC)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = cfg["sim"]
    dt_sim = sim_cfg["dt"]
    duration = sim_cfg["duration"]
    total_steps = int(duration / dt_sim)
    dt_mpc = cfg["mpc"]["dt"]
    mpc_ratio = max(1, int(round(dt_mpc / dt_sim)))
    viz_every = cfg["viz"].get("update_every", 50)

    # Build beam model
    print("Building 2D DER beam model ...")
    beam = DERBeam2D(cfg)
    print(f"  nodes={beam.n_nodes}, n_free={beam.n_free}, "
          f"nx={beam.nx}, nu={beam.nu}")
    print(f"  EI={beam.EI}, EA={beam.EA}, mass/node={beam.node_mass:.4f} kg")

    # Find equilibrium under gravity
    print("Finding static equilibrium ...")
    x_eq = beam.find_equilibrium()
    pos_eq = beam.get_positions(x_eq)
    print(f"  Tip equilibrium y = {pos_eq[-1, 1]:.4f} m")

    # Create simulators (open-loop and closed-loop)
    sim_ol = Simulator(beam, cfg)
    sim_ol.reset(x_eq.copy())

    sim_cl = Simulator(beam, cfg)
    sim_cl.reset(x_eq.copy())

    # Build MPC controller
    mpc = None
    if not args.no_mpc:
        print("Building acados NMPC solver ...")
        mpc = MPCController(beam, cfg,
                            build_dir=str(ROOT / "acados_ocp_build"))
        mpc.set_reference(x_eq)
        print(f"  MPC ready: N={mpc.N}, dt={mpc.dt_mpc}s, nu={beam.nu}")

    # Live visualization (optional)
    live_viz = None
    if args.live and cfg["viz"].get("enabled", True):
        live_viz = LiveViz(beam, cfg)

    # Simulation loop
    print(f"\nRunning simulation: {duration}s, dt_sim={dt_sim}s, "
          f"dt_mpc={dt_mpc}s, steps={total_steps}")
    print("-" * 60)

    u_mpc = np.zeros(beam.n_u)
    n_mpc_ok = 0
    n_mpc_total = 0

    for step in range(total_steps):
        sim_ol.step(u=np.zeros(beam.n_u))

        if mpc is not None and step % mpc_ratio == 0:
            x_obs = sim_cl.observe()
            u_mpc, info = mpc.solve(x_obs)
            n_mpc_total += 1
            if info["status"] == 0:
                n_mpc_ok += 1
            elif n_mpc_total <= 5 or n_mpc_total % 100 == 0:
                print(f"  [MPC] step {step}: status={info['status']}, "
                      f"{info['solve_ms']:.1f} ms")

        sim_cl.step(u=u_mpc if mpc is not None else np.zeros(beam.n_u))

        if live_viz and step % viz_every == 0:
            live_viz.update(sim_ol.x, sim_cl.x, u_mpc, sim_cl.t, x_eq)

        if (step + 1) % (total_steps // 10) == 0:
            pct = 100 * (step + 1) / total_steps
            ke_ol = beam.kinetic_energy(sim_ol.x)
            ke_cl = beam.kinetic_energy(sim_cl.x)
            print(f"  {pct:5.0f}%  t={sim_cl.t:.2f}s  "
                  f"KE_open={ke_ol:.4e}  KE_mpc={ke_cl:.4e}")

    # Results summary
    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)

    h_ol = sim_ol.get_history_arrays()
    h_cl = sim_cl.get_history_arrays()

    ke_ol = np.array(h_ol["ke"])
    ke_cl = np.array(h_cl["ke"])
    print(f"  Mean KE (open-loop):  {np.mean(ke_ol):.6e}")
    print(f"  Mean KE (MPC):        {np.mean(ke_cl):.6e}")
    print(f"  KE reduction:         {100*(1 - np.mean(ke_cl)/max(np.mean(ke_ol), 1e-15)):.1f}%")

    tip_ol = np.array(h_ol["tip_y"])
    tip_cl = np.array(h_cl["tip_y"])
    tip_eq = pos_eq[-1, 1]
    print(f"  RMS tip dev (open):   {np.sqrt(np.mean((tip_ol - tip_eq)**2)):.4e} m")
    print(f"  RMS tip dev (MPC):    {np.sqrt(np.mean((tip_cl - tip_eq)**2)):.4e} m")

    if mpc is not None:
        print(f"  MPC solves:           {n_mpc_ok}/{n_mpc_total} converged")

    plot_summary(h_ol, h_cl, save_dir=str(out_dir))

    if live_viz:
        live_viz.savefig(str(out_dir / "live_dashboard.png"))
        print("\nClose the visualization window to exit.")
        input("Press Enter to close ... ")
        live_viz.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
