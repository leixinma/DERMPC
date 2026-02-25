"""Real-time and post-hoc visualization for DER-MPC."""

from __future__ import annotations

import pathlib
from typing import Dict, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class LiveViz:
    """Real-time 2x2 dashboard updated during the simulation loop."""

    def __init__(self, beam, cfg: dict, max_history: int = 5000):
        self.beam = beam
        for backend in ("QtAgg", "TkAgg", "Qt5Agg"):
            try:
                matplotlib.use(backend)
                break
            except Exception:
                continue
        plt.ion()

        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 8))
        self.fig.suptitle("DER-MPC  (blue = MPC, red = open-loop)", fontsize=12)
        self.ax_shape = self.axes[0, 0]
        self.ax_tip = self.axes[0, 1]
        self.ax_ke = self.axes[1, 0]
        self.ax_ctrl = self.axes[1, 1]

        self.ax_shape.set_title("Beam Shape")
        self.ax_shape.set_xlabel("x [m]")
        self.ax_shape.set_ylabel("y [m]")
        self.ax_shape.set_aspect("equal")
        self.ax_shape.grid(True, ls="--", alpha=0.3)
        self._line_eq, = self.ax_shape.plot([], [], "k--", lw=1, alpha=0.4,
                                            label="rest")
        self._line_ol, = self.ax_shape.plot([], [], "r-o", ms=3, lw=1.2,
                                            label="open-loop")
        self._line_cl, = self.ax_shape.plot([], [], "b-o", ms=3, lw=1.2,
                                            label="MPC")
        self.ax_shape.legend(fontsize=8)

        self.ax_tip.set_title("Tip y-displacement")
        self.ax_tip.set_xlabel("time [s]")
        self.ax_tip.set_ylabel("y [m]")
        self.ax_tip.grid(True, ls="--", alpha=0.3)
        self._tip_ol, = self.ax_tip.plot([], [], "r", lw=1, label="open")
        self._tip_cl, = self.ax_tip.plot([], [], "b", lw=1, label="MPC")
        self.ax_tip.legend(fontsize=8)

        self.ax_ke.set_title("Kinetic Energy")
        self.ax_ke.set_xlabel("time [s]")
        self.ax_ke.set_ylabel("KE [J]")
        self.ax_ke.grid(True, ls="--", alpha=0.3)
        self._ke_ol, = self.ax_ke.plot([], [], "r", lw=1, label="open")
        self._ke_cl, = self.ax_ke.plot([], [], "b", lw=1, label="MPC")
        self.ax_ke.legend(fontsize=8)

        self.ax_ctrl.set_title("Control Inputs")
        self.ax_ctrl.set_xlabel("time [s]")
        self.ax_ctrl.set_ylabel("u")
        self.ax_ctrl.grid(True, ls="--", alpha=0.3)
        self._ctrl_lines = []
        colors = ["C0", "C1", "C2", "C3"]
        for i in range(beam.n_u):
            ln, = self.ax_ctrl.plot([], [], colors[i % len(colors)],
                                    lw=1, label=f"u[{i}]")
            self._ctrl_lines.append(ln)
        self.ax_ctrl.legend(fontsize=8)

        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.fig.canvas.draw()
        plt.pause(0.01)

        self._max_h = max_history
        self._t_buf = []
        self._ke_ol_buf = []
        self._ke_cl_buf = []
        self._tip_ol_buf = []
        self._tip_cl_buf = []
        self._u_bufs = [[] for _ in range(beam.n_u)]

    def update(self, x_ol, x_cl, u_cl, t, x_eq=None):
        pos_ol = self.beam.get_positions(x_ol)
        pos_cl = self.beam.get_positions(x_cl)

        self._line_ol.set_data(pos_ol[:, 0], pos_ol[:, 1])
        self._line_cl.set_data(pos_cl[:, 0], pos_cl[:, 1])
        if x_eq is not None:
            pos_eq = self.beam.get_positions(x_eq)
            self._line_eq.set_data(pos_eq[:, 0], pos_eq[:, 1])
        self.ax_shape.relim()
        self.ax_shape.autoscale_view()

        self._t_buf.append(t)
        self._ke_ol_buf.append(self.beam.kinetic_energy(x_ol))
        self._ke_cl_buf.append(self.beam.kinetic_energy(x_cl))
        self._tip_ol_buf.append(pos_ol[-1, 1])
        self._tip_cl_buf.append(pos_cl[-1, 1])
        for i in range(self.beam.n_u):
            self._u_bufs[i].append(u_cl[i])

        for buf in [self._t_buf, self._ke_ol_buf, self._ke_cl_buf,
                     self._tip_ol_buf, self._tip_cl_buf] + self._u_bufs:
            if len(buf) > self._max_h:
                del buf[0]

        t_arr = np.array(self._t_buf)

        self._tip_ol.set_data(t_arr, self._tip_ol_buf)
        self._tip_cl.set_data(t_arr, self._tip_cl_buf)
        self.ax_tip.relim()
        self.ax_tip.autoscale_view()

        self._ke_ol.set_data(t_arr, self._ke_ol_buf)
        self._ke_cl.set_data(t_arr, self._ke_cl_buf)
        self.ax_ke.relim()
        self.ax_ke.autoscale_view()

        for i, ln in enumerate(self._ctrl_lines):
            ln.set_data(t_arr, self._u_bufs[i])
        self.ax_ctrl.relim()
        self.ax_ctrl.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def savefig(self, path: str):
        self.fig.savefig(path, dpi=200)

    def close(self):
        plt.ioff()
        plt.close(self.fig)


# ---------------------------------------------------------------------------
#  Publication-quality summary plots
# ---------------------------------------------------------------------------

_RC = {
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 1.6,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}

_COLOR_OL = "#d62728"   # red
_COLOR_MPC = "#1f77b4"  # blue


def plot_summary(hist_ol: Dict[str, np.ndarray],
                 hist_cl: Dict[str, np.ndarray],
                 save_dir: str = "results"):
    """Save publication-quality comparison plots after simulation."""
    matplotlib.use("Agg")
    d = pathlib.Path(save_dir)
    d.mkdir(parents=True, exist_ok=True)

    t_ol = hist_ol["t"]
    t_cl = hist_cl["t"]

    with plt.rc_context(_RC):
        # Kinetic energy comparison
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.semilogy(t_ol, np.array(hist_ol["ke"]) + 1e-14,
                    color=_COLOR_OL, lw=1.4, label="Open-loop")
        ax.semilogy(t_cl, np.array(hist_cl["ke"]) + 1e-14,
                    color=_COLOR_MPC, lw=1.4, label="MPC")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Kinetic Energy (J)")
        ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
        ax.grid(True, which="both", ls="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(d / "ke_comparison.png")
        plt.close(fig)

        # Tip y-displacement comparison
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(t_ol, hist_ol["tip_y"], color=_COLOR_OL, lw=1.2, label="Open-loop")
        ax.plot(t_cl, hist_cl["tip_y"], color=_COLOR_MPC, lw=1.2, label="MPC")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Tip y-displacement (m)")
        ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
        ax.grid(True, ls="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(d / "tip_y_comparison.png")
        plt.close(fig)

        # Control inputs
        if "u" in hist_cl and len(hist_cl["u"]) > 0:
            u_arr = np.array(hist_cl["u"])
            fig, ax = plt.subplots(figsize=(8, 3.2))
            labels = ["Force (N)", "Moment (N·m)"]
            colors = ["#1f77b4", "#ff7f0e"]
            for j in range(u_arr.shape[1]):
                lbl = labels[j] if j < len(labels) else f"u[{j}]"
                ax.plot(t_cl, u_arr[:, j], lw=1.2,
                        color=colors[j % len(colors)], label=lbl)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Control Input")
            ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
            ax.grid(True, ls="--", alpha=0.35)
            fig.tight_layout()
            fig.savefig(d / "control.png")
            plt.close(fig)

        # Combined 3-panel summary for README
        fig, axes = plt.subplots(1, 3, figsize=(16, 3.5))

        ax = axes[0]
        ax.semilogy(t_ol, np.array(hist_ol["ke"]) + 1e-14,
                    color=_COLOR_OL, lw=1.4, label="Open-loop")
        ax.semilogy(t_cl, np.array(hist_cl["ke"]) + 1e-14,
                    color=_COLOR_MPC, lw=1.4, label="MPC")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Kinetic Energy (J)")
        ax.set_title("(a) Kinetic Energy")
        ax.legend(fontsize=9, frameon=True, fancybox=False, edgecolor="0.7")
        ax.grid(True, which="both", ls="--", alpha=0.35)

        ax = axes[1]
        ax.plot(t_ol, hist_ol["tip_y"], color=_COLOR_OL, lw=1.2, label="Open-loop")
        ax.plot(t_cl, hist_cl["tip_y"], color=_COLOR_MPC, lw=1.2, label="MPC")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Tip y-displacement (m)")
        ax.set_title("(b) Tip Displacement")
        ax.legend(fontsize=9, frameon=True, fancybox=False, edgecolor="0.7")
        ax.grid(True, ls="--", alpha=0.35)

        if "u" in hist_cl and len(hist_cl["u"]) > 0:
            ax = axes[2]
            u_arr = np.array(hist_cl["u"])
            labels = ["Force (N)", "Moment (N·m)"]
            colors = ["#1f77b4", "#ff7f0e"]
            for j in range(u_arr.shape[1]):
                lbl = labels[j] if j < len(labels) else f"u[{j}]"
                ax.plot(t_cl, u_arr[:, j], lw=1.2,
                        color=colors[j % len(colors)], label=lbl)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Control Input")
            ax.set_title("(c) Control Inputs")
            ax.legend(fontsize=9, frameon=True, fancybox=False, edgecolor="0.7")
            ax.grid(True, ls="--", alpha=0.35)

        fig.tight_layout()
        fig.savefig(d / "summary.png")
        plt.close(fig)

    print(f"Plots saved to {d}/")
