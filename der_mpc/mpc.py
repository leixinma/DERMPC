"""acados NMPC controller for the DER beam.

Builds a LINEAR_LS OCP with configurable observed nodes and cost weights,
compiles the solver, and provides a solve(x) interface.
"""

from __future__ import annotations

import pathlib
import shutil
import time

import numpy as np


class MPCController:
    """Receding-horizon NMPC for the DER beam."""

    def __init__(self, beam, cfg: dict,
                 build_dir: str = "acados_ocp_build"):
        from acados_template import AcadosOcp, AcadosOcpSolver

        self.beam = beam
        mpc_cfg = cfg["mpc"]
        cost_cfg = mpc_cfg["cost"]
        solver_cfg = mpc_cfg.get("solver", {})

        self.N: int = mpc_cfg["N"]
        self.dt_mpc: float = mpc_cfg["dt"]

        obs = cost_cfg.get("observed_nodes", "all")
        if obs == "all":
            self.obs_nodes = list(range(beam.n_fixed, beam.n_nodes))
        else:
            self.obs_nodes = list(obs)
        self.n_obs: int = len(self.obs_nodes)

        self.w_pos = cost_cfg.get("w_pos", 50.0)
        self.w_vel = cost_cfg.get("w_vel", 10.0)
        self.w_ctrl = cost_cfg.get("w_ctrl", 0.01)
        self.w_pos_e = cost_cfg.get("w_pos_terminal", 100.0)
        self.w_vel_e = cost_cfg.get("w_vel_terminal", 20.0)

        nx = beam.nx
        nu = beam.nu
        nq = beam.n_free * 2
        n_obs2 = self.n_obs * 2
        ny = n_obs2 + n_obs2 + nu       # [q_obs; v_obs; u]
        ny_e = n_obs2 + n_obs2           # [q_obs; v_obs]

        # Selection matrices mapping full state to observed outputs
        obs_fidx = [g - beam.n_fixed for g in self.obs_nodes]

        Vx = np.zeros((ny, nx))
        for row, fi in enumerate(obs_fidx):
            Vx[2 * row, 2 * fi] = 1.0
            Vx[2 * row + 1, 2 * fi + 1] = 1.0
        for row, fi in enumerate(obs_fidx):
            r = n_obs2 + 2 * row
            Vx[r, nq + 2 * fi] = 1.0
            Vx[r + 1, nq + 2 * fi + 1] = 1.0

        Vu = np.zeros((ny, nu))
        for j in range(nu):
            Vu[2 * n_obs2 + j, j] = 1.0

        Vx_e = Vx[:ny_e, :]

        W = np.zeros((ny, ny))
        for i in range(n_obs2):
            W[i, i] = self.w_pos
        for i in range(n_obs2, 2 * n_obs2):
            W[i, i] = self.w_vel
        for i in range(2 * n_obs2, ny):
            W[i, i] = self.w_ctrl

        W_e = np.zeros((ny_e, ny_e))
        for i in range(n_obs2):
            W_e[i, i] = self.w_pos_e
        for i in range(n_obs2, ny_e):
            W_e[i, i] = self.w_vel_e

        # Build OCP
        ocp = AcadosOcp()
        ocp.model = beam.get_acados_model()

        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.N * self.dt_mpc

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu
        ocp.cost.Vx_e = Vx_e
        ocp.cost.W = W
        ocp.cost.W_e = W_e
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)

        ocp.constraints.x0 = np.zeros(nx)
        ocp.constraints.lbu = beam.u_lower
        ocp.constraints.ubu = beam.u_upper
        ocp.constraints.idxbu = np.arange(nu)

        so = ocp.solver_options
        so.qp_solver = solver_cfg.get("qp_solver", "PARTIAL_CONDENSING_HPIPM")
        so.nlp_solver_type = solver_cfg.get("nlp_solver_type", "SQP_RTI")
        so.integrator_type = solver_cfg.get("integrator_type", "IRK")
        so.nlp_solver_max_iter = solver_cfg.get("nlp_solver_max_iter", 1)
        so.qp_solver_iter_max = solver_cfg.get("qp_solver_iter_max", 50)
        so.sim_method_num_stages = solver_cfg.get("sim_method_num_stages", 2)
        so.sim_method_num_steps = solver_cfg.get("sim_method_num_steps", 2)
        so.hessian_approx = solver_cfg.get("hessian_approx", "GAUSS_NEWTON")
        so.levenberg_marquardt = solver_cfg.get("levenberg_marquardt", 1e-2)
        so.qp_solver_warm_start = solver_cfg.get("qp_solver_warm_start", 2)
        if "sim_method_newton_iter" in solver_cfg:
            so.sim_method_newton_iter = solver_cfg["sim_method_newton_iter"]

        bd = pathlib.Path(build_dir).resolve()
        if bd.exists():
            shutil.rmtree(bd)
        bd.mkdir(parents=True, exist_ok=True)
        ocp.code_export_directory = str(bd / "c_generated_code")

        self._solver = AcadosOcpSolver(
            ocp, json_file=str(bd / "acados_ocp.json"),
            build=True, generate=True,
        )
        self._ny = ny
        self._ny_e = ny_e
        self._n_obs2 = n_obs2
        self._obs_fidx = obs_fidx
        self._nq = nq
        self._step_count = 0

    def set_reference(self, x_ref: np.ndarray):
        """Set MPC reference (e.g. equilibrium state for vibration suppression)."""
        q_ref = x_ref[: self._nq]
        v_ref = x_ref[self._nq:]

        q_obs = self._select_obs_dofs(q_ref)
        v_obs = self._select_obs_dofs(v_ref)

        self._y_ref = np.concatenate([q_obs, v_obs, np.zeros(self.beam.nu)])
        self._y_ref_e = np.concatenate([q_obs, v_obs])

        for k in range(self.N):
            self._solver.set(k, "yref", self._y_ref)
        self._solver.set(self.N, "yref", self._y_ref_e)

        u0 = np.zeros(self.beam.nu)
        for k in range(self.N + 1):
            self._solver.set(k, "x", x_ref)
        for k in range(self.N):
            self._solver.set(k, "u", u0)

        self._solver.set(0, "lbx", x_ref)
        self._solver.set(0, "ubx", x_ref)
        for _ in range(5):
            self._solver.solve()
        self._step_count = 0

    def solve(self, x_current: np.ndarray) -> tuple[np.ndarray, dict]:
        """Solve MPC for current state. Returns (u_opt, info_dict)."""
        self._solver.set(0, "lbx", x_current)
        self._solver.set(0, "ubx", x_current)

        if self._step_count > 0:
            self._warm_start_shift()

        t0 = time.perf_counter()
        status = self._solver.solve()
        dt_ms = (time.perf_counter() - t0) * 1e3

        u_opt = self._solver.get(0, "u")
        self._step_count += 1

        if status not in (0, 4):
            u_opt = np.zeros(self.beam.nu)

        return u_opt, {"status": status, "solve_ms": dt_ms,
                       "step": self._step_count}

    def _warm_start_shift(self):
        for k in range(self.N):
            self._solver.set(k, "x",
                             self._solver.get(min(k + 1, self.N), "x"))
        for k in range(self.N - 1):
            self._solver.set(k, "u", self._solver.get(k + 1, "u"))

    def _select_obs_dofs(self, vec_free: np.ndarray) -> np.ndarray:
        parts = []
        for fi in self._obs_fidx:
            parts.append(vec_free[2 * fi: 2 * fi + 2])
        return np.concatenate(parts)

    @property
    def step_count(self) -> int:
        return self._step_count
