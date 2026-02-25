"""2D planar Discrete Elastic Rod (DER) beam model.

Provides a CasADi symbolic model (for acados MPC) and a NumPy step
function (for simulation with PBD inextensibility enforcement).
"""

from __future__ import annotations

import numpy as np
import casadi as ca
from scipy.optimize import minimize as sp_minimize


class DERBeam2D:
    """2D planar cantilever beam using Discrete Elastic Rods.

    Nodes 0..n_fixed-1 are clamped.  Nodes n_fixed..n_nodes-1 are free.
    """

    def __init__(self, cfg: dict):
        rod = cfg["rod"]
        ctrl_cfg = cfg["control"]

        self.n_nodes: int = rod["n_nodes"]
        self.n_fixed: int = rod.get("n_fixed", 2)
        self.n_free: int = self.n_nodes - self.n_fixed
        self.length: float = rod["length"]
        self.EI: float = rod["EI"]
        self.EA: float = rod["EA"]
        self.linear_density: float = rod["linear_density"]
        self.damping_coeff: float = rod["damping"]
        self.gravity = np.array(rod["gravity"], dtype=float)
        self.edge_length: float = self.length / (self.n_nodes - 1)
        self.node_mass: float = self.linear_density * self.length / self.n_nodes

        # Rest configuration: straight horizontal beam
        xs = np.linspace(0, self.length, self.n_nodes)
        ys = np.zeros(self.n_nodes)
        self.rest_all = np.column_stack([xs, ys])
        self.rest_fixed = self.rest_all[:self.n_fixed]
        self.rest_free = self.rest_all[self.n_fixed:]

        self.control_inputs = ctrl_cfg["inputs"]
        self.n_u: int = len(self.control_inputs)
        self.u_lower = np.array([c["bounds"][0] for c in self.control_inputs])
        self.u_upper = np.array([c["bounds"][1] for c in self.control_inputs])

        # State: [q_free (2*n_free), v_free (2*n_free)]
        self.nx: int = self.n_free * 2 * 2
        self.nu: int = self.n_u

        self._build_casadi()

    def _build_casadi(self):
        n = self.n_nodes
        nf = self.n_free
        nfx = self.n_fixed
        L0 = self.edge_length
        eps = 1e-8

        q = ca.SX.sym("q", nf * 2)
        v = ca.SX.sym("v", nf * 2)
        u = ca.SX.sym("u", self.n_u)

        q_fixed = ca.DM(self.rest_fixed.flatten())
        q_all = ca.vertcat(q_fixed, q)

        def p(g):
            return q_all[2 * g: 2 * g + 2]

        def safe_norm(e):
            return ca.sqrt(ca.dot(e, e) + eps ** 2)

        # Stretching energy (included in MPC model; simulation uses PBD instead)
        E_stretch = ca.SX(0)
        for i in range(n - 1):
            e = p(i + 1) - p(i)
            L = safe_norm(e)
            E_stretch += (self.EA / (2 * L0)) * (L - L0) ** 2

        # Bending energy
        E_bend = ca.SX(0)
        for j in range(1, n - 1):
            e_prev = p(j) - p(j - 1)
            e_next = p(j + 1) - p(j)
            Lp = safe_norm(e_prev)
            Ln = safe_norm(e_next)
            cross = e_prev[0] * e_next[1] - e_prev[1] * e_next[0]
            dot = ca.dot(e_prev, e_next)
            kappa = 2 * cross / (Lp * Ln + dot + eps)
            l_vor = 0.5 * (Lp + Ln)
            E_bend += (self.EI / 2) * kappa ** 2 / l_vor

        # Gravity potential
        g_vec = ca.DM(self.gravity)
        E_grav = ca.SX(0)
        for i in range(nf):
            qi = q[2 * i: 2 * i + 2]
            E_grav -= self.node_mass * ca.dot(g_vec, qi)

        # Control forces (applied force and moment inputs)
        F_ctrl = ca.SX.zeros(nf * 2)
        for idx, inp in enumerate(self.control_inputs):
            gnode = inp["node"]
            fi = gnode - nfx

            if inp["type"] == "force":
                d = np.array(inp["direction"], dtype=float)
                d = d / np.linalg.norm(d)
                if 0 <= fi < nf:
                    F_ctrl[2 * fi] += u[idx] * d[0]
                    F_ctrl[2 * fi + 1] += u[idx] * d[1]

            elif inp["type"] == "moment":
                if 1 <= gnode <= n - 2:
                    e_prev = p(gnode) - p(gnode - 1)
                    e_next = p(gnode + 1) - p(gnode)
                    cross_val = e_prev[0] * e_next[1] - e_prev[1] * e_next[0]
                    dot_val = ca.dot(e_prev, e_next)
                    theta = ca.atan2(cross_val, dot_val)
                    dtheta_dq = ca.jacobian(theta, q)
                    F_ctrl += u[idx] * dtheta_dq.T

        # Full dynamics (with stretching) for MPC
        E_full = E_stretch + E_bend + E_grav
        F_full = -ca.jacobian(E_full, q).T - self.damping_coeff * v + F_ctrl
        a_full = F_full / self.node_mass

        x = ca.vertcat(q, v)
        xdot_sym = ca.SX.sym("xdot", self.nx)
        f_expl = ca.vertcat(v, a_full)

        self.x_sym = x
        self.u_sym = u
        self.xdot_sym = xdot_sym
        self.f_expl = f_expl
        self.f_impl = xdot_sym - f_expl

        # Soft dynamics (bending + gravity only) for simulation
        E_soft = E_bend + E_grav
        F_soft = -ca.jacobian(E_soft, q).T - self.damping_coeff * v + F_ctrl
        a_soft = F_soft / self.node_mass
        f_expl_sim = ca.vertcat(v, a_soft)

        self._dynamics_fn = ca.Function("f_full", [x, u], [f_expl])
        self._sim_dynamics_fn = ca.Function("f_sim", [x, u], [f_expl_sim])
        self._energy_fn = ca.Function("E_total", [q], [E_full])
        self._grad_fn = ca.Function("dE_dq", [q], [ca.jacobian(E_full, q)])

    def get_acados_model(self):
        from acados_template import AcadosModel
        m = AcadosModel()
        m.name = "der_beam_2d"
        m.x = self.x_sym
        m.u = self.u_sym
        m.xdot = self.xdot_sym
        m.f_expl_expr = self.f_expl
        m.f_impl_expr = self.f_impl
        return m

    def dynamics_full(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array(self._dynamics_fn(x, u)).flatten()

    def dynamics_sim(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array(self._sim_dynamics_fn(x, u)).flatten()

    def step(self, x: np.ndarray, u: np.ndarray, dt: float,
             f_ext: np.ndarray | None = None,
             pbd_iter: int = 10) -> np.ndarray:
        """Semi-implicit Euler + PBD inextensibility projection."""
        nq = self.n_free * 2
        q = x[:nq].copy()
        v = x[nq:].copy()

        xdot = self.dynamics_sim(x, u)
        a = xdot[nq:]

        if f_ext is not None:
            a = a + f_ext / self.node_mass

        v_new = v + a * dt
        q_new = q + v_new * dt

        if pbd_iter > 0:
            q_new, v_new = self._pbd_correct(q_new, v_new, dt, pbd_iter)

        return np.concatenate([q_new, v_new])

    def _pbd_correct(self, q: np.ndarray, v: np.ndarray,
                     dt: float, n_iter: int) -> tuple:
        """Gauss-Seidel PBD: project edge lengths to rest length."""
        L0 = self.edge_length
        nfx = self.n_fixed
        q_before = q.copy()
        q_fixed_flat = self.rest_fixed.flatten()

        for _ in range(n_iter):
            for i in range(self.n_nodes - 1):
                q_all = np.concatenate([q_fixed_flat, q])
                pi = q_all[2 * i: 2 * i + 2]
                pj = q_all[2 * (i + 1): 2 * (i + 1) + 2]
                delta = pj - pi
                L = np.linalg.norm(delta)
                if L < 1e-12:
                    continue
                corr = (L - L0) / L * delta

                i_free = (i >= nfx)
                j_free = (i + 1 >= nfx)

                if i_free and j_free:
                    fi = 2 * (i - nfx)
                    fj = 2 * (i + 1 - nfx)
                    q[fi: fi + 2] += 0.5 * corr
                    q[fj: fj + 2] -= 0.5 * corr
                elif j_free:
                    fj = 2 * (i + 1 - nfx)
                    q[fj: fj + 2] -= corr
                elif i_free:
                    fi = 2 * (i - nfx)
                    q[fi: fi + 2] += corr

        if dt > 0:
            v += (q - q_before) / dt
        return q, v

    def find_equilibrium(self) -> np.ndarray:
        """Static equilibrium via energy minimization (L-BFGS-B)."""
        q0 = self.rest_free.flatten().copy()

        def obj_and_grad(q_flat):
            E = float(self._energy_fn(q_flat))
            dE = np.array(self._grad_fn(q_flat)).flatten()
            return E, dE

        res = sp_minimize(obj_and_grad, q0, method="L-BFGS-B", jac=True,
                          options={"maxiter": 5000, "ftol": 1e-14,
                                   "gtol": 1e-10})
        q_eq = res.x
        return np.concatenate([q_eq, np.zeros_like(q_eq)])

    def rest_state(self) -> np.ndarray:
        q = self.rest_free.flatten()
        return np.concatenate([q, np.zeros_like(q)])

    def get_positions(self, x: np.ndarray) -> np.ndarray:
        """Return all node positions (n_nodes, 2) from state vector."""
        q = x[: self.n_free * 2]
        q_all = np.concatenate([self.rest_fixed.flatten(), q])
        return q_all.reshape(-1, 2)

    def get_free_positions(self, x: np.ndarray) -> np.ndarray:
        return x[: self.n_free * 2].reshape(-1, 2)

    def get_free_velocities(self, x: np.ndarray) -> np.ndarray:
        return x[self.n_free * 2:].reshape(-1, 2)

    def kinetic_energy(self, x: np.ndarray) -> float:
        v = x[self.n_free * 2:]
        return 0.5 * self.node_mass * float(np.sum(v ** 2))

    def tip_displacement(self, x: np.ndarray, x_eq: np.ndarray) -> float:
        pos = self.get_positions(x)
        pos_eq = self.get_positions(x_eq)
        return float(np.linalg.norm(pos[-1] - pos_eq[-1]))
