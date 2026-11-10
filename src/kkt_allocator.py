"""Mixed-precision bit allocation on the VU13P: KKT water-filling, exact discrete solution and
Lagrangian dual decomposition (primal-dual with shadow prices). See docs/theory_and_formulation.md.

Problem (proxy of 1 - SP, Sec. 3-4 of the theory doc):

    min_x  Omega(x) = sum_l Omega^w_l(w_l) + sum_m Omega^a_m(a_m)
    s.t.   g_j(x) = (R_j(x) - B_j) / B_j <= 0,     j in {DSP, LUT, FF}
           w_l, a_m in a finite set of bit widths

with R_j from the calibrated cost model (src.hardware). Layer l's cost depends on (w_l, a_in,l),
where a_in,l is the ReLU precision of the previous layer, so grouping each compute layer with the
activation that feeds it makes both Omega and R separable over "choice groups". Then:

* ``solve_exact``: enumeration of all combinations (the design space is small) -> global optimum,
  ground truth for the other methods;
* ``LagrangianDualAllocator``: for fixed shadow prices mu the Lagrangian minimization splits into
  independent per-group argmins (exact primal step); mu follows projected subgradient ascent on the
  concave dual function q(mu) (Polyak step). It reports the dual bound, the best feasible primal and
  the duality gap, and certifies optimality only when the Lagrangian sufficient conditions hold;
* ``solve_kkt_bit_allocation``: the continuous relaxation (weights only, linear cost) solved in closed
  form by water-filling, with bisection or safeguarded Newton on t = log2(lambda).

This replaces ``LagrangeKKTQuantizerOptimizer`` / ``check_kkt_stopping_criteria``, whose primal step
did not minimize the Lagrangian (hard-coded budget factors and bit rules) and whose multiplier decay
enforced complementary slackness artificially.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.hardware import CONSTRAINED_RESOURCES

# ============================================================================ continuous relaxation


def _water_level_bits(c, a, t, b_min, b_max):
    """b_i(t) = clip(1/2 log2(c_i ln4 / a_i) - t/2, b_min, b_max) with t = log2(lambda)."""
    raw = 0.5 * np.log2(c * np.log(4.0) / a) - 0.5 * t
    return np.clip(raw, b_min, b_max), raw


def solve_kkt_bit_allocation(
    layer_names: Sequence[str],
    c_coeffs: Sequence[float],
    a_costs: Sequence[float],
    budget_bops: float,
    b_min: int = 2,
    b_max: int = 8,
    tol: float = 1e-9,
    max_iter: int = 200,
    method: str = "bisection",
) -> Dict[str, Any]:
    """Water-filling solution of  min sum c_i 4^-b_i  s.t.  sum a_i b_i <= B,  b_min <= b_i <= b_max.

    Stationarity: b_i*(lambda) = clip(1/2 log2(c_i ln4 / (lambda a_i)), b_min, b_max); lambda* solves
    h(lambda) = sum a_i b_i*(lambda) - B = 0 (h non-increasing). With t = log2 lambda, h is piecewise
    linear with slope -1/2 sum_{interior} a_i, so Newton is exact on each linear piece; it is
    safeguarded by a bisection bracket. The integer solution is floor + greedy fill of the leftover
    budget by marginal efficiency dOmega/da (compare with the exact discrete solvers).
    """
    c = np.maximum(np.asarray(c_coeffs, dtype=np.float64), 1e-300)
    a = np.asarray(a_costs, dtype=np.float64)
    if np.any(a <= 0):
        raise ValueError("a_costs must be positive")
    min_cost, max_cost = float(np.sum(a * b_min)), float(np.sum(a * b_max))
    if budget_bops < min_cost - 1e-9:
        raise ValueError(f"Budget {budget_bops} below the minimum cost {min_cost} (all layers at {b_min} bits).")

    if budget_bops >= max_cost:
        cont = np.full_like(a, float(b_max))
        integer_b = np.full(a.shape, b_max, dtype=int)
        lam, t_star, iters, active = 0.0, -np.inf, 0, False
    else:
        def h(t):
            return float(np.sum(a * _water_level_bits(c, a, t, b_min, b_max)[0]) - budget_bops)

        # bracket: h(lo) > 0 (bits towards b_max), h(hi) <= 0 (bits towards b_min)
        lo, hi = -10.0, 10.0
        while h(lo) <= 0:
            lo -= 10.0
        while h(hi) > 0:
            hi += 10.0
        t = 0.5 * (lo + hi)
        iters = 0
        for iters in range(1, max_iter + 1):
            val = h(t)
            if abs(val) <= tol * max(1.0, budget_bops):
                break
            if val > 0:
                lo = t
            else:
                hi = t
            t_next = 0.5 * (lo + hi)
            if method == "newton":
                _, raw = _water_level_bits(c, a, t, b_min, b_max)
                interior = (raw > b_min) & (raw < b_max)
                slope = -0.5 * float(np.sum(a[interior]))
                if slope < 0:
                    cand = t - val / slope
                    if lo < cand < hi:
                        t_next = cand
            elif method != "bisection":
                raise ValueError(f"unknown method {method!r}")
            t = t_next
        t_star, lam, active = t, float(2.0 ** t), True
        cont, _ = _water_level_bits(c, a, t_star, b_min, b_max)

        integer_b = np.clip(np.floor(cont + 1e-9).astype(int), b_min, b_max)
        spent = float(np.sum(a * integer_b))
        while True:
            gains = np.where(
                (integer_b < b_max) & (spent + a <= budget_bops + 1e-9),
                c * (4.0 ** (-integer_b) - 4.0 ** (-(integer_b + 1))) / a,
                -np.inf,
            )
            i = int(np.argmax(gains))
            if not np.isfinite(gains[i]):
                break
            integer_b[i] += 1
            spent += a[i]

    # KKT residuals of the continuous solution (Lagrangian with lambda, nu-, nu+)
    grad = -c * np.log(4.0) * 4.0 ** (-cont) + lam * a
    kkt = {}
    for name, bi, gi in zip(layer_names, cont, grad):
        if bi <= b_min + 1e-9:
            kkt[name] = {"bound": "lower", "nu_lower": float(gi), "stationarity_residual": 0.0}
        elif bi >= b_max - 1e-9:
            kkt[name] = {"bound": "upper", "nu_upper": float(-gi), "stationarity_residual": 0.0}
        else:
            kkt[name] = {"bound": "interior", "stationarity_residual": float(gi)}
    return {
        "method": method,
        "iterations": int(iters),
        "lambda_optimal": float(lam),
        "continuous_bits": {n: float(b) for n, b in zip(layer_names, cont)},
        "integer_bits": {n: int(b) for n, b in zip(layer_names, integer_b)},
        "achieved_bops": float(np.sum(a * integer_b)),
        "continuous_cost": float(np.sum(a * cont)),
        "budget_bops": float(budget_bops),
        "budget_active": bool(active),
        "complementary_slackness": float(lam * (np.sum(a * cont) - budget_bops)),
        "kkt": kkt,
    }


def linear_cost_coefficients(
    layer_cost_fn: Callable[[str, int, int], Dict[str, float]],
    layer: str,
    w_options: Sequence[int],
    a_in_bits: int,
    resource: str = "LUT",
) -> Tuple[float, float]:
    """Least-squares line R(w) ~ p + q w of a layer's cost over the weight bit widths (for water-filling)."""
    w = np.asarray(w_options, dtype=float)
    r = np.array([layer_cost_fn(layer, int(b), a_in_bits)[resource] for b in w_options])
    q, p = np.polyfit(w, r, 1)
    return float(p), float(q)


# ============================================================================ discrete problem


@dataclass
class ChoiceGroup:
    """One compute layer together with the activation precision that feeds it."""

    name: str
    options: List[Dict[str, Dict[str, int]]]  # [{"w": {layer: b}, "a": {act_layer: b}}, ...]
    omega: np.ndarray                         # (n_options,)
    resources: Dict[str, np.ndarray]          # resource -> (n_options,)


def build_choice_groups(
    layer_order: Sequence[str],
    activation_layers: Sequence[str],
    omega_w: Dict[str, Dict[int, float]],
    omega_a: Dict[str, Dict[int, float]],
    layer_cost_fn: Callable[[str, int, int], Dict[str, float]],
    w_options: Sequence[int],
    a_options: Sequence[int],
    input_bits: int,
    fixed_a_bits: Optional[Dict[str, int]] = None,
    resources: Sequence[str] = CONSTRAINED_RESOURCES,
) -> List[ChoiceGroup]:
    """Separable groups: group l = (w_l, a of the activation entering l). The network input is fixed."""
    fixed_a_bits = fixed_a_bits or {}
    groups, feeding = [], None
    for name in layer_order:
        if feeding is None:
            a_opts = [None]
        elif feeding in fixed_a_bits:
            a_opts = [int(fixed_a_bits[feeding])]
        else:
            a_opts = list(a_options)
        options, om, res = [], [], {r: [] for r in resources}
        for w in w_options:
            for a in a_opts:
                a_in = input_bits if a is None else a
                cost = layer_cost_fn(name, int(w), int(a_in))
                options.append({"w": {name: int(w)}, "a": {} if a is None else {feeding: int(a)}})
                om.append(omega_w[name][int(w)] + (0.0 if a is None else omega_a[feeding][int(a)]))
                for r in resources:
                    res[r].append(cost[r])
        groups.append(ChoiceGroup(name, options, np.asarray(om), {r: np.asarray(v, float) for r, v in res.items()}))
        feeding = name if name in activation_layers else None
    return groups


def assignment_from_indices(groups: Sequence[ChoiceGroup], idx: Sequence[int]) -> Tuple[Dict[str, int], Dict[str, int]]:
    w_bits, a_bits = {}, {}
    for g, i in zip(groups, idx):
        w_bits.update(g.options[int(i)]["w"])
        a_bits.update(g.options[int(i)]["a"])
    return w_bits, a_bits


def evaluate_indices(groups, idx, overhead: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    omega = float(sum(g.omega[int(i)] for g, i in zip(groups, idx)))
    res = {r: float(overhead.get(r, 0.0) + sum(g.resources[r][int(i)] for g, i in zip(groups, idx)))
           for r in groups[0].resources}
    return omega, res


def normalized_constraints(resources: Dict[str, float], budgets: Dict[str, float]) -> Dict[str, float]:
    """g_j = (R_j - B_j) / B_j (dimensionless; g_j = 0.2 means 20 % over budget)."""
    return {r: (resources[r] - budgets[r]) / max(budgets[r], 1.0) for r in budgets}


def solve_exact(groups: Sequence[ChoiceGroup], budgets: Dict[str, float], overhead: Dict[str, float]) -> Dict[str, Any]:
    """Global optimum of the discrete problem by enumerating every combination of group options."""
    n = len(groups)
    shape = [len(g.omega) for g in groups]

    def broadcast(vec, k):
        s = [1] * n
        s[k] = len(vec)
        return vec.reshape(s)

    total_omega = sum(broadcast(g.omega, k) for k, g in enumerate(groups))
    feasible = np.ones(shape, dtype=bool)
    for r, b in budgets.items():
        tot = overhead.get(r, 0.0) + sum(broadcast(g.resources[r], k) for k, g in enumerate(groups))
        feasible &= tot <= b * (1 + 1e-12)
    n_feasible = int(feasible.sum())
    if n_feasible == 0:
        return {"feasible": False, "n_candidates": int(np.prod(shape)), "n_feasible": 0}
    masked = np.where(feasible, total_omega, np.inf)
    idx = np.unravel_index(int(np.argmin(masked)), shape)
    omega, res = evaluate_indices(groups, idx, overhead)
    w_bits, a_bits = assignment_from_indices(groups, idx)
    return {
        "feasible": True,
        "indices": [int(i) for i in idx],
        "w_bits": w_bits,
        "a_bits": a_bits,
        "omega": omega,
        "resources": res,
        "g": normalized_constraints(res, budgets),
        "n_candidates": int(np.prod(shape)),
        "n_feasible": n_feasible,
    }


def kkt_certificate(
    omega_x: float,
    dual_value: float,
    mu: Dict[str, float],
    g: Dict[str, float],
    tol_cs: float = 1e-9,
) -> Dict[str, Any]:
    """Lagrangian sufficient optimality conditions for x = argmin_x L(x, mu) (true by construction).

    If x is primal feasible, mu >= 0 and mu_j g_j(x) = 0 for all j, then Omega(x) = q(mu) <= Omega*,
    so x is a global optimum -- for integer problems too (Everett's theorem). When no such pair exists
    there is a duality gap and only bounds are available.
    """
    cs = {r: float(mu[r] * g[r]) for r in g}
    feasible = all(v <= 1e-12 for v in g.values())
    dual_ok = all(v >= 0 for v in mu.values())
    cs_ok = all(abs(v) <= tol_cs * max(abs(omega_x), 1e-30) for v in cs.values())
    return {
        "stationarity": True,
        "primal_feasible": bool(feasible),
        "dual_feasible": bool(dual_ok),
        "complementary_slackness": cs,
        "complementary_slackness_ok": bool(cs_ok),
        "duality_gap": float(omega_x - dual_value),
        "certified_optimal": bool(feasible and dual_ok and cs_ok),
    }


class LagrangianDualAllocator:
    """Dual decomposition with projected subgradient ascent on the shadow prices mu_j >= 0.

    L(x, mu) = Omega(x) + sum_j mu_j g_j(x),  q(mu) = min_x L(x, mu)  (concave, q(mu) <= Omega*).
    Primal step: x(mu) = argmin_x L(x, mu), exact and separable per group.
    Dual step:   mu_j <- max(0, mu_j + eta_t g_j(x(mu))),  eta_t = theta_t (UB - q(mu_t)) / ||g||^2 (Polyak),
                 theta_t halved when the dual bound stalls (Held-Karp).
    Upper bound: best feasible point among the x(mu_t), each repaired/improved by a greedy Lagrangian
                 heuristic (``repair_and_improve``). For an integer problem the dual bound need not reach
                 the optimum (duality gap), so the result reports LB <= Omega* <= UB.
    """

    def __init__(self, groups: Sequence[ChoiceGroup], budgets: Dict[str, float], overhead: Dict[str, float],
                 use_heuristic: bool = True):
        self.groups = list(groups)
        self.budgets = {r: float(b) for r, b in budgets.items()}
        self.overhead = {r: float(overhead.get(r, 0.0)) for r in self.budgets}
        self.use_heuristic = use_heuristic  # repair/improve each x(mu) into a feasible upper bound
        self.history: List[Dict[str, Any]] = []

    def _scale(self, r):
        return max(self.budgets[r], 1.0)

    def lagrangian_argmin(self, mu: Dict[str, float]) -> Tuple[List[int], float]:
        idx, q = [], 0.0
        for g in self.groups:
            val = g.omega + sum(mu[r] * g.resources[r] / self._scale(r) for r in self.budgets)
            i = int(np.argmin(val))
            idx.append(i)
            q += float(val[i])
        q += sum(mu[r] * (self.overhead[r] - self.budgets[r]) / self._scale(r) for r in self.budgets)
        return idx, q

    def repair_and_improve(self, idx: Sequence[int]) -> Optional[List[int]]:
        """Lagrangian heuristic: make x(mu) feasible greedily, then improve it by feasible local moves.

        Repair: while some g_j > 0, switch the group option with the smallest Omega increase per unit of
        (normalized) violated resource released. Improve: repeatedly apply the single-group change with
        the largest Omega decrease that keeps every constraint satisfied.
        """
        idx = list(idx)
        resources = list(self.budgets)
        # per-group resource matrices (n_options, n_resources), and the totals of the current point
        mats = [np.stack([g.resources[r] for r in resources], axis=1) for g in self.groups]
        budget = np.array([self.budgets[r] for r in resources])
        scale = np.array([self._scale(r) for r in resources])
        base = np.array([self.overhead[r] for r in resources])

        def totals(ix):
            return base + sum(m[i] for m, i in zip(mats, ix))

        def violation(tot):
            return np.maximum((tot - budget) / scale, 0.0).sum(axis=-1)

        tot = totals(idx)
        while violation(tot) > 1e-12:
            best, v_now = None, violation(tot)
            for k, (grp, m) in enumerate(zip(self.groups, mats)):
                cand_tot = tot - m[idx[k]] + m              # totals if group k switches to each option
                dv = v_now - violation(cand_tot)            # strict decrease of the total violation
                for i in np.nonzero(dv > 1e-15)[0]:
                    ratio = (grp.omega[i] - grp.omega[idx[k]]) / dv[i]
                    if best is None or ratio < best[0]:
                        best = (ratio, k, int(i))
            if best is None:
                return None
            idx[best[1]] = best[2]
            tot = totals(idx)
        while True:
            best = None
            for k, (grp, m) in enumerate(zip(self.groups, mats)):
                cand_tot = tot - m[idx[k]] + m
                ok = np.all(cand_tot <= budget * (1 + 1e-12), axis=1) & (grp.omega < grp.omega[idx[k]])
                if ok.any():
                    i = int(np.argmin(np.where(ok, grp.omega, np.inf)))
                    gain = grp.omega[idx[k]] - grp.omega[i]
                    if best is None or gain > best[0]:
                        best = (gain, k, i)
            if best is None:
                return idx
            idx[best[1]] = best[2]
            tot = totals(idx)
        while True:
            omega_now, _ = evaluate_indices(self.groups, idx, self.overhead)
            best = None
            for k, grp in enumerate(self.groups):
                for i in np.argsort(grp.omega):
                    if grp.omega[i] >= grp.omega[idx[k]]:
                        break
                    cand = list(idx)
                    cand[k] = int(i)
                    if self._feasible(cand):
                        gain = grp.omega[idx[k]] - grp.omega[i]
                        if best is None or gain > best[0]:
                            best = (gain, cand)
                        break  # options are visited by increasing Omega: first feasible is the best for k
            if best is None:
                return idx
            idx = best[1]

    def _cheapest_feasible(self):
        idx = [int(np.argmin(sum(g.resources[r] / self._scale(r) for r in self.budgets))) for g in self.groups]
        omega, res = evaluate_indices(self.groups, idx, self.overhead)
        g = normalized_constraints(res, self.budgets)
        return (idx, omega) if all(v <= 1e-12 for v in g.values()) else (None, np.inf)

    def solve(self, max_iter: int = 500, tol_gap: float = 1e-4, theta0: float = 1.0, patience: int = 10,
              theta_min: float = 1e-4, mu0: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        mu = {r: float((mu0 or {}).get(r, 0.0)) for r in self.budgets}
        best_idx, ub = self._cheapest_feasible()
        if best_idx is None:
            raise ValueError("No feasible configuration: even the cheapest bit widths violate a budget.")
        lb, theta, stall = -np.inf, theta0, 0
        self.history = []
        stop = "max_iter"
        cert = None
        for t in range(1, max_iter + 1):
            idx, q = self.lagrangian_argmin(mu)
            omega_x, res = evaluate_indices(self.groups, idx, self.overhead)
            g = normalized_constraints(res, self.budgets)
            feasible = all(v <= 1e-12 for v in g.values())
            if feasible and omega_x < ub:
                ub, best_idx = omega_x, idx
            if self.use_heuristic:
                cand = self.repair_and_improve(idx)
                if cand is not None:
                    omega_c, _ = evaluate_indices(self.groups, cand, self.overhead)
                    if omega_c < ub:
                        ub, best_idx = omega_c, cand
            if not np.isfinite(lb) or q > lb + 1e-12 * max(abs(lb), 1e-30):
                lb, stall = q, 0
            else:
                stall += 1
                if stall >= patience:
                    theta, stall = theta / 2.0, 0
            cert = kkt_certificate(omega_x, q, mu, g)
            w_bits, a_bits = assignment_from_indices(self.groups, idx)
            self.history.append({
                "iteration": t, "mu": dict(mu), "g": g, "resources": res, "dual_value": q, "omega_x": omega_x,
                "upper_bound": ub, "lower_bound": lb, "theta": theta, "feasible": feasible,
                "w_bits": w_bits, "a_bits": a_bits,
            })
            if cert["certified_optimal"]:
                stop = "kkt_certificate"
                break
            if ub - lb <= tol_gap * abs(ub):
                stop = "duality_gap"
                break
            if theta < theta_min:
                stop = "dual_stalled"  # remaining gap is (mostly) the integer duality gap
                break
            norm2 = sum(v * v for v in g.values())
            if norm2 == 0.0:
                stop = "zero_subgradient"
                break
            eta = theta * (ub - q) / norm2
            mu = {r: max(0.0, mu[r] + eta * g[r]) for r in mu}

        omega_b, res_b = evaluate_indices(self.groups, best_idx, self.overhead)
        w_bits, a_bits = assignment_from_indices(self.groups, best_idx)
        return {
            "w_bits": w_bits,
            "a_bits": a_bits,
            "omega": omega_b,
            "resources": res_b,
            "g": normalized_constraints(res_b, self.budgets),
            "mu": dict(mu),
            "lower_bound": float(lb),
            "upper_bound": float(ub),
            "relative_gap": float((ub - lb) / abs(ub)) if ub else 0.0,
            "iterations": len(self.history),
            "stop_reason": stop,
            "last_certificate": cert,
        }


def budget_frontier(
    groups: Sequence[ChoiceGroup],
    budgets: Dict[str, float],
    overhead: Dict[str, float],
    resource: str,
    values: Sequence[float],
) -> List[Dict[str, Any]]:
    """Exact optimum as a function of one budget (others fixed): the Omega*-vs-B_j Pareto front."""
    out = []
    for v in values:
        b = dict(budgets)
        b[resource] = float(v)
        sol = solve_exact(groups, b, overhead)
        sol["budget_value"] = float(v)
        out.append(sol)
    return out
