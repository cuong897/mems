from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from math import isfinite, log, sqrt
from pathlib import Path
from statistics import linear_regression


UC = 1.0 / 3.0
ALPHA_C_EXACT = Fraction(4, 27)
ALPHA_C = float(ALPHA_C_EXACT)
METHODS = ("bisection", "newton", "secant")
FORMS = ("centered", "direct")
E2_LOADS = tuple(Decimal(r) for r in ("0.9", "0.99", "0.999", "0.999999"))
STABLE_BRACKET = (0.0, UC)
OUTSIDE_BRACKET = (0.0, 2.0)
BENCHMARK_TOLERANCE = 1e-12
REFERENCE_DIGITS = 80
MATCH_TOLERANCE = Decimal("1e-9")
MAX_ITERATIONS = 200
TERMINATED = ("stop_met", "computed_zero")
PHI = (1 + sqrt(5)) / 2


def derivative(u: float) -> float:
    return (1.0 - u) * (1.0 - 3.0 * u)


@dataclass
class Evaluator:
    delta: float
    form: str
    alpha: float
    n_f: int = 0
    n_df: int = 0

    def value(self, u: float) -> float:
        self.n_f += 1
        if self.form == "centered":
            t = u - UC
            return ALPHA_C * self.delta + t * t * (t - 1.0)
        gap = 1.0 - u
        return u * gap * gap - self.alpha

    def slope(self, u: float) -> float:
        self.n_df += 1
        return derivative(u)


@dataclass(frozen=True)
class Iterate:
    u: float
    residual: float
    step: float | None = None
    half_width: float | None = None


@dataclass(frozen=True)
class Result:
    status: str
    iterations: int
    point: Iterate | None = None


@dataclass(frozen=True)
class Stop:
    rule: str
    tolerance: float

    def reached(self, point: Iterate) -> bool:
        if self.rule == "residual":
            return abs(point.residual) <= self.tolerance
        distance = point.step if self.rule == "step" else point.half_width
        return distance is not None and distance <= self.tolerance


def bisect(function: Evaluator, stop: Stop, bracket: tuple[float, float]) -> Result:
    a, b = bracket
    fa, fb = function.value(a), function.value(b)
    if not (fa < 0.0 < fb or fb < 0.0 < fa):
        return Result("no_sign_change", 0)

    previous = None
    for count in range(1, MAX_ITERATIONS + 1):
        half_width = (b - a) / 2.0
        u = a + half_width
        step = None if previous is None else abs(u - previous.u)
        point = Iterate(u, function.value(u), step, half_width)
        if stop.reached(point):
            return Result("stop_met", count, point)
        if point.residual == 0.0:
            return Result("computed_zero", count, point)
        if u in (a, b):
            return Result("stagnation", count, point)
        if (point.residual < 0.0) == (fa < 0.0):
            a, fa = u, point.residual
        else:
            b = u
        previous = point
    return Result("max_iterations", MAX_ITERATIONS, previous)


def open_solver(
    method: str, function: Evaluator, stop: Stop, u0: float, u1: float | None
) -> Result:
    previous = None
    current = Iterate(u0, function.value(u0))
    if stop.reached(current):
        return Result("stop_met", 0, current)
    if method == "secant":
        assert u1 is not None
        previous = current
        current = Iterate(u1, function.value(u1))
        if stop.reached(current):
            return Result("stop_met", 0, current)

    for count in range(1, MAX_ITERATIONS + 1):
        if method == "newton":
            denominator = function.slope(current.u)
            if denominator == 0.0:
                status = "computed_zero" if current.residual == 0.0 else "zero_derivative"
                return Result(status, count - 1, current)
            correction = current.residual / denominator
        else:
            assert previous is not None
            denominator = current.residual - previous.residual
            if denominator == 0.0:
                status = (
                    "computed_zero" if current.residual == 0.0 else "zero_secant_denominator"
                )
                return Result(status, count - 1, current)
            correction = (current.residual / denominator) * (current.u - previous.u)

        if not isfinite(denominator):
            return Result("nonfinite", count - 1, current)
        u = current.u - correction
        if not isfinite(u):
            return Result("nonfinite", count - 1, current)
        value = function.value(u)
        if not isfinite(value):
            return Result("nonfinite", count - 1, current)
        point = Iterate(u, value, abs(u - current.u))
        if stop.reached(point):
            return Result("stop_met", count, point)
        if point.step == 0.0:
            status = "computed_zero" if value == 0.0 else "stagnation"
            return Result(status, count, point)
        previous, current = current, point
    return Result("max_iterations", MAX_ITERATIONS, current)


@dataclass(frozen=True)
class Reference:
    delta: Decimal
    r: Decimal
    alpha: Decimal
    roots: dict[str, Decimal]
    match_tolerance: Decimal
    direct_alpha: float
    direct_delta: Decimal
    direct_roots: dict[str, Decimal]

    @property
    def target(self) -> Decimal | None:
        return self.roots.get("stable", self.roots.get("critical"))

    @classmethod
    def build(cls, delta: Decimal) -> Reference:
        with localcontext() as context:
            context.prec = REFERENCE_DIGITS
            r = 1 - delta
            alpha = (Decimal(4) * r) / 27
            direct_alpha = float(ALPHA_C_EXACT * (1 - Fraction(delta)))
            direct_alpha_exact = Decimal.from_float(direct_alpha)
            direct_delta = 1 - direct_alpha_exact * 27 / 4
            roots = reference_roots(delta, alpha)
            direct_roots = reference_roots(direct_delta, direct_alpha_exact)
            values = list(roots.values())
            gaps = [b - a for a, b in zip(values, values[1:])]
            match_tolerance = min(MATCH_TOLERANCE, min(gaps) / 4) if gaps else MATCH_TOLERANCE
            return cls(delta, r, alpha, roots, match_tolerance, direct_alpha, direct_delta, direct_roots)

    def diagnose(self, result: Result, form: str) -> dict:
        target = self.target
        direct_target = self.direct_roots.get("stable") if form == "direct" else None
        diagnostics = {
            "target_branch": "stable" if self.delta > 0 else "critical" if self.delta == 0 else None,
            "u_target_ref": target,
            "u_stable_ref": self.roots.get("stable"),
            "u_unstable_ref": self.roots.get("unstable"),
            "u_outside_ref": self.roots["outside"],
            "u_stable_direct_ref": direct_target,
            "root_match_tolerance": self.match_tolerance,
            "valid_stable": False,
        }
        with localcontext() as context:
            context.prec = REFERENCE_DIGITS
            if target is not None and direct_target is not None:
                diagnostics["abs_shift_target_direct"] = abs(direct_target - target)
            if result.point is None:
                return diagnostics

            u = result.point.u
            physical = 0.0 <= u < 1.0
            positive_slope = derivative(u) > 0.0
            postcheck = physical and positive_slope
            x = Decimal.from_float(u)
            exact_residual = x * (1 - x) ** 2 - self.alpha
            nearest, nearest_error = nearest_root(x, self.roots)
            matched = nearest if nearest_error <= self.match_tolerance else "unresolved"
            stopped = result.status in TERMINATED
            stable_match = matched == "stable"
            diagnostics.update(
                abs_residual_ref=abs(exact_residual),
                nearest_root=nearest,
                abs_error_nearest=nearest_error,
                abs_error_target=None if target is None else abs(x - target),
                root_match=matched,
                in_physical_domain=physical,
                positive_slope=positive_slope,
                postcheck_pass=postcheck,
                postcheck_agrees=(postcheck == stable_match) if stopped else None,
                valid_stable=stopped and stable_match and postcheck,
            )
            if form == "direct":
                direct_nearest, direct_error = nearest_root(x, self.direct_roots)
                diagnostics.update(
                    nearest_direct_root=direct_nearest,
                    abs_error_nearest_direct=direct_error,
                    abs_error_target_direct=None if direct_target is None else abs(x - direct_target),
                )
        return diagnostics


def reference_roots(delta: Decimal, alpha: Decimal) -> dict[str, Decimal]:
    uc = Decimal(1) / 3
    peak_offset = Decimal(4) * delta / 27
    centered_radius = Decimal("0.125")

    def isolate(a: Decimal, b: Decimal, increasing: bool) -> Decimal:
        for _ in range(280):
            u = (a + b) / 2
            t = u - uc
            if abs(t) < centered_radius:
                value = peak_offset + t * t * (t - 1)
            else:
                value = u * (1 - u) ** 2 - alpha
            if value == 0 or u in (a, b):
                return u
            if (value < 0) == increasing:
                a = u
            else:
                b = u
        return (a + b) / 2

    roots = {}
    if delta > 0:
        roots["stable"] = isolate(Decimal(0), uc, increasing=True)
        roots["unstable"] = isolate(uc, Decimal(1), increasing=False)
    elif delta == 0:
        roots["critical"] = uc
    roots["outside"] = 4 * uc if delta == 0 else isolate(Decimal(1), Decimal(2), increasing=True)
    return roots


def nearest_root(x: Decimal, roots: dict[str, Decimal]) -> tuple[str, Decimal]:
    return min(((name, abs(x - root)) for name, root in roots.items()), key=lambda pair: pair[1])


FIELDS = (
    "experiment", "method", "form", "regime", "delta", "r", "r_float64",
    "direct_alpha_float64", "direct_effective_delta", "stop_rule", "tolerance",
    "u0", "u1", "bracket_a", "bracket_b", "exact_stable_bracket_exists",
    "status", "requested_stop_met", "iterations", "n_f", "n_df", "u",
    "abs_residual_float64", "step", "half_width", "repeated_iterate",
    "target_branch", "u_target_ref", "u_stable_ref", "u_unstable_ref",
    "u_outside_ref", "abs_residual_ref", "abs_error_target", "nearest_root",
    "abs_error_nearest", "root_match_tolerance", "root_match",
    "in_physical_domain", "positive_slope", "postcheck_pass",
    "postcheck_agrees", "valid_stable",
    "u_stable_direct_ref", "abs_shift_target_direct", "abs_error_target_direct",
    "nearest_direct_root", "abs_error_nearest_direct",
)


def measure(
    experiment: str, reference: Reference, method: str, form: str,
    stop: Stop, u0: float = 0.0, u1: float = 0.1,
    bracket: tuple[float, float] = STABLE_BRACKET,
) -> dict:
    function = Evaluator(float(reference.delta), form, reference.direct_alpha)
    is_bisection = method == "bisection"
    if is_bisection:
        result = bisect(function, stop, bracket)
    else:
        result = open_solver(method, function, stop, u0, u1 if method == "secant" else None)
    point = result.point
    row = {
        "experiment": experiment,
        "method": method,
        "form": form,
        "regime": "below" if reference.delta > 0 else "critical" if reference.delta == 0 else "above",
        "delta": reference.delta,
        "r": reference.r,
        "r_float64": float(reference.r),
        "direct_alpha_float64": function.alpha if form == "direct" else None,
        "direct_effective_delta": reference.direct_delta if form == "direct" else None,
        "stop_rule": stop.rule,
        "tolerance": stop.tolerance,
        "u0": None if is_bisection else u0,
        "u1": u1 if method == "secant" else None,
        "bracket_a": bracket[0] if is_bisection else None,
        "bracket_b": bracket[1] if is_bisection else None,
        "exact_stable_bracket_exists": (
            reference.delta > 0 and bracket == STABLE_BRACKET if is_bisection else None
        ),
        "status": result.status,
        "requested_stop_met": result.status == "stop_met",
        "iterations": result.iterations,
        "n_f": function.n_f,
        "n_df": function.n_df,
    }
    if point is not None:
        row.update(
            u=point.u,
            abs_residual_float64=abs(point.residual),
            step=point.step,
            half_width=point.half_width,
            repeated_iterate=None if point.step is None else point.step == 0.0,
        )
    row.update(reference.diagnose(result, form))
    return row


def experiment_one() -> list[dict]:
    deltas = [
        Decimal("0.5"),
        *(Decimal(f"1e-{k}") for k in range(1, 19)),
        Decimal(0),
        Decimal("-1e-10"),
        Decimal("-1e-4"),
    ]
    rows = []
    for delta in deltas:
        reference = Reference.build(delta)
        for form in FORMS:
            for method in METHODS:
                distance_rule = "half_width" if method == "bisection" else "step"
                for rule in ("residual", distance_rule):
                    stop = Stop(rule, BENCHMARK_TOLERANCE)
                    rows.append(measure("E1", reference, method, form, stop))
    return rows


def summarize_iteration_growth(rows: list[dict]) -> list[dict]:
    fits = []
    for form in FORMS:
        for method, convergence_factor in (("newton", 2), ("secant", PHI)):
            for rule in ("residual", "step"):
                candidates = [
                    row for row in rows
                    if row["form"] == form and row["method"] == method and row["stop_rule"] == rule
                    and Decimal("1e-10") <= row["delta"] <= Decimal("1e-3")
                    and row["requested_stop_met"] and row["postcheck_pass"]
                ]
                decades = [-row["delta"].adjusted() for row in candidates]
                counts = [row["iterations"] for row in candidates]
                fit = linear_regression(decades, counts) if len(decades) >= 2 else None
                fits.append({
                    "form": form,
                    "method": method,
                    "stop_rule": rule,
                    "runs_in_fit": len(decades),
                    "valid_stable_runs": sum(row["valid_stable"] for row in candidates),
                    "first_decade": min(decades, default=None),
                    "last_decade": max(decades, default=None),
                    "iterations_per_decade": None if fit is None else fit.slope,
                    "theoretical_iterations_per_decade": log(10) / (2 * log(convergence_factor)),
                })
    return fits


def experiment_two(grid_size: int, loads: tuple[Decimal, ...] = E2_LOADS) -> list[dict]:
    rows = []
    stop = Stop("step", BENCHMARK_TOLERANCE)
    bracket_stop = Stop("half_width", BENCHMARK_TOLERANCE)
    seeds = sorted({i / grid_size for i in range(grid_size)} | {UC})
    for r in loads:
        with localcontext() as context:
            context.prec = REFERENCE_DIGITS
            reference = Reference.build(1 - r)
        for bracket in (STABLE_BRACKET, OUTSIDE_BRACKET):
            rows.append(measure("E2", reference, "bisection", "centered", bracket_stop, bracket=bracket))
        for u0 in seeds:
            u1 = u0 + 0.01 if u0 + 0.01 < 1.0 else u0 - 0.01
            for method in ("newton", "secant"):
                rows.append(measure("E2", reference, method, "centered", stop, u0, u1))
    return rows


def error_scale(reference: Reference, method: str, stop: Stop) -> float | None:
    tolerance = stop.tolerance
    if stop.rule == "residual":
        if reference.delta == 0:
            return sqrt(tolerance)
        with localcontext() as context:
            context.prec = REFERENCE_DIGITS
            stable = reference.roots["stable"]
            slope = (1 - stable) * (1 - 3 * stable)
        return tolerance / float(slope)
    if method == "bisection":
        return tolerance if reference.delta > 0 else None
    if reference.delta == 0:
        return tolerance if method == "newton" else PHI * tolerance
    return None


def experiment_three(max_decade: int = 14) -> list[dict]:
    rows = []
    deltas = tuple(Decimal(s) for s in ("0.1", "0.01", "0.001", "1e-6", "1e-10", "0"))
    for delta in deltas:
        reference = Reference.build(delta)
        for form in FORMS:
            for method in METHODS:
                for decade in range(2, max_decade + 1):
                    for rule in ("residual", "step"):
                        stop = Stop(rule, 10.0 ** -decade)
                        row = measure("E3", reference, method, form, stop)
                        estimate = error_scale(reference, method, stop)
                        error = row.get("abs_error_target")
                        row.update(
                            error_scale_estimate=estimate,
                            abs_error_over_estimate=(
                                error / Decimal.from_float(estimate)
                                if row["requested_stop_met"] and error is not None and estimate is not None
                                else None
                            ),
                        )
                        rows.append(row)
    return rows


def summarize_branches(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (row["r"], row["method"], row["bracket_a"], row["bracket_b"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (r, method, a, b), subset in groups.items():
        stopped = [row for row in subset if row["status"] in TERMINATED]
        matches = Counter(row["root_match"] for row in stopped)
        correct = sum(row["postcheck_agrees"] for row in stopped)
        summary.append({
            "method": method,
            "r": r,
            "bracket_a": a,
            "bracket_b": b,
            "runs": len(subset),
            "requested_stop_met": sum(row["requested_stop_met"] for row in subset),
            "stable": matches["stable"],
            "unstable": matches["unstable"],
            "outside": matches["outside"],
            "unresolved": matches["unresolved"],
            "failed": len(subset) - len(stopped),
            "postcheck_pass": sum(row["postcheck_pass"] for row in stopped),
            "postcheck_false_positive": sum(
                row["postcheck_pass"] and row["root_match"] != "stable" for row in stopped
            ),
            "postcheck_false_negative": sum(
                not row["postcheck_pass"] and row["root_match"] == "stable" for row in stopped
            ),
            "postcheck_accuracy": correct / len(stopped) if stopped else None,
        })
    return summary


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path}: {len(rows):,} rows")


def plot_iteration_growth(plt, rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for ax, form in zip(axes, FORMS):
        for method in METHODS:
            data = [row for row in rows if row["form"] == form and row["method"] == method
                    and row["delta"] > 0 and row["stop_rule"] != "residual" and row["requested_stop_met"]]
            ax.plot([-log(float(row["delta"]), 10) for row in data],
                    [row["iterations"] for row in data], "o-", markersize=3, label=method)
        ax.set(title=form, xlabel=r"$\log_{10}(1/\delta)$")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Iterations")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_initial_points(plt, rows: list[dict], path: Path) -> None:
    bands = {}
    points = {}
    for row in rows:
        if row["u0"] is None:
            continue
        key = (row["r"], row["method"])
        band = bands.setdefault(key, len(bands))
        label = row["root_match"] if row["status"] in TERMINATED else "failed"
        points.setdefault(label, []).append((row["u0"], band))
    colors = {"stable": "tab:blue", "unstable": "tab:orange", "outside": "tab:purple",
              "unresolved": "tab:brown", "failed": "tab:gray"}
    fig, ax = plt.subplots(figsize=(10, max(2.8, 1.8 + 0.42 * len(bands))), constrained_layout=True)
    for label, positions in points.items():
        xs, ys = zip(*positions)
        ax.scatter(xs, ys, s=35, marker="|", linewidths=0.8, color=colors[label], label=label)
    ax.axvline(UC, color="0.3", linewidth=1, linestyle=":")
    ax.set(xlim=(0, 1), xlabel=r"Initial point $u_0$")
    ax.set_yticks(list(bands.values()))
    ax.set_yticklabels([f"r={r} {method}" for r, method in bands])
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=len(points))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_critical_errors(plt, rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, constrained_layout=True)
    for ax, form in zip(axes, FORMS):
        for method, color in zip(METHODS, ("tab:gray", "tab:blue", "tab:orange")):
            for rule, style in (("residual", "-"), ("step", "--")):
                data = [row for row in rows if row["regime"] == "critical" and row["form"] == form
                        and row["method"] == method and row["stop_rule"] == rule
                        and row["requested_stop_met"]]
                if data:
                    ax.loglog([row["tolerance"] for row in data],
                              [float(row["abs_error_target"]) for row in data],
                              linestyle=style, marker="o", markersize=3, color=color, label=f"{method} {rule}")
        ax.set(title=form, xlabel=r"Stopping tolerance $\tau$")
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Absolute error at r=1")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_load(value: str) -> Decimal:
    try:
        r = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid load: {value}") from exc
    if not r.is_finite() or not 0 < r < 1:
        raise argparse.ArgumentTypeError("E2 loads must satisfy 0 < r < 1")
    with localcontext() as context:
        context.prec = REFERENCE_DIGITS
        if not 0 < 1 - r < 1:
            raise argparse.ArgumentTypeError("E2 load is too close to an endpoint for the reference precision")
    return r


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark E1-E3 for the static MEMS pull-in problem (Python 3.10+).",
        epilog="E1 sweeps load, E2 sweeps initial points, and E3 compares stopping rules.",
    )
    parser.add_argument("experiment", nargs="?", choices=("all", "e1", "e2", "e3"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"), help="output directory")
    parser.add_argument("--grid-size", type=int, default=1000, help="number of uniform E2 seeds")
    parser.add_argument("--e2-r", type=parse_load, nargs="+", default=E2_LOADS, help="loads to scan in E2")
    parser.add_argument("--e3-max-decade", type=int, default=14, help="smallest E3 tolerance is 10**(-N)")
    parser.add_argument("--plots", action="store_true", help="also write PNG plots (requires matplotlib)")
    args = parser.parse_args()
    if args.grid_size < 2:
        parser.error("--grid-size must be at least 2")
    if not 2 <= args.e3_max_decade <= 323:
        parser.error("--e3-max-decade must produce a positive float64 tolerance (2 to 323)")

    plt = None
    if args.plots:
        try:
            import matplotlib

            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
        except ImportError as exc:
            parser.error(f"--plots requires matplotlib: {exc}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.experiment in ("all", "e1"):
        e1 = experiment_one()
        write_csv(args.output_dir / "e1_near_pullin.csv", e1, FIELDS)
        fits = summarize_iteration_growth(e1)
        write_csv(args.output_dir / "e1_iteration_fit.csv", fits, tuple(fits[0]))
        if plt is not None:
            plot_iteration_growth(plt, e1, args.output_dir / "e1_iteration_growth.png")
    if args.experiment in ("all", "e2"):
        e2 = experiment_two(args.grid_size, tuple(dict.fromkeys(args.e2_r)))
        write_csv(args.output_dir / "e2_initial_points.csv", e2, FIELDS)
        summary = summarize_branches(e2)
        write_csv(args.output_dir / "e2_branch_summary.csv", summary, tuple(summary[0]))
        if plt is not None:
            plot_initial_points(plt, e2, args.output_dir / "e2_initial_points.png")
    if args.experiment in ("all", "e3"):
        e3 = experiment_three(args.e3_max_decade)
        fields = (*FIELDS, "error_scale_estimate", "abs_error_over_estimate")
        write_csv(args.output_dir / "e3_stopping_rules.csv", e3, fields)
        if plt is not None:
            plot_critical_errors(plt, e3, args.output_dir / "e3_critical_errors.png")


if __name__ == "__main__":
    main()
