from .dataset import EVALUATION_DATASET, EvalCase


def __getattr__(name: str):
    if name in ("BenchmarkReport", "run_benchmark"):
        from . import benchmark
        return getattr(benchmark, name)
    if name in ("AblationSuiteResult", "ConfigMetrics", "run_ablation_suite"):
        from . import ablation
        return getattr(ablation, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "EvalCase",
    "BenchmarkReport",
    "EVALUATION_DATASET",
    "run_benchmark",
    "AblationSuiteResult",
    "ConfigMetrics",
    "run_ablation_suite",
]
