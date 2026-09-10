import sys

from src.sim.executor import ScenarioExecutor
from src.sim.scenario_registry import REGISTRY


def main():
    results = []
    for name, cls in REGISTRY.items():
        print(f"\n===== SCENARIO: {name} =====")
        results.append(ScenarioExecutor(cls()).run(save_plot=f"scenario_{name}.png"))

    print("\n===== SUMMARY =====")
    all_pass = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        all_pass &= r.passed
        print(f"{r.name:20s} {status}  {r.metrics}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()