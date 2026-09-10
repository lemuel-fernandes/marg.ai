import sys

from src.sim.executor import ScenarioExecutor
from src.sim.scenario_registry import REGISTRY


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "static_obstacle"
    if name not in REGISTRY:
        print(f"Unknown scenario '{name}'. Available: {list(REGISTRY)}")
        sys.exit(2)

    result = ScenarioExecutor(REGISTRY[name]()).run(save_plot=f"scenario_{name}.png")
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()