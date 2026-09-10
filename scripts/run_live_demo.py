import sys
from src.sim.executor import ScenarioExecutor
from src.sim.scenario_registry import REGISTRY

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "multi_animal"
    if name not in REGISTRY:
        print(f"Unknown scenario '{name}'. Available: {list(REGISTRY)}")
        sys.exit(2)
        
    print(f"🚀 Starting LIVE DASHBOARD for scenario: {name}")
    print("👀 Watch the Matplotlib window. Close it to end the scenario.")
    ScenarioExecutor(REGISTRY[name](), live=True).run(save_plot=f"scenario_{name}.png")

if __name__ == "__main__":
    main()