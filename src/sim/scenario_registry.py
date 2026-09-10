from .scenarios.crossing_animal import CrossingAnimalScenario
from .scenarios.static_obstacle import StaticObstacleScenario

REGISTRY = {
    "static_obstacle": StaticObstacleScenario,
    "crossing_animal": CrossingAnimalScenario,
}