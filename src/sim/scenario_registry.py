from .scenarios.crossing_animal import CrossingAnimalScenario
from .scenarios.static_obstacle import StaticObstacleScenario
from .scenarios.multi_animal import MultiAnimalScenario
from .scenarios.sensor_noise import SensorNoiseScenario

REGISTRY = {
    "static_obstacle": StaticObstacleScenario,
    "crossing_animal": CrossingAnimalScenario,
    "multi_animal": MultiAnimalScenario,
    "sensor_noise": SensorNoiseScenario,
}