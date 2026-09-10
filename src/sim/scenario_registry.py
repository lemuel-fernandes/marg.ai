from .scenarios.crossing_animal import CrossingAnimalScenario
from .scenarios.static_obstacle import StaticObstacleScenario
from .scenarios.multi_animal import MultiAnimalScenario
from .scenarios.sensor_noise import SensorNoiseScenario
from .scenarios.indian_road import IndianRoadScenario
from .scenarios.city_roads import CityRoadsScenario

REGISTRY = {
    "static_obstacle": StaticObstacleScenario,
    "crossing_animal": CrossingAnimalScenario,
    "multi_animal": MultiAnimalScenario,
    "sensor_noise": SensorNoiseScenario,
    "indian_road": IndianRoadScenario,
    "city_roads": CityRoadsScenario,
}