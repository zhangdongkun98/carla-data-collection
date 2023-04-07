import carla_utils as cu
from carla_utils import carla, rl_template

from .params import num_steps, num_vehicles



class ScenarioUnstructured(rl_template.ScenarioSingleAgent):

    time_tolerance = num_steps

    # map_name = 'Town01'
    map_name = 'Town03'
    # map_name = 'Town01_Opt'
    # max_velocity = 6
    max_velocity = 8
    num_vehicles = num_vehicles
    num_static = 0 # 4
    obstacle_color = (255,255,255)
    type_id = 'vehicle.tesla.model3'
    obstacle_type_id = 'vehicle.*'
    # obstacle_type_id = 'vehicle.tesla.model3'


    def __init__(self, config):
        super().__init__(config)

        self.core.traffic_manager.global_percentage_speed_difference(80)
        # self.core.world.unload_map_layer(carla.MapLayer.Buildings)


    # def _generate_spawn_points(self):
    #     if self.map_name == 'Town01' or self.map_name == "Town01_Opt":
    #         spawn_points = [w.transform.location for w in self.town_map.generate_waypoints(20)]
    #         print(cu.basic.prefix(self) + 'Num of spawn_points: ', len(spawn_points))
    #     else: raise NotImplementedError
    #     return spawn_points


    @property
    def type_ids(self):
        _type_ids = [self.type_id] + [self.obstacle_type_id] * (self.num_vehicles-1)
        return _type_ids
    @property
    def colors(self):
        _colors = [(255,0,0)]
        _colors += [(0,0,255)] * self.num_static
        _colors += [self.obstacle_color] * (self.num_vehicles-1-self.num_static)
        return _colors
    @property
    def role_names(self):
        _role_names = [cu.Role(vi=0, atype=self.agent_role,)]
        _role_names += [cu.Role(vi=vi, atype=cu.ScenarioRole.static,) for vi in range(1, 1+self.num_static)]
        _role_names += [cu.Role(vi=vi, atype=cu.ScenarioRole.obstacle,) for vi in range(1+self.num_static, self.num_vehicles)]
        return _role_names


