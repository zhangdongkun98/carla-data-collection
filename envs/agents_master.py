import carla_utils as cu



class Agent(cu.BaseAgent):
    dim_action = 1

    def __init__(self, config, vehicle, sensors_master, global_path):
        super().__init__(config, vehicle, sensors_master, global_path)
        vehicle.set_autopilot(True, self.core.tm_port)

    def get_target(self, reference):
        return reference

    def get_control(self, target):
        return target

    def forward(self, control):
        return control

    def extend_route(self):
        return



class AgentListMaster(cu.AgentListMaster):
    from .perception import PerceptionSemanticLidar as Perception
    agent_type = Agent

    dim_state = Perception.dim_state
    dim_action = agent_type.dim_action

    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.num_vehicles = config.num_vehicles
        self.dim_state = config.dim_state
        self.max_velocity = config.max_velocity
        self.perception_range = config.perception_range

        self.perp = self.Perception(config)


    def get_agent_type(self, learnable=True):
        if learnable:
            agent_type = self.agent_type
        else:
            raise NotImplementedError
        return agent_type
    

    def perception(self, index, timestamp):
        obstacles = self.agents[1:] + self.obstacles
        state = self.perp.run_step(index, timestamp, self.agents_learnable[0], obstacles)
        return state


    def visualize(self, state):
        self.perp.visualize(state.bbxs, state.points)



    def destroy(self):
        super().destroy()
        self.perp.destroy()

