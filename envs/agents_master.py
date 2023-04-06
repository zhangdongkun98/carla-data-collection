import carla_utils as cu

import numpy as np


class Agent(cu.BaseAgent):
    dim_action = 1

    def get_target(self, reference):
        return reference

    def get_control(self, target):
        return target

    def forward(self, control):
        return control




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
        state = self.perp.run_step(index, timestamp, self.agents_learnable)
        if timestamp > 0: self.perp.viz(state)
        return state.to_tensor().unsqueeze(0)

