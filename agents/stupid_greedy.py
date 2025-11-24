from agents.base_agent import BaseAgent
from utils.constants import Style, Actions, ACTIONS_STR


class StupidGreedy(BaseAgent):
    def __init__(self, id, initial_position):
        super().__init__(id, initial_position)
        self.agent_type = 'Stupid-Greedy'

    @staticmethod
    def compute_shortest_paths(self, weights, objects):
        pass

    def step(self, env):
        if self.cooldown > 0:
            self.cooldown -= 1
            return Actions.NO_OP, None

        return Actions.NO_OP, None
