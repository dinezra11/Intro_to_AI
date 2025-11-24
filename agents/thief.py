from agents.base_agent import BaseAgent
from utils.constants import Actions
from utils.greedy import dijkstra


class Thief(BaseAgent):
    def __init__(self, id, initial_position):
        super().__init__(id, initial_position)
        self.agent_type = 'Thief'
        self.is_rescuing = False

    def step(self, env):
        # Using Dijkstra's algorithm. TODO: Break tie of dijkstra by prefering the vertex with more people
        if self.cooldown > 0:
            self.cooldown -= 1
            return Actions.NO_OP, None

        if self.is_holding_amphibian:
            # TODO: Running away from other agents when equipping with the amphibian kit
            return Actions.NO_OP, None
        else:
            if env.check_amphibian_availability(self.position):
                return Actions.EQUIP, None

            # Get weights and filter edges that goes into flooded vertices.
            W = env.weights.copy()
            for i in range(env.n_vertices):
                for j in range(env.n_vertices):
                    if env.check_flooded(i, j):
                        W[i, j] = -1

            target_vertices = []
            for i, objects in enumerate(env.objects):
                for obj in objects:
                    if 'K' in obj:
                        target_vertices.append(i)
                        break

            distance, path = dijkstra(self.position, W, target_vertices)

            if len(path) > 1:
                return Actions.TRAVERSE, path[1]
            else:
                return Actions.NO_OP, None
