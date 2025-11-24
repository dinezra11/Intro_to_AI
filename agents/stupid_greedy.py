from agents.base_agent import BaseAgent
from utils.constants import Actions
from utils.greedy import dijkstra


class StupidGreedy(BaseAgent):
    def __init__(self, id, initial_position):
        super().__init__(id, initial_position)
        self.agent_type = 'Stupid-Greedy'

    def step(self, env):
        # Using Dijkstra's algorithm. TODO: Break tie of dijkstra by prefering the vertex with more people
        if self.cooldown > 0:
            self.cooldown -= 1
            return Actions.NO_OP, None

        # Get weights and filter edges that goes into flooded vertices. TODO: Account of flooded vertices and amphibian kit collecting. For now the function ignore it and assumes that there are no flooded vertices.
        W = env.weights.copy()
        for i in range(env.n_vertices):
            for j in range(env.n_vertices):
                if env.check_flooded(i, j):
                    W[i, j] = -1

        target_vertices = []
        for i, objects in enumerate(env.objects):
            for obj in objects:
                if 'P' in obj:
                    target_vertices.append(i)
                    break

        distance, path = dijkstra(self.position, W, target_vertices)

        if len(path) > 1:
            return Actions.TRAVERSE, path[1]
        else:
            return Actions.NO_OP, None
