import numpy as np
import yaml
from utils.constants import Style, Actions
from agents.human import Human


class Environment:
    def __init__(self, yaml_path):
        try:
            self.steps = 1
            self.evacuated_people = 0
            self.agents = []

            with open(yaml_path, 'r') as file:
                configs = yaml.safe_load(file)

            self.n_vertices = configs['vertices']['N']
            self.weights = -1 * np.ones((self.n_vertices, self.n_vertices)).astype(int)
            self.objects = [[] for i in range(self.n_vertices)]
            self.flooded_flag = [False for i in range(self.n_vertices)]
            self.action_duration = configs['action_duration']

            # Parse objects from yaml
            for obj_list in configs['vertices']['objects']:
                obj_list = obj_list.split(',')
                vertex = int(obj_list[0])
                objects = obj_list[1:]

                self.objects[vertex] = objects

            # Parse edges
            for edge in configs['edges']:
                edge = edge.split(',')
                edge[0] = int(edge[0])
                edge[1] = int(edge[1])
                edge[2] = int(edge[2])
                self.weights[edge[0]][edge[1]] = edge[2]
                self.weights[edge[1]][edge[0]] = edge[2]

                if len(edge) > 3:
                    if edge[3] == 'F':
                        self.flooded_flag[edge[0]] = True
                    else:
                        raise ValueError('Error - 4th value of edge is invalid.')

            # Populate agents
            self.agents.append(Human(id=1, initial_position=0))
            self.objects[0].append('Agent1')

        except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
            print(e)
            print(f"Error: The file {yaml_path} was not found or not readable. Creating a default environment.")
            # TODO: Default configs for env

    def step(self):
        for agent in self.agents:
            action, info = agent.step(env=self)

            if action == Actions.NO_OP:
                print(f'{Style.MAGENTA}Agent {agent.id} took no action.')
            elif action == Actions.TRAVERSE:
                old_pos, new_pos = agent.position, info
                self.objects[old_pos].remove(f'Agent{agent.id}')
                self.objects[new_pos].append(f'Agent{agent.id}')
                agent.position = new_pos
                if agent.is_holding_amphibian: # if it holds the amphibian kit
                    agent.cooldown = self.action_duration['amphibian'] * self.weights[old_pos][new_pos] - 1
                else:
                    agent.cooldown = self.weights[old_pos][new_pos] - 1

                print(f'{Style.MAGENTA}Agent {agent.id} is moving from {old_pos} to {new_pos}.{Style.RESET}')
            elif action == Actions.EQUIP:
                pos = agent.position
                self.objects[pos].remove('K')
                agent.is_holding_amphibian = True
                agent.cooldown = self.action_duration['equip'] - 1
            elif action == Actions.UNEQUIP:
                pos = agent.position
                self.objects[pos].append('K')
                agent.is_holding_amphibian = False
                agent.cooldown = self.action_duration['unequip'] - 1


        self.steps += 1

    def get_adjacent_vertices(self, vertex):
        adjacents = []

        for i in range(self.n_vertices):
            if i != vertex and self.weights[vertex][i] > 0:
                adjacents.append(i)

        return adjacents

    def check_flooded(self, vertex):
        return self.flooded_flag[vertex]

    def check_amphibian_availability(self, vertex):
        return 'K' in self.objects[vertex]

    def log_environment(self):
        print()
        print(f'{Style.BLUE}Step {self.steps}:{Style.RESET}')
        print(f'{Style.UNDERLINE}Number of vertices:{Style.RESET}', self.n_vertices)
        print(f'{Style.UNDERLINE}Objects in Vertices:{Style.RESET}')
        print(self.objects)
        print(f'{Style.UNDERLINE}Flooded flags:{Style.RESET}')
        print(self.flooded_flag)
        print(f'{Style.UNDERLINE}Weights:{Style.RESET}')
        for i in range(self.n_vertices):
            for j in range(self.n_vertices):
                if self.weights[i][j] == -1:
                    print(Style.RED, end='')
                print(self.weights[i][j], Style.RESET, end='\t')
            print()

        print(f'{Style.UNDERLINE}Agent States:{Style.RESET}')
        for agent in self.agents:
            agent.log()

        print(f'{Style.UNDERLINE}Agent Actions in Current Step:{Style.RESET}')
