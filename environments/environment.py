import numpy as np
import yaml
from utils.constants import Style, Actions
from utils.heuristic import precompute_distances
from agents.human import Human
from agents.stupid_greedy import StupidGreedy
from agents.thief import Thief


class Environment:
    def __init__(self, yaml_path):
        try:
            self.steps = 1
            self.total_rescued_people = 0
            self.total_people_to_be_rescued = 0
            self.agents = []

            with open(yaml_path, 'r') as file:
                configs = yaml.safe_load(file)

            self.n_vertices = configs['vertices']['N']
            self.weights = -1 * np.ones((self.n_vertices, self.n_vertices)).astype(int)
            self.objects = [[] for i in range(self.n_vertices)]
            self.flooded_flag = np.zeros((self.n_vertices, self.n_vertices)).astype(bool)
            self.action_duration = configs['action_duration']

            # Parse objects from yaml
            for obj_list in configs['vertices']['objects']:
                obj_list = obj_list.split(',')
                vertex = int(obj_list[0])
                objects = obj_list[1:]

                self.objects[vertex] = objects

                # Count total people that needs to be rescued
                for obj in objects:
                    if 'P' in obj:
                        self.total_people_to_be_rescued += int(obj[1:])

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
                        self.flooded_flag[edge[0]][edge[1]] = True
                        self.flooded_flag[edge[1]][edge[0]] = True
                    else:
                        raise ValueError('Error - 4th value of edge is invalid.')

            # set the optimistic distances matrix, which will be used by the heuristic
            self.optimistic_dist = precompute_distances(self.weights)

            # Populate agents
            for i, agent in enumerate(configs['agents']):
                agent_type, agent_initial_position = agent.split(',')
                if agent_type == 'human':
                    agent_type = Human
                elif agent_type == 'stupid-greedy':
                    agent_type = StupidGreedy
                elif agent_type == 'thief':
                    agent_type = Thief
                else: # Default
                    agent_type = Human

                agent_initial_position = int(agent_initial_position)
                agent_id = i + 1

                self.agents.append(agent_type(id=agent_id, initial_position=agent_initial_position))
                self.objects[agent_initial_position].append(f'Agent{agent_id}')

        except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
            print(e)
            print(f"Error: The file {yaml_path} was not found or not readable. Creating a default environment.")
            # TODO: Default configs for env

    def step(self):
        for agent in self.agents:
            # Action handling
            action, info = agent.step(env=self)

            if action == Actions.NO_OP:
                print(f'{Style.MAGENTA}Agent {agent.id} took no action.{Style.RESET}')
            elif action == Actions.TRAVERSE:
                old_pos, new_pos = agent.position, info
                self.objects[old_pos].remove(f'Agent{agent.id}')
                self.objects[new_pos].append(f'Agent{agent.id}')
                agent.position = new_pos
                if agent.is_holding_amphibian: # if it holds the amphibian kit
                    action_cooldown = self.action_duration['amphibian'] * self.weights[old_pos][new_pos]
                else:
                    action_cooldown = self.weights[old_pos][new_pos]
                agent.cooldown = action_cooldown - 1

                print(f'{Style.MAGENTA}Agent {agent.id} is moving from {old_pos} to {new_pos} (action duration is {action_cooldown} steps).{Style.RESET}')
            elif action == Actions.EQUIP:
                pos = agent.position
                self.objects[pos].remove('K')
                agent.is_holding_amphibian = True
                agent.cooldown = self.action_duration['equip'] - 1

                print(f'{Style.MAGENTA}Agent {agent.id} is equipping the amphibian kit (action duration is {self.action_duration['equip']} steps).{Style.RESET}')
            elif action == Actions.UNEQUIP:
                pos = agent.position
                self.objects[pos].append('K')
                agent.is_holding_amphibian = False
                agent.cooldown = self.action_duration['unequip'] - 1

                print(f'{Style.MAGENTA}Agent {agent.id} is unequipping the amphibian kit (action duration is {self.action_duration['unequip']} steps).{Style.RESET}')

            # Reward handling
            if agent.is_rescuing and agent.cooldown == 0:
                for i, object in enumerate(self.objects[agent.position]):
                    if 'P' in object:
                        rescued_amount = int(object[1:])
                        new_score = rescued_amount * 1000
                        agent.score += new_score
                        agent.rescued_amount += rescued_amount
                        self.total_rescued_people += rescued_amount
                        self.objects[agent.position].pop(i)

            agent.score -= 1 # Drop 1 point for each step


        self.steps += 1

    def get_adjacent_vertices(self, vertex):
        adjacents = []

        for i in range(self.n_vertices):
            if i != vertex and self.weights[vertex][i] > 0:
                adjacents.append(i)

        return adjacents

    def check_flooded(self, vertex_from, vertex_to):
        return self.flooded_flag[vertex_from][vertex_to]

    def check_amphibian_availability(self, vertex):
        return 'K' in self.objects[vertex]

    def log_environment(self):
        print()
        print(f'{Style.CYAN}Step {self.steps}:{Style.RESET}')
        print(f'{Style.UNDERLINE}Total Rescued People:{Style.RESET}', f'{self.total_rescued_people}/{self.total_people_to_be_rescued}')
        print(f'{Style.UNDERLINE}Number of vertices:{Style.RESET}', self.n_vertices)
        print(f'{Style.UNDERLINE}Objects in Vertices:{Style.RESET}')
        print(self.objects)
        print(f'{Style.UNDERLINE}Weights:{Style.RESET}')
        for i in range(self.n_vertices):
            for j in range(self.n_vertices):
                if self.weights[i][j] == -1: # Edge doesn't exist - mark as red
                    print(Style.RED, end='')
                elif self.flooded_flag[i][j]: # Edge is flooded - mark as blue
                    print(Style.BLUE, end='')
                print(self.weights[i][j], Style.RESET, end='\t')
            print()

        print(f'{Style.UNDERLINE}Agent States:{Style.RESET}')
        for agent in self.agents:
            agent.log()

        print(f'{Style.UNDERLINE}Agent Actions in Current Step:{Style.RESET}')
