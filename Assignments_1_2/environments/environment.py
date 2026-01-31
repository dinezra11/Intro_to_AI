import numpy as np
import yaml
from utils.constants import Style, Actions
from Assignments_1_2.utils.heuristic import precompute_distances

from Assignments_1_2.agents.human import Human
from agents.stupid_greedy import StupidGreedy
from agents.thief import Thief
from Assignments_1_2.agents.greedy_search import GreedySearch
from agents.a_star_search import AStarSearch
from agents.a_star_rt_search import RealTimeAStar
from agents.minimax_agent import MinimaxAgent


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
            self.objects = [[] for _ in range(self.n_vertices)]
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

            # optimistic distances (used by heuristics / evaluations)
            self.optimistic_dist = precompute_distances(self.weights)

            # ---------------------------------------------------------
            # Populate agents
            # ---------------------------------------------------------
            agent_classes = []
            for i, agent in enumerate(configs['agents']):
                agent_type, agent_initial_position = agent.split(',')

                if agent_type == 'human':
                    cls = Human
                elif agent_type == 'stupid-greedy':
                    cls = StupidGreedy
                elif agent_type == 'thief':
                    cls = Thief
                elif agent_type == 'greedy-search':
                    cls = GreedySearch
                elif agent_type == 'a-star':
                    cls = AStarSearch
                elif agent_type == 'a-star-rt':
                    cls = RealTimeAStar
                elif agent_type == 'minimax':
                    cls = MinimaxAgent
                else:
                    cls = Human

                agent_classes.append(cls)

                agent_initial_position = int(agent_initial_position)
                agent_id = i

                self.agents.append(cls(id=agent_id, initial_position=agent_initial_position))
                self.objects[agent_initial_position].append(f'Agent{agent_id}')

            # ---------------------------------------------------------
            # Turn-based mode for Assignment 2 (minimax game)
            # If ANY minimax agent exists -> enable turn-based stepping.
            # ---------------------------------------------------------
            self.turn_based = any(cls is MinimaxAgent for cls in agent_classes)
            self.turn = 0  # whose turn to act (0/1) when turn_based=True

        except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
            print(e)
            print(f"Error: The file {yaml_path} was not found or not readable. Creating a default environment.")
            # TODO: Default configs for env

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------
    def _apply_action(self, agent, action, info):
        """Apply an action chosen by an agent."""
        if action == Actions.NO_OP:
            print(f'{Style.MAGENTA}Agent {agent.id} took no action.{Style.RESET}')
            return

        if action == Actions.TRAVERSE:
            old_pos, new_pos = agent.position, info

            # If agent tried to traverse illegal edge, treat as NO_OP (robustness)
            if self.weights[old_pos][new_pos] == -1:
                print(f'{Style.MAGENTA}Agent {agent.id} tried illegal move {old_pos}->{new_pos} (NO_OP).{Style.RESET}')
                return

            self.objects[old_pos].remove(f'Agent{agent.id}')
            self.objects[new_pos].append(f'Agent{agent.id}')
            agent.position = new_pos

            if agent.is_holding_amphibian:
                action_cooldown = self.action_duration['amphibian'] * self.weights[old_pos][new_pos]
            else:
                action_cooldown = self.weights[old_pos][new_pos]

            agent.cooldown = action_cooldown - 1
            print(f'{Style.MAGENTA}Agent {agent.id} is moving from {old_pos} to {new_pos} '
                  f'(action duration is {action_cooldown} steps).{Style.RESET}')
            return

        if action == Actions.EQUIP:
            pos = agent.position
            # Only equip if a kit is available on ground at this vertex
            if 'K' in self.objects[pos]:
                self.objects[pos].remove('K')
                agent.is_holding_amphibian = True
                agent.cooldown = self.action_duration['equip'] - 1
                print(f'{Style.MAGENTA}Agent {agent.id} is equipping the amphibian kit '
                      f'(action duration is {self.action_duration["equip"]} steps).{Style.RESET}')
            else:
                print(f'{Style.MAGENTA}Agent {agent.id} tried EQUIP but no kit here (NO_OP).{Style.RESET}')
            return

        if action == Actions.UNEQUIP:
            pos = agent.position
            # Drop kit on ground
            self.objects[pos].append('K')
            agent.is_holding_amphibian = False
            agent.cooldown = self.action_duration['unequip'] - 1
            print(f'{Style.MAGENTA}Agent {agent.id} is unequipping the amphibian kit '
                  f'(action duration is {self.action_duration["unequip"]} steps).{Style.RESET}')
            return

    def _tick_cooldowns_and_rescue(self):
        """
        One unit of time passes in the world.
        IMPORTANT for turn-based mode: even the agent that did NOT act still spends time,
        so cooldowns must decrease for everyone.
        """
        for agent in self.agents:
            # cooldown counts remaining "busy" ticks
            if agent.cooldown > 0:
                agent.cooldown -= 1

            # If rescue completes now, apply it
            if getattr(agent, "is_rescuing", False) and agent.cooldown == 0:
                # remove ALL people objects at vertex (robust when multiple P exist)
                new_objs = []
                for obj in self.objects[agent.position]:
                    if isinstance(obj, str) and obj.startswith('P'):
                        rescued_amount = int(obj[1:])
                        agent.score += rescued_amount * 1000
                        agent.rescued_amount += rescued_amount
                        self.total_rescued_people += rescued_amount
                    else:
                        new_objs.append(obj)
                self.objects[agent.position] = new_objs

            # cost of time passing
            agent.score -= 1

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def step(self):
        """
        Advance the simulation by 1 time unit.

        - Assignment 1 mode (turn_based=False): all agents are queried each step (your original behavior).
        - Assignment 2 mode (turn_based=True): only the 'turn' agent is queried; the other agent just waits,
          but time still passes for BOTH (cooldowns decrease, score decreases, rescue completion happens).
        """
        if not self.turn_based:
            # Original behavior: everyone acts each tick
            for agent in self.agents:
                action, info = agent.step(env=self)
                self._apply_action(agent, action, info)

            # time passes for everyone
            self._tick_cooldowns_and_rescue()
            self.steps += 1
            return

        # Turn-based behavior: only one agent chooses an action
        acting_agent = self.agents[self.turn]

        # Let the acting agent choose (if it's busy, its step() should likely return NO_OP,
        # but we still allow it to decide; environment will apply NO_OP).
        action, info = acting_agent.step(env=self)
        self._apply_action(acting_agent, action, info)

        # time passes for everyone (including the non-acting agent!)
        self._tick_cooldowns_and_rescue()

        # switch turn and advance time counter
        self.turn = 1 - self.turn
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
        print(f'{Style.UNDERLINE}Total Rescued People:{Style.RESET}',
              f'{self.total_rescued_people}/{self.total_people_to_be_rescued}')
        print(f'{Style.UNDERLINE}Number of vertices:{Style.RESET}', self.n_vertices)
        print(f'{Style.UNDERLINE}Objects in Vertices:{Style.RESET}')
        print(self.objects)
        print(f'{Style.UNDERLINE}Weights:{Style.RESET}')
        for i in range(self.n_vertices):
            for j in range(self.n_vertices):
                if self.weights[i][j] == -1:
                    print(Style.RED, end='')
                elif self.flooded_flag[i][j]:
                    print(Style.BLUE, end='')
                print(self.weights[i][j], Style.RESET, end='\t')
            print()

        print(f'{Style.UNDERLINE}Agent States:{Style.RESET}')
        for agent in self.agents:
            agent.log()

        print(f'{Style.UNDERLINE}Agent Actions in Current Step:{Style.RESET}')
