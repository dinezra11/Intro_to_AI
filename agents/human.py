from utils.constants import Style, Actions, ACTIONS_STR
from agents.base_agent import BaseAgent


class Human(BaseAgent):
    def __init__(self, id, initial_position):
        super().__init__(id, initial_position)
        self.agent_type = 'Human'

    def step(self, env):
        if self.cooldown > 0:
            self.cooldown -= 1
            return Actions.NO_OP, None

        while True:
            action = int(input(f'Input action for Agent {self.id} ({ACTIONS_STR}):'))

            if action == Actions.NO_OP:
                return Actions.NO_OP, None
            if action == Actions.TRAVERSE:
                valid_moves = env.get_adjacent_vertices(self.position)
                go_to = int(input(f'Where to? Available options: {valid_moves}'))

                if go_to not in valid_moves:
                    print(f'Illegal move.')
                    continue
                if env.check_flooded(go_to) and not self.is_holding_amphibian:
                    print('Illegal move. Agent must have amphibian kit to move across flooded vertex.')
                    continue

                return Actions.TRAVERSE, go_to

            if action == Actions.EQUIP:
                if not env.check_amphibian_availability(self.position):
                    print("Illegal move. No amphibian kit in this position.")
                    continue

                return Actions.EQUIP, None

            if action == Actions.UNEQUIP:
                if not self.is_holding_amphibian:
                    print("Illegal move. Agent doesn't hold amphibian kit.")
                    continue

                return Actions.UNEQUIP, None

            return action, None
