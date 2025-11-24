from utils.constants import Actions, ACTIONS_STR


class Human:
    def __init__(self, id, initial_position=0):
        self.id = id
        self.position = initial_position
        self.item_hold = None
        self.score = 0
        self.cooldown = 0

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

                if go_to in valid_moves:
                    return Actions.TRAVERSE, go_to
                else:
                    print(f'Illegal move.')
                    continue
            if action == Actions.EQUIP:
                pass
            if action == Actions.UNEQUIP:
                pass

            return action, None

    def log(self):
        log = f'Human Agent (ID {self.id}), Current Position: {self.position}, Score: {self.score}. '
        if self.cooldown > 0:
            log += f'Agent is currently in action. ({self.cooldown} steps to finish)'
        print(log)


