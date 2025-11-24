from utils.constants import Style


class BaseAgent:
    def __init__(self, id, initial_position):
        self.id = id
        self.position = initial_position
        self.is_holding_amphibian = False
        self.score = 0
        self.rescued_amount = 0
        self.cooldown = 0

        self.agent_type = '??'

    def step(self, env):
        pass

    def log(self):
        log = f'{self.agent_type} Agent (ID {self.id}), Current Position: {self.position}, {Style.YELLOW}Score: {self.score}, rescued {self.rescued_amount} people{Style.RESET} '
        if self.is_holding_amphibian:
            log += ' | With Amphibian Kit.'
        if self.cooldown > 0:
            log += f' | {Style.RED}Agent is currently in action. ({self.cooldown} steps left to finish){Style.RESET}'
        else:
            log += f' | {Style.GREEN}Agent is ready to take an action.{Style.RESET}'
        print(log)
