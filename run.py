from environments.environment import Environment
import os
os.system('')

env = Environment(yaml_path='environments/environment_minimax_config.yaml')
for i in range(7):
    env.log_environment()
    env.step()