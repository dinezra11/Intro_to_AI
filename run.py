from environments.environment import Environment
import os
os.system('')

env = Environment(yaml_path='environments/environment_minimax_large_config.yaml')
for i in range(100):
    env.log_environment()
    env.step()