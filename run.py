from environments.environment import Environment

env = Environment(yaml_path='environments/environment_config.yaml')
for i in range(100):
    env.log_environment()
    env.step()