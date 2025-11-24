from environments.environment import Environment

env = Environment(yaml_path='environments/environment_config.yaml')
for i in range(10):
    env.log_environment()
    env.step()