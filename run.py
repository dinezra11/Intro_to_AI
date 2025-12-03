from environments.environment import Environment

env = Environment(yaml_path='environments/example_from_chat.yaml')
for i in range(7):
    env.log_environment()
    env.step()