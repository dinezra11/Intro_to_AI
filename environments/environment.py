import numpy as np
import yaml
from utils.colors_code import Style


class Environment:
    def __init__(self, yaml_path='environment_config.yaml'):
        try:
            self.step = 1
            self.evacuated_people = 0
            self.agents = []

            with open(yaml_path, 'r') as file:
                configs = yaml.safe_load(file)

            self.n_vertices = configs['vertices']['N']
            self.weights = -1 * np.ones((self.n_vertices, self.n_vertices)).astype(int)
            self.objects = [[] for i in range(self.n_vertices)]
            self.flooded_flag = [False for i in range(self.n_vertices)]

            # Parse objects from yaml
            for obj_list in configs['vertices']['objects']:
                obj_list = obj_list.split(',')
                vertex = int(obj_list[0]) - 1
                objects = obj_list[1:]

                self.objects[vertex] = objects

            # Parse edges
            for edge in configs['edges']:
                edge = edge.split(',')
                edge[0] = int(edge[0]) - 1
                edge[1] = int(edge[1]) - 1
                edge[2] = int(edge[2])
                self.weights[int(edge[0])][int(edge[1])] = edge[2]
                self.weights[int(edge[1])][int(edge[0])] = edge[2]

                if len(edge) > 3:
                    if edge[3] == 'F':
                        self.flooded_flag[edge[0]] = True
                    else:
                        raise ValueError('Error - 4th value of edge is invalid.')

        except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
            print(e)
            print(f"Error: The file {yaml_path} was not found or not readable. Creating a default environment.")
            # TODO: Default configs for env

    def log_environment(self):
        print(f'{Style.BLUE}Step {self.step}:{Style.RESET}')
        print('Number of vertices:', self.n_vertices)
        print('Objects in Vertices:')
        print(self.objects)
        print('Flooded flag:')
        print(self.flooded_flag)
        print('Weights:')
        for i in range(self.n_vertices):
            for j in range(self.n_vertices):
                if self.weights[i][j] == -1:
                    print(Style.RED, end='')
                print(self.weights[i][j], Style.RESET, end='\t')
            print()


Environment().log_environment()
