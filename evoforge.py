from random import random

from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np


# expand np array to 512 by 512, to represent an image.
# image = np.kron(grid, np.ones((8, 8, 1), dtype=np.uint8))

class Evolution:
    def __init__(self, population_size=20):
        self.population = []
        self.generation = 0
        self.population_size = 1
        self.image = None

    def open_image(self, path):
        image = Image.open(path).convert('RGB')
        image = ImageOps.contain(image, (512, 512))
        self.image = np.array(image)

    def create_population(self):
        for _ in range(self.population_size):
            individual = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            self.population.append(individual)

    # TODO ensure that at least one mutation is happening
    @staticmethod
    def mutate(individual):
        if np.random.random() < 0.5:
            col, row = np.random.randint(0, 64), np.random.randint(0, 64)
            color = np.random.randint(0, 256, (1, 1, 3), dtype=np.uint8)
            individual[row:(row + 1), col:(col + 1)] = color
            print('random pixel')
        if np.random.random() < 0.5:
            col, row = np.random.randint(0, 64), np.random.randint(0, 64)
            delta = np.random.randint(-30, 31, (1, 1, 3))
            individual[row, col] = np.clip(individual[row, col] + delta, 0, 255)
            print('random adjustment')
        if np.random.random() < 0.5:
            col1, row1 = np.random.randint(0, 64), np.random.randint(0, 64)
            col2, row2 = np.random.randint(0, 64), np.random.randint(0, 64)
            individual[row1, col1], individual[row2, col2] = individual[row2, col2].copy(), individual[row1, col1].copy()
            print('random swap')
        if np.random.random() < 0.5:
            col, row = np.random.randint(0, 64), np.random.randint(0, 64)
            individual[row, col] = np.mean(
                [individual[row, col], individual[(row + 1) % 64, col],
                 individual[row, (col + 1) % 64]], axis=0).astype(np.uint8)
            print('random blending')

        return individual

    def crossover(self, individual1, individual2):
        pass

    def show_individual(self, idx):
        img = Image.fromarray(self.population[idx], "RGB")
        img.show()
