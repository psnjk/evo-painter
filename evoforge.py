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
            x, y = np.random.randint(0, 64), np.random.randint(0, 64)
            color = np.random.randint(0, 256, (1, 1, 3), dtype=np.uint8)
            individual[y:(y + 1), x:(x + 1)] = color
            print('random pixel')
        if np.random.random() < 0.5:
            x, y = np.random.randint(0, 64), np.random.randint(0, 64)
            delta = np.random.randint(-30, 31, (1, 1, 3))
            individual[y, x] = np.clip(individual[y, x] + delta, 0, 255)
            print('random adjustment')
        if np.random.random() < 0.5:
            x1, y1 = np.random.randint(0, 64), np.random.randint(0, 64)
            x2, y2 = np.random.randint(0, 64), np.random.randint(0, 64)
            individual[x1, y1], individual[x2, y2] = individual[x2, y2].copy(), individual[x1, y1].copy()
            print('random swap')
        if np.random.random() < 0.5:
            x, y = np.random.randint(0, 64), np.random.randint(0, 64)
            individual[y, x] = np.mean([individual[y, x], individual[(y + 1) % 64, x], individual[y, (x + 1) % 64]],
                                       axis=0).astype(np.uint8)
            print('random blending')

        return individual

    def crossover(self, individual1, individual2):
        pass

    def show_individual(self, idx):
        img = Image.fromarray(self.population[idx], "RGB")
        img.show()
