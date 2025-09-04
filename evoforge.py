from random import random

from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from skimage.color import rgb2lab
from enum import Enum, auto


# expand np array to 512 by 512, to represent an image.
# image = np.kron(grid, np.ones((8, 8, 1), dtype=np.uint8))

class EvoTools:
    @staticmethod
    def mae_fitness(individual, target):
        return np.mean(np.abs(individual.astype(np.int32) - target.astype(np.int32)))

    @staticmethod
    def mse_fitness(individual, target):
        return np.mean((individual.astype(np.int32) - target.astype(np.int32)) ** 2)

    @staticmethod
    def delta_fitness(individual, target):
        individual_lab = rgb2lab(individual / 255.0)
        target_lab = rgb2lab(target / 255.0)
        return np.mean(np.linalg.norm(individual_lab - target_lab, axis=2))

    @staticmethod
    def evaluate(population, target, func=mae_fitness):
        return [func(individual, target) for individual in population]


class MosaicEvolution:
    def __init__(self, population_size, elite_proportion, block_size):
        self.population = []
        self.generation = 0
        self.population_size = population_size
        self.image = None
        self.elite_proportion = elite_proportion
        self.pooled_image = None
        self.block_size = block_size
        self.image_size = 512 // block_size

    def open_image(self, image):
        self.image = np.array(ImageOps.contain(image.convert('RGB'), (512, 512)))

        h, w, c = self.image.shape
        self.pooled_image = self.image.reshape(h // self.block_size, self.block_size, w // self.block_size,
                                               self.block_size, c).mean(axis=(1, 3)).astype(np.uint8)

    def reset_population(self):
        self.generation = 0
        for _ in range(self.population_size):
            individual = np.random.randint(0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8)
            self.population.append(individual)

    def mutate(self, individual):
        mutation = np.random.randint(4)
        if mutation == 0 or np.random.random() < 0.5:
            col, row = np.random.randint(0, self.image_size), np.random.randint(0, self.image_size)
            color = np.random.randint(0, 256, (1, 1, 3), dtype=np.uint8)
            individual[row:(row + 1), col:(col + 1)] = color
            # print('random pixel')
        if mutation == 1 or np.random.random() < 0.5:
            col, row = np.random.randint(0, self.image_size), np.random.randint(0, self.image_size)
            delta = np.random.randint(-30, 31, (1, 1, 3))
            individual[row, col] = np.clip(individual[row, col] + delta, 0, 255)
            # print('random adjustment')
        if mutation == 2 or np.random.random() < 0.5:
            col1, row1 = np.random.randint(0, self.image_size), np.random.randint(0, self.image_size)
            col2, row2 = np.random.randint(0, self.image_size), np.random.randint(0, self.image_size)
            individual[row1, col1], individual[row2, col2] = individual[row2, col2].copy(), individual[
                row1, col1].copy()
            # print('random swap')
        if mutation == 3 or np.random.random() < 0.5:
            col, row = np.random.randint(0, self.image_size), np.random.randint(0, self.image_size)
            individual[row, col] = np.mean(
                [individual[row, col], individual[(row + 1) % self.image_size, col],
                 individual[row, (col + 1) % self.image_size]], axis=0).astype(np.uint8)
            # print('random blending')

        return individual

    def crossover(self, individual1, individual2):
        crossover_chance = np.random.random()
        if crossover_chance < 0.1:
            alpha = np.random.rand()
            return (alpha * individual1 + (1 - alpha) * individual2).astype(np.uint8)
        elif crossover_chance < 0.3:
            if np.random.random() < 0.5:
                mask = np.random.randint(0, 2, (self.image_size, 1, 1), dtype=bool)
                return np.where(mask, individual1, individual2)
            else:
                mask = np.random.randint(0, 2, (1, self.image_size, 1), dtype=bool)
                return np.where(mask, individual1, individual2)
        elif crossover_chance < 0.6:
            block_size = 16
            y = np.random.randint(0, self.image_size - block_size + 1)
            x = np.random.randint(0, self.image_size - block_size + 1)
            child = individual2.copy()
            child[y:y + block_size, x:x + block_size] = individual1[y:y + block_size, x:x + block_size]
            return child
        else:
            mask = np.random.randint(0, 2, (self.image_size, self.image_size, 1), dtype=bool)
            return np.where(mask, individual1, individual2)

    def select_parent(self, fitness, k=3):
        idx = np.random.choice(len(self.population), k, replace=False)
        best_idx = min(idx, key=lambda i: fitness[i])
        return self.population[best_idx]

    def show_individual(self, idx):
        img = Image.fromarray(self.population[idx], "RGB")
        img.show()

    def save_individual(self, individual, filename):
        image = np.kron(individual, np.ones((8, 8, 1), dtype=np.uint8))
        image = Image.fromarray(image, "RGB")
        image.save(filename)

    def select_best(self):
        fitness = EvoTools.evaluate(self.population, self.pooled_image)
        best_index = np.argsort(fitness)[:1][0]
        return self.population[best_index]

    def step(self):

        fitness = EvoTools.evaluate(self.population, self.pooled_image, EvoTools.mae_fitness)

        elite_indices = np.argsort(fitness)[:int(self.population_size * self.elite_proportion)]
        # print('Generation {}'.format(self.generation), 'with fitness', fitness[elite_indices[0]])
        new_population = [self.population[i].copy() for i in elite_indices]

        while len(new_population) < self.population_size:
            parent1 = self.select_parent(fitness)
            parent2 = self.select_parent(fitness)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population
        self.generation += 1

        return fitness[elite_indices[0]]


class Algorithm(Enum):
    MOSAIC = auto()
    TEXT = auto()
    FRACTAL = auto()


class Forge:
    def __init__(self, algorithm: Algorithm = Algorithm.MOSAIC, **kwargs):
        self.best_fit = 0
        self.running = False
        self.paused = False
        self.algorithm = algorithm
        if self.algorithm == Algorithm.MOSAIC:
            if 'population_size' in kwargs:
                population_size = kwargs['population_size']
            else:
                population_size = 20

            if 'elite_proportion' in kwargs:
                elite_proportion = kwargs['elite_proportion']
            else:
                elite_proportion = 0.5

            if 'block_size' in kwargs:
                block_size = kwargs['block_size']
            else:
                block_size = 8
            self.evo = MosaicEvolution(population_size, elite_proportion, block_size)

        elif self.algorithm == Algorithm.TEXT:
            raise NotImplementedError
        elif self.algorithm == Algorithm.FRACTAL:
            raise NotImplementedError

    def run(self):
        self.running = True

    def open_image(self, target: Image.Image):
        self.evo.open_image(target)

    def pause(self):
        self.paused = True

    def stop(self):
        self.running = False

    def unpause(self):
        self.paused = False

    def get_generation(self):
        return self.evo.generation

    def reset_algorithm(self):
        self.evo.reset_population()

    def get_best_fit(self):
        return self.best_fit

    def get_best(self):
        grid = self.evo.select_best()
        image = np.kron(grid, np.ones((8, 8, 1), dtype=np.uint8))
        return Image.fromarray(image)

    def loop(self):
        while True:
            if self.running:
                if not self.paused:
                    self.best_fit = self.evo.step()