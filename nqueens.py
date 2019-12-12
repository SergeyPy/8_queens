# -*- coding: utf-8 -*-
import numpy as np
import random


class Solver_8_queens:

    def __init__(self, cross_prob=0.4, mut_prob=0.3, pop_size=160):
        self.pop_size = pop_size
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob

    def initial_population(self):

        random.seed()
        popul = []
        index = []

        for i in range(self.pop_size):
            position_q = []
            individuals = np.zeros((8, 8)).astype(int)

            for i, row in enumerate(individuals):
                queen_pos = random.choices(np.arange(0, 8))
                position_q.append((i, queen_pos[0]))
                row[queen_pos] = 1

            index.append(position_q)
            popul.append(individuals.tolist())

        return popul, index

    def fitness(self, index_pop):
        valued_chroms = []
        count_h = 0

        for index in index_pop:
            count = 0
            intersection = 0
            while count != 8:
                i = index.pop(0)
                hits = 0

                for j in index:
                    if abs(i[0] - j[0]) == 0 or abs(i[1] - j[1]) == 0 or abs(i[0] - j[0]) == abs(i[1] - j[1]):
                        hits = 1
                        break

                if hits == 0:
                    intersection += 1

                index.append(i)
                count += 1

            valued_chroms.append((count_h, intersection))
            count_h += 1

        return valued_chroms

    def roulette_select(self, fit, population):

        random.seed()
        prob = 0
        probs = []
        total_fitness = sum([el[1] for el in fit])
        rel_fitness = [f[1] / total_fitness for f in fit]
        probs = [sum(rel_fitness[:i + 1]) for i in range(len(rel_fitness))]

        new_population = []

        for n in np.arange(self.cross_prob * self.pop_size):
            r = random.random()
            for (i, individual) in enumerate(population):
                if r <= probs[i]:
                    new_population.append(individual)
                    break

        return new_population

    def crossover(self, mating):
        random.seed()
        childrens = []
        while len(mating) > 0:
            descendant_1, descendant_2 = [], []
            rand_1 = random.choices(np.arange(0, len(mating) - 1))
            rand_2 = random.choices(np.arange(0, len(mating) - 1))
            if rand_1 > rand_2:
                ancestor_1 = mating.pop(rand_1[0])
                ancestor_2 = mating.pop(rand_2[0])
            elif len(mating) == 2:
                ancestor_1 = mating.pop(0)
                ancestor_2 = mating.pop(0)
            else:
                ancestor_2 = mating.pop(rand_2[0])
                ancestor_1 = mating.pop(rand_1[0])

            cross_index = random.choices(np.arange(1, 8))

            descendant_1.extend(ancestor_1[0:cross_index[0]])
            descendant_1.extend(ancestor_2[cross_index[0]:])

            descendant_2.extend(ancestor_2[0:cross_index[0]])
            descendant_2.extend(ancestor_1[cross_index[0]:])

            childrens.append(descendant_1)
            childrens.append(descendant_2)

        return childrens

    def mutate(self, children):
        random.seed()

        for child in children:

            if random.random() < self.mut_prob:
                i = random.randrange(8)
                ind = child[i].index(1)
                child[i][ind] = 0

                j = random.randrange(8)
                child[i][j] = 1

    def reduction(self, fit):
        to_reduce = []
        for i in range(int(self.pop_size * self.cross_prob)):
            to_reduce.append(fit[i][0])
        return to_reduce

    def next_pop(self, red, population, childrens):
        for el in red:
            population[el] = childrens.pop(0)
        return population

    def solve(self, min_fitness=7, max_epochs=100):
        population, index = self.initial_population()
        epochs = 0
        max_fitness = 0

        while epochs < max_epochs and max_fitness <= min_fitness:
            fit = self.fitness(index)
            fit.sort(key=lambda tup: tup[1])

            mating = self.roulette_select(fit, population)
            childrens = self.crossover(mating)
            self.mutate(childrens)

            red = self.reduction(fit)
            population = self.next_pop(red, population, childrens)
            epochs += 1

        visualization = ""
        for el in population[int(fit[-1][0])]:
            for num in el:
                if num == 1:
                    visualization += "Q"
                else:
                    visualization += "+"
            visualization += "\n"

        return fit[-1][0],fit[-1][1], epochs, visualization