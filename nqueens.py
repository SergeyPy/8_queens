#!/usr/bin/env python
# coding: utf-8

# In[2855]:


# -*- coding: utf-8 -*-

import random
import numpy as np
import copy
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

            for j, row in enumerate(individuals):
                queen_pos = random.choices(np.arange(0, 8))[0]
                row[queen_pos] = 1

            popul.append(individuals.tolist())

        return popul

    def hits(self, index):
        intersection = 0
        count = 0

        while count != 8:
            i = index.pop(0)
            hits = 0

            for j in range(len(index)):
                if (i[0] - index[j][0]) == 0 or (i[1] - index[j][1]) == 0 or abs(index[j][0] - i[0]) == abs(
                        index[j][1] - i[1]):
                    hits = 1
                    break

            if not hits:
                intersection += 1

            index.append(i)
            count += 1

        return intersection

    def pos(self, ch):
        index = []
        for i in np.arange(0, 8):
            index.append((i, ch[i].index(1)))

        return index

    def fitness(self, pop):
        valued_chroms = []
        count_h = 0

        for ch in pop:
            index = self.pos(ch)
            intersection = self.hits(index)
            valued_chroms.append((count_h, intersection))
            count_h += 1

        return valued_chroms

    def roulette(self, fit):
        random.seed()


        mating = []
        total_fitness = sum([el[1] for el in fit])

        rel_fitness = [f[1] / total_fitness for f in fit]
        probs = [sum(rel_fitness[:i + 1]) for i in range(len(rel_fitness))]

        for i in range(int(self.pop_size * self.cross_prob)):
            r = random.random()
            for prob in probs:
                if prob > r:
                    ind = probs.index(prob)
                    mating.append(fit[ind][0])
                    break

        return mating

    def crossover(self, mating):
        random.seed()
        childrens = []
        while len(mating) > 0:
            descendant_1, descendant_2 = [], []
            rand_1 = random.choices(np.arange(0, len(mating) - 1))
            rand_2 = random.choices(np.arange(0, len(mating) - 1))

            if len(mating) == 2:
                ancestor_1 = mating.pop(rand_1[0])
                ancestor_2 = mating.pop(rand_2[0])
            else:
                ancestor_1 = mating.pop(0)
                ancestor_2 = mating.pop(0)

            cross_index = random.choices(np.arange(1, 8))

            descendant_1.extend(ancestor_1[0:cross_index[0]])
            descendant_1.extend(ancestor_2[cross_index[0]:])

            descendant_2.extend(ancestor_1[0:cross_index[0]])
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
        population = self.initial_population()
        epochs = 0
        max_fitness = 0

        while epochs < max_epochs and max_fitness <= min_fitness:

            fit = self.fitness(population)
            fit.sort(key=lambda tup: tup[1])

            mating = self.roulette(fit)
            mating_chromes = []

            pop = copy.deepcopy(population)
            for chrom in mating:
                mating_chromes.append(pop[chrom])
            pop.clear()

            childrens = self.crossover(mating_chromes)
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

        return fit[-1][1], epochs, visualization




