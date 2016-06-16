# coding=utf-8
from functools import reduce
from random import randint, random
from operator import add

# The size of the genetic string


class GeneticModel:
    """
        Something
    """
    def __init__(self, N, min, max, count):
        self.N = N
        self.min_ = min
        self.max_ = max
        self.count = count

    def individual(self):
        return [randint(self.min_, self.max_) for x in range(self.N)]

    def population(self):
        return [self.individual() for x in range(self.count)]

    def fitness(self, individual, targ):
        summ = reduce(add, individual, 0)
        return abs(targ - summ)

    def grade(self, targ, population):
        summed = reduce(add, (self.fitness(individual, targ) for individual in population), 0)
        return summed/(len(population) * 1.0)

    def breed(self, father, mother):
        return father[:int(self.N/2)] + mother[int(self.N/2):]

    def mutate(self, population):
        mutate_prob = 0.01
        for item in population:
            if mutate_prob > random():
                mutation_pos = randint(0, len(item) - 1)
                item[mutation_pos] = randint(min(item), max(item))

    def evolve(self, pop, target, retain=0.2, random_select=0.05, mutate=0.01):
        graded = [(self.fitness(x, target), x) for x in pop]
        graded = [x[1] for x in sorted(graded)]
        retain_len = int(len(graded)*retain)
        parents = graded[:retain_len]

        # genetic diversity
        for individual in graded[retain_len:]:
            if random_select > random():
                parents.append(individual)

        # mutation of some individuals
        self.mutate(parents)

        # crossover parents
        parents_len = len(parents)
        desired_len = len(pop) - parents_len
        children = []
        while len(children) < desired_len:
            father = randint(0, parents_len-1)
            mother = randint(0, parents_len-1)
            if father != mother:
                father = parents[father]
                mother = parents[mother]
                children.append(self.breed(father, mother))
        parents.extend(children)
        return parents

if __name__ == "__main__":
    generation = GeneticModel(6, 0, 100, 100)
    target = 371

    pop = generation.population()
    print("Population: ", pop, len(pop))

    fitness_history = [generation.grade(target, pop)]
    print("Fitness History", fitness_history)

    for i in range(100):
        pop = generation.evolve(pop, target)
        fitness_history.append(generation.grade(target, pop))

    print("Fitness History after evolve: ", fitness_history)