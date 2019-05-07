#!/usr/bin/env python

"""
__author__ = "Lech Szymanski"
__copyright__ = "Copyright 2019, COSC343"
__license__ = "GPL"
__version__ = "2.0.1"
__maintainer__ = "Lech Szymanski"
__email__ = "lechszym@cs.otago.ac.nz"
"""

from cosc343world import Creature, World
import numpy as np
import time
import matplotlib.pyplot as plt
import random

# You can change this number to specify how many generations creatures are going to evolve over...
numGenerations = 1

# You can change this number to specify how many turns in simulation of the world for given generation
numTurns = 100

# You can change this number to change the world type.  You have two choices - world 1 or 2 (described in
# the assignment 2 pdf document)
worldType=1

# You can change this number to change the world size
gridSize=24

# You can set this mode to True to have same initial conditions for each simulation in each generation.  Good
# for development, when you want to have some determinism in how the world runs from generatin to generation.
repeatableMode=True

# This is a class implementing you creature a.k.a MyCreature.  It extends the basic Creature, which provides the
# basic functionality of the creature for the world simulation.  Your job is to implement the AgentFunction
# that controls creature's behavoiur by producing actions in respons to percepts.
class MyCreature(Creature):

    # Initialisation function.  This is where you creature
    # should be initialised with a chromosome in random state.  You need to decide the format of your
    # chromosome and the model that it's going to give rise to
    #
    # Input: numPercepts - the size of percepts list that creature will receive in each turn
    #        numActions - the size of actions list that creature must create on each turn
    def __init__(self, numPercepts, numActions):
        
        # Chromosome is a numActions X numPercepts matrix. Each element in the matrix acts a weight
        self.chromosome = np.random.normal(0, 1, size = (numActions, numPercepts))


        # Do not remove this line at the end.  It calls constructors
        # of the parent classes.
        Creature.__init__(self)


    # This is the implementation of the agent function that is called on every turn, giving your
    # creature a chance to perform an action.  You need to implement a model here, that takes its parameters
    # from the chromosome and it produces a set of actions from provided percepts
    #
    # Input: percepts - a list of percepts
    #        numAction - the size of the actions list that needs to be returned
    def AgentFunction(self, percepts, numActions):

        
        # Recoding percepts so a linear model makes sense
        # 0 = monster, 1 = other creature, 2 = empty, 3 = strawberry        
        zeros = np.where(percepts == 0)
        ones = np.where(percepts == 1)
        twos = np.where(percepts == 2)
            
        
        percepts[zeros] = 2
        percepts[ones] = 0
        percepts[twos] = 1
        
        # Calculating all weights        
        action_weights = np.array(percepts) * self.chromosome
        
        # There is one row coding for each action, taking row sums essentially works as a linear model
        actions  = action_weights.sum(axis=1)
                
        #actions = np.random.uniform(0, 1, size=numActions)
        return actions.tolist()


# Simple fitness function: If the individual survived fitness is arbitrarily high (500), if death = time of death
def fitness_func(individual):
    if individual.isDead() == False:
        fitness = 500
    else:
        fitness = individual.timeOfDeath()
    
    return(fitness)

# Function for tournament selection
def tournament(old_population, tournament_size):
    # Choosing n random individuals
    tour = random.sample(population = old_population, k = tournament_size)
    fitness = []  
    
    # Calculating fitness for each individual
    for i in range(len(tour)):
        fitness.append(fitness_func(i))
       
    
    
    # Finding the indices of the two fittest individuals
    indices = np.argpartition(fitness, -2)[-2:]
    indices = indices.tolist()
    
    winners = [tour[i] for i in indices]
    
    winners = tour[indices]
   
    
    
    return(winners)


def crossover(pair, mutation):
    
    # Making an empty chromosome to fill with parent values
    child_chromosome  = np.zeros(pair[0].chromosome.shape)
    
    p1_chromosome = pair[0].chromosome
    p2_chromosome = pair[1].chromosome
    
    # Random point crossover
    #cross_point = np.random.randint(pair.shape[1])
    # Middle point crossover (4 first columns from p1, 5 last from p2)
    cross_point = int(pair.shape[1] / 2)
    
    # Creating child
    # First half from first parent, second half from second parent
    for i in range(child_chromosome.shape[0]):        
        child_chromosome[i, 0:cross_point] = p1_chromosome[i, 0:cross_point]
        child_chromosome[i, cross_point:] = p2_chromosome[i, cross_point:]
    
    # Mutation 
    # Method 1: mutate 1 gene at a certain rate    
    if np.random.uniform(0, 1, 1)[0] <= mutation:
        
        
        # choose random index to mutate
        ind = np.random.randint(child.shape[0], size = 1)
        # choose random number to mutate it to 
        mut = np.random.randint(8, size = 1) + 1
    
        # mutating
        child[ind] = mut[0]
    
    return(child)


for i in range(ch.shape[0]):
    ch[i, 0:cross_point] = c1[i, 0:cross_point]
    ch[i, cross_point: ] = c2[i, cross_point: ]

# This function is called after every simulation, passing a list of the old population of creatures, whose fitness
# you need to evaluate and whose chromosomes you can use to create new creatures.
#
# Input: old_population - list of objects of MyCreature type that participated in the last simulation.  You
#                         can query the state of the creatures by using some built-in methods as well as any methods
#                         you decide to add to MyCreature class.  The length of the list is the size of
#                         the population.  You need to generate a new population of the same size.  Creatures from
#                         old population can be used in the new population - simulation will reset them to starting
#                         state.
#
# Returns: a list of MyCreature objects of the same length as the old_population.
def newPopulation(old_population, tourny_size = 5):
    global numTurns

    nSurvivors = 0
    avgLifeTime = 0
    fitnessScore = 0
    fitness = []

    # For each individual you can extract the following information left over
    # from evaluation to let you figure out how well individual did in the
    # simulation of the world: whether the creature is dead or not, how much
    # energy did the creature have a the end of simualation (0 if dead), tick number
    # of creature's death (if dead).  You should use this information to build
    # a fitness function, score for how the individual did
    for individual in old_population:

        # You can read the creature's energy at the end of the simulation.  It will be 0 if creature is dead
        energy = individual.getEnergy()

        # This method tells you if the creature died during the simulation
        dead = individual.isDead()
        
        if dead == False:
            fitness.append(500)
        
        # If the creature is dead, you can get its time of death (in turns)
        if dead:
            timeOfDeath = individual.timeOfDeath()
            avgLifeTime += timeOfDeath
            fitness.append(timeOfDeath)
        else:
            nSurvivors += 1
            avgLifeTime += numTurns
    

    new_gen = []        
    for i in range(len(old_population)):
        tourny = tournament(old_population, tourny_size)
            
        

    # Here are some statistics, which you may or may not find useful
    avgLifeTime = float(avgLifeTime)/float(len(population))
    print("Simulation stats:")
    print("  Survivors    : %d out of %d" % (nSurvivors, len(population)))
    print("  Avg life time: %.1f turns" % avgLifeTime)
    print(fitness)

    # The information gathered above should allow you to build a fitness function that evaluates fitness of
    # every creature.  You should show the average fitness, but also use the fitness for selecting parents and
    # creating new creatures.


    # Based on the fitness you should select individuals for reproduction and create a
    # new population.  At the moment this is not done, and the same population with the same number
    # of individuals
    new_population = old_population

    return new_population

plt.close('all')
fh=plt.figure()

# Create the world.  Representaiton type choses the type of percept representation (there are three types to chose from);
# gridSize specifies the size of the world, repeatable parameter allows you to run the simulation in exactly same way.
w = World(worldType=worldType, gridSize=gridSize, repeatable=repeatableMode)

#Get the number of creatures in the world
numCreatures = w.maxNumCreatures()

#Get the number of creature percepts
numCreaturePercepts = w.numCreaturePercepts()

#Get the number of creature actions
numCreatureActions = w.numCreatureActions()

# Create a list of initial creatures - instantiations of the MyCreature class that you implemented
population = list()
for i in range(numCreatures):
   c = MyCreature(numCreaturePercepts, numCreatureActions)
   population.append(c)

# Pass the first population to the world simulator
w.setNextGeneration(population)

# Runs the simulation to evalute the first population
w.evaluate(numTurns)

# Show visualisation of initial creature behaviour
#w.show_simulation(titleStr='Initial population', speed='normal')

for i in range(numGenerations):
    print("\nGeneration %d:" % (i+1))

    # Create a new population from the old one
    population = newPopulation(population)

    # Pass the new population to the world simulator
    w.setNextGeneration(population)

    # Run the simulation again to evalute the next population
    w.evaluate(numTurns)

    # Show visualisation of final generation
    #if i==numGenerations-1:
    #    w.show_simulation(titleStr='Final population', speed='normal')


