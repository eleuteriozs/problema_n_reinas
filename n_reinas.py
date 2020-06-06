# -*- coding: utf-8 -*-
#Se importan las librerias que se van a utilizar
import random
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

#numero de reinas (poblacion) 
NB_QUEENS = 8

# funcion para obtener el fitnes
def evalNQueens(individual):

    # en esta funcion se evalua el problema donde se calcula el número de reinas y 
    #determinando los ataques que solo pueden ser en diagonales

    #el primer ciclo for se determina el numero de reinas en cada diagonal, esto se recorre 
    #desde i hasta el valor de size

    #posteriormente el siguiente siclo for determina el numero de ataques en cada diagonal, al 
    #determinar un ataque se va almacenando en la variable suma determinada en 0 antes del  
    #ciclo

    #finalmente se retorna la variable suma

    size = len(individual)
    diagonal_izquierda_derecha = [0] * (2*size-1)
    diagonal_derecha_izquierda = [0] * (2*size-1)

    for i in range(size): 
        diagonal_izquierda_derecha[i+individual[i]] += 1 
        diagonal_derecha_izquierda[size-1-i+individual[i]] += 1 

    suma = 0
    for i in range(2*size-1): 
        if diagonal_izquierda_derecha[i] > 1: 
            suma += diagonal_izquierda_derecha[i] - 1 
        if diagonal_derecha_izquierda[i] > 1:
            suma += diagonal_derecha_izquierda[i] - 1
    return suma,


# se define el problema con ayuda del modulo creator del framework deap donde se realiza la creacion de tipos 
#se crea un tipo FitnessMin para probkemas de minimizacion
#se crea un tipo Individual que se deriba de una lista con un atributo de aptitud
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

#deap nos brinda toolbox que es un contenedor para herramienras donde registramos las funciones que son necesarias
toolbox = base.Toolbox()
toolbox.register("permutation", random.sample, range(NB_QUEENS), NB_QUEENS)

#registo de las funciones de inicilización del individuo y de la población
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.permutation)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#registro de la función de evaluación
toolbox.register("evaluate", evalNQueens)

#tambien el registro para los operadores genéticos
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0/NB_QUEENS)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
#en esta funcion se declaran algunas variables las cuales se utilizan en una algoritmo que recibe estas variables, donde se tiene el objeto que almacena el mejor individuo, asi como el objeto para calcular estadisticas(stats).
#retornando finalmente pop, stats y hof
    seed=0
    random.seed(seed)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1) # objeto que almacena el mejor individuo
    stats = tools.Statistics(lambda ind: ind.fitness.values) # objeto para calcular estadísticas
    stats.register("Avg", numpy.mean)
    stats.register("Std", numpy.std)
    stats.register("Min", numpy.min)
    stats.register("Max", numpy.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats,
                        halloffame=hof, verbose=True) # algoritmo genético como "caja negra"

    return pop, stats, hof

#en esta seccion se realiza la imprecion de lasvariables en consola, mostrando un arreglo de valores
#tambien se utiliza la libreria matplotlib para mostrar los datos en una grafica haciendo mas visual el resultado de los procesos que se llevaron a cavo
if __name__ == "__main__":
    pop, stats, best = main()
    print(best)
    print(best[0].fitness.values)
    y = best[0]
    x= range(NB_QUEENS)
    x= numpy.array(x)
    print(x)
    y = numpy.array(y)
    print(y)    
    x = x + 0.5
    y = y + 0.5
    plt.figure()
    plt.scatter(x,y)
    plt.xlim(0,NB_QUEENS)
    plt.ylim(0,NB_QUEENS)
    plt.xticks(x-0.5)
    plt.yticks(x-0.5)
    plt.grid(True)
    plt.title(u"Mejor Individuo")
    plt.show()
