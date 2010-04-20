# -*- coding: utf-8 -*-
from neuron import Neuron

class Perceptron(object):
    """ Однослойный n-нейронный перцептрон
    """

    def __init__(self, n, m):
        """ Конструктор
        @param n - число нейронов
        @param m - число входов каждого нейрона скрытого слоя
        """
        self.n = n
        self.m = m
        self.neurons = []
        for i in range(n):
            self.neurons.append(Neuron(m))

    def recognize(self, x, raw=False):
        """ Распознавание образа
        @param x - входной вектор
        @return - выходной образ
        """
        y = []
        for i in range(self.n):
            y.append(self.neurons[i].transfer(x, raw))
        return y

    def initWeights(self):
        """ Инициализация начальных весов малым случайным значением
        """
        for i in range(self.n):
            self.neurons[i].initWeights(10)

    def teach(self, x, y):
        """ Обучение перцептрона
        @param x - входной вектор
        @param y - правильный выходной вектор
        """
        v = 1 # скорость обучения

        t = self.recognize(x)
        while t != y:
            # подстройка весов каждого нейрона
            for i in range(self.n):
                d = y[i] - t[i]
                self.neurons[i].changeWeights(v, d, x)
            t = self.recognize(x)
