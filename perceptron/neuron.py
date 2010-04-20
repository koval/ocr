# -*- coding: utf-8 -*-
import random

class Neuron(object):
    """Нейрон с пороговой функцией
    """

    def __init__(self, m):
        self.m = m
        self.w = []
        self.s = 50

    def transfer(self, x, raw=False):
        if raw:
            return self.adder(x)
        else:
            return self.activator(self.adder(x))

    def initWeights(self, n):
        """ Инициализация начальных весов синапсов небольшими случайными значениями
            @param n - от 0 до n
        """
        for i in range(self.m):
            self.w.append(random.randint(0, n))

    def changeWeights(self, v, d, x):
        """Модификация весов синапсов для обучения
        @param v - скорость обучения
        @param d - разница между выходом нейрона и нужным выходом
        @param x - входной вектор
        """
        for i in range(self.m):
            self.w[i] += v*d*x[i];

    def adder(self, x):
        """Сумматор
        @param x - входной вектор
        @return - невзвешенная сумма nec (биас не используется)
        """
        nec = 0;
        for i in range(len(x)):
            nec += x[i] * self.w[i]
        return nec

    def activator(self, nec):
        """ Нелинейный преобразователь или функция активации,
        в данном случае - жесткая пороговая функция,
        имеющая область значений {0;1}
        @param nec - выход сумматора
        @return
        """
        if nec >= self.s:
            return 1
        else:
            return 0
