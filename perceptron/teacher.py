# -*- coding: utf-8 -*-
import os
import Image

from perceptron import Perceptron

def get_in_vector(p):
    """ Трансформация пикселей изображения в вектор из 0 и 1
    1 на том месте, где есть цвет, 0 - там где белый
    @param p - пиксели изображения
    @return - вектор для входа перцептрона
    """
    x = []
    for c in p:
        if c == (255, 255, 255):
            x.append(0)
        else:
            x.append(1)
    return x


class Teacher(object):
    """ Учитель учит перцептрон распознаванию цифр
    """

    def __init__(self, perceptron):
        """ Конструктор @param perceptron
        """
        self.perceptron = perceptron

    def teach(self, path, n, debug=False):
        """ Обучение перцептрона
        @param path
        @param n - количество циклов обучения
        """

        # загрузка всех тестовых изображений в массив img
        img = []
        for fname in os.listdir(path):
            if not fname.endswith('.jpg'):
                continue
            img.append(Image.open(os.path.join(path, fname)))

        # инициализация начальных весов
        self.perceptron.initWeights()

        # получение пиксельных массивов каждого изображения
        # и обучение n раз каждой выборке
        for c in range(n):
            if debug:
                print 'Teaching cycle #%d' % c
            for i in img:
                w, h = i.size
                if w*h > self.perceptron.m:
                    continue

                pixels = i.getdata()
                # получение векторов и обучение перцептрона
                x = get_in_vector(pixels)
                y = self.getOutVector(int(os.path.basename(i.filename)[0]))
                if debug:
                    print '  %s image...' % i.filename
                self.perceptron.teach(x, y)

    def getOutVector(self, n):
        """ Генерация правильного выходного вектора
        @param n - цифра, в соответствии с которой
        нужно построить вектор, другими словами:
        на каком месте должна быть 1, остальные 0
        @return - выходной вектор для перцептрона
        """
        y = []
        for i in range(self.perceptron.n):
            if i == n:
                y.append(1)
            else:
                y.append(0)
        return y
