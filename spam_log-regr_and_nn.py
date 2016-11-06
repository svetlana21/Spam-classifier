'''
Класс Classifier включает в себя и логистическую регрессию, и многослойный персептрон.
'''

# -*- coding: utf-8 -*-
import pprint
import numpy as np
import math
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import pickle
import random
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score

class Classifier():
    '''
    Класс для бинарной классификации с помощью логистической регрессии.
    '''
    
    def __init__(self):
        self.__st_model = None
        
    def load_data(self, filename):
        '''
        Загрузка данных из файла. Преобразование строковых данных в числовые.
        '''
        with open(filename) as f:
            lines=f.readlines()     # список строк
            general = []
            for line in lines:
                line = line.strip()     # удаление лишних символов в строке
                small_list = line.split(',')    # список значений (тип str)
                digits = []
                for string in range(0,len(small_list)-3):       # значения первых 55 признаков - вещественные числа
                    digits.append(float(small_list[string]))    
                for string in range(len(small_list)-3,len(small_list)):     # значения последних 2 признаков и метка класса - целые числа
                    digits.append(int(small_list[string]))    
                general.append(digits)  # список векторов признаков (тоже списков значений)
        return general

    def division(self, general_list):
        '''
        Разделение на обучающую и тестовую выборку. Отделение векторов от меток классов.
        '''
        new_data_train = []
        new_data_test = []
        new_target_train = []
        new_target_test = []
        random.shuffle(general_list)    # перемешивание массива
        splitter = math.trunc(len(general_list)*0.8)    # число, равное 80% от длины всех данных и округленное до целого
        gen_data_train = general_list[:splitter]    # разделение на 80% и 20%
        gen_data_test = general_list[splitter:]
        for lst in gen_data_train:              # отделение векторов от меток классов для обучения
            new_data_train.append(lst[0:-1])
            new_target_train.append(lst[-1])
        for lst in gen_data_test:               # отделение векторов от меток классов для тестирования
            new_data_test.append(lst[0:-1])
            new_target_test.append(lst[-1])
        data_train = np.array(new_data_train)   # перевод всех данных в тип ndarray
        data_test = np.array(new_data_test)
        target_train = np.array(new_target_train)
        target_test = np.array(new_target_test)
        return data_train, target_train, data_test, target_test
        
    def simple_division(self, general_list):
        '''
        Отделение векторов от меток классов.
        '''
        new_data_train = []
        new_target_train = []
        random.shuffle(general_list)    # перемешивание массива
        for lst in general_list:              # отделение векторов от меток классов для обучения
            new_data_train.append(lst[0:-1])
            new_target_train.append(lst[-1])
        data_train = np.array(new_data_train)   # перевод всех данных в тип ndarray
        target_train = np.array(new_target_train)
        return data_train, target_train        

    def normalization(self, data, coefs=None):
        '''
        Нормализация векторов признаков - l2-norm. Формула: coef = np.sqrt(sum(x**2))
                                                            x_norm = x/coef).
        Для решения задачи классификации "спам/не спам" не используется, т.к. значительно ухудшает результаты.
        '''
        data_transposed = np.transpose(data)    # транспонирование матрицы с данными, чтобы построчно производить операции с каждым признаком
        arrays_1d = []
        if coefs is None:   # коэффициенты можно передать в функцию (как в случае с тестовым множеством), если они не переданы, то считаем
            coefs = []
            for i in range(0,data_transposed.shape[0]):
                coef = np.sqrt(sum(data_transposed[i]**2))
                coefs.append(coef)      # список коэффициентов
        for i in range(0,data_transposed.shape[0]):     # делим значения на коэффициент
            i_norm = data_transposed[i]/coefs[i]
            arrays_1d.append(i_norm)    # добавляем полученный нормализованный признак в список
        data_row = np.array(arrays_1d)  # перевод списка в тип ndarray
        data_norm = np.transpose(data_row) # транспонируем обратно       
        return data_norm, coefs
     
    def log_reg_train(self, data_train, target_train):
        '''
        Функция для обучения логистической регрессии.
        '''
        logreg = linear_model.LogisticRegression()
        logreg.fit(data_train, target_train)  # обучение
        self.__st_model = pickle.dumps(logreg)  # сохранение модели
        
    def log_reg_test(self, data_test, target_test):
        '''
        Функция для тестирования.
        '''
        logreg = pickle.loads(self.__st_model)      # загрузка сохранённой модели
        predicted = logreg.predict(data_test)       # тестирование
        print('LogisticRegression: ', classification_report(target_test, predicted))    # вывод результатов
        print('Accuracy: ', accuracy_score(target_test, predicted))
        
    def cross_val(self, data_train, target_train):
        '''
        Кросс-валидация
        '''
        logreg = pickle.loads(self.__st_model)      # загрузка сохранённой модели
        scores = cross_val_score(logreg, data_train, target_train, cv=5) # кросс-валидация
        print('Cross_validation_scores:', scores)
        
    def neural_n(self, data_train, target_train, data_test, target_test):
        '''
        Метод для обучения с помощью персептрона.
        '''
        nw = MLPClassifier(activation='tanh', max_iter=1000)
        nw.fit(data_train, target_train)
        predicted = nw.predict(data_test)
        print('Perseptron: ', classification_report(target_test, predicted))    # вывод результатов
        print('Accuracy: ', accuracy_score(target_test, predicted))
        
if __name__ == '__main__':
    classifier = Classifier()   
    inp = classifier.load_data('spambase.data.txt')     # загрузка данных
    data = classifier.division(inp)                     # разделение на обучающую выборку и тестовую
    
    train = classifier.log_reg_train(data[0], data[1])   # обучение с помощью логистической регрессии
    test = classifier.log_reg_test(data[2], data[3])      # тестирование
    
    data_for_CV = classifier.simple_division(inp)       # оценка с помощью кросс-валидации
    cv = classifier.cross_val(data_for_CV[0], data_for_CV[1])
    
    perceptron = classifier.neural_n(data[0], data[1], data[2], data[3])   # обучение с помощью персептрона
