#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Библиотеки

# Стандартный стэк
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Игнорирование предупреждений
from pandas.core.common import SettingWithCopyWarning
import warnings

# Препроцессинг
import sklearn.impute
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

# Метрики, кросс-валидация, отбор признаков
from sklearn.metrics import  mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.inspection import permutation_importance

# Модели
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import catboost as ctb

### Данные
data_full = pd.read_csv('C:/Users/nicka/OneDrive/Рабочий стол/pet_housing/data_ordered/data3.csv', index_col = 0)
data_model = data_full.copy()
data_model = data_model.drop(['id_c', 'living_square', 'ceiling', 'consultant', 'name', 'address', 'description','street', 'number', 'id_a', ], axis = 1)

data_model['total_log'] = np.log(data_model['total_square'])
data_model['kitchen_log'] = np.log(data_model['kitchen_square'])

data_model = data_model.loc[:, data_model.columns [list(map(lambda x: x not in ['type', 'rooms', 'toilet', 'view', 'renovation', 'elevator', 'building', 'overlap', \
                                                                                'parking', 'area'], data_model.columns.tolist()))].tolist() + \
                            ['type', 'rooms', 'toilet', 'view', 'renovation', 'elevator', 'building', 'overlap', 'parking', 'area']]

global data_train, data_test, predictions, next_preds, predictions_bstr, next_preds_bstr
data_train, data_test = train_test_split(data_model, test_size = 309, shuffle = True, random_state = 42)
predictions = pd.DataFrame(index = data_train.index)
next_preds = -1
predictions_bstr = pd.DataFrame(index = data_train.index)
next_preds_bstr = -1

### №1 Заполнение пропусков средним/медианой/наиболее часто встречающимся значениями, создание дамми на то, был ли в наблюдении пропуск
class filler:
    '''
    ПЕРЕМЕННЫЕ:
        *** strat (str) - стратегия заполнения пропусков, 'mean'/'median'/'most_frequent'
        
        *** col (int) - номер столбца с пропусками
        
        *** integer (bool) - приведение к целой части числа при заполнении, True/False
    МЕТОДЫ:
        *** indicate - создать дамми-переменную, указывающую на то, было ли значение пропущено для каждого наблюдения
        
        *** fit_transform - обучить imputer на трейне и трансформировать его
        
        *** transform - трансформировать тестовую/валидационную части
    '''
    def __init__(self, strat, col, integer = False):
        self.strat = strat
        self.col = col
        self.integer = integer
        self.zerone = pd.Series([], dtype = np.int64)
        self.imputer = sklearn.impute.SimpleImputer(missing_values = np.nan, strategy = self.strat)
        
    def indicate(self, setdf): 
        self.zerone = pd.Series(np.zeros(setdf.shape[0]))
        self.zerone.index = setdf.index
        self.zerone.loc[setdf.loc[:, self.col].isnull()] = 1
        return(self.zerone)
    
    def fit_transform(self, setdf0): # Обучение и заполнение пропущенных значений для трейна
        self.imputer.fit(np.transpose(np.asmatrix(setdf0[self.col])))
        if self.integer == False:
            return(np.ravel(self.imputer.transform(np.transpose(np.asmatrix(setdf0[self.col])))))
        else:
            return(np.int64(np.ravel(self.imputer.transform(np.transpose(np.asmatrix(setdf0[self.col]))))))
    
    def transform(self, setdf1): # Заполнение пропущенных значений для теста
        if self.integer == False:
            return(np.ravel(self.imputer.transform(np.transpose(np.asmatrix(setdf1[self.col])))))
        else:
            return(np.int64(np.ravel(self.imputer.transform(np.transpose(np.asmatrix(setdf1[self.col]))))))

### №2 Создание сетов train, test, val для кросс-валидации и train для финальной модели, заполнение в них пропусков, \
### создание признаков по комнатам, масштабирование числовых регрессоров, расчет коэффициентов для обратного масштабирования
def processing(scaler, cv, index_train = None, index_test = None, scaling_return = False, data_train = data_train, data_test = data_test):
    '''
    *** scaler (str) - 'standard'/'minmax'/'none', тип масштабирования числовых признаков
    
    *** cv (bool) - подготовить данные для кросс-валидации (возвращает train+test+val) или обучения на полном сете (возвращает train)
    
    *** index_train (np.array/NoneType) - индексы обучающей выборки (None работает только для cv == False)
    
    *** index_test (np.array/NoneType) - индексы тестовой выборки (--//--)
    
    *** scaling_return (bool) - возвращает имена признаков и соответствующие коэффициенты для обратного масштабирования, по умолчанию = False
    
    *** data_train (pd.DataFrame) - тренировочный датафрейм с наблюдиениями и соответствующими признаками  
    
    *** data_test (pd.DataFrame) - тестовый датафрейм с наблюдиениями и соответствующими признаками  
    '''
    # Делим фичи на категориальные и числовые
    global dum
    global other
    dum = np.r_[84:94]
    other = np.r_[1:84]
    if cv == True:
        # Деление на трейн+тест+валидацию
        train = data_train.iloc[index_train]
        test = data_train.iloc[index_test]
        val = data_test.copy()
        ### Пропущенные значения
        for i in data_train.columns:
            # Проверяем, есть ли в столбце пропуски
            if (train.loc[:, i].isnull().values.any() or test.loc[:, i].isnull().values.any() or val.loc[:, i].isnull().values.any()): 
                if i == 'year': # Заполняем год целыми числами
                    fill_mean = filler(strat = 'mean', col = i, integer = True)
                    train.loc[:, f'{i}_miss'] = fill_mean.indicate(train)
                    test.loc[:, f'{i}_miss'] = fill_mean.indicate(test)
                    train.loc[:, f'{i}_meaned'] = fill_mean.fit_transform(train)
                    test.loc[:, f'{i}_meaned'] = fill_mean.transform(test)
                    fill_median = filler(strat = 'median', col = i, integer = True)
                    train.loc[:, f'{i}_medianed'] = fill_median.fit_transform(train)
                    test.loc[:, f'{i}_medianed'] = fill_median.transform(test)
                    fill_freq = filler(strat = 'most_frequent', col = i, integer = True)
                    train.loc[:, f'{i}_freq'] = fill_freq.fit_transform(train)
                    test.loc[:, f'{i}_freq'] = fill_freq.transform(test)
                    val.loc[:, f'{i}_miss'] = fill_mean.indicate(val)
                    val.loc[:, f'{i}_meaned'] = fill_mean.transform(val)
                    val.loc[:, f'{i}_medianed'] = fill_median.transform(val)
                    val.loc[:, f'{i}_freq'] = fill_freq.transform(val)
                elif i == 'kitchen_square': # Заполняем кухонную площадь
                    fill_mean = filler(strat = 'mean', col = i, integer = False)
                    train.loc[:, f'{i}_miss'] = fill_mean.indicate(train)
                    test.loc[:, f'{i}_miss'] = fill_mean.indicate(test)
                    train.loc[:, f'{i}_meaned'] = fill_mean.fit_transform(train)
                    test.loc[:, f'{i}_meaned'] = fill_mean.transform(test)
                    fill_median = filler(strat = 'median', col = i, integer = False)
                    train.loc[:, f'{i}_medianed'] = fill_median.fit_transform(train)
                    test.loc[:, f'{i}_medianed'] = fill_median.transform(test)
                    val.loc[:, f'{i}_miss'] = fill_mean.indicate(val)
                    val.loc[:, f'{i}_meaned'] = fill_mean.transform(val)
                    val.loc[:, f'{i}_medianed'] = fill_median.transform(val)
                elif i in 'kitchen_log': # Заполняем логарифм кухонной площади
                    fill_mean = filler(strat = 'mean', col = i, integer = False)
                    train.loc[:, f'{i}_meaned'] = fill_mean.fit_transform(train)
                    test.loc[:, f'{i}_meaned'] = fill_mean.transform(test)
                    fill_median = filler(strat = 'median', col = i, integer = False)
                    train.loc[:, f'{i}_medianed'] = fill_median.fit_transform(train)
                    test.loc[:, f'{i}_medianed'] = fill_median.transform(test)
                    val.loc[:, f'{i}_meaned'] = fill_mean.transform(val)
                    val.loc[:, f'{i}_medianed'] = fill_median.transform(val)
                else: # Заполняем оставшиеся категориальные признаки
                    fill_freq = filler(strat = 'most_frequent', col = i, integer = False)
                    train.loc[:, f'{i}_freq'] = fill_freq.fit_transform(train)
                    test.loc[:, f'{i}_freq'] = fill_freq.transform(test)
                    train.loc[:, f'{i}_miss'] = fill_freq.indicate(train)
                    test.loc[:, f'{i}_miss'] = fill_freq.indicate(test)
                    val.loc[:, f'{i}_freq'] = fill_freq.transform(val)
                    val.loc[:, f'{i}_miss'] = fill_freq.indicate(val)

        ### Комнаты          
        # Дамми на специальный вид комнат
        train['special_k'] = 0
        train['special_c'] = 0
        test['special_k'] = 0
        test['special_c'] = 0
        val['special_k'] = 0
        val['special_c'] = 0
        train.loc[train.rooms == 'К', 'special_k'] = 1
        train.loc[train.rooms == 'С', 'special_c'] = 1
        test.loc[test.rooms == 'К', 'special_k'] = 1
        test.loc[test.rooms == 'С', 'special_c'] = 1
        val.loc[val.rooms == 'К', 'special_k'] = 1
        val.loc[val.rooms == 'С', 'special_c'] = 1

        # Вариант второй (присваиваем нули спец. комнатам)
        train['rooms_v1'] = train.rooms.apply(lambda x: 0 if (x == 'С' or x == 'К')  else x)
        test['rooms_v1'] = test.rooms.apply(lambda x: 0 if (x == 'С' or x == 'К')  else x)
        val['rooms_v1'] = val.rooms.apply(lambda x: 0 if (x == 'С' or x == 'К')  else x)

        # Вариант третий (наиболее близкое в среднеем по группам)
        train['rooms_v2'] = train.rooms
        test['rooms_v2'] = test.rooms
        val['rooms_v2'] = val.rooms

        for j in ['С', 'К']:
            comp = []
            mmean = train.loc[train.rooms == j, 'total_square'].mean()
            for k in ['1', '2', '3', '4', '5']:
                comp.append(train.loc[train.rooms == k, 'total_square'].mean())
                comp[np.int64(k) - 1] = np.abs(comp[np.int64(k) - 1] - mmean)
            train.loc[train.rooms == j, 'rooms_v2'] = np.argmin(comp) + 1
            test.loc[test.rooms == j, 'rooms_v2'] = np.argmin(comp) + 1
            val.loc[val.rooms == j, 'rooms_v2'] = np.argmin(comp) + 1

        # Вариант четвертый (наиболее близкое среднее для каждой квартиры отдельно)
        train['rooms_v3'] = train.rooms
        test['rooms_v3'] = test.rooms
        val['rooms_v3'] = val.rooms

        a = [train.loc[train.rooms == j, 'total_square'].mean() for j in ['1', '2', '3', '4', '5']]
        train.loc[train.rooms.isin(['С', 'К']), 'rooms_v3'] = train.loc[train.rooms.isin(['С', 'К']), :] \
        .apply(lambda x: np.argmin([np.abs(x.total_square - a[i]) for i in range(0, 5)]) + 1, axis = 1)

        test.loc[test.rooms.isin(['С', 'К']), 'rooms_v3'] = test.loc[test.rooms.isin(['С', 'К']), :] \
        .apply(lambda x: np.argmin([np.abs(x.total_square - a[i]) for i in range(0, 5)]) + 1, axis = 1)

        val.loc[val.rooms.isin(['С', 'К']), 'rooms_v3'] = val.loc[val.rooms.isin(['С', 'К']), :] \
        .apply(lambda x: np.argmin([np.abs(x.total_square - a[i]) for i in range(0, 5)]) + 1, axis = 1)

        train.rooms_v1 = train.rooms_v1.astype(np.int64)
        test.rooms_v1 = test.rooms_v1.astype(np.int64)
        val.rooms_v1 = val.rooms_v1.astype(np.int64)
        train.rooms_v2 = train.rooms_v2.astype(np.int64)
        test.rooms_v2 = test.rooms_v2.astype(np.int64)
        val.rooms_v2 = val.rooms_v2.astype(np.int64)
        train.rooms_v3 = train.rooms_v3.astype(np.int64)
        test.rooms_v3 = test.rooms_v3.astype(np.int64)
        val.rooms_v3 = val.rooms_v3.astype(np.int64)

        # Избавляемся от колонок с пропусками
        train = train.dropna(axis = 1)
        test = test.dropna(axis = 1)
        val = val.dropna(axis = 1)
        
        # Порядок колонок: сначала числовые, потом категориальные
        train = train.loc[:, ['price'] + train.select_dtypes(exclude = 'O').columns.drop('price').tolist() + train.select_dtypes(include = 'O').columns.tolist()]
        test = test.loc[:, ['price'] + test.select_dtypes(exclude = 'O').columns.drop('price').tolist() + test.select_dtypes(include = 'O').columns.tolist()]
        val = val.loc[:, ['price'] + val.select_dtypes(exclude = 'O').columns.drop('price').tolist() + val.select_dtypes(include = 'O').columns.tolist()]
        
        ### Масштабирование регрессоров
        if scaler == 'standard':
            scaler_num = StandardScaler()
            train.iloc[:, other] = pd.DataFrame(scaler_num.fit_transform(train.iloc[:, other]), columns = train.columns[other], index = train.index)
            test.iloc[:, other] = pd.DataFrame(scaler_num.transform(test.iloc[:, other]), columns = test.columns[other], index = test.index)
            val.iloc[:, other] = pd.DataFrame(scaler_num.transform(val.iloc[:, other]), index = val.index)
        elif scaler == 'minmax':
            scaler_num = MinMaxScaler()
            train.iloc[:, other] = pd.DataFrame(scaler_num.fit_transform(train.iloc[:, other]), columns = train.columns[other], index = train.index)
            test.iloc[:, other] = pd.DataFrame(scaler_num.transform(test.iloc[:, other]), columns = test.columns[other], index = test.index)
            val.iloc[:, other] = pd.DataFrame(scaler_num.transform(val.iloc[:, other]), columns = val.columns[other], index = val.index)
        return(train, test, val)
    else:
        train = data_train.copy()
        for i in data_train.columns:
            if (train.loc[:, i].isnull().values.any()):
                if i == 'year':
                    fill_mean = filler(strat = 'mean', col = i, integer = True)
                    train.loc[:, f'{i}_miss'] = fill_mean.indicate(train)
                    train.loc[:, f'{i}_meaned'] = fill_mean.fit_transform(train)
                    fill_median = filler(strat = 'median', col = i, integer = True)
                    train.loc[:, f'{i}_medianed'] = fill_median.fit_transform(train)
                    fill_freq = filler(strat = 'most_frequent', col = i, integer = True)
                    train.loc[:, f'{i}_freq'] = fill_freq.fit_transform(train)
                elif i == 'kitchen_square':
                    fill_mean = filler(strat = 'mean', col = i, integer = False)
                    train.loc[:, f'{i}_miss'] = fill_mean.indicate(train)
                    train.loc[:, f'{i}_meaned'] = fill_mean.fit_transform(train)
                    fill_median = filler(strat = 'median', col = i, integer = False)
                    train.loc[:, f'{i}_medianed'] = fill_median.fit_transform(train)
                elif i == 'kitchen_log':
                    train.loc[:, f'{i}_meaned'] = fill_mean.fit_transform(train)
                    fill_median = filler(strat = 'median', col = i, integer = False)
                    train.loc[:, f'{i}_medianed'] = fill_median.fit_transform(train)
                else:
                    fill_freq = filler(strat = 'most_frequent', col = i, integer = False)
                    train.loc[:, f'{i}_freq'] = fill_freq.fit_transform(train)
                    train.loc[:, f'{i}_miss'] = fill_freq.indicate(train)
        ### Разбираемся с комнатами            
        # Дамми на специальный вид комнат
        train['special_k'] = 0
        train['special_c'] = 0

        train.loc[train.rooms == 'К', 'special_k'] = 1
        train.loc[train.rooms == 'С', 'special_c'] = 1

        # Вариант второй (присваиваем нули спец. комнатам)
        train['rooms_v1'] = train.rooms.apply(lambda x: 0 if (x == 'С' or x == 'К')  else x)
        # Вариант третий (наиболее близкое в среднеем по группам)
        train['rooms_v2'] = train.rooms

        for j in ['С', 'К']:
            comp = []
            mmean = train.loc[train.rooms == j, 'total_square'].mean()
            for k in ['1', '2', '3', '4', '5']:
                comp.append(train.loc[train.rooms == k, 'total_square'].mean())
                comp[np.int64(k)-1] = np.abs(comp[np.int64(k)-1] - mmean)
            train.loc[train.rooms == j, 'rooms_v2'] = np.argmin(comp) + 1

        # Вариант четвертый (наиболее близкое среднее для каждой квартиры)
        train['rooms_v3'] = train.rooms
        
        a = [train.loc[train.rooms == j, 'total_square'].mean() for j in ['1', '2', '3', '4', '5']]
        train.loc[train.rooms.isin(['С', 'К']), 'rooms_v3'] = train.loc[train.rooms.isin(['С', 'К']), :] \
        .apply(lambda x: np.argmin([np.abs(x.total_square - a[i]) for i in range(0, 5)]) + 1, axis = 1)
        
        train.rooms_v1 = train.rooms_v1.astype(np.int64)
        train.rooms_v2 = train.rooms_v2.astype(np.int64)
        train.rooms_v3 = train.rooms_v3.astype(np.int64)
        train = train.dropna(axis = 1)
        # Порядок колонок: сначала числовые, потом категориальные
        train = train.loc[:, ['price'] + train.select_dtypes(exclude = 'O').columns.drop('price').tolist() + train.select_dtypes(include = 'O').columns.tolist()]
        # Масштабирование
        if scaler == 'standard':
            scaler_num = StandardScaler()
            train.iloc[:, other] = pd.DataFrame(scaler_num.fit_transform(train.iloc[:, other]), columns = train.columns[other], index = train.index)
        elif scaler == 'minmax':
            scaler_num = MinMaxScaler()
            train.iloc[:, other] = pd.DataFrame(scaler_num.fit_transform(train.iloc[:, other]), columns = train.columns[other], index = train.index)
        if scaling_return == False:
            return(train)
        else:
            return(pd.DataFrame({'names' : train.iloc[:, other].columns, 'scale' : scaler_num.scale_}))


        
### №3 Функция для обучения линейных моделей , при кросс-валидации  возвращает табоицу с результатами, при обучении на всем сете - пандас серию с результатами \
### для лучшего шага отбора признаков
def linear_model(model, alpha = 0.001, l1_ratio = 0.5, max_features = 93, cv = True, iters = None, scaler = None,
                 normalizing = False, verbose = True): 
    '''
    *** model (str) = 'linear'/'ridge'/'lasso'/'elastic_net' (выбор линейной модели);
    
    *** alpha (float) = коэффициент регуляризации, по умолчанию = 0.001;
    
    *** l1_ratio (float [0:1]) = доля L1 регуляризации для elastic_net, доля L2 = (1 - l1_ratio) соответственно, по умолчанию = 0.5;
    
    *** max_features (int [1:93]), максимальное число регрессоров в модели, по умолчанию = 93;
    
    *** cv (bool) = True/False, проводить ли кросс-валидацию или оценить регрессию на полном дата-сете, по умолчанию = True;
    
    ***  iters (NoneType/int) = максимальное число итераций оптимизационного алгоритма модели, по умолчанию:
            -linear = None
            -ridge = None ('auto')
            -lasso = None (1000)
            -elastic_net = None (1000);
        
    *** scaler (NoneType/str) = None/'standard'/'minmax' (тип предварительного масштабирования данных), по умолчанию = None;
    
    *** normalizing (bool) = True/False (нормализация данных в модели), по умолчанию = False;
    
    *** verbose (bool) = True/False (оповещать о выполнении расчетов на текущем фолде и переходе к следующему), по умолчанию = True.
        
    ''' 
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning) # Игнорируем вылезающие напоминалки от pandas
    kf = KFold(n_splits = 10, shuffle = True, random_state = 42) # Делим на фолды
    
    ### Готовим таблицу для результатов
    a = []
    for i in range(0, 10): 
        a.append(f'mse{i}')
        a.append(f'mse_test{i}')
    for i in range(0, 10): # Два цикла, чтобы у столбцов был заданный порядок
        a.append(f'r2_adj{i}')
        a.append(f'r2_adj_test{i}')
    a.append('features')
    results = pd.DataFrame(columns = a)
    if cv == False:
        results = pd.concat([results, pd.DataFrame(columns=['coefs', 'intercept', 'names'])])
        
    ### Инициализируем обучение    
    step = 0
    for index_train, index_test in kf.split(data_train):
        # Разбиваем выборку
        train, test, val = processing(scaler = scaler, cv = True, index_train = index_train, index_test = index_test)
        # Создаем лист регрессоров, заполненный минус единицами
        regressors = - np.ones(max_features, dtype = int)
        k = 0
        while True:
            if len(regressors[regressors != -1]) == max_features: # Останавливаемся, отобрав все max_features признаков
                break
            j = 1 # Так как нулевой столбец в train - целевая переменная, начинаем с первой фичи
            mses = [1000] # По той же причине mse для нулевой переменной присваиваем большое значение
            while True:
                if j > 93: # Заканчиваем перебор, когда были расчитаны mse для каждого из признаков (всего их 93)
                    break
                if j in regressors: 
                    # Если признак был выбран в одной из прошлых регрессий, его mse присваиваем большое число
                    mses.append(1000)
                    j += 1 # И готовимся перебирать следующие фичи
                else: # Если нет, начинаем для него расчет
                    regressors[k] = j
                    # Текущие используемые дамми
                    dum_now = regressors[regressors != -1][list(map(lambda x: x in dum, regressors[regressors != -1]))]
                    # Текущие используемые числовые переменные
                    other_now = regressors[regressors != -1][list(map(lambda x: x not in dum, regressors[regressors != -1]))]
                    
                    if len(dum_now) > 0 and len(other_now) > 0:
                        X_train = pd.concat([train.iloc[:, other_now], pd.get_dummies(train.iloc[:, dum_now], drop_first = True)], axis = 1)
                        X_test = pd.concat([test.iloc[:, other_now], pd.get_dummies(test.iloc[:, dum_now], drop_first = True)], axis = 1)
                        X_val = pd.concat([val.iloc[:, other_now], pd.get_dummies(val.iloc[:, dum_now], drop_first = True)], axis = 1)
                    elif len(dum_now) > 0:
                        X_train = pd.get_dummies(train.iloc[:, dum_now], drop_first = True)
                        X_test = pd.get_dummies(test.iloc[:, dum_now], drop_first = True)
                        X_val = pd.get_dummies(val.iloc[:, dum_now], drop_first = True)
                    else:
                        X_train = train.iloc[:, other_now]
                        X_test = test.iloc[:, other_now]
                        X_val = val.iloc[:, other_now]

                    # Если вдруг в каком-то из сетов нет колонок, которые есть в других (так может случиться из-за дамми)
                    X_test[X_val.columns.difference(X_test.columns).append(X_train.columns.difference(X_test.columns))] = 0
                    X_train[X_val.columns.difference(X_train.columns).append(X_test.columns.difference(X_train.columns))] = 0
                    X_val[X_train.columns.difference(X_val.columns).append(X_test.columns.difference(X_val.columns))] = 0                   
                    # Избавляемся от случайных дубликатов (иногда дамми на комнаты могут повторяться)
                    X_train = X_train.loc[:, ~X_train.columns.duplicated()]
                    X_test = X_test.loc[:, ~X_test.columns.duplicated()]
                    X_val = X_val.loc[:, ~X_val.columns.duplicated()]
                    # Задаем объясняемую переменную
                    y_train = np.log(train.price)
                    y_test = np.log(test.price)
                    y_val = np.log(val.price)
                    ### Модель из списка 'linear', 'ridge', 'lasso', 'elastic_net'
                    if model == 'linear':
                        rdg = LinearRegression(normalize = normalizing)
                    elif model == 'ridge':
                        rdg = Ridge(alpha = alpha, normalize = normalizing, max_iter = iters, random_state = 42)
                    elif model == 'lasso':
                        if iters == None:
                            iters = 1000                        
                        rdg = Lasso(alpha = alpha, normalize = normalizing, max_iter = iters, random_state = 42)
                    else:
                        if iters == None:
                            iters = 1000                        
                        rdg = ElasticNet(alpha = alpha, l1_ratio = l1_ratio, normalize = normalizing, max_iter = iters, random_state = 42)
                    rdg.fit(X_train, y_train) # Обучаем
                    mses.append(mean_squared_error(rdg.predict(X_val), y_val)) # Считаем mse на валидационном сете
                    j += 1 # Готовимся перебирать следующую фичу
            regressors[k] = np.argmin(mses) # Выбираем минимизирующий mse на данном этапе признак

            ### Запись результатов с новым регрессором в табличку results
            dum_now = regressors[regressors != -1][list(map(lambda x: x in dum, regressors[regressors != -1]))]
            other_now = regressors[regressors != -1][list(map(lambda x: x not in dum, regressors[regressors != -1]))]
            if len(dum_now) > 0 and len(other_now) > 0:
                X_train = pd.concat([train.iloc[:, other_now], pd.get_dummies(train.iloc[:, dum_now], drop_first = True)], axis = 1)
                X_test = pd.concat([test.iloc[:, other_now], pd.get_dummies(test.iloc[:, dum_now], drop_first = True)], axis = 1)
            elif len(dum_now) > 0:
                X_train = pd.get_dummies(train.iloc[:, dum_now], drop_first = True)
                X_test = pd.get_dummies(test.iloc[:, dum_now], drop_first = True)
            else:
                X_train = train.iloc[:, other_now]
                X_test = test.iloc[:, other_now]
            X_test[X_train.columns.difference(X_test.columns)] = 0
            X_train[X_test.columns.difference(X_train.columns)] = 0                     
            X_train = X_train.loc[:, ~X_train.columns.duplicated()]
            X_test = X_test.loc[:, ~X_test.columns.duplicated()]
            y_train = np.log(train.price)
            y_test = np.log(test.price)
            rdg.fit(X_train, y_train)
            # MSE и R2_adj для трейна и тестовой выборок соответственно
            results.loc[k, f'mse{step}'] = mean_squared_error(rdg.predict(X_train), y_train)
            results.loc[k, f'mse_test{step}'] = mean_squared_error(rdg.predict(X_test), y_test)
            if X_train.shape[1] != 1: # Если переменных больше одной, считаем R2_adj
                results.loc[k, f'r2_adj{step}'] = (1-(1-r2_score(rdg.predict(X_train), y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
                results.loc[k, f'r2_adj_test{step}'] = (1-(1-r2_score(rdg.predict(X_test), y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
            else: # Если нет, то обычный R2
                results.loc[k, f'r2_adj{step}'] = r2_score(rdg.predict(X_train), y_train)
                results.loc[k, f'r2_adj_test{step}'] = r2_score(rdg.predict(X_test), y_test)
            results.at[k, 'features'] = regressors[regressors != -1]
            results.at[k, 'features_names'] = list(train.columns[regressors[regressors != -1]])

            # Средние и стандартные ошибки по фолдам для mse и r2 соответственно
            results['mean_mse'] = results.loc[:, [f'mse{i}' for i in range(10)]].mean(axis = 1)
            results['mean_mse_test'] = results.loc[:, [f'mse_test{i}' for i in range(10)]].mean(axis = 1)
            results['mean_r2'] = results.loc[:, [f'r2_adj{i}' for i in range(10)]].mean(axis = 1)
            results['mean_r2_test'] = results.loc[:, [f'r2_adj_test{i}' for i in range(10)]].mean(axis = 1)
            results['std_mse'] = results.loc[:, [f'mse{i}' for i in range(10)]].std(axis = 1)
            results['std_mse_test'] = results.loc[:, [f'mse_test{i}' for i in range(10)]].std(axis = 1)
            results['std_r2'] = results.loc[:, [f'r2_adj{i}' for i in range(10)]].std(axis = 1)
            results['std_r2_test'] = results.loc[:, [f'r2_adj_test{i}' for i in range(10)]].std(axis = 1)
            k += 1 # Переходим к следующему слоту для признака
        if verbose == True:
            print(step) # Оповещаем о выполнении расчетов для фолда № step
        step += 1 # Переходим к следующему фолду валидации
    if cv == True:
        return(results)
    else: # В случае, когда возвращаем лучшую модель без кросс-валидации, понадобится ее перерасчет, здесь все как и раньше
        results = pd.DataFrame(results['features'])
        train = processing(scaler = scaler, cv = False)
        # Переоцениваем лучшую регрессию на всех данных
        best_reg = np.array(results['features'][max_features-1].tolist())
        best_dum = best_reg[list(map(lambda x: x in dum, best_reg))]
        best_other = best_reg[list(map(lambda x: x not in dum, best_reg))]
        
        if len(best_dum) > 0 and len(best_other) > 0:
            X_train = pd.concat([train.iloc[:, best_other], pd.get_dummies(train.iloc[:, best_dum], drop_first = True)], axis = 1)
        elif len(dum_now) > 0:
            X_train = pd.get_dummies(train.iloc[:, best_dum], drop_first = True)
        else:
            X_train = train.iloc[:, best_other]
        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
        y_train = np.log(train.price)
        if model == 'linear':
            rdg = LinearRegression(normalize = normalizing)
        elif model == 'ridge':
            rdg = Ridge(alpha = alpha, normalize = normalizing, max_iter = iters, random_state = 42)
        elif model == 'lasso':
            if iters == None:
                iters = 1000                        
            rdg = Lasso(alpha = alpha, normalize = normalizing, max_iter = iters, random_state = 42)
        else:
            if iters == None:
                iters = 1000                        
            rdg = ElasticNet(alpha = alpha, l1_ratio = l1_ratio, normalize = normalizing, max_iter = iters, random_state = 42)
        rdg.fit(X_train, y_train)
        results.at[:, 'coefs'] = 0
        results.at[:, 'intercept'] = 0
        results.at[:, 'names'] = 0
        results = results.dropna(axis = 1).iloc[-1, :]
        results.at['coefs'] = rdg.coef_
        results.at['intercept'] = rdg.intercept_
        results['names'] = X_train.columns.tolist()
        results['features'] = list(train.columns[best_reg])
        return(results)

    
    
### №4 Функция для обучения лесов Функция для обучения линейных моделей , при кросс-валидации  возвращает табоицу с результатами, при обучении на всем сете - пандас серию \
### с результатами для лучшего шага отбора признаков
def forest_model(n_estimators = 100,
                 max_depth = None,
                 min_samples_split = 2,
                 min_samples_leaf = 1,
                 max_features = 'auto',
                 max_leaf_nodes = None,
                 bootstrap = True,
                 max_samples = None,
                 threshold = 0.01,
                 cv = True,
                 verbose = True):
    '''
    Параметры от n_estimators до max_samples соотвествуют параметрам sklearn.ensemble.RandomForestRegressor():
     (значения по умолчанию тоже совпадают)
     
        *** n_estimators (int (1:+∞)) = максимальное число используемых в ансамбле деревьев, by default = None; 
        
        *** max_depth (int (1:+∞)) = максимальная глубина дерева, by default = None;
        
        *** min_samples_split (int (1:+∞)) = минимальное число объектов в вершине дерева, при котором будет выполнено ее
            разбиение, by default = 2;
        
        *** min_samples_leaf (int (1:+∞)) = минимальное количество объектов в листе дерева, при котором будет выполнено
            разбиение вершины, by default = 1;
        
        *** max_features (str/int (1:+∞)/float (0:1]) = ("auto", "sqrt", "log2"), максимальное число/доля признаков,
            рассматриваемых при разбиении узла, by default = 'auto';
        
        *** max_leaf_nodes (NoneType/int) = максимальное число листовых вершин дерева, by default = None;
        
        *** bootstrap (bool) = True/False, бутстрапировать ли выборку при построении дерева, by default = True;
        
        *** max_samples (NoneType/float (0:1) (можно и целые числа, на ограничимся дробными)) = при boostrap = True 
            доля выборки, которая будет бутстрапирована для тренировки каждого дерева, by default = None;
    
    threshold (float (0:+∞)) = порог удаления переменных в модели (признаки с feature_importances_ < threshold исключаются
        из нее), by default = 0.01;
    
    cv (bool) = True/False, проводить ли кросс-валидацию или оценить модель на полном дата-сете, by default = True;
    
    include_targeted (bool) = False/True, включать ли в качестве признаков таргет-закодированные колонки улиц и комнат 
        (черевато переобучением), by default = False;
    
    verbose (bool) = True/False, оповещать о выполнении расчетов на текущем шаге оптимизации и переходе к следующему,
        by default = True.
        
    ''' 
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    kf = KFold(n_splits = 10, shuffle = True, random_state = 42) # Делим на фолды
    global next_preds
    next_preds += 1
    # Готовим таблицу для результатов
    a = []
    # Несколько циклов, чтобы установить желаемый порядок колонок
    for i in range(0, 10): 
        a.append(f'mse{i}')
        a.append(f'mse_test{i}')
    for i in range(0, 10):
        a.append(f'r2_adj{i}')
        a.append(f'r2_adj_test{i}')
    for i in range(0, 10):    
        a.append(f'feat_imps{i}')
    a.append('features')
    a.append('feat_imps_mean')
    results = pd.DataFrame(columns = a)
    global predictions
    e = 0
    while True:
        step = 0
        for index_train, index_test in kf.split(data_train):
            # Делим фичи на категориальные и числовые
            train, test, val = processing(scaler = None, cv = True, index_train = index_train, index_test = index_test)
            
            X_train = pd.concat([train.iloc[:, other], pd.get_dummies(train.iloc[:, dum], drop_first = True)], axis = 1)
            X_test = pd.concat([test.iloc[:, other], pd.get_dummies(test.iloc[:, dum], drop_first = True)], axis = 1)
            X_val = pd.concat([val.iloc[:, other], pd.get_dummies(val.iloc[:, dum], drop_first = True)], axis = 1)
            
            # Если вдруг в каком-то из сетов нет колонок, которые есть в других
            X_test[X_val.columns.difference(X_test.columns).append(X_train.columns.difference(X_test.columns))] = 0
            X_train[X_val.columns.difference(X_train.columns).append(X_test.columns.difference(X_train.columns))] = 0
            X_val[X_train.columns.difference(X_val.columns).append(X_test.columns.difference(X_val.columns))] = 0
            
            # Избавляемся от случайных дубликатов, возникающих при некоторых условиях
            X_train = X_train.loc[:, ~X_train.columns.duplicated()]
            X_test = X_test.loc[:, ~X_test.columns.duplicated()]
            X_val = X_val.loc[:, ~X_val.columns.duplicated()]
            
            # Здесь не понадобятся логарифмы площадей, так как с ними результаты хуже
            X_train = X_train.drop(['kitchen_log_meaned', 'total_log', 'kitchen_log_medianed'], axis = 1)
            X_test = X_test.drop(['kitchen_log_meaned', 'total_log', 'kitchen_log_medianed',], axis = 1)
            X_val = X_val.drop(['kitchen_log_meaned', 'total_log', 'kitchen_log_medianed',], axis = 1)
            
            # Задаем объясняемую переменную
            y_train = np.log(train.price)
            y_test = np.log(test.price)
            y_val = np.log(val.price)
            # После нулевого шага оставляем только те признаки, которые проходят порог по важности
            if e != 0:
                X_train = X_train.loc[:, chosen_feats]
                X_test = X_test.loc[:, chosen_feats]
                X_val = X_val.loc[:, chosen_feats]
            shape_stopper = X_train.shape[1] # Запоминаем, сколько фичей используется на данном этапе
            # Определяем случайный лес
            frst = RandomForestRegressor(n_estimators = n_estimators,
                                         max_depth = max_depth,
                                         min_samples_split = min_samples_split,
                                         min_samples_leaf = min_samples_leaf,
                                         max_features = max_features,
                                         max_leaf_nodes = max_leaf_nodes,
                                         bootstrap = bootstrap,
                                         max_samples = max_samples,
                                         n_jobs = -1,
                                         random_state = 42)
            frst.fit(X_train, y_train)
            prs = frst.predict(X_test)
            prs_train = frst.predict(X_train)
            # Сохраняем пресдказания в датафрейм predictions
            if cv == True:
                predictions.at[test.index, f'preds{next_preds}_{e}'] = prs
                if next_preds % 100 == 0: # Так как иногда происходит дефрагментация, перидоически перезаписываем фрейм
                    predictions = predictions.copy()
                    
            ### Сохраняем метрики
            results.loc[e, f'mse{step}'] = mean_squared_error(prs_train, y_train)
            results.loc[e, f'mse_test{step}'] = mean_squared_error(prs, y_test)
            if X_train.shape[1] != 1:
                results.loc[e, f'r2_adj{step}'] = (1-(1-r2_score(prs_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
                results.loc[e, f'r2_adj_test{step}'] = (1-(1-r2_score(prs, y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
            else:
                results.loc[e, f'r2_adj{step}'] = r2_score(prs_train, y_train)
                results.loc[e, f'r2_adj_test{step}'] = r2_score(prs, y_test)
            results.at[e, f'feat_imps{step}'] = frst.feature_importances_ # Запоминаем важности признаков
            step += 1 # Переходим к следующему шагу кросс-валидации
        results.at[e, 'features'] = np.array(X_train.columns) # Запоминаем имена признаков
        # Находим средние важности по фолдам
        results.loc[e, 'feat_imps_mean'] = (results.loc[e, 'feat_imps0'] + results.loc[e, 'feat_imps1'] + results.loc[e, 'feat_imps2'] + results.loc[e, 'feat_imps3'] + \
                                            results.loc[e, 'feat_imps4'] + results.loc[e, 'feat_imps5'] + results.loc[e, 'feat_imps6'] + results.loc[e, 'feat_imps7'] + \
                                            results.loc[e, 'feat_imps8'] + results.loc[e, 'feat_imps9'])/10
        # Оставляем только фичи, средняя важность для которых выше определенного порога
        chosen_feats = results.features[e][results.feat_imps_mean[e] >= threshold]
        if verbose == True: # Оповещаем о переходе к следующему шагу алгоритма
            print(e)
        e += 1 # Переходим к следующему шагу отбора признаков
        if len(chosen_feats) == shape_stopper: # Заканчиваем оптимизировать, если не отобралось новое число фичей
            break
    results[['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'max_leaf_nodes', 'bootstrap', 'max_samples', 'threshold']] = \
    [n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, max_leaf_nodes, bootstrap, max_samples, threshold]
    
    results['mean_mse'] = results.apply(lambda x: np.mean([x[f'mse{i}'] for i in range(10)]), axis = 1)
    results['mean_mse_test'] = results.apply(lambda x: np.mean([x[f'mse_test{i}'] for i in range(10)]), axis = 1)
    results['mean_r2'] = results.apply(lambda x: np.mean([x[f'r2_adj{i}'] for i in range(10)]), axis = 1)
    results['mean_r2_test'] = results.apply(lambda x: np.mean([x[f'r2_adj_test{i}'] for i in range(10)]), axis = 1)
    results['std_mse'] = results.apply(lambda x: np.std([x[f'mse{i}'] for i in range(10)]), axis = 1)
    results['std_mse_test'] = results.apply(lambda x: np.std([x[f'mse_test{i}'] for i in range(10)]), axis = 1)
    results['std_r2'] = results.apply(lambda x: np.std([x[f'r2_adj{i}'] for i in range(10)]), axis = 1)
    results['std_r2_test'] = results.apply(lambda x: np.std([x[f'r2_adj_test{i}'] for i in range(10)]), axis = 1)
    # Возвращаем результаты
    if cv == True:
        return(results)
    else:
        train = processing(scaler = None, cv = False)
        X_train = pd.concat([train.iloc[:, other], pd.get_dummies(train.iloc[:, dum], drop_first = True)], axis = 1)    
        # Избавляемся от случайных дубликатов, возникающих при некоторых условиях
        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
        # Включаем только те переменные, которые соответствуют лучшей регрессии
        X_train = X_train.loc[:, results.loc[results.mean_mse_test.idxmin(), 'features']]
        # Задаем объясняемую переменную
        y_train = np.log(train.price)
       
        frst.fit(X_train, y_train)
        results = results.drop(results.columns[np.r_[0:52, 61:69]], axis =  1).loc[results.mean_mse_test.idxmin(), :]
        results.at['features'] = np.array(X_train.columns)
        results.at['importance'] = frst.feature_importances_
        return(results)



### № 5 Функция для обучения бустинговФункция для обучения линейных моделей , при кросс-валидации  возвращает табоицу с результатами, при обучении на всем сете - пандас серию \
### с результатами для лучшего шага отбора признаков
def boosting_model(iterations = None,
                   depth = None,
                   learning_rate = None,
                   random_strength = None,
                   bagging_temperature = None,
                   border_count = None,
                   l2_leaf_reg = None,
                   grow_policy = 'SymmetricTree',
                   threshold = 0.01,
                   cv = True,
                   verbose = True):
    '''
    Параметры от iterations до od_type соотвествуют параметрам catboost.CatBoostRegressor():
     (значения по умолчанию тоже совпадают)
     
        *** iterations (int (1:+∞)) = максимальное число используемых в ансамбле деревьев, по умолчанию = None; 
        
        *** depth (int (1:+∞)) = максимальная глубина дерева, по умолчанию = None;
        
        *** learning_rate (float (0:1]) = скорость сходимости = None;
        
        *** random_strength (float (0:1]) = коэффициент, на который умножается дисперсия шума, добавляющегося к метрике 
            при выборе лучших разбиений, по умолчанию = 1;
        
        *** bagging_temperature (float [0:+∞]) = доля выборки, которая будет бутстрапирована, по умолчанию = None;
        
        *** border_count (int [1,65535]) = количество разбиений для числовых признаков, по умолчанию = 254;
        
        *** l2_leaf_reg (float [0,+inf]) = коэффициент L2 регуляризации, по умолчанию = 3;
        
        *** grow_policy (str 'SymmetricTree'/'Depthwise') = стратегия построения деревьев: Symmetric для симметричных
            деревьев, Depthwise для построение по уровням до достижения выбранной глубины, по умолчанию = 'SymmetricTree';
    
    threshold (float (0:+∞)) = порог удаления переменных в модели (признаки с feature_importances_ < threshold исключаются
        из нее), по умолчанию = 0.01;
    
    cv (bool) = True/False, проводить ли кросс-валидацию или оценить модель на полном дата-сете, по умолчанию = True;
    
    verbose (bool) = True/False, оповещать о выполнении расчетов на текущем шаге оптимизации и переходе к следующему,
        по умолчанию = True.
        
    ''' 
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    kf = KFold(n_splits = 10, shuffle = True, random_state = 42) # Делим на фолды
    global next_preds_bstr
    next_preds_bstr += 1
    # Готовим таблицу для результатов
    a = []
    # Несколько циклов, чтобы установить желаемый порядок колонок
    for i in range(0, 10): 
        a.append(f'mse{i}')
        a.append(f'mse_test{i}')
    for i in range(0, 10):
        a.append(f'r2_adj{i}')
        a.append(f'r2_adj_test{i}')
    for i in range(0, 10):    
        a.append(f'feat_imps{i}')
    a.append('features')
    a.append('feat_imps_mean')
    results = pd.DataFrame(columns = a)
    global predictions_bstr
    e = 0
    while True:
        step = 0
        for index_train, index_test in kf.split(data_train):
            train, test, val = processing(scaler = None, cv = True, index_train = index_train, index_test = index_test)
            
            # Так как для catboost-а не нужно ohe-кодирование, пропускаем эти шаги, удаляем логарифмы и цену
            X_train = train.drop(['price', 'kitchen_log_meaned', 'total_log', 'kitchen_log_medianed'], axis = 1)
            X_test = test.drop(['price', 'kitchen_log_meaned', 'total_log', 'kitchen_log_medianed',], axis = 1)
            X_val = val.drop(['price', 'kitchen_log_meaned', 'total_log', 'kitchen_log_medianed',], axis = 1)
            # Если вдруг в каком-то из сетов нет колонок, которые есть в других
            X_test[X_val.columns.difference(X_test.columns).append(X_train.columns.difference(X_test.columns))] = 0
            X_train[X_val.columns.difference(X_train.columns).append(X_test.columns.difference(X_train.columns))] = 0
            X_val[X_train.columns.difference(X_val.columns).append(X_test.columns.difference(X_val.columns))] = 0
            
            # Избавляемся от случайных дубликатов, возникающих при некоторых условиях
            X_train = X_train.loc[:, ~X_train.columns.duplicated()]
            X_test = X_test.loc[:, ~X_test.columns.duplicated()]
            X_val = X_val.loc[:, ~X_val.columns.duplicated()]
            
            X_val_st = X_val.copy()
            # Задаем объясняемую переменную
            y_train = np.log(train.price)
            y_test = np.log(test.price)
            y_val = np.log(val.price)
            
            # После нулевого шага оставляем только те признаки, которые проходят порог по важности
            if e != 0:
                X_train = X_train.loc[:, chosen_feats]
                X_test = X_test.loc[:, chosen_feats]
                X_val = X_val.loc[:, chosen_feats]
                X_train.info()
                print(chosen_feats)
            shape_stopper = X_train.shape[1] # Запоминаем, сколько фичей используется на данном этапе
            # Определяем бустинг
            bstr = ctb.CatBoostRegressor(iterations = iterations, 
                                         depth = depth, 
                                         learning_rate = learning_rate,
                                         random_strength = random_strength,
                                         bagging_temperature = bagging_temperature,
                                         border_count = border_count,
                                         l2_leaf_reg = l2_leaf_reg, 
                                         grow_policy = grow_policy,
                                         cat_features = list(X_train.columns[X_train.dtypes == 'O']),
                                         early_stopping_rounds = 30,
                                         random_state = 42)
            bstr.fit(X_train, y_train, verbose = False, eval_set = ctb.Pool(X_val, y_val, cat_features = list(X_val.columns[X_val.dtypes == 'O'])))
            prs = bstr.predict(X_test)
            prs_train = bstr.predict(X_train)
            if cv == True:
                predictions_bstr.at[test.index, f'preds{next_preds_bstr}_{e}'] = prs
                if next_preds_bstr % 100 == 0:
                    predictions_bstr = predictions_bstr.copy()
            # Считаем метрики
            results.loc[e, f'mse{step}'] = mean_squared_error(prs_train, y_train)
            results.loc[e, f'mse_test{step}'] = mean_squared_error(prs, y_test)
            if X_train.shape[1] != 1:
                results.loc[e, f'r2_adj{step}'] = (1-(1-r2_score(prs_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
                results.loc[e, f'r2_adj_test{step}'] = (1-(1-r2_score(prs, y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
            else:
                results.loc[e, f'r2_adj{step}'] = r2_score(prs_train, y_train)
                results.loc[e, f'r2_adj_test{step}'] = r2_score(prs, y_test)
            # Запоминаем важности признаков
            results.at[e, f'feat_imps{step}'] = permutation_importance(bstr, X_val, y_val, n_repeats = 10, random_state = 42).importances_mean
            # Переходим к следующему шагу кросс-валидации
            step += 1 
        # Запоминаем имена признаков
        results.at[e, 'features'] = bstr.feature_names_ 
        # Находим средние важности по фолдам
        results.loc[e, 'feat_imps_mean'] = (results.loc[e, 'feat_imps0'] + results.loc[e, 'feat_imps1'] + results.loc[e, 'feat_imps2'] + results.loc[e, 'feat_imps3'] + \
                                            results.loc[e, 'feat_imps4'] + results.loc[e, 'feat_imps5'] + results.loc[e, 'feat_imps6'] + results.loc[e, 'feat_imps7'] + \
                                            results.loc[e, 'feat_imps8'] + results.loc[e, 'feat_imps9'])/10
        # Оставляем только фичи, средняя важность для которых выше определенного порога
        chosen_feats = np.array(results.features[e])[results.feat_imps_mean[e] >= threshold]
        if verbose == True: # Оповещаем о переходе к следующему шагу алгоритма
            print(e)
        e += 1 # Переходим к следующему шагу отбора признаков
        if len(chosen_feats) == shape_stopper: # Заканчиваем оптимизировать, если не отобралось новое число фичей
            break
    results[['iterations', 'depth', 'learning_rate', 'random_strength', 'bagging_temperature', 'border_count', 'l2_leaf_reg', 'grow_policy', 'threshold']] = \
    [iterations, depth, learning_rate, random_strength, bagging_temperature, border_count, l2_leaf_reg, grow_policy, threshold]
    
    results['mean_mse'] = results.apply(lambda x: np.mean([x[f'mse{i}'] for i in range(10)]), axis = 1)
    results['mean_mse_test'] = results.apply(lambda x: np.mean([x[f'mse_test{i}'] for i in range(10)]), axis = 1)
    results['mean_r2'] = results.apply(lambda x: np.mean([x[f'r2_adj{i}'] for i in range(10)]), axis = 1)
    results['mean_r2_test'] = results.apply(lambda x: np.mean([x[f'r2_adj_test{i}'] for i in range(10)]), axis = 1)
    results['std_mse'] = results.apply(lambda x: np.std([x[f'mse{i}'] for i in range(10)]), axis = 1)
    results['std_mse_test'] = results.apply(lambda x: np.std([x[f'mse_test{i}'] for i in range(10)]), axis = 1)
    results['std_r2'] = results.apply(lambda x: np.std([x[f'r2_adj{i}'] for i in range(10)]), axis = 1)
    results['std_r2_test'] = results.apply(lambda x: np.std([x[f'r2_adj_test{i}'] for i in range(10)]), axis = 1)
    # Возвращаем результаты
    if cv == True:
        return(results)
    else:
        train = processing(scaler = None, cv = False)
        X_train = train.drop(['price', 'kitchen_log_meaned', 'total_log', 'kitchen_log_medianed'], axis = 1)
        # Избавляемся от случайных дубликатов, возникающих при некоторых условиях
        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
        # Включаем только те переменные, которые соответствуют лучшей регрессии
        X_train = X_train.loc[:, results.loc[results.mean_mse_test.idxmin(), 'features']]
        X_val = X_val_st.loc[:, results.loc[results.mean_mse_test.idxmin(), 'features']]
        # Задаем объясняемую переменную
        y_train = np.log(train.price)
        
        bstr = ctb.CatBoostRegressor(iterations = iterations, 
                                     depth = depth, 
                                     learning_rate = learning_rate, 
                                     random_strength = random_strength, 
                                     bagging_temperature = bagging_temperature, 
                                     border_count = border_count, 
                                     l2_leaf_reg = l2_leaf_reg, 
                                     grow_policy = grow_policy, 
                                     cat_features = list(X_train.columns[X_train.dtypes == 'O']), 
                                     early_stopping_rounds = 30, 
                                     random_state = 42)
        bstr.fit(X_train, y_train, verbose = False, eval_set = ctb.Pool(X_val, y_val, cat_features = list(X_val.columns[X_val.dtypes == 'O'])))
        return(bstr, results.loc[results.mean_mse_test.idxmin(), 'feat_imps_mean'])

