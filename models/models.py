import inspect
import models.classic
import models.trends

import pandas as pd
from statsmodels.tsa.seasonal import STL
from trendacd_py import trendacd


class Model:
    def __init__(self, N, dataset, seasonal_median, is_seasonal=True, auto=True): 
        self.N = N
        self.dataset = dataset
        self.seasonal_median = seasonal_median
        self.is_seasonal = is_seasonal
        self.auto = auto
        
        self.train, self.test = self.data_preprocessing(self.dataset)
        
        self.init_types_of_models()
        self.init_list_of_models()
        
    def available_models(self):
        return self.models_types.keys()
    
    def available_predicted_models(self):
        return self.models_list.keys()
    
    def init_types_of_models(self):
        classes_classic = inspect.getmembers(models.classic, inspect.isclass)
        classes_trends = inspect.getmembers(models.trends, inspect.isclass)

        d = dict(classes_classic)
        d.update(dict(classes_trends))
        classes = list(d.items())

        self.models_types = {name: cls for name, cls in classes if cls.__module__ == models.classic.__name__ or cls.__module__ == models.trends.__name__}
        
    def init_list_of_models(self):        
        models_list = dict()
        for name, clss in self.models_types.items():
            if name in ['Model_ETS','Model_SARIMAX']:
                elem = clss(self.N, self.dataset, self.train, self.test, self.seasonal_median, auto=self.auto)
                models_list[elem.name_eng] = elem
                #print(f'Created model : {name}')
                
                dataACD, trainACD, testACD = self.dataACD_preprocessing(self.dataset)
                elem = clss(self.N, dataACD, trainACD, self.test, self.seasonal_median, auto=self.auto, type_data_ACD=True)
                models_list[elem.name_eng] = elem
                #print(f'Created model : {name}')
                
            elif name in ['Poly_LinearRegression','Poly_SARIMAX']:
                for degree in range(1,6):
                    elem = clss(self.N, self.dataset, self.train, self.test, self.seasonal_median, degree=degree, auto=self.auto)
                    models_list[elem.name_eng] = elem
                    #print(f'Created model : {name}')
            else:
                elem = clss(self.N, self.dataset, self.train, self.test, self.seasonal_median, auto=self.auto)
                models_list[elem.name_eng] = elem
                #print(f'Created model : {name}')
                
        self.models_list = models_list
    
    def get_last_year_data(self, data):
        split_last_year = pd.to_datetime(str(data.index.year[-1]))
        if len(data.iloc[data.index.year==data.index.year[-1]]) < 12:
            split_last_year = pd.to_datetime(str(data.index.year[-2]))
        return split_last_year
    
    def split(self, dataset, split_date):
        train = dataset.loc[dataset.index < split_date]
        test  = dataset.loc[dataset.index >= split_date]
        return train, test
    
    def trend_ACD(self, data):
        return pd.Series(data=trendacd(data.values, ddof=0), index=data.index)
    
    def data_preprocessing(self, data):
        #Разделение выборки
        split_last_year = self.get_last_year_data(data)
        train, test = self.split(data, split_last_year)
        return train, test
    
    def dataACD_preprocessing(self, data):
        #Данные с трендом ACD
        stlfit = STL(data).fit()
        dataACD = stlfit.seasonal + self.trend_ACD(data)
        
        #Разделение выборки
        split_last_year = self.get_last_year_data(data)
        trainACD, testACD = self.split(dataACD, split_last_year)
        return dataACD, trainACD, testACD