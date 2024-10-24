from models.base import Pattern

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import ETSModel


class Model_SARIMAX(Pattern):
    def __init__(self, N, dataset, train, test, seasonal_median, is_seasonal=True, auto=True, type_data_ACD=False):
        self.type_data_ACD = type_data_ACD
        if self.type_data_ACD:
            self.name_eng = 'DataACD SARIMAX'
            self.name_rus = 'Модель ARIMA для данных с трендом ACD'
        else:
            self.name_eng = 'Data SARIMAX'
            self.name_rus = 'Модель ARIMA для исходных данных'
            
        super().__init__(N, dataset, train, test, seasonal_median, is_seasonal=is_seasonal, auto=auto)
        
    def fit(self):
        self.model = self.train
        
    def pred(self):
        #SARIMAX
        self.predict, self.fitted_model, self.order, self.seasonal_order = super().sarimax_predictions(self.model, self.test, ''.join([str(self.N),self.name_eng]), self.auto)
        
        if self.is_seasonal:
            self.predict = self.seasonal_median + self.predict
        

class Model_ETS(Pattern):
    def __init__(self, N, dataset, train, test, seasonal_median, is_seasonal=False, auto=True, type_data_ACD=False):
        self.type_data_ACD = type_data_ACD
        if self.type_data_ACD:
            self.name_eng = 'DataACD ETS'
            self.name_rus = 'Модель ETS для данных с трендом ACD'
        else:
            self.name_eng = 'Data ETS'
            self.name_rus = 'Модель ETS для исходных данных'
        
        super().__init__(N, dataset, train, test, seasonal_median, is_seasonal=is_seasonal, auto=auto)
        
    def fit(self):
        self.model = self.train
        self.fitted_model = ETSModel(self.model, error="add", trend="add", seasonal="add", damped_trend=True, seasonal_periods=12).fit()
    
    def pred(self):
        self.predict = self.fitted_model.forecast(len(self.test))