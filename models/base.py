from abc import ABC, abstractmethod

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Pattern(ABC):
    def __init__(self, N, dataset, train, test, seasonal_median, is_seasonal=True, auto=True):
        self.N = N
        self.dataset = dataset
        self.train = train
        self.test = test
        self.seasonal_median = seasonal_median
        self.is_seasonal = is_seasonal
        self.auto = auto
        
        self.fitted_model = None
        self.model = None
        self.predict = None
        
        self.fit()
        self.pred()
    
    def sarimax_predictions(self, train, test, name, auto=True):
        order = (0,0,0)
        seasonal_order = (0,0,0,0)
        
        order_dict_without_d = {            
        '1Data SARIMAX' : (3, 0, 4),
        '1DataACD SARIMAX' : (2, 0, 4),
        '1ACD' : (3, 0, 1),
        '1fft' : (1, 0, 4),
        '1LOESS' : (2, 0, 0),
        '1polyf1 SARIMAX' : (4, 0, 0),
        '1polyf2 SARIMAX' : (2, 0, 0),
        '1polyf3 SARIMAX' : (2, 0, 0),
        '1polyf4 SARIMAX' : (4, 0, 0),
        '1polyf5 SARIMAX' : (3, 0, 0),
        '1spline' : (3, 0, 4),
            
        '2Data SARIMAX' : (2, 0, 4),
        '2DataACD SARIMAX' : (2, 0, 4),
        '2ACD' : (2, 0, 0),
        '2fft' : (1, 0, 4),
        '2LOESS' : (2, 0, 1),
        '2polyf1 SARIMAX' : (4, 0, 0),
        '2polyf2 SARIMAX' : (4, 0, 0),
        '2polyf3 SARIMAX' : (4, 0, 0),
        '2polyf4 SARIMAX' : (3, 0, 0),
        '2polyf5 SARIMAX' : (2, 0, 1),
        '2spline' : (4, 0, 4),
            
        '3Data SARIMAX' : (2, 0, 3),
        '3DataACD SARIMAX' : (3, 0, 4),
        '3ACD' : (3, 0, 2),
        '3fft' : (4, 0, 0),
        '3LOESS' : (3, 0, 0),
        '3polyf1 SARIMAX' : (2, 0, 4),
        '3polyf2 SARIMAX' : (4, 0, 0),
        '3polyf3 SARIMAX' : (4, 0, 0),
        '3polyf4 SARIMAX' : (1, 0, 4),
        '3polyf5 SARIMAX' : (1, 0, 4),
        '3spline' : (2, 0, 3),
            
        '4Data SARIMAX' : (4, 0, 2),
        '4DataACD SARIMAX' : (4, 0, 4),
        '4ACD' : (4, 0, 1),
        '4fft' : (1, 0, 4),
        '4LOESS' : (2, 0, 4),
        '4polyf1 SARIMAX' : (4, 0, 0),
        '4polyf2 SARIMAX' : (4, 0, 0),
        '4polyf3 SARIMAX' : (1, 0, 3),
        '4polyf4 SARIMAX' : (1, 0, 3),
        '4polyf5 SARIMAX' : (1, 0, 4),
        '4spline' : (4, 0, 2),
            
        '5Data SARIMAX' : (3, 0, 3),
        '5DataACD SARIMAX' : (3, 0, 3),
        '5ACD' : (2, 0, 0),
        '5fft' : (4, 0, 2),
        '5LOESS' : (2, 0, 0),
        '5polyf1 SARIMAX' : (1, 0, 3),
        '5polyf2 SARIMAX' : (2, 0, 4),
        '5polyf3 SARIMAX' : (1, 0, 3),
        '5polyf4 SARIMAX' : (1, 0, 3),
        '5polyf5 SARIMAX' : (3, 0, 0),
        '5spline' : (3, 0, 3),
            
        '6Data SARIMAX' : (3, 0, 3),
        '6DataACD SARIMAX' : (3, 0, 3),
        '6ACD' : (2, 0, 3),
        '6fft' : (2, 0, 4),
        '6LOESS' : (2, 0, 4),
        '6polyf1 SARIMAX' : (1, 0, 3),
        '6polyf2 SARIMAX' : (1, 0, 3),
        '6polyf3 SARIMAX' : (4, 0, 0),
        '6polyf4 SARIMAX' : (4, 0, 2),
        '6polyf5 SARIMAX' : (1, 0, 3),
        '6spline' : (3, 0, 3),
            
        '7Data SARIMAX' : (3, 0, 3),
        '7DataACD SARIMAX' : (3, 0, 3),
        '7ACD' : (2, 0, 4),
        '7fft' : (2, 0, 1),
        '7LOESS' : (1, 0, 4),
        '7polyf1 SARIMAX' : (1, 0, 3),
        '7polyf2 SARIMAX' : (2, 0, 4),
        '7polyf3 SARIMAX' : (2, 0, 4),
        '7polyf4 SARIMAX' : (1, 0, 3),
        '7polyf5 SARIMAX' : (1, 0, 3),
        '7spline' : (3, 0, 3),
        }
        
        order_dict_with_d = {
        '1Data SARIMAX' : (4, 2, 4),
        '1DataACD SARIMAX' : (4, 2, 4),
        '1ACD' : (0, 2, 1),
        '1fft' : (4, 1, 3),
        '1LOESS' : (4, 2, 0),
        '1polyf1 SARIMAX' : (4, 3, 3),
        '1polyf2 SARIMAX' : (3, 3, 3),
        '1polyf3 SARIMAX' : (3, 4, 0),
        '1polyf4 SARIMAX' : (2, 4, 2),
        '1polyf5 SARIMAX' : (3, 0, 0),
        '1spline' : (4, 2, 4),
        
        '2Data SARIMAX' : (2, 2, 4),
        '2DataACD SARIMAX' : (3, 2, 4),
        '2ACD' : (0, 2, 0),
        '2fft' : (1, 0, 4),
        '2LOESS' : (4, 2, 0),
        '2polyf1 SARIMAX' : (4, 3, 4),
        '2polyf2 SARIMAX' : (1, 3, 3),
        '2polyf3 SARIMAX' : (3, 3, 2),
        '2polyf4 SARIMAX' : (0, 4, 1),
        '2polyf5 SARIMAX' : (3, 5, 1),
        '2spline' : (2, 2, 4),
            
        '3Data SARIMAX' : (2, 0, 3),
        '3DataACD SARIMAX' : (3, 0, 4),
        '3ACD' : (3, 0, 2),
        '3fft' : (4, 1, 0),
        '3LOESS' : (3, 1, 0),
        '3polyf1 SARIMAX' : (3, 3, 2),
        '3polyf2 SARIMAX' : (2, 3, 2),
        '3polyf3 SARIMAX' : (0, 3, 0),
        '3polyf4 SARIMAX' : (1, 4, 0),
        '3polyf5 SARIMAX' : (1, 5, 2),
        '3spline' : (2, 0, 3),
        
        '4Data SARIMAX' : (3, 1, 3),
        '4DataACD SARIMAX' : (4, 0, 4),
        '4ACD' : (0, 2, 1),
        '4fft' : (2, 4, 2),
        '4LOESS' : (3, 1, 3),
        '4polyf1 SARIMAX' : (1, 3, 0),
        '4polyf2 SARIMAX' : (4, 3, 0),
        '4polyf3 SARIMAX' : (1, 3, 1),
        '4polyf4 SARIMAX' : (3, 4, 0),
        '4polyf5 SARIMAX' : (3, 5, 2),
        '4spline' : (3, 1, 3),
            
        '5Data SARIMAX' : (4, 1, 4),
        '5DataACD SARIMAX' : (4, 1, 4),
        '5ACD' : (0, 2, 1),
        '5fft' : (4, 0, 2),
        '5LOESS' : (2, 1, 0),
        '5polyf1 SARIMAX' : (2, 3, 2),
        '5polyf2 SARIMAX' : (0, 4, 2),
        '5polyf3 SARIMAX' : (0, 4, 2),
        '5polyf4 SARIMAX' : (3, 4, 3),
        '5polyf5 SARIMAX' : (1, 5, 2),
        '5spline' : (4, 1, 4),
            
        '6Data SARIMAX' : (3, 1, 4),
        '6DataACD SARIMAX' : (3, 2, 3),
        '6ACD' : (2, 2, 0),
        '6fft' : (3, 3, 1),
        '6LOESS' : (3, 1, 4),
        '6polyf1 SARIMAX' : (4, 2, 0),
        '6polyf2 SARIMAX' : (1, 3, 1),
        '6polyf3 SARIMAX' : (1, 3, 3),
        '6polyf4 SARIMAX' : (2, 4, 2),
        '6polyf5 SARIMAX' : (1, 0, 3),
        '6spline' : (2, 1, 4),

        '7Data SARIMAX' : (2, 1, 4),
        '7DataACD SARIMAX' : (1, 2, 3),
        '7ACD' : (2, 1, 3),
        '7fft' : (3, 1, 2),
        '7LOESS' : (3, 1, 2),
        '7polyf1 SARIMAX' : (1, 3, 1),
        '7polyf2 SARIMAX' : (3, 3, 2),
        '7polyf3 SARIMAX' : (3, 3, 0),
        '7polyf4 SARIMAX' : (0, 4, 1),
        '7polyf5 SARIMAX' : (1, 0, 3),
        '7spline' : (3, 1, 4),
        }
        
        if auto: order = self.get_params_hqic(self.model)
        else: order = order_dict_with_d[name]
        
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)

        predictions = model_fit.predict(start=test.index[0], end=test.index[-1])
        return predictions, model_fit, order, seasonal_order
    
    # Подбор коэффициентов с ниаменьшим HQIC.
    def get_params_hqic(self, train, d=0, p_min=0, p_max=4, q_min=0, q_max=4):
        d = self.find_d(train)
        p = -1
        q = -1
        minHQIC = float('+inf')
        for j in range(p_min, p_max+1):
            for i in range(q_min, q_max+1):
                #print(f'p={j}, q={i}')
                try:
                    model = SARIMAX(train, order=(j,d,i), seasonal_order=(0,0,0,0))#, enforce_stationarity=False, enforce_invertibility=False)
                    hqic = model.fit().hqic
                except:
                    pass
                if hqic < minHQIC:
                    minHQIC = hqic
                    p = j
                    q = i
        return (p,d,q)
    
    def find_d(self, data):
        d = 0
        dif_data = data.copy()
        while not(self.adf_test(dif_data)): #пока не выполнен ADF тест
            d += 1
            dif_data = dif_data.diff().dropna()
        return d
    
    def adf_test(self, data):
        adf = adfuller(data)
        hyp = (adf[1] < 0.05)
        return hyp
    
    def metric(self, a, b):
        MAE = mean_absolute_error(a, b)
        RMSE = mean_squared_error(a, b, squared=False)
        return round(MAE, 5), round(RMSE, 5)
    
    def LLF(self, data):
        # Calculate the log-likelihood value
        n = len(data)
        sigma2 = np.sum(data ** 2) / n
        result = -n / 2 * np.log(2 * np.pi * sigma2) - n / 2
        return round(result, 5)
        
    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def pred(self):
        pass

    
class Trend(Pattern):
    def __init__(self, N, dataset, train, test, seasonal_median, is_seasonal=True, auto=True):
        super().__init__(N, dataset, train, test, seasonal_median, is_seasonal=is_seasonal, auto=auto)
    
    def get_params_hqic(self, train, d=0, p_min=0, p_max=4, q_min=0, q_max=4):
        return super().get_params_hqic(train, d=d, p_min=p_min, p_max=p_max, q_min=q_min, q_max=q_max)
    
    def sarimax_predictions(self, train, test, order, seasonal_order):
        return super().sarimax_predictions(train, test, order, seasonal_order)
        
    def metric(self, a, b):
        return super().metric(a, b)
    
    def LLF(self, data):
        return super().LLF(data)
    
'''
    @abstractmethod
    def fit(self):
        #Построение модели тренда
        pass
    
    @abstractmethod
    def pred(self):
        #Предсказание модели тренда
        pass
'''