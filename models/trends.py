from models.base import Trend

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
#import pylab as pl     #fourier
#from numpy import fft  #fourier
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from trendacd_py import trendacd

    
### Полиномиальная функция (линейная, параболическая,...)
class Poly_LinearRegression(Trend):    
    def __init__(self, N, dataset, train, test, seasonal_median, degree=1, is_seasonal=True, auto=True):
        self.degree = degree
        self.name_eng = ''.join(['polyf', str(self.degree)])
        self.name_rus = ' '.join(['Многочлен', str(self.degree), 'степени'])
        
        super().__init__(N, dataset, train, test, seasonal_median, is_seasonal=is_seasonal, auto=auto)
    
    def fit(self):
        x_train = range(len(self.train))
        y_train = self.train.values
        
        z = np.polyfit(x_train, y_train, self.degree)
        self.fitted_model = np.poly1d(z)

        train_px = pd.Series(data=self.fitted_model(x_train), index=self.train.index)
        self.model = train_px
        
    def pred(self):
        x_test  = [i+len(self.train) for i in range(len(self.test))]
        
        test_px = pd.Series(data=self.fitted_model(x_test), index=self.test.index)
        
        if self.is_seasonal:
            self.predict = self.seasonal_median + test_px
        else:
            self.predict = test_px


### Полиномиальный тренд + SARIMAX
class Poly_SARIMAX(Trend):
    def __init__(self, N, dataset, train, test, seasonal_median, degree=1, is_seasonal=True, auto=True):
        self.degree = degree
        self.name_eng = ''.join(['polyf', str(self.degree), ' ', 'SARIMAX'])
        self.name_rus = ' '.join(['Многочлен', str(self.degree), 'степени','+','ARIMA'])
        
        super().__init__(N, dataset, train, test, seasonal_median, is_seasonal=is_seasonal, auto=auto)
    
    def fit(self):
        x = range(len(self.dataset))
        y = self.dataset.values

        z = np.polyfit(x, y, self.degree)
        self.fitted_model = np.poly1d(z)
        
        px = pd.Series(data=self.fitted_model(x), index=self.dataset.index)
        self.model = px
        
    def pred(self):
        #SARIMAX
        self.predict, self.fitted_model, self.order, self.seasonal_order = super().sarimax_predictions(self.model, self.test, ''.join([str(self.N),self.name_eng]), self.auto)
        
        if self.is_seasonal:
            self.predict = self.seasonal_median + self.predict


### Гиперболическая функция
class Hyperbolic(Trend):
    def __init__(self, N, dataset, train, test, seasonal_median, is_seasonal=True, auto=True):
        self.name_eng = 'hypf'
        self.name_rus = 'Гиперболическая функция'
        
        super().__init__(N, dataset, train, test, seasonal_median, is_seasonal=is_seasonal, auto=auto)
        
    def fit(self):
        x_train = range(1,len(self.train)+1)
        y_train = self.train.values
        div_x_train = [1/t for t in x_train]
        
        z = np.polyfit(div_x_train, y_train, 1)
        self.fitted_model = np.poly1d(z)

        train_px = pd.Series(data=self.fitted_model(div_x_train), index=self.train.index)
        self.model = train_px
        
    def pred(self):
        x_test  = [i+len(self.train) for i in range(len(self.test))]
        div_x_test  = [1/t for t in x_test]
        
        test_px = pd.Series(data=self.fitted_model(div_x_test), index=self.test.index)
        
        if self.is_seasonal:
            self.predict = self.seasonal_median + test_px
        else:
            self.predict = test_px

            
### Логарифмическая функция
class Logarithmic(Trend):
    def __init__(self, N, dataset, train, test, seasonal_median, is_seasonal=True, auto=True):
        self.name_eng = 'logf'
        self.name_rus = 'Логарифмическая функция'
        
        super().__init__(N, dataset, train, test, seasonal_median, is_seasonal=is_seasonal, auto=auto)
    
    def fit(self):
        x_train = range(1,len(self.train)+1)
        y_train = self.train.values
        
        self.fitted_model = self.logFit(x_train, y_train)
        
        train_px = pd.Series(data=self.logFunc(x_train, *self.fitted_model), index=self.train.index)
        self.model = train_px
    
    def pred(self):
        x_test  = [i+len(self.train) for i in range(len(self.test))]
        
        test_px = pd.Series(data=self.logFunc(x_test, *self.fitted_model), index=self.test.index)
        
        if self.is_seasonal:
            self.predict = self.seasonal_median + test_px
        else:
            self.predict = test_px

    def logFit(self, x, y):
        # cache some frequently reused terms
        sumy = np.sum(y)
        sumlogx = np.sum(np.log(x))

        b = (len(x) * np.sum(y * np.log(x)) - sumy * sumlogx) / (len(x) * np.sum(np.log(x)**2) - sumlogx**2)
        a = (sumy - b * sumlogx) / len(x)

        return (a, b)

    def logFunc(self, x, a, b):
        return a + b * np.log(x)
    

#Экспоненциальная функция
class Exponential(Trend):
    def __init__(self, N, dataset, train, test, seasonal_median, is_seasonal=True, auto=True):
        self.name_eng = 'expf'
        self.name_rus = 'Экспоненциальная функция'
        
        super().__init__(N, dataset, train, test, seasonal_median, is_seasonal=is_seasonal, auto=auto)
    
    def fit(self):
        x_train = range(1,len(self.train)+1)
        y_train = self.train.values

        z = np.polyfit(x_train, np.log(y_train), 1)
        self.fitted_model = np.poly1d(z)

        train_px = pd.Series(data=np.exp(self.fitted_model(x_train)), index=self.train.index)
        self.model = train_px
        
    def pred(self):
        x_test  = [i+len(self.train) for i in range(len(self.test))]
        test_expf  = np.exp(self.fitted_model(x_test))

        test_px = pd.Series(data=test_expf, index=self.test.index)
        
        if self.is_seasonal:
            self.predict = self.seasonal_median + test_px
        else:
            self.predict = test_px
            

#Функция сплайнов
class Spline(Trend):
    def __init__(self, N, dataset, train, test, seasonal_median, is_seasonal=True, auto=True):
        self.name_eng = 'spline'
        self.name_rus = 'Функция сплайнов'
        
        super().__init__(N, dataset, train, test, seasonal_median, is_seasonal=is_seasonal, auto=auto)
    
    def fit(self):
        x = range(len(self.dataset))
        y = self.dataset.values
        
        spl = CubicSpline(x, y) #Кубические сплайны
    
        trend_px = pd.Series(data=spl(x), index=self.dataset.index)
        self.model = trend_px
    
    def pred(self):
        #SARIMAX
        self.predict, self.fitted_model, self.order, self.seasonal_order = super().sarimax_predictions(self.model, self.test, ''.join([str(self.N),self.name_eng]), self.auto)
    
        if self.is_seasonal:
            self.predict = self.seasonal_median + self.predict
            

#Разложение в ряд Фурье
#https://gist.github.com/tartakynov/83f3cd8f44208a1856ce
class Fourier(Trend):
    def __init__(self, N, dataset, train, test, seasonal_median, is_seasonal=True, auto=True):
        self.name_eng = 'fft'
        self.name_rus = 'Разложение в ряд Фурье'
        
        super().__init__(N, dataset, train, test, seasonal_median, is_seasonal=is_seasonal, auto=auto)
    
    def fit(self):
        trend_px = pd.Series(data=self.fourierExtrapolation(self.dataset.values, 0), index=self.dataset.index)
        self.model = trend_px
    
    def pred(self):
        #SARIMAX
        self.predict, self.fitted_model, self.order, self.seasonal_order = super().sarimax_predictions(self.model, self.test, ''.join([str(self.N),self.name_eng]), self.auto)
    
        if self.is_seasonal:
            self.predict = self.seasonal_median + self.predict

    def fourierExtrapolation(self, x, n_predict):
        n = x.size
        n_harm = 10                     # number of harmonics in model
        t = np.arange(0, n)
        p = np.polyfit(t, x, 1)         # find linear trend in x
        x_notrend = x - p[0] * t        # detrended x
        x_freqdom = np.fft.fft(x_notrend)  # detrended x in frequency domain
        f = np.fft.fftfreq(n)              # frequencies
        indexes = list(range(n))
        # sort indexes by frequency, lower -> higher
        indexes.sort(key = lambda i: np.absolute(f[i]))

        t = np.arange(0, n + n_predict)
        restored_sig = np.zeros(t.size)
        for i in indexes[:1 + n_harm * 2]:
            ampli = np.absolute(x_freqdom[i]) / n   # amplitude
            phase = np.angle(x_freqdom[i])          # phase
            restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
        return restored_sig + p[0] * t

    
### Экспоненциальное сглаживание
class ExpSmooth(Trend):
    def __init__(self, N, dataset, train, test, seasonal_median, is_seasonal=True, auto=True):
        self.name_eng = 'exps'
        self.name_rus = 'Экспоненциальное сглаживание'
        
        super().__init__(N, dataset, train, test, seasonal_median, is_seasonal=is_seasonal, auto=auto)
    
    def fit(self):
        model = ExponentialSmoothing(self.train, initialization_method="estimated")
        smootharr = [np.round(i,1) for i in np.arange(1,0,-0.1)]
        fitt = [model.fit(smoothing_level=i) for i in smootharr]
        fcast = [fitt[i].forecast(len(self.test)) for i in range(len(fitt))]
        
        maer  = np.zeros(len(smootharr))
        rmser = np.zeros(len(smootharr))
        for i in range(len(fitt)):
            maer[i], rmser[i] = super().metric(self.test, fcast[i])
            
        i = np.where(rmser==min(rmser))[0][0]
        self.fitted_model = fitt[i]
        self.model = fitt[i].fittedvalues
    
    def pred(self):
        if self.is_seasonal:
            self.predict = self.seasonal_median + self.fitted_model.forecast(len(self.test))
        else:
            self.predict = self.fitted_model.forecast(len(self.test))
    
    def plot_levels(self, show_flag=True, save_flag=False, save_path='Data/'):
        model = ExponentialSmoothing(self.train, initialization_method="estimated")
        smootharr = [np.round(i,1) for i in np.arange(1,0,-0.1)]
        fitt = [model.fit(smoothing_level=i) for i in smootharr]
        fcast = [fitt[i].forecast(len(test)) for i in range(len(fitt))]
        
        maer  = np.zeros(len(smootharr))
        rmser = np.zeros(len(smootharr))
        
        cmap = plt.get_cmap('Greys')#'gist_gray')
        colors = [cmap(i) for i in np.linspace(0.5, 1, len(fitt))]
        markers = ['o', 'v', '^', '1', '>', 's', 'p', '*', '+', '3', 'd', 'h', '4', 'H', '.', '<', '2', 'x', 'o', 'D', 'v', '^']

        fig = plt.figure(figsize=(15, 10), dpi=100)
        plt.plot(data, color='black', alpha=0.8, linestyle='-', label='data')
        xlabel = ''
        for i in range(len(fitt)):
            maer[i], rmser[i] = super().metric(self.test, fcast[i])
            plt.plot(fitt[i].fittedvalues, color=colors[i], alpha=0.5, linestyle='-', marker = markers[i], markersize=3, label='='.join(['sm',str(smootharr[i])]))
            plt.plot(fcast[i],             color=colors[i], alpha=0.5, linestyle='-', marker = markers[i], markersize=3)
            xlabel = '\n'.join([xlabel,f'smooth = {smootharr[i]} : (MAE, RMSE) = {maer[i], rmser[i]}'])

        #best smoothing
        #i = np.where(maer==min(maer))[0][0]
        i = np.where(rmser==min(rmser))[0][0]
        xlabel = '\n'.join([xlabel,f'\nBest smooth = {smootharr[i]} with (MAE, RMSE) = {maer[i], rmser[i]}'])

        font2 = {'family':'serif','color':'black','size':14}
        plt.xlabel(xlabel, loc='left', fontdict = font2)
        plt.title('Графики для всех уровней экспоненциального сглаживания.', fontdict = font2)
        plt.legend()
        
        if not show_flag: plt.close()
        if save_flag: fig.savefig(''.join([save_path,'exps_smoothing_level.png']), bbox_inches='tight')


#Метод LOESS
class LOESS(Trend):
    def __init__(self, N, dataset, train, test, seasonal_median, is_seasonal=True, auto=True):
        self.name_eng = 'LOESS'
        self.name_rus = 'Метод LOESS'
        
        super().__init__(N, dataset, train, test, seasonal_median, is_seasonal=is_seasonal, auto=auto)
    
    def fit(self):
        stlfit = STL(self.dataset).fit()
        self.model = stlfit.trend
    
    def pred(self):
        #SARIMAX
        self.predict, self.fitted_model, self.order, self.seasonal_order = super().sarimax_predictions(self.model, self.test, ''.join([str(self.N),self.name_eng]), self.auto)
    
        if self.is_seasonal:
            self.predict = self.seasonal_median + self.predict
            
            
#Алгоритм ACD
class ACD(Trend):
    def __init__(self, N, dataset, train, test, seasonal_median, is_seasonal=True, auto=True):
        self.name_eng = 'ACD'
        self.name_rus = 'Алгоритм ACD'
        
        super().__init__(N, dataset, train, test, seasonal_median, is_seasonal=is_seasonal, auto=auto)
    
    def fit(self):
        trend = pd.Series(data=trendacd(self.dataset.values, ddof=0), index=self.dataset.index)
        self.model = trend
    
    def pred(self):
        #SARIMAX
        self.predict, self.fitted_model, self.order, self.seasonal_order = super().sarimax_predictions(self.model, self.test, ''.join([str(self.N),self.name_eng]), self.auto)
    
        if self.is_seasonal:
            self.predict = self.seasonal_median + self.predict