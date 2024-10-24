import pandas as pd
from Get_Dataset import Get_Dataset


class Data:
    def __init__(self, N):
        pass

    def split(dataset, split_date):
        train = dataset.loc[dataset.index < split_date]
        test  = dataset.loc[dataset.index >= split_date]

        return train, test

    ### trendacd
    from trendacd_py import trendacd
    def trend_ACD(data):
        trend = pd.Series(trendacd(data.values, ddof=0))
        trend.index = data.index
        return trend

    def dannye(N, cdata, save_flag=False, save_path='Data/'):
        plt.figure(figsize=(15, 10), dpi=100)
        font2 = {'family':'serif','color':'black','size':14}

        type_dataset = ''
        if N == 1:
            type_dataset = 'заработной плате по месяцам'
        if N == 2:
            type_dataset = 'денежным доходам населения по месяцам'
        if N == 3:
            type_dataset = 'индексу цен на грузовые перевозки по месяцам'
        if N == 4:
            type_dataset = 'индексу реального объема сельскохозяйственного производства по месяцам'
        if N == 5:
            type_dataset = 'индексу производства и распределения электроэнергии, газа и воды по месяцам'
        if N == 6:
            type_dataset = 'акциям Сбербанка по дням'
        if N == 7:
            type_dataset = 'акциям Яндекса по дням'

        '''
        plt.plot(cdata, color='black')
        plt.title(''.join(['График временного ряда для набора данных по ',type_dataset,'.']), fontdict = font2)
        if save_flag:
            plt.savefig(''.join([save_path,'dataset']))
        '''
        
        from statsmodels.tsa.seasonal import STL
        # Декомпозиция
        stl = STL(cdata)
        stlfit = stl.fit()
        data = stlfit.seasonal + stlfit.trend
        dataACD = stlfit.seasonal + trend_ACD(data)

        # Разделение выборки
        split_last_year = pd.to_datetime(str(data.index.year[-1]))
        if len(data.iloc[data.index.year==data.index.year[-1]]) < 12:
            split_last_year = pd.to_datetime(str(data.index.year[-2]))
        train, test = split(data, split_last_year)
        trainACD, testACD = split(dataACD, split_last_year)

        return stlfit, data, train, test, dataACD, trainACD, testACD

    def seasonality(N, stlfit, data, test, save_flag=False, save_path='Data/'):
        seasonal = stlfit.seasonal
        #plot_compare2(data, seasonal, legend1='data', legend2='seasonal', title='seasonality', title_rus='Сезонность временного ряда.', save_flag=save_flag, save_path=save_path)

        if N<6:
            vl = []
            av = seasonal.groupby(seasonal.index.month).median()
            for i in test.index.month:
                vl.append(av.iloc[av.index == i].iloc[0])
        else:
            vl = np.zeros(len(test.index.day))

        seasonal_median = pd.Series(data=vl, index=test.index)
        #plot_compare2(seasonal, seasonal_median, legend1='seasonal', legend2='seasonal_median', title='seasonal_median', title_rus='Медианная сезонность для одного сезонного интервала.', save_flag=save_flag, save_path=save_path)

        return seasonal_median