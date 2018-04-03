import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, MinMaxScaler
try:
    profile  # throws an exception when profile isn't defined
except NameError:
    def profile(x): return x


class financial_data:
    """
    financial data class contains multiple different
    datasources from Quandl, CSVs and other sources.
    Which are cleaned, scaled and split in this class.
    This data will then pass into the prediction model.
    TODO:
    -Expand datasources available
    -Perform more error checking
    -Allow more customizable test train data
    """
    @profile
    def __init__(self, input_size, split=0.5):
        """[financial data object that parses various data formats]

        Arguments:
            input_size {[int]} -- [Number of values used for prediction in RNN]

        """
        self.input_size = input_size
        self.data_ls = []  # Data List Containing all stock dataframes in each entry
        # Normalized Data List Containing all stock dataframes in each entry
        self.norm_data_ls = []
        self.candle_data = []  # Candle Data List Containing RB, UB, LS
        self.x_train = []  # Training data in observations
        self.y_train = []  # Training data result
        self.x_test = []  # Testing data in observations
        self.y_test = []  # Training data result
        self.split = split  # Test Training Split Percent
        self.data_ls = self.get_data('Stock')  # Initialize stock data
        self.prepare_data()  # Prepare and parse the data

    @profile
    def split_data(self):
        """[method to split training and test data]

        Keyword Arguments:
            split {float} -- [percentage to split train vs test] (default: {0.7})
        """

        for stock in self.norm_data_ls:
            for val in range(0, len(stock)-self.input_size-1):
                self.x_train.append(
                    stock.Close.values[val:val+self.input_size])
                self.y_train.append(stock.Close.values[val+self.input_size+1])
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x_train, self.y_train, test_size=self.split, random_state=0)

    @profile
    def get_data(self, typeDat, queryDic={}):
        """[router function to get data from different datasources]

        Arguments:
            typeDat {[str]} -- [datasource to use string]

        Keyword Arguments:
            queryDic {dict} -- [contains query fields for each datasource] (default: {{}})
        """

        if typeDat == 'Stock':
            res = self.__get_stock_data()
        if res:
            return res
        else:
            print('No Data Found')

    @profile
    def prepare_data(self):
        """[will clean, normalize, create candle sticks and split up the data]
        """

        self.clean_data()
        self.normalize_data()

        for stock in self.data_ls:
            RB = 100.0 * (stock.Close.values -
                          stock.Open.values) / (stock.Open.values)
            RB = RB.reshape(1, -1)
            US = 100.0 * (stock.Close.values - stock.Open.values) / \
                (stock.High.values - stock.Open.values)
            US = US.reshape(1, -1)
            LS = 100.0 * (stock.Close.values - stock.Open.values) / \
                (stock.Close.values - stock.Low.values)
            LS = LS.reshape(1, -1)
            candle_data = [self.clean_and_scale(
                RB), self.clean_and_scale(US), self.clean_and_scale(LS)]
            self.candle_data.append(candle_data)

        self.split_data()

    def clean_and_scale(self, candle):
        scaler = MinMaxScaler()
        imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        candle[np.isinf(candle)] = max(candle[np.isfinite(candle)])
        candle = candle.reshape(-1, 1)
        imputer = imputer.fit(candle)
        candle = imputer.transform(candle)
        scaler.fit(candle)
        candle = scaler.transform(candle)
        return candle.reshape(1, -1)[0]

    @profile
    def clean_data(self):
        """[will remove any NaNs inside dataset with mean value near it]
        """

        for stock in self.data_ls:
            imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
            imputer = imputer.fit(stock.iloc[:, 1:6])
            stock.iloc[:, 1:6] = imputer.transform(stock.iloc[:, 1:6])

    @profile
    def normalize_data(self):
        """[will perform a min max scaler transformation on dataset]
        """

        for stock in self.data_ls:
            scaler = MinMaxScaler()
            norm_st = stock
            scaler.fit(stock.iloc[:, 1:6])
            norm_st.iloc[:, 1:6] = scaler.transform(stock.iloc[:, 1:6])
            self.norm_data_ls.append(norm_st)

    @profile
    def __get_stock_data(self):
        """[gets stock data from pickled file of 5 year historic stocks]

        Returns:
            [list] -- [stock dataframes in each list entry]
        """

        stock_val = pickle.load(open(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), 'data', 'stock_vals.p'), 'rb'))
        return stock_val


if __name__ == '__main__':
    data_fd = financial_data(10)

    '''
    Possible Future Addition for other data sources
    if typeDat == 'Wiki':
            res = self.__get_wiki_data(queryDic)
        elif typeDat == 'Quandl':
            res = self.__get_quandl_data(queryDic)
        el
     def __get_quandl_data(self, queryDic):
        try:
            stock_val = quandl.get(queryDic['Query'])
            return stock_val
        except:
            print('Error in attempting to get '+stock)
            return ''   
    def __get_wiki_data(self, queryDic):
        try:
            stock = queryDic['Stock']
            if stock+'.csv' in os.listdir('data'):
                print('Getting Cached')
                stock_val = pd.read_csv(os.path.join('data', stock)+'.csv')
            else:
                print('Getting QUANDL')
                stock_val = quandl.get_table('WIKI/PRICES', ticker=stock)
                stock_val.columns = ['ticker', 'date', 'Open', 'High', 'Low', 'Close', 'volume',
                                     'ex-dividend', 'split_ratio', 'adj_open', 'adj_high', 'adj_low',
                                     'adj_close', 'adj_volume']
                stock_val.to_csv(os.path.join('data', stock)+'.csv')
            return stock_val
        except:
            print('Error in attempting to get '+stock)
            return ''
    '''