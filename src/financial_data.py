import os
import pickle
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, MinMaxScaler
import quandl
import csv

try:
    profile  # throws an exception when profile isn't defined
except NameError:
    def profile(x): return x
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
        self.ticker_ls = [stock['Name'].values[1] for stock in self.data_ls]

    @profile
    def import_sector(self):
        """
        imports sector data as indices of 25 heavily traded securities
        per sector for the 5 major trading sectors. Data from Quandl WIKI/PRICES
        and includes open, close, high, low, volume, dividend, split, and all 
        adjusted values
        """
        #TODO UPDATE PYTHONIC SYNTAX
        sectors = {}
        finance = {}
        health = {}
        re = {}
        tech = {}
        energy = {}
        k = 1
        for filename in os.listdir('Financial'):
            with open('Financial/'+filename) as fin:
                reader=csv.reader(fin, skipinitialspace=True, quotechar="'")
                security = filename.split('.')[0]
                temp_dict = {}
                for row in reader:
                    temp_dict[row[0]]=row[1:]
            #add temp dict to complete finance dict
            print('temp length = '+str(len(temp_dict))+" for the security"+security)
            finance[security] = temp_dict
            k = k+1
        print('finance length = '+str(len(finance)))
        #print(finance)

        for filename in os.listdir('Health'):
            with open('Health/'+filename) as fin:
                reader=csv.reader(fin, skipinitialspace=True, quotechar="'")
                security = filename.split('.')[0]
                temp_dict = {}
                for row in reader:
                        temp_dict[row[0]]=row[1:]
        #add temp dict to complete finance dict
        print('temp length = '+str(len(temp_dict))+" for the security"+security)
        health[security] = temp_dict
        k = k+1
        print('health length = '+str(len(health)))

        for filename in os.listdir('RE'):
            with open('RE/'+filename) as fin:
                reader=csv.reader(fin, skipinitialspace=True, quotechar="'")
                security = filename.split('.')[0]
                temp_dict = {}
                for row in reader:
                        temp_dict[row[0]]=row[1:]
        #add temp dict to complete finance dict
        print('temp length = '+str(len(temp_dict))+" for the security"+security)
        re[security] = temp_dict
        k = k+1
        print('Real Estate length = '+str(len(re)))

        for filename in os.listdir('Tech'):
            with open('Tech/'+filename) as fin:
                reader=csv.reader(fin, skipinitialspace=True, quotechar="'")
                security = filename.split('.')[0]
                temp_dict = {}
                for row in reader:
                        temp_dict[row[0]]=row[1:]
        #add temp dict to complete finance dict
        print('temp length = '+str(len(temp_dict))+" for the security"+security)
        tech[security] = temp_dict
        k = k+1
        print('tech length = '+str(len(tech)))

        for filename in os.listdir('Energy'):
            with open('Energy/'+filename) as fin:
                reader=csv.reader(fin, skipinitialspace=True, quotechar="'")
                security = filename.split('.')[0]
                #print("parsing energy security "+security)
                temp_dict = {}
                for row in reader:
                        temp_dict[row[0]]=row[1:]
        #add temp dict to complete finance dict
        print('temp length = '+str(len(temp_dict))+" for the security"+security)
        energy[security] = temp_dict
        k = k+1
        print('energy length = '+str(len(energy)))
        #merge all 5 dicts
        sectors["energy"] = energy
        sectors["health"] = health
        sectors["re"] = re
        sectors["finance"] = finance
        sectors["tech"] = tech
        print('sectors length = '+str(len(sectors)))

        return sectors
    
    def split_data(self, norm_data_ls):
        """[method to split training and test data]

        Keyword Arguments:
            split {float} -- [percentage to split train vs test] (default: {0.7})
        """
        x_train = []
        y_train = []
        for stock in norm_data_ls:
            for val in range(0, len(stock) - self.input_size - 1):
                x_train.append(
                    stock.Close.values[val:val + self.input_size])
                y_train.append(stock.Close.values[val + self.input_size + 1])
        return x_train, y_train

    def shuffle_data(self, x_train, y_train):
        """[shuffles testing and training data ]

        Arguments:
            x_train {[PyTorch Tensor]} -- [pytorch input data of training]
            y_train {[PyTorch Tensor]} -- [pytorch output data of training]]

        Returns:
            [list] -- [lists of shuffled training and testing data]
        """
        x_train, x_test, y_train, y_test = train_test_split(
            x_train, y_train, test_size=self.split, random_state=0)
        return x_train, y_train, x_test, y_test

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
        elif typeDat == 'Wiki':
            res = self.__get_wiki_data(queryDic)
            return res
        if res:
            return res
        else:
            print('No Data Found')

    @profile
    def prepare_data(self):
        """[will clean, normalize, create candle sticks and split up the data]
        """

        self.data_ls = self.clean_data(self.data_ls)
        self.norm_data_ls = self.normalize_data(self.data_ls)

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
        #self.x_train, self.y_train = self.split_data(self.norm_data_ls)
        #self.x_train, self.y_train, self.x_test, self.y_test = self.shuffle_data(
        #    self.x_train, self.y_train)

    def clean_and_scale(self, candle):
        """[clean data in candle stick]

        Arguments:
            candle {[list]} -- [candle stick info]

        Returns:
            [List] -- [list of candle data]
        """
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
    def clean_data(self, data_ls):
        """[will remove any NaNs inside dataset with mean value near it]
        """

        for stock in data_ls:
            imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
            imputer = imputer.fit(stock.iloc[:, 1:6])
            stock.iloc[:, 1:6] = imputer.transform(stock.iloc[:, 1:6])
        return data_ls

    @profile
    def normalize_data(self, data_ls):
        """[will perform a min max scaler transformation on dataset]
        """
        norm_data_ls = []
        for stock in data_ls:
            scaler = MinMaxScaler()
            norm_st = stock
            scaler.fit(stock.iloc[:, 1:6])
            norm_st.iloc[:, 1:6] = scaler.transform(stock.iloc[:, 1:6])
            norm_data_ls.append(norm_st)
        return norm_data_ls

    @profile
    def __get_stock_data(self):
        """[gets stock data from pickled file of 5 year historic stocks]

        Returns:
            [list] -- [stock dataframes in each list entry]
        """

        stock_val = pickle.load(open(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), 'data', 'stock_vals.p'), 'rb'))
        return stock_val

    def __get_wiki_data(self, queryDic):
        stock = queryDic['Stock']
        if stock in self.ticker_ls:
            print('Stock is used in triaining data')

        stock_val = quandl.get_table('WIKI/PRICES', ticker=stock)
        if stock_val.empty:
            print('Values undefined')
            return
        stock_val.columns = ['ticker', 'date', 'Open', 'High', 'Low', 'Close', 'volume',
                             'ex-dividend', 'split_ratio', 'adj_open', 'adj_high', 'adj_low',
                             'adj_close', 'adj_volume']
        return stock_val[['date', 'Open', 'High',
                          'Low', 'Close', 'volume', 'ticker']]


if __name__ == '__main__':
    data_fd = financial_data(10)
