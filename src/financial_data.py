import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split

class financial_data:
    """
    financial data class contains mutiple different
    datasources from Quandl, CSVs and other sources.
    Which are cleaned, scaled and split in this class.
    This data will then pass into the predicition model.
    TODO:
    -Expand datasources available
    -Perform more erro checking
    -Allow more customizable test train data
    """

    def __init__(self, input_size):
        """[finacial data object that parses various data formats]

        Arguments:
            input_size {[int]} -- [Number of values used for prediciton in RNN]

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
        self.get_data('Stock')  # Initialize stock data
        self.prepare_data()  # Prepare and parse the data

    def split_data(self, split=0.7):
        """[method to split training and test data]

        Keyword Arguments:
            split {float} -- [percentage to split train vs test] (default: {0.7})
        """

        for stock in self.norm_data_ls:
            for val in range(0, len(stock)-self.input_size-1):
                self.x_train.append(
                    [stock.Close.values[val:val+self.input_size]])
                self.y_train.append(stock.Close.values[val+self.input_size+1])
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x_train, self.y_train, test_size=split, random_state=0)

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
            self.data_ls = res
        else:
            print('No Data Found')

    def prepare_data(self):
        """[will clean, normalize, create candle sticks and split up the data]
        """

        self.clean_data()
        self.normalize_data()

        for stock in self.norm_data_ls:
            RB = 100.0 * (stock.Close - stock.Open) / (stock.Open)
            US = 100.0 * (stock.Close - stock.Open) / (stock.High - stock.Open)
            LS = 100.0 * (stock.Close - stock.Open) / (stock.Close - stock.Low)
            candle_data = [RB, US, LS]
            self.candle_data.append(candle_data)
        self.split_data(split=0.7)

    def clean_data(self):
        """[will remove any NaNs inside dataset with mean value near it]
        """

        for stock in self.data_ls:
            imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
            imputer = imputer.fit(stock.iloc[:, 1:6])
            stock.iloc[:, 1:6] = imputer.transform(stock.iloc[:, 1:6])

    def normalize_data(self):
        """[will perform a min max scaler transformation on dataset]
        """

        for stock in self.data_ls:
            scaler = MinMaxScaler()
            norm_st = stock
            scaler.fit(stock.iloc[:, 1:6])
            norm_st.iloc[:, 1:6] = scaler.transform(stock.iloc[:, 1:6])
            self.norm_data_ls.append(norm_st)

    def __get_stock_data(self):
        """[gets stock data from pickled file of 5 year historic stocks]

        Returns:
            [list] -- [stock dataframes in each list entry]
        """

        stock_val = pickle.load(open('data/stock_vals.p', 'rb'))
        return stock_val

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
