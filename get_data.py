import quandl

quandl.ApiConfig.api_key = 'gBFpXP2-svdq_izibeJS'
quandl.get_table('WIKI/PRICES', start_date='1999-11-18', end_date='1999-12-31', ticker='A')
