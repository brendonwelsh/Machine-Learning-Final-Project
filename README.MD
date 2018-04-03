# Machine Learning Final Project
## Trading Algorithm Python 3.5 

![Alt text](src/lines.png)
Github repository for the machine learning trading algorithm. Still a work in progress. The goal is to use candle sticks in order to determine when to buy or hold a stock. 

### Setup
1. Have pip installed on your [system](https://pip.pypa.io/en/stable/) 
2. Also you will need to install Ta-Lib at C++ Lib(Dependency), checkout [directions](https://github.com/mrjbq7/ta-lib)
3. Run pip install -r requirements.txt, make sure this is for python3, may need to specify pip3
4. Go to the pytorch website and install for [your OS](http://pytorch.org)
5. Run the command jupyter lab inside src directory
6. Go into the ipython notebook candle stick plot and play around

## Running Code
The code consists of two python classes. One financial data which will clean and split up stock data for training of the recurrent neural network. Additionally, the class prediction_model defines the recurrent neural network. Further work is being done on reinforcement learning from this prediction model. Tensorboard can be used to visualize training accuracy. Pytest is being used in order to test the code is working.
```
tensorboard --logdir runs
python src/prediction_model.py
```

Northeastern University 2018
