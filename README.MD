# Machine Learning Final Project
Find out more [here](https://docs.google.com/presentation/d/1aCeB7Fr_KRg486TcoJ9SwiK-Yt1HUlEglGdI1NzHV6Q/edit?usp=sharing)
![Alt text](https://abisulco.com/img/stocks.gif)


Github repository for the machine learning trading algorithm. Still a work in progress. The goal is to use reinforcement learning in order to determine when to buy or hold a stock. 

### Setup
1. Have pip installed on your [system](https://pip.pypa.io/en/stable/) 
2. Run pip install -r requirements.txt, make sure this is for python3, may need to specify pip3
3. Go to the pytorch website and install for [your OS](http://pytorch.org)
4. Run the command jupyter lab inside src directory
5. Go into the ipython notebook for reinforcement learning and play around

## Running Code
The main class that contains the data for processing in this code is financial data. This class imports 500 different stocks, cleans and scales the data. After this process is performed reinforcment learning is used in order to train to buy, sell or hold stocks. This code can be seen int he RL ipython notebook. 

```
jupyter lab
```

Northeastern University 2018
