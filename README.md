# pm25_exercise
# Bartlomiej Wieczorek


## Table of contents

* [Setup](##Setup)
* [Ex1](##Ex1)
* [Ex2a](##Ex2a)
* [Ex2b](##Ex2b)


#### Setup

To run this project, clone it on your device and run:

```
$ pip install -r requirements.txt
```

## Ex 1
TBA

## Ex2a
### GARCH volatility modelling

For the second exercise, I decided to formulate two entirely different problems.

Problem 2a) - forecast the "volatility" of PM2.5 for selected city; train the model using the data till 2014, and evaluate the model it on 2015.

So, assumptions - one city (I chose Beijing), train data - 2010 - 2014, validation data - 2015; target - propose some kind of estimation of the "possible change" of PM2.5.

To solve such problem, I decided to look to my statistics book and recall a ARCH (Autoregressive conditional heteroskedasticity) family of models.

I considered several models - `ARCH(1)` - `ARCH(4)`, and GARCH(p,q) for p,q in [1,2,3], respectively.

The best model, for my case, was the GARCH(1,3) model - the coefficients from the model were significant;
however, one can change the model using `config.py` file (`GARCH_P`, `GARCH_Q` variables, respectively).

After training the model, I forecasted the volatility 1 year ahead and created two confidence intervals - 95% and 99%. In my case, 95.48% observations qualified for the first inverval, and all of them qualified for the 2nd one.

(I actually cheated a bit; PM2.5 is a negative phenomenon, so all the changes "downwards" were considered "inside" of all intervals; I was only interested in checking whether "positive" changes of PM2.5 don't exceed the forecasted value)

Running the solution : 

```
$ python garch.py
```

## Ex2b

For the 2nd approach to the 2nd exercise (Problem 2b), I propose simple time series forecasting. 

So - assumptions - I will use the lagged numerical features (temperature, pressure, humidity, wind speed, precipitation and cumulated precipitation) and lagged PM2.5 values to forecast present value of PM2.5.

To do so, I, again, split the dataset on the train and validation sample (data till the end of 2014 will be my train sample, the rest will be the validation sample).

Hovewer, this time, I use one city as an independent, test sample. It is, again, Beijing.

In order to make a forecast, I decided to propose a LSTM (Long Short-Term Memory) Neural Network, which I implemented using Pytorch. As a neat feature, I made sure that the network can be trained using GPU (I have a good, old RTX 2070, so I gave it a try).

Again, the hyperparameters of the model, such as batch size, number of epochs, learning rate etc., can be modified by changing the corresponding variables in the `config.py` file.

One can also look at some plots in the `img` subfolder.

Some things that should be added to this solution - selected seed (I forgot about it during the training, so the solution is not entirely reproducible), Early Stopping mechanism, probably the train/validation vs epochs graph, maybe some more.

I must say, I'm not 100% satisfied with my forecast, as my training MSE error was really large (6864). However, on the test sample, the forecasting went kind of okay (MSE = 2841, lower than 3051 on validation sample). However, I also don't feel like it has a significant forecasting power, I probably felt to a trap that model just guesses the values around past values.

Running the solution : 

```
$ python ts_solution.py
```