# pm25_exercise
## Bartlomiej Wieczorek


### Table of contents

* [Setup](####Setup)
* [Ex1](####Ex1)
* [Ex2a](####Ex2a)
* [Ex2b](####Ex2b)


#### Setup

To run this project, clone it on your device and run:

```
$ pip install -r requirements.txt
```

#### Ex 1
TBA

#### Ex2a
##### GARCH volatility modelling

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
