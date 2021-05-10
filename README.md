# solid-kinetics
a software package to model and predict solid-state reactions



arguments



```python
# limit conversion fraction
low = 0.05
high = 0.95
# polynomial degree and interpolation points for the polynomial fit of the experimental conversion fraction
pdeg = 9
npoints = 1000
# export csv with the desicions
measure = 'resREr' # choose measureS
fitExp = False
```

### steps

```python
# plot and export solely the experimental data
graph_experimental_data(DATA,OUTPUT)
```

it reads the CSV files from the data directory and scatter plots the experimental conversion over time for all temperatures in a single graph

```python
# perform linear regression on the integral rate experimental data
data2integralFit(DATA,OUTPUT,modelNames,low,high)
```

it fits the linear fit kt to the experimental integral reaction rate and calculates all the metrics and Arrhenius constant for all data sets. Then it saves separate csv files with all this info with the suffix *_integral_regression_accuracy.csv*

```python
# perform non-linear regression on the exact conversion
data2conversionFit(DATA,OUTPUT,modelNames,low,high)
```

it fits the model conversion to the experimental conversion and export the relative metrics data and Arrhenius constant for all data sets. It saves separate csv files with the suffix *_conversion_regression_accuracy.csv*

if the model is D2 or D4 the metrics from the integral conversion are used

```python
# perform non-linear regression on the differential rate experimental data
data2differentialFit(DATA,OUTPUT,modelNames,low,high)
```

it fits the experimental conversion fraction with the solution of the ODE regarding the differential reaction rate. It saves separate files with all the metrics using the *suffix _differential_regression_accuracy.csv*

if the model is D2 or D4 the metrics from the integral conversion are used

```python
# export reaction rate data
export_experimental_reaction(DATA,OUTPUT,pdeg,npoints)
```

it calculates the experimental reaction rates for all data sets in two ways: i. by differentiating the original experimental conversion over time, ii. by fitting the original conversion with a polynomial and then differentiate the polynomial. It saves separate csv files with the suffix *_reaction_rate.csv*

```python
# calculate accuracy metrics for the actual reaction experimental rate fit
ratedata2Fit(DATA,OUTPUT,modelNames,low,high,pdeg,npoints,True)
```

it takes the Arrhenius constant by fitting the experimental conversion with models conversion and then calculates the modeled reaction rate as k*f(a). Then, we calculate the metrics of the fitting of the experimental reaction rate with the modeled. If the model is D2, D4 the Arrhenius constant from the fitting of the integral reaction rate is used.

It saves separate files with the suffix *_experimental_rate_fit_accuracy.csv* if the experimental conversion is directly being differentiated or *_polynomial_rate_fit_accuracy.csv* if the polynomial of the experimental conversion is differentiated.