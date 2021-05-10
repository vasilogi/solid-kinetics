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
measure = 'resREr' # choose measure
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

