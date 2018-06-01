# Notes for Book:   *Mastering pandas for Finance*
## Ch 1 Get started   
Several useful packages:   

- Zipline: backtesting/trading simulator from Quantopian
- Quandl: data API 
- Mibian: small library that computes B-S and its derivatives  
  
Installing packages through: Shell mode--> pip/conda  

## Ch 2 Series and DataFrame

### Series
It extends the NumPy *ndarray* by adding an labels for indexing;  
It can hold SINGLE data type;
### DataFrame
A dictionary-like container of one or more *Series* objects;  
Like a spreadsheet;  
### Basic Operation of Series
#### -Create, Access, Size:
3 means of creation:  

```
#01 By default
np.random.seed(1)
s=pd.Series(np.random.randn(100))   #Create with 100 elements

#02 Constructor property
s2=pd.Series([1,2,3,4],index=['a','b','c','d'])

#03 From a Python dictionary
s3=pd.Series({'a':1,'b':2,'c':3,'d':4,'e':5}) 
```
basic access and slice:  

```
s[0]          #Index start from 0 in Python
s[[2,5,20]]   #Can achieve multiply using an array of label values
s[3:8]        #Slice   NOTE: 1. the end value not inclusive!!! (3,4,5,6,7) 2.return both LABEL and VALUE

s.head()
s.tail()      #By default display 5 rows
s.index
s.values
```
size and unique value:  

```
len(s)
s.shape   #return a tuple, since Series only 1-D, return (10,) only len
s.count()    # rows without NaN
s.unique()
s.value_counts()  #counts the frequency, decending order

```
### Basic Operation of DataFrame
#### -Create:  

```
#01 From a Numpy array, default label
pd.DataFrame(np.array([ [10,11],[20,21] ]))

#02 Passing a list of Series, default label
pd.DataFrame( [pd.Series(np.arange(10,15) ),pd.Series(np.arange(15,20) )] )

#03 By constructor
df=pd.DataFrame( np.array([ [0,1],[2,3] ]), columns=['c1','c2'], index=['r1','r2'])

#04 By dictionary
s1=pd.Series(np.arange(1,6,1))
s2=pd.Series(np.arange(6,11,1))
pd.DataFrame({'c1':s1,'c2':s2})

#NOTE: if we want to add a column by index, pandas automatically fills in NaN for values not supplied
```
DataFrame also have:  
  
```
df.columns
df.index
df.values
```
#### -Selecting columns 
It is a little special, as there are 2 different ways: location-based and value-based. They return different results.  

```
sp500[[1,2]].head(3)   #The [ [] ] structure returns DataFrame still
sp500[[1]].head(3)
sp500['price']         #This results in a Series, since it's value-based 
sp500.price            #Column can also be accessed through property; however,
                              #'Book Value' cannot be accessed since there is a space in between
```
#### -Selecting rows  
Note that different from *Series*, DataFrame overload [ ] to select columns instead of rows except for slicing!  
Thus selecting rows breaks down into different cases:   
1/ Slicing using [ ]  
Can either by position or by label; (But people avoid using integer since it may be confusing as accessing columns)

```
sp500[:3]
sp500['XYL':'YUM']     
```
2/ Label- or location-based lookup using .loc[ ], .loc[ ], and  .ix[ ]  

```
sp500.loc['MMM']    #retrieved via the index label
sp500.iloc[[0,2]]   #retrieved by location

#can look up the location first, and then retrieve
i1=sp500.index.get_loc('MMM')
i2=sp500.index.get_loc('A')
sp500.iloc[[i1,i2]]

# .ix can look up both by label or location, but is not recommended due to confusion
sp500.ix[['MSFT',ZTS'']]
sp500.ix[[10,200,450]]
```
3/  Scalar loolup by label/location using .at[ ] and .iat[ ]

```
sp500.at['MMM','Price']    #passing the row label and column name
sp500.iat[0,1]  #passing the row and column location(Preferred method!!)
```
4/ Boolean selection

```
sp500.Price<100
sp500[sp500.Price<100]

#also support multiple conditions:
sp500[(sp500.Price<10) & (sp500.Price>0)] [['Price']]   #using parenthese
```
#### -Arithmetic on a DataFrame  
- By default, the operation be applied across all rows and columns, and return a new DataFrame, 
leaving the original unchanged;  
- Operation between a DataFrame and Series will be a row-wise broadcast;  
- Operation between two DataFrame will align both the column and index labels;  
- Use provided function by DataFrame: .sub() method

```
df=pd.DataFrame(np.random.randn(5,4),columns=['A','B','C','D'])
df*2
df-df.iloc[0]

# with 2 DataFrame, those not aligned are set to NaN
subframe=df[1:4][['B','C']]
df-subframe  # A,D column 和B C column 没cover到的地方都是NaN

# substract A column from every column
a_col=df['A']
df.sub(a_col,axis=0)
```
### Reindexing for Series and DataFrame
simple way:  
`s.index=['a','b','c','d','e']`  

greater flexible way: .reindex()  

- allow different length than original number of rows in Series
- the result is a new Series; If there exists matching label, value is copied  
- if not having existing label, the value be assigned NaN(By default)

```
s1 = pd.Series([0,1,2], index=[0,1,2])
s2 = pd.Series([3,4,5], index=['0','1','2'])
s1+s2  # the result is all NaN in values, since the index doesn't align.

#Solutions  using .reindex():
s2.index=s2.index.values.astype(int)
s1+s2
```
By default, fills NaN for missing value; we can self define:  

```
s2=s.copy()
s2.reindex(['a','f'],fill_value=0) 
```
Can also perform interpolation:  

```
s3=pd.Series(['red','green','blue'],index=[0,3,5])
s3.reindex(np.arange(0,7),method='ffill')  #forward filling, =last value
s3.reindex(np.arange(0,7),method='bfill')  #backwards filling, =next value (index为6的value就是NaN了)
```

## Ch 3  Reshape, Reorganize, and Aggregate
### Load data
Load data using pandas's DataReader(the following example may be different now) 

```
import datetime
import pandas.io.data as web

start=datetime.datetime(2012,1,1)
end=datetime.datetime(2012,12,30)
msft=web.DataReader("MSFT",'yahoo',start,end)
aapl=web.DataReader("AAPL",'yahoo',start,end)

msft.to_csv("msft.csv")
aapl.to_csv("aapl.csv")  # store it into csv in case that value changed online
msft=pd.read_csv("msft.csv",index_col=0,parse_dates=True)
aapl=pd.read_csv("aapl.csv",index_col=0,parse_dates=True)
```
### Concatenating
Create new pandas object by combining multiple pandas object along a **single, specified axis**; (row by default)  
Realize by first extract the labels along the specified axis, use it as index of new DataFrame, and copy values;  
Methods and properties:

```
# During a concatenation along the row axis, pandas will not align index labels:
msftA01=msft['2012-01'][['Adj Close']]
aaplA01=aapl['2012-01'][['Adj Close']]
withDups=pd.concat([msftA01[:3],aaplA01[:3]])   #so the result has two set of same labels, but don't know the source of each row 

# Can add additional level of index (make it MultiIndex) identifying the source:
closes=pd.concat([msftA01[:3],aaplA01[:3]], keys=['MSFT','AAPL'])
closes.ix['MSFT'][:3]  # now we can access specifically

# Can concate multiple columns, missing filled with NaN. If want intersection, use inner method:
msftAV=msft[['Adj Close','Volumn']]
aaplA=aapl[['Adj Close']]
pd.concat([msftAV,aaplA],join='inner')  # the result then only have one column: Adj Close

# Change axis, now by columns, and the above operation is similar:
msftA=msft[['Adj Close']]
aaplAV=aapl[['Adj Close','Volumn']]
closes=pd.concat([msftA,aaplA],axis=1)
pd.concat([msftAV[:5],aaplAV[:3]],axis=1,keys=['MSFT','AAPL']) #also MultiIndex for columns

# Can ignore previous index, create a default zero-based index:
pd.concat([msftA[:3],aaplA[:3]],ignore_index=True)   #now the index is only 0,1,2,3,4,5 
```  
### Merging
Merging is different from concatenation that pd.merge() combines data based on one or more columns instead of a specific axis;  
Default process: first identify which columns to base(default is the common column), and then perform an **inner** join on that;  

```
# first reset the index, means that the previous labels become a new column, 
# and the new labels are by default zero-based integer:
msftAR=msftA.reset_index()
msftVR=msft[['Volumn']].reset_index()

# merge them so that they have 3 columns:
msftCVR=pd.merge(msftAR,msftVR)  #common column is Date, they inner join Date first, then copy values

# support many types of joins: left(use keys from the left DataFrame), right, outer, inner
pd.merge(msftAR0_5,msftVR2_4,how='outer')  
```
### Pivoting
Used for extracting information and reorganize;  
Assign the Index, split a column as label of columns, select values from previous column;

```
# initialize data
msft.insert(0, 'Symbol', 'MSFT')
aapl.insert(0, 'Symbol', 'AAPL')
combined = pd.concat([msft, aapl]).sort_index()
s4p = combined.reset_index();
s4p[:5]
##########
Out[27]:
        Date Symbol    Open    High     Low   Close    Volume  Adj Close
0 2012-01-03   MSFT   26.55   26.96   26.39   26.77  64731500      24.42
1 2012-01-03   AAPL  409.40  412.50  409.00  411.23  75555200      55.41
2 2012-01-04   MSFT   26.82   27.47   26.78   27.40  80516100      25.00
3 2012-01-04   AAPL  410.00  414.68  409.28  413.44  65005500      55.71
4 2012-01-05   MSFT   27.38   27.73   27.29   27.68  56081400      25.25
###########

# pivot the data:
closes = s4p.pivot(index='Date', columns='Symbol', 
                   values='Adj Close')
closes[:3]
##########
Out[28]:
Symbol       AAPL   MSFT
Date                    
2012-01-03  55.41  24.42
2012-01-04  55.71  25.00
2012-01-05  56.33  25.25
##########
```
### Stacking and Unstacking
Stacking unpivots column labels into another level of the index;  
Unstacking performs the opposite function, it pivots a level of index into a column;

```
# .stack()
stackedCloses = closes.stack()
#############
Out[29]:
Date            Symbol
2012-01-03  AAPL      55.41
                     MSFT      24.42
2012-01-04  AAPL      55.71
                     MSFT      25.00
                                       ...  
2012-12-27  AAPL      70.02
                     MSFT      25.29
2012-12-28  AAPL      69.28
                     MSFT      24.91
dtype: float64
#############

# since it's MultiIndex, can look up value easily:
stackedCloses.ix['2012-01-03','AAPL']
stackedCloses.ix['2012-01-03']
stackedCloses.ix[:,'AAPL'] #Note that : refers to all the dates

# after unstacking, the result goes back to the previous one(close)
unstackedCloses=stackedCloses.unstack()
```
### Melting
To transform a DataFrame into a  unique id-variable combination;  
Only columns specified as id remain columns, all other columns have their name mapped to the values in the *variable* column;  
Useful to select chunks of info based on ID;

```
melted=pd.melt(s4p,id_vars=['Date','Symbol'])  #id_vars specifies which still be columns
melted[(metled.Date=='2012-01-03')&(metled.Symbol=='MSFT')]   #select all measurement for this ID
##############
Out[35]:
           Date Symbol   variable        value
0    2012-01-03   MSFT       Open        26.55
498  2012-01-03   MSFT       High        26.96
996  2012-01-03   MSFT        Low        26.39
1494 2012-01-03   MSFT      Close        26.77
1992 2012-01-03   MSFT     Volume  64731500.00
2490 2012-01-03   MSFT  Adj Close        24.42
##############
```
### Grouping
Objects can be split into groups using .groupby() method, which return a subclass of **GroupBy** object. The object has many properties to use for specific purpose.

```
# To illustrate, first initialize the data
s4g = combined[['Symbol', 'Adj Close']].reset_index()
s4g.insert(1, 'Year', pd.DatetimeIndex(s4g['Date']).year)
s4g.insert(2, 'Month',pd.DatetimeIndex(s4g['Date']).month)  #add two columns,using 2 portions of the Date as integers
s4g[:5]
##################
Out[36]:
        Date  Year  Month Symbol  Adj Close
0 2012-01-03  2012      1   MSFT      24.42
1 2012-01-03  2012      1   AAPL      55.41
2 2012-01-04  2012      1   MSFT      25.00
3 2012-01-04  2012      1   AAPL      55.71
4 2012-01-05  2012      1   MSFT      25.25
##################

# Grouping and its property
grouped = s4g.groupby('Symbol')

type(grouped.groups)    # returns dict;
grouped.groups             # keys are the name of each group(if MultiIndex, then it's a tuple), values are the index labels
len(grouped)        # number of groups
grouped.ngroups  # same result as len()
grouped.size()      #summary of size of all groups
grouped.get_group('MSFT')  #retrieve the specific group

# Grouping can be done on multiple columns
mcg = s4g.groupby(['Symbol', 'Year', 'Month'])   # in this case, the index are still 0-based interger
mi = s4g.set_index(['Symbol', 'Year', 'Month'])     #set MultiIndex

# Grouping can use levels of the hierarchical index
mig_l1 = mi.groupby(level=0)     #level 0 means symbol
mig_l12 = mi.groupby(level=['Symbol', 'Year', 'Month'])    #can also pass by name
```
### Aggregating
Aggregation is performed using function .aagregate() or .agg() as a method of the GroupBy object; parameter is the reference to a function that is applied to each group.

```
mig_l12.agg(np.mean)   # calculate the mean across symbol, year and month
mig_l12.agg([np.mean,np.std])  # apply multiple aggregation function by passing a list

# If don't want previous index, can use back to numerical index
s4g.groupby(['Symbol', 'Year', 'Month'], as_index=False).agg(np.mean)[:5] 
################
Out[50]:
  Symbol  Year  Month  Adj Close
0   AAPL  2012      1      57.75
1   AAPL  2012      2      67.05
2   AAPL  2012      3      77.82
3   AAPL  2012      4      81.66
4   AAPL  2012      5      76.09
################
```
## Ch 4 Time series
- Pandas's time function is more powerful than SciPy and NumPy;  
- Pandas uses *Timestamp* class, which is based on NumPy 's dtype datetime64 and has higher precision than Python's built-in *datetime* object;     
- Sequences of timestamp objects are represented as a *DatetimeIndex*, which is a type of pandas index optimised for indexing by dates and times;

### Creation of DatetimeIndex
- through .DatetimeIndex()
- through .Series() passed time index
- though .to_datetime() function
- through .date_range() function

```
import datetime
from datetime import datetime

# Create a DatetimeIndex from an array of datetime objects:
dates=[datetime(2014,8,1), datetime(2014,8,2)]
dti=pd.DatetimeIndex(dates)

# Series would automatically constrct DatetimeIndex when we pass a list of datetime objects as its index
ts=pd.Series(np.random.rand(2),dates)
type(ts.index)
# then we can access it through its Timestamp object
ts[datetime(2014,8,2)]
ts['2014-8-2']

# Series can also recognize strings and create a DatetimeIndex
dates=['2014-08-01','2014-08-02']
ts=pd.Series(np.random.randn(2),dates)

# pandas also provide function to convert mixed type into DatetimeIndex
dti=pd.to_datetime(['Aug 1, 2014', '2014-08-02', '2014.8.3', None])  # return NaT for None denoting not-a-time value

dti2 = pd.to_datetime(['Aug 1, 2014', 'foo'])  
type(dti2)  # return numpy.ndarray if it cannot parse a value('foo')
pd.to_datetime(['Aug 1, 2014', 'foo'], coerce=True)   #force it to convert to DatetimeIndex, use NaT for other items

dti2 = pd.to_datetime(['1/8/2014'], dayfirst=True)   #still 2014-08-01  00:00:00

# a range of timestamps at a specific frequency
np.random.seed(123456)
dates = pd.date_range('8/1/2014', periods=10)
s1 = pd.Series(np.random.randn(10), dates)
# by default the frequency is daily, we can set to one-minute:
bymin = pd.Series(np.arange(0, 90*60*24),pd.date_range('2014-08-01','2014-10-29 23:59:00',freq='T'))  
```
### Selecting using DatetimeIndex
Note that Series and DataFrame is different when doing slicing:

```
# msft is a DataFrame and msftAC is a Series
msft.loc['2012-01-03']     #return a Series where index labels are the column names of msft
msft['2012-01-03']     # throw error since DataFrame search a column named '2012-01-03' by default
msftAC['2012-01-03']  #return a value, since Series support search by index label 
```
One advantage is to select based on **partial** datetime:

```
msft['2012-02'].head(5)   #NOTE: this don't need .loc method as pandas identify it as partial dates
msft['2012-02':'2012-02-09'][:5]  # supports the beginning of specific month and ending with specific day 
```
### Periods
- use the start time as anchor, and the end time is calculated using given interval length;  
- mathematical operations are overloaded for periods, based on the interval(frequency);  
- also has PeriodIndex similar to DatetimeIndex;

```
aug2014=pd.Period('2014-08',freq='M')
aug2014.start_time, aug2014.end_time   # end time is the last second of aug
sep2014=aug2014+1   #not simply add days, it knows how many days are in a month

# Create PeriodIndex with .period_range()
mp2013 = pd.period_range('1/1/2013', '12/31/2013', freq='M')
ps = pd.Series(np.random.randn(12), mp2013)
```
### Shifting and Lagging
- by .shift() change the alignment of data
- by .tshift() shift the value of time index
```
msftAC.shift(1)  #shift forward, cause the  first value be NaN
msftAC.shift(-2)  #opposite direction
msftAC. shift(1,freq="S")   #can shift by different frequencies
msftAC.tshift(1,freq="D")   
```
### Frequency Conversion
- using the .asfreq() method, can use it for resample
```
sample.asfreq("H") #the newly created index would be  NaN
sample.asfreq("H",method="ffill") #fill with last known value
sample.asfreq("H",method="bfill")  #use the next known value
```
### Resampling
- more elaborate control than just frequency conversion
- downsampling(dtd->mtm) or upsampling are calculated by a function provided to pandas instead of simple filing
- by .resample() method, the calculation method controlled by *how* parameter, the index type controlled by *kind* (Times)
```
msft_cum_ret.resample("M")  #from dtd to month-to-month, index be the last day of a month, value be the mean of the month
msft_cum_ret.resample("M",how="ohlc") #give the summary of  the open(first value), high, low, close(the last value)
msft_cum_ret.resample("M",how="mean",kind="period") 
```

## Ch5 Time-series Stock Data
get data->visualizing->analyze
### Obtaining data
- data can be retrieved using the DataReader class
```
# get the data for MSFT
start=datetime.date(2012,1,1)
end=datetime.date(2014,12,31)
msft=pd.io.data.DataReader('MSFT',"yahoo",start,end)
# we want to get all the stocks in a single DataFrame, so write a function to do so:
def get(tickers, start, end):
    def data(ticker):
        return pd.io.data.DataReader(ticker,'yahoo',start, end) 
    datas=map(data,tickers)
    return pd.concat(datas,keys=tickers, names=['Ticker','Date'])
```
### Visualizing
#### Plotting closing price 
- with .plot(), for either a single stock or multiple stocks
```
#reset the index to make everything columns
just_closing_prices=all_data[['Adj Close']].reset_index()
# next step we want to pivot Date into the index and each Ticker value into a column
daily_close_px=just_closing_prices.pivot('Date','Ticker','Adj Close')
# now plot:
_=daily_close_px['AAPL'].plot(figsize=(12,8))  #select specific column and call .plot()
_=daily_close_px.plot(figsize=(12,8)) #calling .plot() on the entire DataFrame
``` 
#### Plotting volume data 
- is easy to plot using pandas and the .bar() function
```
msftV=all_data.Volume.loc['MSFT']
plt.bar(msftV.index,msftV)
plt.gcf().set_size_inches(12,6)
```
- combines price and volumes:
```
# subdivide the whole plot
top = plt.subplot2grid((4,4), (0, 0), rowspan=3, colspan=4)
# plot the price chart on the top
top.plot(daily_close_px.index, 
         daily_close_px['MSFT'], 
         label='Adjusted Close')
plt.title('MSFT Adjusted Close Price from 2011 - 2014')
plt.legend(loc=2)
# pick the bottom
bottom = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=4)
# draw the bar at the bottom
bottom.bar(msftV.index, msftV)
plt.title('Microsoft Daily Trading Volume')
plt.gcf().set_size_inches(12,8) 
# set the size
plt.subplots_adjust(hspace=0.75)
plt.savefig('5104OS_05_04.png', bbox_inches='tight', dpi=300)
```
#### Plotting candlestick
- process contains formatting the date, choose which date on the x axis, transform DataFrame to tuples
```
# subset to MSFT in Dec 2014
subset = all_data.loc['MSFT'].loc['2014-12':'2014-12'] \
                 .reset_index()   #\ means chaging the line
# convert our dates matplotlib formatters representation
import matplotlib.dates as mdates
subset['date_num'] = subset['Date'].apply(lambda date: mdates.date2num(date.to_pydatetime()))
# extract only values required, in order, as tuples
subset_as_tuples = [tuple(x) for x in subset[['date_num', 
                                              'Open', 
                                              'High', 
                                              'Low', 
                                              'Close']].values]
# required imports for fomatting
from matplotlib.dates import DateFormatter
week_formatter = DateFormatter('%b %d')  # e.g., Jan 12
# We want to only display labels for Mondays
from matplotlib.dates import (WeekdayLocator, MONDAY)
mondays = WeekdayLocator(MONDAY) # major ticks on the Mondays
# now draw the plot
plt.figure(figsize(12,8))
fig, ax = plt.subplots()
# set the locator and formatter for the x-axis
ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_major_formatter(week_formatter)

# draw the candlesticks
from matplotlib.finance import candlestick_ohlc 
_ = candlestick_ohlc(ax, subset_as_tuples, width=0.6, 
                     colorup='g',colordown='r');
plt.savefig('5104OS_05_05.png', bbox_inches='tight', dpi=300)
```
### Fundamental financial calculation
- simple percentage change can use .pct_change() or divide by .shift()
```
daily_pct_change= daily_close_px.pct_change()
# set NaN's to 0 
daily_pct_change.fillna(0, inplace=True)
```
- cumulative return 
```
cum_daily_return = (1 + daily_pct_change).cumprod()
```
### Analyzing the distribution
- via Histograms, Q-Q plots, Box and whisker plots
#### Histograms
```
# Histograms using .hist() method of a pandas Series
aapl.hist(bins=50,figsize=(12,8))
daily_pct_change.hist(bins=50,sharex=True,figsize=(12,8))  #Histogram for all stocks; sharex means using the same x axis range
# can use .describe() to further specify the percentiles,so that get the confidence intervals
aapl.describe(percentiles=[0.025,0.5,0.975])
```
#### Q-Q plots
- plotting their quantiles against each other
```
# create a qq-plot of AAPl returns vs normal
import scipy.stats as stats
f = plt.figure(figsize=(12,8))
ax = f.add_subplot(111)
stats.probplot(aapl, dist='norm', plot=ax)
plt.show();
```
#### Box-and-whisker plots
- the box portion shows from low quantile to high, whiskers show the amount of variability outside of quartiles
- the dashed line extended out is IQR multiply 1.5, with IQR=Q3-Q1
```
_ = daily_pct_change[['AAPL']].plot(kind='box', figsize=(3,6));
daily_pct_change.plot(kind='box', figsize=(12,8));  
```
### Comparison between stocks
```
# all stocks against each other, with a KDE in the diagonal
_ = pd.scatter_matrix(daily_pct_change, diagonal='kde', alpha=0.1,
                      figsize=(12,12));  #kde means kernel density estimation
```
### Moving windows
- many functions are provided to calculate moving(rolling) stats, such as .rolling_mean(), ,rolling_std()
```
sample.plot(figsize=(12,8))
pd.rolling_mean(sample, 5).plot(figsize=(12,8));
```
- can use pd.rolling_apply function
```
mean_abs_dev = lambda x: np.fabs(x - x.mean()).mean()
pd.rolling_apply(sample, 5, mean_abs_dev).plot(figsize=(12,8));
```
- Rolling correlation
```
rolling_corr = pd.rolling_corr(daily_pct_change['AAPL'], 
                               daily_pct_change['MSFT'], 
                               window=252).dropna()
```
### Least Squares
```
model = pd.ols(y=daily_pct_change['AAPL'], 
               x={'MSFT': daily_pct_change['MSFT']}, 
               window=250)
model.beta[0:5]
```

## Ch 6 Trading Using Google Trends
- "Is it possible to predict efficient trading strategies based upon the frequency of certain words in Google searches?"--paper titled *Quantifying Trading Behavior in Financial Markets Using Google Trends*
- basic idea: if the search volume is greater than the average of the past three weeks, then short, else long.
- can use Quandl api for retrieving data
- for trends data, go to **www.google.com/trends/**
```
# this is the same DJIA data from the authors
paper = pd.read_csv('PreisMoatStanley2013.dat', 
                    delimiter=' ', 
                    parse_dates=[0,1,100,101])
data = pd.DataFrame({'GoogleWE': paper['Google End Date'],
                     'debt': paper['debt'].astype(np.float64),
                     'DJIADate': paper['DJIA Date'],
                     'DJIAClose': paper['DJIA Closing Price']
                         .astype(np.float64)})
# get the DJIA from Quandl for 2004-01-01 to 2011-02-28
import Quandl 
djia = Quandl.get("YAHOO/INDEX_DJI", 
                  trim_start='2004-01-01', 
                  trim_end='2011-03-05')
djia_closes = djia['Close'].reset_index()
data = pd.merge(data, djia_closes, 
                left_on='DJIADate', right_on='Date')
data.drop(['DJIADate'], inplace=True, axis=1)
data = data.set_index('Date')

# examine authors versus our DJIA data
data[['DJIAClose', 'Close']].plot(figsize=(12,8));
data[['DJIAClose', 'Close']].corr()

# type trends_report_debt.csv # on windows
from StringIO import StringIO
with open("trends_report_debt.csv") as f:
    data_section = f.read().split('\n\n')[1]
    trends_data = pd.read_csv(
        StringIO(data_section),
        header=1, index_col='Week',
        converters={
            'Week': lambda x: pd.to_datetime(x.split(' ')[-1])
        }
    )
our_debt_trends = trends_data['2004-01-01':'2011-02-28'] \
                              .reset_index()
final = pd.merge(data.reset_index(), our_debt_trends, 
                 left_on='GoogleWE', right_on='Week',
                 suffixes=['P', 'O'])
final.drop('Week', inplace=True, axis=1)
final.set_index('Date', inplace=True)
combined_trends = final[['GoogleWE', 'debtP', 'debtO']] \
                        .set_index('GoogleWE')
combined_trends.corr()

# visualize them together
fig, ax1 = plt.subplots(figsize=(12,8))
ax1.plot(combined_trends.index,
         combined_trends.debtP, color='b')
ax2 = ax1.twinx()
ax2.plot(combined_trends.index,
         combined_trends.debtO, color='r')
         
# generate the order signals
base = final.reset_index().set_index('GoogleWE')
base.drop(['DJIAClose'], inplace=True, axis=1)
# calculate the rolling mean of the previous three weeks for each week
base['PMA'] = pd.rolling_mean(base.debtP.shift(1), 3)
base['OMA'] = pd.rolling_mean(base.debtO.shift(1), 3)
# calculate the order signals
# for the papers data
base['signal0'] = 0 # default to 0
base.loc[base.debtP > base.PMA, 'signal0'] = -1 
base.loc[base.debtP < base.PMA, 'signal0'] = 1

# and for our trend data
base['signal1'] = 0
base.loc[base.debtO > base.OMA, 'signal1'] = -1
base.loc[base.debtO < base.OMA, 'signal1'] = 1
base[['debtP', 'PMA', 'signal0', 'debtO', 'OMA', 'signal1']]
      
# add in next week's percentage change to each week of data
base['PctChg'] = base.Close.pct_change().shift(-1)

# calculate the returns
base['ret0'] = base.PctChg * base.signal0
base['ret1'] = base.PctChg * base.signal1

# calculate and report the cumulative returns
base['cumret0'] = (1 + base.ret0).cumprod() - 1
base['cumret1'] = (1 + base.ret1).cumprod() - 1
base[['cumret0', 'cumret1']]

# show graph of growth for the papers data
base['cumret0'].plot(figsize=(12,4));
# show graph of growth for the papers data
base[['cumret0', 'cumret1']].plot(figsize=(12,4));             
```
## Ch 7 Algorithmic Trading
- two broad categories for predicting movements in the market: momemtum strategies(gives better results in a rising market) and mean-reversion strategies
### simple moving average DRAWBACKS:
- The shorter the window used, the more the noise in the signal feeds
- It doesn't tell you anything about the future
### exponential weighted moving average
```
# calculate EWMA relative to MA for 90 days
span = 90
msft_ewma = msft[['Adj Close']].copy()
msft_ewma['MA90'] = pd.rolling_mean(msft_ewma, span)
msft_ewma['EWMA90'] = pd.ewma(msft_ewma['Adj Close'], 
                              span=span)
msft_ewma['2014'].plot(figsize=(12, 8));
```
### Technical analysis techniques
#### 01 crossovers
- the simplest form is when the price of an asset moves from one side of a moving average to the other (represents a change in momentum)
- the second type, referred to as a dual moving average crossover, occurs when a short-term average crosses a long-term average
#### 02 pairs trading
- two highly correlated stocks will converge

### Code with Zipline
#### Buy Apple
```
import zipline as zp
class BuyApple(zp.TradingAlgorithm):
    """ Simple trading algorithm that does nothing
    but buy one share of AAPL every trading period.
    """
    
    trace=False
    
    def __init__(self, trace=False):
        BuyApple.trace = trace
        super(BuyApple, self).__init__()
    
    def initialize(context):
        if BuyApple.trace: print("---> initialize")
        if BuyApple.trace: print(context)
        if BuyApple.trace: print("<--- initialize")
        
    def handle_data(self, context):
        if BuyApple.trace: print("---> handle_data")
        if BuyApple.trace: print(context)
        self.order("AAPL", 1)
        if BuyApple.trace: print("<-- handle_data")  
        
import zipline.utils.factory as zpf
# zipline has its own method to load data from Yahoo! Finance
data = zpf.load_from_yahoo(stocks=['AAPL'], 
                           indexes={}, 
                           start=datetime(1990, 1, 1),
                           end=datetime(2014, 1, 1), 
                           adjusted=False)
data.plot(figsize=(12,8));
result = BuyApple(trace=True).run(data['2000-01-03':'2000-01-07'])
```
#### Dual Moving Average Cross-Over
```
# reminder of the AAPL data
sub_data = data['1990':'2002-01-01']
sub_data.plot(figsize=(12,8));
"""
The following algorithm implements a double moving average cross 
over. Investments will be made whenever the short moving average 
moves across the long moving average. We will trade only at the 
cross, not continuously buying or selling until the next cross. 
If trending down, we will sell all of our stock.  If trending up, 
we buy as many shares as possible up to 100. The strategy will 
record our buys and sells in extra data return from the simulation.
"""
class DualMovingAverage(zp.TradingAlgorithm):
    def initialize(context):
        # we need to track two moving averages, so we will set
        #these up in the context the .add_transform method 
        # informs zipline to execute a transform on every day 
        # of trading
        
        # the following will set up a MovingAverge transform, 
        # named short_mavg, accessing the .price field of the 
        # data, and a length of 100 days
        context.add_transform(zp.transforms.MovingAverage, 
                              'short_mavg', ['price'],
                              window_length=100)

        # and the following is a 400 day MovingAverage
        context.add_transform(zp.transforms.MovingAverage,
                              'long_mavg', ['price'],
                              window_length=400)

        # this is a flag we will use to track the state of 
        # whether or not we have made our first trade when the 
        # means cross.  We use it to identify the single event 
        # and to prevent further action until the next cross
        context.invested = False

    def handle_data(self, data):
        # access the results of the transforms
        short_mavg = data['AAPL'].short_mavg['price']
        long_mavg = data['AAPL'].long_mavg['price']
        
        # these flags will record if we decided to buy or sell
        buy = False
        sell = False

        # check if we have crossed
        if short_mavg > long_mavg and not self.invested:
            # short moved across the long, trending up
            # buy up to 100 shares
            self.order_target('AAPL', 100)
            # this will prevent further investment until 
            # the next cross
            self.invested = True
            buy = True # records that we did a buy
        elif short_mavg < long_mavg and self.invested:
            # short move across the long, tranding down
            # sell it all!
            self.order_target('AAPL', -100)
            # prevents further sales until the next cross
            self.invested = False
            sell = True # and note that we did sell

        # add extra data to the results of the simulation to 
        # give the short and long ma on the interval, and if 
        # we decided to buy or sell
        self.record(short_mavg=short_mavg,
                    long_mavg=long_mavg,
                    buy=buy,
                    sell=sell)
# run the simulation
results = DualMovingAverage().run(sub_data)
# draw plots of the results
def analyze(data, perf):
    fig = plt.figure() # create the plot
    
    # the top will be a plot of long/short ma vs price
    ax1 = fig.add_subplot(211,  ylabel='Price in $')
    data['AAPL'].plot(ax=ax1, color='r', lw=2.)
    perf[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

    # the following puts an upward triangle at each point 
    # we decided to buy
    ax1.plot(perf.ix[perf.buy].index, perf.short_mavg[perf.buy],
             '^', markersize=10, color='m')
    # and the following a downward triangle where we sold
    ax1.plot(perf.ix[perf.sell].index, perf.short_mavg[perf.sell],
             'v', markersize=10, color='k')

    # bottom plot is the portfolio value
    ax2 = fig.add_subplot(212, ylabel='Portfolio value in $')
    perf.portfolio_value.plot(ax=ax2, lw=2.)

    # and also has the marks for buy and sell points
    ax2.plot(perf.ix[perf.buy].index, 
             perf.portfolio_value[perf.buy],
             '^', markersize=10, color='m')
    ax2.plot(perf.ix[perf.sell].index, 
             perf.portfolio_value[perf.sell],
             'v', markersize=10, color='k')

    # and set the legend position and size of the result
    plt.legend(loc=0)
    plt.gcf().set_size_inches(14, 10)
```
#### pairs trading
```
# load data for Coke and Pepsi and visualize
data = zpf.load_from_yahoo(stocks=['PEP', 'KO'], 
                           indexes={},
                           start=datetime(1997, 1, 1), 
                           end=datetime(1998, 6, 1), 
                           adjusted=True)
data.plot(figsize=(12,8));
# calculate and plot the spread
data['Spread'] = data.PEP - data.KO
data['1997':].Spread.plot(figsize=(12,8))
plt.ylabel('Spread')
plt.axhline(data.Spread.mean());
import statsmodels.api as sm
@zp.transforms.batch_transform
def ols_transform(data, ticker1, ticker2):
    """Compute the ordinary least squares of two series.
    """
    p0 = data.price[ticker1]
    p1 = sm.add_constant(data.price[ticker2], prepend=True)
    slope, intercept = sm.OLS(p0, p1).fit().params

    return slope, intercept
class Pairtrade(zp.TradingAlgorithm):
    """ Pairtrade algorithm for two stocks, using a window 
    of 100 days for calculation of the z-score and 
    normalization of the spread. We will execute on the spread 
    when the z-score is > 2.0 or < -2.0. If the absolute value 
    of the z-score is < 0.5, then we will empty our position 
    in the market to limit exposure.
    """
    def initialize(self, window_length=100):
        self.spreads=[]
        self.invested=False
        self.window_length=window_length
        self.ols_transform= \
            ols_transform(refresh_period=self.window_length,
                          window_length=self.window_length)

    def handle_data(self, data):
        # calculate the regression, will be None until 100 samples
        params=self.ols_transform.handle_data(data, 'PEP', 'KO')
        if params:
            # get the intercept and slope
            intercept, slope=params

            # now get the z-score
            zscore=self.compute_zscore(data, slope, intercept)

            # record the z-score
            self.record(zscore=zscore)

            # execute based upon the z-score
            self.place_orders(data, zscore)

    def compute_zscore(self, data, slope, intercept):
        # calculate the spread
        spread=(data['PEP'].price-(slope*data['KO'].price+ 
                                       intercept))
        self.spreads.append(spread) # record for z-score calc
        self.record(spread = spread)
        
        # now calc the z-score
        spread_wind=self.spreads[-self.window_length:]
        zscore=(spread - np.mean(spread_wind))/np.std(spread_wind)
        return zscore

    def place_orders(self, data, zscore):
        if zscore>=2.0 and not self.invested:
            # buy the spread, buying PEP and selling KO
            self.order('PEP', int(100/data['PEP'].price))
            self.order('KO', -int(100/data['KO'].price))
            self.invested=True
            self.record(action="PK")
        elif zscore<=-2.0 and not self.invested:
            # buy the spread, buying KO and selling PEP
            self.order('PEP', -int(100 / data['PEP'].price))
            self.order('KO', int(100 / data['KO'].price))
            self.invested = True
            self.record(action='KP')
        elif abs(zscore)<.5 and self.invested:
            # minimize exposure
            ko_amount=self.portfolio.positions['KO'].amount
            self.order('KO', -1*ko_amount)
            pep_amount=self.portfolio.positions['PEP'].amount
            self.order('PEP', -1*pep_amount)
            self.invested=False
            self.record(action='DE')
        else:
            # take no action
            self.record(action='noop')
perf = Pairtrade().run(data['1997':])
# what actions did we take?
selection = ((perf.action=='PK') | (perf.action=='KP') |
             (perf.action=='DE'))
actions = perf[selection][['action']]
# plot prices
ax1 = plt.subplot(411)
data[['PEP', 'KO']].plot(ax=ax1)
plt.ylabel('Price')

# plot spread
ax2 = plt.subplot(412, sharex=ax1)
data.Spread.plot(ax=ax2)
plt.ylabel('Spread')

# plot z-scores
ax3 = plt.subplot(413)
perf['1997':].zscore.plot()
ax3.axhline(2, color='k')
ax3.axhline(-2, color='k')
plt.ylabel('Z-score')

# plot portfolio value
ax4 = plt.subplot(414)
perf['1997':].portfolio_value.plot()
plt.ylabel('Protfolio Value')

# draw lines where we took actions
for ax in [ax1, ax2, ax3, ax4]:
    for d in actions.index[actions.action=='PK']:
        ax.axvline(d, color='g')
    for d in actions.index[actions.action=='KP']:
        ax.axvline(d, color='c')
    for d in actions.index[actions.action=='DE']:
        ax.axvline(d, color='r')

plt.gcf().set_size_inches(16, 12)
```
### Ch 8 Working with Options
#### data
```
aapl_options = pd.read_csv('aapl_options.csv', 
                           parse_dates=['Expiry'])
# let's restructure and tidy this data to be useful in the examples
aos = aapl_options.sort(['Expiry', 'Strike'])[
    ['Expiry', 'Strike', 'Type', 'IV', 'Bid', 
     'Ask', 'Underlying_Price']] 
aos['IV'] = aos['IV'].apply(lambda x: float(x.strip('%')))
```
#### implied volatility
- can see the volatility smile when we plot this
```
# all calls on expiry date of 2015-02-27
calls1 = aos[(aos.Expiry=='2015-02-27') & (aos.Type=='call')]
# IV tends to be minimized at the underlying price
ax = aos[(aos.Expiry=='2015-02-27') & (aos.Type=='call')] \
        .set_index('Strike')[['IV']].plot(figsize=(12,8))
ax.axvline(calls1.Underlying_Price.iloc[0], color='g'); 
```
#### smirks: reverse skew and forward skew
for reverse-skew: 
- volatility for options at lower strikes is higher than at higher strikes.
- This means that in-the-money calss and out-the-money puts are more expensive than the relative.
- Possible explanation: worried about crash, so buy out-the-money puts; buy in-the-money call for the alternative to stock purchase
![reverse skewness](PATH)
for forward-skew:
- the IV for options at lower strikes is lower than the IV at higher strikes
- This pattern is common for options in the commodities market. When the supply is tight, businesses world would rather pay more to secure supply than to risk supply disruption.
![forward skewness](PATH)

#### Calculating payoff on options
```
# for call
def call_payoff(price_at_maturity, strike_price):
    return max(0, price_at_maturity - strike_price)
def call_payoffs(min_maturity_price, max_maturity_price, 
                 strike_price, step=1):
    """
    Calculate the payoffs for a range of maturity prices at 
    a given strike price
    """
    maturities = np.arange(min_maturity_price, 
                           max_maturity_price + step, step)
    """
    uses np.vectorize() to efficiently apply the call_payoff() function 
    to each item in the specific column of the DataFrame
    """
    payoffs = np.vectorize(call_payoff)(maturities, strike_price)
    df = pd.DataFrame({'Strike': strike_price, 'Payoff': payoffs}, 
                      index=maturities)
    df.index.name = 'Maturity Price'
    return df
```
#### Pricing of Options
- European easier to price: BS; American harder: binomial tree
- American option is generally higher than European due to the flexibility and increased risk on the counterparty side
- Black-Scholes's Assumptions:
1. There is no arbitrage
2. There is the ability to borrow money at a constant risk-free interest rate
throughout the life of the option
3. There are no transaction costs
4. The pricing of the underlying security follows a Brownian motion with
constant drift and volatility
5. No dividends are paid from the underlying security
#### Black-scholes using Mibian
Mibian provides several methods of option price calculation, one of which is B-S.
```
aos[aos.Expiry=='2016-01-15'][:2]
"""
OUT:
      Expiry  Strike  Type     IV    Bid    Ask  Underlying_Price
0 2016-01-15   34.29  call  57.23  94.10  94.95            128.79
1 2016-01-15   34.29   put  52.73   0.01   0.07            128.79
"""
date(2016, 1, 15) - date(2015, 2, 25) #OUT: datetime.timedelta(324)
import mibian
c = mibian.BS([128.79, 34.29, 1, 324], 57.23)
c.callPrice   #OUT: 94.878970089456217
c.putPrice
c = mibian.BS([128.79, 34.29, 1, 324], 
              callPrice=94.878970089456217 )
c.impliedVolatility
```
- Charting option price change over time:
```
df = pd.DataFrame({'DaysToExpiry': np.arange(364, 0, -1)})
bs_v1 = mibian.BS([128.79, 34.29, 1, 324], volatility=57.23)
calc_call = lambda r: mibian.BS([128.79, 34.29, 1, 
                                 r.DaysToExpiry], 
                                volatility=57.23).callPrice
df['CallPrice'] = df.apply(calc_call, axis=1)
df[['CallPrice']].plot(figsize=(12,8));
```
#### The Greeks
```
greeks = pd.DataFrame()
delta = lambda r: mibian.BS([r.Price, 60, 1, 180], 
                            volatility=30).callDelta
gamma = lambda r: mibian.BS([r.Price, 60, 1, 180], 
                            volatility=30).gamma
theta = lambda r: mibian.BS([r.Price, 60, 1, 180], 
                            volatility=30).callTheta
vega = lambda r: mibian.BS([r.Price, 60, 1, 365/12], 
                           volatility=30).vega

greeks['Price'] = np.arange(10, 70)
greeks['Delta'] = greeks.apply(delta, axis=1)
greeks['Gamma'] = greeks.apply(gamma, axis=1)
greeks['Theta'] = greeks.apply(theta, axis=1)
greeks['Vega'] = greeks.apply(vega, axis=1)
greeks[['Delta', 'Gamma', 'Theta', 'Vega']].plot(figsize=(12,8));
```
### Ch 9 Portfolios and Risk



