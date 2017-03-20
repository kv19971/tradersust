import pandas as pd 
import csv
from pandas_datareader import data as web
import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import os
from datetime import datetime as dt
import statsmodels.api as sm
from scipy.stats import norm
import math
import seaborn as sns
import time
import tkinter
from tkinter import ttk
import webbrowser
from datetime import datetime, timedelta
'''todo - add dashboard with current holdings and their prices 
dashboard - hit ratios, win ratios and profit ratios, average return per trade - current holdings - quit button 
add execution tab which allows users to add stocks and show the price when adding stocks

'''

top = tkinter.Tk()

def addEntry():
    code = e1.get()
    qty = int(e2.get())
    curdate = (datetime.now() - timedelta(days=1)).strftime("%d/%m/%Y")
    with open('traderAction.csv', 'a') as fl:
        writer = csv.writer(fl, delimiter=',')
        writer.writerow([curdate, code, qty])
    os.execl(sys.executable, sys.executable, *sys.argv)



n = ttk.Notebook(top)
fr_list = []
for i in range(8):
    f = ttk.Frame(n)
    fr_list.append(f)

n.add(fr_list[0], text="Dashboard")
n.add(fr_list[1], text="Regression")
n.add(fr_list[2], text="Portfolio Changes")
n.add(fr_list[3], text="Backtesting without overlapping")
n.add(fr_list[4], text="Backtesting with overlapping")
n.add(fr_list[5], text="Backtesting without fixed start")
n.add(fr_list[6], text="Stocks")

n.pack()
raw_csvDF = pd.read_csv("traderAction.csv")
#---Dashboard---#
dash = tkinter.Frame(fr_list[0])

lf = tkinter.LabelFrame(fr_list[0], text="Performance Overview", padx=20, pady=20)
lf.pack(fill=tkinter.BOTH)
hr = tkinter.LabelFrame(lf, text="Hit Ratios", padx=50, pady=20)
hr.pack(side=tkinter.LEFT)
wr = tkinter.LabelFrame(lf, text="Other Ratios", padx=50, pady=20)
wr.pack(side=tkinter.RIGHT)
def observeStock(stockName, start_date, attribute = 'Close'):
    'observe attribute of an arbitrary stock within the time specified'
    'start_date and end_date must be in format %Y-%m-%d'
    'attribute can be Open, High, Low, Close, Volume or Adj Close'
    start_date = dt.strptime(start_date, "%d/%m/%Y").strftime("%Y-%m-%d")
    end_date = curdate = (datetime.now() - timedelta(days=1)).strftime("%Y/%m/%d")
    sepwin = tkinter.Toplevel()
    confr = tkinter.Frame(sepwin)
    f = plt.figure(1, figsize=(10,8))
    df = web.DataReader(stockName, "yahoo", start_date, end_date)
    X = df.index
    Y = df[attribute]
    plt.plot(X, Y)
    plt.margins(0.02)
    plt.xlabel('Date')
    plt.ylabel(attribute)
    canvas = FigureCanvasTkAgg(f, master=confr)
    canvas.show()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
    canvas.update_idletasks()
    confr.pack()


stocklist = tkinter.LabelFrame(fr_list[0], text="Recent Transactions", padx=20, pady=20)
for d in raw_csvDF.tail(5).iterrows():
    act = ""
    if (d[1]['Quantity'] < 0):
        act="Sold"
    else:
        act="Bought"
    tkinter.Button(stocklist, text=act+" "+str(abs(d[1]['Quantity']))+" of "+str(d[1]['Symbol'])+" on "+d[1]['Date'], anchor=tkinter.W, justify=tkinter.LEFT, command=lambda sym=str(d[1]['Symbol']), date=str(d[1]['Date']): observeStock(sym, date)).pack(fill=tkinter.X)


stocklist.pack(fill=tkinter.BOTH)
tkinter.Button(dash, text="Quit", command=dash.quit).pack()
dash.pack()
#---Display stocks bought and sold + Form for adding stocks ---#
stform = tkinter.Frame(fr_list[6])
tkinter.Label(stform, text="Stock Code").grid(row=0, column=0)
tkinter.Label(stform, text="   ").grid(row=0, column=2)
tkinter.Label(stform, text="Quantity").grid(row=0, column=3)
tkinter.Label(stform, text="   ").grid(row=0, column=5)
tkinter.Button(stform, text="Add Entry", command=addEntry).grid(row=0, column=6)


e1 = tkinter.Entry(stform)
e2 = tkinter.Entry(stform)

e1.grid(row=0, column=1)
e2.grid(row=0, column=4)

stform.pack()

stlist = tkinter.Frame(fr_list[6])

#for index, row in df.iterrows():

stlist.pack()

#------------Data Extraction Functions------------#

def extractStockName(raw_csvDF):
    'Return list of stockName involved in traderAction.csv'
    stockName = []

    for i in raw_csvDF["Symbol"]:
        if i not in stockName:
            stockName.append(i)

    return stockName

def extractDateRange(raw_csvDF):
    'Return dateMin, dateMax, and also converts date to datetime object'
    dateMax = dt.strptime(raw_csvDF["Date"][0], "%d/%m/%Y")
    dateMin = dt.strptime(raw_csvDF["Date"][0], "%d/%m/%Y")

    for i in raw_csvDF["Date"]:
        i = dt.strptime(i, "%d/%m/%Y")
        if i < dateMin:
            dateMin = i
        if i > dateMax:
            dateMax = i

    return dateMin, dateMax

def assignTradeDates(raw_csvDF, spyDF):
    'Trade dates inputted by user in traderAction.csv may not be valid trading dates'
    'This function changes non trading dates in traderAction.csv to the next nearest trading date available'
    'Return the modified csv as csvDF'
    csvDF = raw_csvDF
    trueIndexList = []

    for i in range(len(raw_csvDF)):   
        recordDate =  dt.strptime(raw_csvDF.Date[i], "%d/%m/%Y").strftime("%Y-%m-%d")
        if recordDate in spyDF.index:
            trueIndexList.append(pd.to_datetime(recordDate, format='%Y-%m-%d', errors='ignore'))
        else:
            actualDate = spyDF.loc[lambda df: df.index >= recordDate, :].iloc[0].name
            trueIndexList.append(actualDate)

    csvDF['Date'] = trueIndexList

    return csvDF

def extractTrades(csvDF, rawDF):
    'Extract trades in csvDF and calculate the returns of each trade'
    'Returns two dataframes:'
    'tradesOnHold records all unexcited positions and unrealized returns'
    'finishedTrades records all realized returns'
    sortedDF = csvDF.sort_values(by='Symbol')
    sortedDF.index = range(len(csvDF))

    tradesOnHold = []
    finishedTrades = []
    curStockAmt = 0

    for i in range(1, len(sortedDF)):
        if sortedDF['Symbol'][i] == sortedDF['Symbol'][i-1]:
            stockName = sortedDF['Symbol'][i]
            start_date = sortedDF['Date'][i-1]
            end_date = sortedDF['Date'][i]
            returns = float(format((float(rawDF[stockName][end_date][0]) - float(rawDF[stockName][start_date][0])) * abs(sortedDF['Quantity'][i-1]), '.7f'))
            
            finishedTrades.append([start_date, sortedDF['Symbol'][i-1], curStockAmt, sortedDF['Quantity'][i-1] + curStockAmt, returns])
            curStockAmt = sortedDF['Quantity'][i-1] + curStockAmt
        else:
            stockName = sortedDF['Symbol'][i-1]

            if sortedDF['Quantity'][i-1] + curStockAmt == 0: # trader has exited position
                start_date = sortedDF['Date'][i-1]
                end_date = sortedDF['Date'][i]
                returns = float(format((float(rawDF[stockName][end_date][0]) - float(rawDF[stockName][start_date][0])) * abs(sortedDF['Quantity'][i-1]), '.7f'))
                
                finishedTrades.append([start_date, stockName, curStockAmt, 0, returns])
            else: # stock involved in (i-1)-th row remains on hold till the end
                start_date = sortedDF['Date'][i-1]
                end_date = csvDF['Date'][len(csvDF)-1]
                returns = float(format((float(rawDF[stockName][end_date][0]) - float(rawDF[stockName][start_date][0])) * abs(sortedDF['Quantity'][i-1]), '.7f'))

                tradesOnHold.append([start_date, sortedDF['Symbol'][i-1], curStockAmt, sortedDF['Quantity'][i-1] + curStockAmt, returns])
            curStockAmt = 0

    tradesOnHoldDF = pd.DataFrame(tradesOnHold)
    finishedTradesDF = pd.DataFrame(finishedTrades)
    tradesOnHoldDF.columns = ['Date', 'Symbol', 'QTY Before Trade', 'QTY After Trade', 'Returns']
    finishedTradesDF.columns = ['Date', 'Symbol', 'QTY Before Trade', 'QTY After Trade', 'Returns']

    return tradesOnHoldDF, finishedTradesDF

def extractRawDF(dateRange, stockName, csvDF):
    'Return rawDF'
    'Every cell has format [float1, float2]; float1 is daily closing price, float2 is action perform on specific day'
    # dateRange = pd.date_range(dateMin.strftime("%Y-%m-%d"), dateMax.strftime("%Y-%m-%d"))
    rawDF = pd.DataFrame(index = dateRange)

    # 1. initialize all [float1, float2] to [0, 0]
    for i in stockName:
        ls = []
        for j in range(len(dateRange)):
            ls.append([0,0])
        rawDF[i] = ls

    # 2. insert float2 values based on information from csvDF 
    for i in csvDF.index:
        rawDF.ix[csvDF["Date"][i], csvDF["Symbol"][i]][1] = csvDF["Quantity"][i]

    # 3. extract data frame yahoo
    for i in stockName:
        tempDF = web.DataReader(i, "yahoo", dateMin, dateMax)
        for j in range(len(tempDF.index)):
            rawDF[i][tempDF.index[j]][0] = format(tempDF["Close"][tempDF.index[j]], '.2f')

    return rawDF

def convertDF(rawDF, stockName, numOfDays):
    'Convert rawDF to cumulativeDF'
    'Every slot is [float1, float2], float1 is daily closing price, float2 is stocks on hand'
    cumulativeDF = rawDF

    for i in stockName:
        for j in range(1, len(rawDF)):   # Starting from second index
            cumulativeDF[i][j][1] = cumulativeDF[i][j][1] + cumulativeDF[i][j - 1][1]

    return cumulativeDF

def overallPortfolioChanges(rawDF, stockName, portfolioChanges, cumulativeDF):
    'Return daily portfolio changes in a list'
    portfolioChanges = []
    numerator = []
    denominator = []

    for i in stockName:
        numeratorList = []
        denominatorList = []
        preIndex = 0
        for j in range(1, len(cumulativeDF)):
            if cumulativeDF[i][j][1] == 0:
                numeratorList.append(0)
                denominatorList.append(0) 
            else:
                if cumulativeDF[i][preIndex][1] == 0:
                    numeratorList.append(0)
                    denominatorList.append(0)
                else:
                    numeratorList.append(float(cumulativeDF[i][j][0]) * float(cumulativeDF[i][j][1]) - float(cumulativeDF[i][preIndex][0]) * float(cumulativeDF[i][preIndex][1]))
                    denominatorList.append(float(cumulativeDF[i][preIndex][0]) * float(cumulativeDF[i][preIndex][1]))
                preIndex = j
        numerator.append(numeratorList)
        denominator.append(denominatorList)

    for i in range(len(numerator[0])):
        numeratorSum = 0
        denominatorSum = 0

        for j in range(len(numerator)):
            numeratorSum = numeratorSum + numerator[j][i]
            denominatorSum = denominatorSum + denominator[j][i]
        
        if denominatorSum != 0:    # Only add to the list if denominator is nonzero value
            portfolioChanges.append(float(format(numeratorSum / denominatorSum, '.7f'))) 

    return portfolioChanges


def extractSPY(dateMin, dateMax):
    'Return S&P 500 index for period of portfolio changes in a dataframe'
    'Return changes of S&P 500 index as daily benchmark changes in a list'
    spyDF = web.DataReader("SPY", "yahoo", dateMin, dateMax)
    spyChanges = []

    for i in range(1,len(spyDF.index)):
        spyChanges.append(float(format((spyDF["Close"][i] - spyDF["Close"][i - 1]) / spyDF["Close"][i - 1], '.7f')))

    return spyDF, spyChanges

#------------Graph Plotting and VaR functions------------#

def plotRegression(portfolioChanges, spyChanges, fr):
    'Plot daily benchmark changes vs daily portfolio changes'
    X = portfolioChanges
    Y = spyChanges
    n2 = ttk.Notebook(fr)
    grph = ttk.Frame(n2)
    smry = ttk.Frame(n2)
    n2.add(grph, text="Graph")
    n2.add(smry, text="Summary")
    n2.pack()
    f = plt.figure(1, figsize=(10,8))
    #f1.add_subplot(111)
    #plt.ion()
    results = sm.OLS(Y, sm.add_constant(X)).fit()
    sns.set()
    plt.scatter(X, Y)
    fit = np.polyfit(X, Y, 1)
    fit_fn = np.poly1d(fit) # fit_fn is now a function which takes in x and returns an estimate for y
    #plt.ion()
    plt.plot(X, Y, 'yo', X, fit_fn(X))
    plt.margins(0.02)
    plt.xlabel('Daily Portlio Changes')
    plt.ylabel('Daily Benchmark Changes')
    # show this shit in gui
    print(results.summary())
    tkinter.Label(smry, text=results.summary(), anchor=tkinter.W).pack()
    #plt.show()
    canvas = FigureCanvasTkAgg(f, master=grph)
    canvas.show()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

def plotVaR(portfolioChanges, investmentValue, confidenceLevel, _window, fr):
    f = plt.figure(2, figsize=(10,8))
    #plt.ion()
    'Plot VaR vs time'
    returns = pd.DataFrame(portfolioChanges) # Create dataframe of returns

    returnsMean = returns.rolling(center = False, window = _window).mean()
    returnsVolatility = returns.rolling(center = False, window = _window).std()

    var = VaR(investmentValue, confidenceLevel, returnsMean, returnsVolatility)
    varToDF = pd.DataFrame(var, columns = ["VaR"])
    

    plt.margins(0.02)
    plt.plot(varToDF)
    plt.xlabel('Day')
    plt.ylabel('VaR')
    #print(results.summary())
    #plt.show()
    canvas = FigureCanvasTkAgg(f, master=fr)
    canvas.show()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    return var

def VaR(value, confidenceLevel, returnsMean, returnsVolatility):
    'Formula to calculate Value at Risk'
    alpha = norm.ppf(1 - confidenceLevel, returnsMean, returnsVolatility)
    return (value - value * (alpha + 1))

def calculateHitRatio(varlist, threshold):
    'Return hit ratio'
    lenOfHit = 0
    lenOfVar = 0

    for i in range(len(varlist)):
        if math.isnan(float(varlist[i])) == False: # Eliminate NULL value
            lenOfVar = lenOfVar + 1
            if float(varlist[i]) > float(threshold):
                lenOfHit = lenOfHit + 1

    if lenOfVar == 0:
        raise ValueError("Cannot be divided by zero")

    return float(format(float(lenOfHit) / lenOfVar, '.7f'))


#------------backtesting functions------------#

def porfolioChangesSumList(portfolioChanges):
    'Returns a list S such that S[j] is the sum of portfolioChanges[0] + portfolioChanges[1] + ... + portfolioChanges[j-1] + portfolioChanges[j]'
    sum_of_first_i_items = []
    sum_of_first_i_items.append(float(format(portfolioChanges[0], '.7f')))

    for i in range(1, len(portfolioChanges)):
        sum_of_first_i_items.append(float(format(sum_of_first_i_items[i-1] + portfolioChanges[i], '.7f')))

    return sum_of_first_i_items

def porfolioChangesSquaredSumList(portfolioChanges):
    'Returns a list S such that S[j] is the sum of portfolioChanges[0]^2 + portfolioChanges[1]^2 + ... + portfolioChanges[j-1]^2 + portfolioChanges[j]^2'
    sum_of_first_i_items = []
    sum_of_first_i_items.append(float(format(portfolioChanges[0] * portfolioChanges[0], '.7f')))

    for i in range(1, len(portfolioChanges)):
        sum_of_first_i_items.append(float(format(sum_of_first_i_items[i-1] + portfolioChanges[i] * portfolioChanges[i], '.7f')))

    return sum_of_first_i_items

def backtestWithoutOverlapping(investmentValue, confidenceLevel, portfolioChanges, sumList, squaredSumList, intervalSize = 7):
    i = 0
    varList = []

    while i <= (len(portfolioChanges) - intervalSize):
        if (i == 0):
            returnsMean = sumList[i + intervalSize - 1]/intervalSize
            returnsVolatility = math.sqrt(squaredSumList[i + intervalSize - 1]/intervalSize - returnsMean * returnsMean)
            varList.append(float(format(VaR(investmentValue, confidenceLevel, returnsMean, returnsVolatility), '.7f')))

        else:
            returnsMean = (sumList[i + intervalSize - 1] - sumList[i-1])/intervalSize
            returnsVolatility = math.sqrt((squaredSumList[i + intervalSize - 1] - squaredSumList[i-1])/intervalSize - returnsMean * returnsMean)
            varList.append(float(format(VaR(investmentValue, confidenceLevel, returnsMean, returnsVolatility), '.7f')))
            
        i = i + intervalSize

    return varList

def backtestWithOverlapping(investmentValue, confidenceLevel, portfolioChanges, sumList, squaredSumList, testFrequency = 3, intervalSize = 7):
    i = 0
    varList = []

    while i <= (len(portfolioChanges) - intervalSize):
        if (i == 0):
            returnsMean = sumList[i + intervalSize - 1]/intervalSize
            returnsVolatility = math.sqrt(squaredSumList[i + intervalSize - 1]/intervalSize - returnsMean * returnsMean)
            varList.append(format(VaR(investmentValue, confidenceLevel, returnsMean, returnsVolatility), '.7f'))

        else:
            returnsMean = (sumList[i + intervalSize - 1] - sumList[i-1])/intervalSize
            #print((squaredSumList[i + intervalSize - 1] - squaredSumList[i-1])/intervalSize - returnsMean * returnsMean)
            returnsVolatility = math.sqrt((squaredSumList[i + intervalSize - 1] - squaredSumList[i-1])/intervalSize - returnsMean * returnsMean)
            varList.append(float(format(VaR(investmentValue, confidenceLevel, returnsMean, returnsVolatility), '.7f')))
            
        i = i + testFrequency

    return varList

def backtestWithFixedStart(investmentValue, confidenceLevel, portfolioChanges, sumList, squaredSumList, intervalSize = 7):
    i = 0
    varList = []

    while i <= (len(portfolioChanges) - intervalSize):
        returnsMean = sumList[i + intervalSize - 1]/(i + intervalSize + 1)
        returnsVolatility = math.sqrt(squaredSumList[i + intervalSize - 1]/(i + intervalSize + 1) - returnsMean * returnsMean)
        varList.append(float(format(VaR(investmentValue, confidenceLevel, returnsMean, returnsVolatility), '.7f')))
            
        i = i + intervalSize

    return varList

def manualPlotVaR(varList, fr, ind):
    varToDF = pd.DataFrame(varList, columns = ["VaR"])
    f = plt.figure(ind, figsize=(10,8))
    plt.margins(0.02)
    plt.plot(varToDF)
    plt.xlabel('Interval')
    plt.ylabel('VaR')
    canvas = FigureCanvasTkAgg(f, master=fr)
    canvas.show()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    return None

# ----------- Other Metrics ----------- #

def winRatio(finishedTrades):
    'Calculate the ratio of profitable trades to total number of trades'
    count = 0

    for i in range(len(finishedTrades)):
        if finishedTrades['Returns'][i] > 0:
            count = count + 1
    
    return float(count)/len(finishedTrades) 


def averageReturnPerTrade(finishedTrades):
    'Calculate the average returns generated from portfolio'
    return sum(finishedTrades['Returns'])/len(finishedTrades)

def profitFactor(finishedTrades):
    'Calculate the ratio of profitable trades to unprofitable trades'
    loss = 0.0
    profit = 0.0

    for i in range(len(finishedTrades)):
        if finishedTrades['Returns'][i] > 0:
            profit = profit + finishedTrades['Returns'][i]
        else:
            loss = loss + finishedTrades['Returns'][i]

    return profit/math.fabs(loss)

# ----------- main ----------- #
stockName = extractStockName(raw_csvDF)

dateMin, dateMax = extractDateRange(raw_csvDF)

numOfDays = (dateMax - dateMin).days + 1

# ---------------------------- #

start_time = time.time()

spyDF, spyChanges = extractSPY(dateMin, dateMax)
print(spyDF)

csvDF = assignTradeDates(raw_csvDF, spyDF)

print("--- extractSPY() takes %s seconds ---" % (time.time() - start_time))

# ---------------------------- #

start_time = time.time()

rawDF = extractRawDF(spyDF.index, stockName, csvDF)

print(rawDF)

print("--- extractRawDF() takes %s seconds ---" % (time.time() - start_time))

# ---------------------------- #

start_time = time.time()

cumulativeDF = convertDF(rawDF, stockName, numOfDays)

print("--- convertDF() takes %s seconds ---" % (time.time() - start_time))

# ---------------------------- #

start_time = time.time()

portfolioChanges = overallPortfolioChanges(rawDF, stockName, [], cumulativeDF)

print("--- overallPortfolioChanges() takes %s seconds ---" % (time.time() - start_time))

# ---------------------------- #

start_time = time.time()

plotRegression(portfolioChanges, spyChanges, fr_list[1])

print("--- plotRegression() takes %s seconds ---" % (time.time() - start_time))

# --------------backtesting with moving average-------------- #

start_time = time.time()

var = plotVaR(portfolioChanges, 500000, 0.99, 3, fr_list[2])

print("--- plotVaR() takes %s seconds ---" % (time.time() - start_time))

# --------------backtesting with intervals-------------- #

start_time = time.time()

sumList = porfolioChangesSumList(portfolioChanges)

squaredSumList = porfolioChangesSquaredSumList(portfolioChanges)

varList = backtestWithoutOverlapping(500000, 0.99, portfolioChanges, sumList, squaredSumList, 7)

varList1 = backtestWithOverlapping(500000, 0.99, portfolioChanges, sumList, squaredSumList, 3, 7)

varList2 = backtestWithFixedStart(500000, 0.99, portfolioChanges, sumList, squaredSumList, 7)

manualPlotVaR(varList, fr_list[3], 3)

manualPlotVaR(varList1, fr_list[4], 4)

manualPlotVaR(varList2, fr_list[5], 5)

print("--- backtesting with intervals takes %s seconds ---" % (time.time() - start_time))

# -------------Calculate hit ratio for each backtest--------------- #

start_time = time.time()

hitRatio = calculateHitRatio(var, 5000)

hitRatio_0 = calculateHitRatio(varList, 5000)

hitRatio_1 = calculateHitRatio(varList1, 5000)

hitRatio_2 = calculateHitRatio(varList2, 5000)

tkinter.Label(hr, text="Hit Ratio:    "+str(hitRatio)+"", anchor=tkinter.W, justify=tkinter.LEFT).pack(fill=tkinter.X)
tkinter.Label(hr, text="Hit Ratio 2: "+str(hitRatio_0)+"", anchor=tkinter.W, justify=tkinter.LEFT).pack(fill=tkinter.X)
tkinter.Label(hr, text="Hit Ratio 3: "+str(hitRatio_1)+"", anchor=tkinter.W, justify=tkinter.LEFT).pack(fill=tkinter.X)
tkinter.Label(hr, text="Hit Ratio 4: "+str(hitRatio_2)+"", anchor=tkinter.W, justify=tkinter.LEFT).pack(fill=tkinter.X)

print(hitRatio, hitRatio_0, hitRatio_1, hitRatio_2)

print("--- calculateHitRatio() takes %s seconds ---" % (time.time() - start_time))

# -------------calculate returns of each trade-------------#

tradesOnHold, finishedTrades = extractTrades(csvDF, rawDF)
tkinter.Label(wr, text="Win Ratio :    "+str(winRatio(finishedTrades)), anchor=tkinter.W, justify=tkinter.LEFT).pack(fill=tkinter.X)
tkinter.Label(wr, text="Profit Ratio : "+str(profitFactor(finishedTrades)), anchor=tkinter.W, justify=tkinter.LEFT).pack(fill=tkinter.X)
tkinter.Label(wr, text="Average Return per Trade : "+str(averageReturnPerTrade(finishedTrades)), anchor=tkinter.W, justify=tkinter.LEFT).pack(fill=tkinter.X)

top.mainloop()

top.destroy()
