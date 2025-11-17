#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 15:45:52 2025

@author: theo.cammisar
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

tickers = ['AI.PA','MC.PA','^FCHI']

DATA_Stock = yf.download(tickers=tickers,start='2020-07-10', end='2025-07-10',interval='1d',auto_adjust=True)

print(DATA_Stock.columns.tolist())
print(DATA_Stock['Close']['^FCHI'].head())

df_DATA = pd.DataFrame(DATA_Stock)
print(df_DATA)

#coder CAPM

return_AI_PA = DATA_Stock['Close']['AI.PA'].pct_change().dropna() #permet d'avoir les rendements
return_FCHI = DATA_Stock['Close']['^FCHI'].pct_change().dropna()

model = sm.OLS(return_FCHI,return_AI_PA)

result_capm = model.fit()
print(result_capm.summary())

beta0,beta1= result_capm.params
help(sm)
print("le beta de air liquide ou airbus je sais plus est :",beta1)

plt.scatter(return_AI_PA,return_FCHI,color="red")
plt.xlabel("Rendement de air liquide")
plt.ylabel("Rendement du CAC 40")
plt.title("Rendement du CAC et d'air liquide sur une période de 5 ans")

#test graphique

plt.plot(DATA_Stock.index,return_AI_PA)

plt.plot(DATA_Stock.index,DATA_Stock['Close']['MC.PA'],color='blue')
plt.xlabel("Cours de LVMH sur 5 ans")

plt.boxplot(DATA_Stock['Close']['MC.PA'])

plt.boxplot(return_AI_PA)

# ARIMA model
# Regardons ce qu'il se passe sur le rendement de LVMH
plt.plot(DATA_Stock.index,DATA_Stock['Close']['MC.PA'],color='blue')
plt.xlabel("Cours de LVMH sur 5 ans")

# il faut maintenant vérifier si la série ets stationnaire au sens faible du terme
#c'est-à-dire si sa moyenne est indep du temps, la variance et cov constante pour tous les 
#intervalles de la série

MC_PA = DATA_Stock['Close']['MC.PA']
print(adfuller(MC_PA)[0],adfuller(MC_PA)[1]) #série non stationnaire car p-value > 0.05
Log_MC_PA = np.log(MC_PA)
print(adfuller(Log_MC_PA)[1]) #série toujours stationnaire

mc_pa_return = DATA_Stock['Close']['MC.PA'].pct_change().dropna()
print(mc_pa_return.head())
result_stationnary = adfuller(mc_pa_return)
print(result_stationnary[1])

#Passage en log si cela ne fonctionne pas 
def get_stationnary(serie):
    resultat = adfuller(serie)
    if resultat[1] < 0.05:
        print(f"La série est stationnaire et la p-value est de {resultat[1]:.10f}")
    else:
        print("La série n'est pas stationnaire, nous allons corriger cela")
        modified_serie = np.log(serie)-np.log(serie.shift[1])
        print(f'nous avons comme nouvelle p-value {adfuller(modified_serie)[1]:.15f}')
        plt.plot(modified_serie.index,modified_serie['Close'],color='red')
        plt.title("Cours sur 5 ans de", serie)
        plt.show()
        return modified_serie
get_stationnary(return_FCHI)

#Passage au delta normal dXt = Xt - Xt-1
def get_stationnary_2(serie):
    resultat = adfuller(serie)
    if resultat[1] < 0.05:
        print(f"La série est stationnaire et la p-value est de {resultat[1]:.10f}")
    else:
        print("La série n'est pas stationnaire, nous allons corriger cela")
        modified_serie_2 = serie.diff().dropna()
        print(f"comme nouvelle p-value {adfuller(modified_serie_2)[1]:.12f}")
        graph = plt.plot(modified_serie_2)
        print(graph)
        return modified_serie_2
get_stationnary_2(MC_PA)

## Refaire avec des séries aux rendements mensuels sur 30 ans 

stock = ['BNP.PA','TTE.PA','OR.PA']
df_stocks = yf.download(tickers = stock,start = '2000-01-01',end='2025-01-01',
                        interval='1mo',auto_adjust=True)

print(df_stocks.head(10))
print(df_stocks['FCHI'])

def tickers(stock):
    rdt = df_stocks['Close'][stock].pct_change().dropna()
    print('Voici les 10 premiers rendemendt de ',stock, rdt.head(10))
    rdt_[stock]=rdt[stock]
tickers(stock)

or_pa_return = df_stocks['Close']['OR.PA'].pct_change().dropna()
tte_pa_return = df_stocks['Close']['TTE.PA'].pct_change().dropna()
bnp_pa_return = df_stocks['Close']['BNP.PA'].pct_change().dropna()

def get_stationnary_2(serie):
    resultat = adfuller(serie)
    if resultat[1] < 0.05:
        print(f"La série est stationnaire et la p-value est de {resultat[1]:.10f}")
    else:
        print("La série n'est pas stationnaire, nous allons corriger cela")
        modified_serie_2 = serie.diff().dropna()
        print(f"comme nouvelle p-value {adfuller(modified_serie_2)[1]:.12f}")
        graph = plt.plot(modified_serie_2)
        print(graph)
        return modified_serie_2

get_stationnary_2(or_pa_return)
get_stationnary_2(bnp_pa_return)
get_stationnary_2(tte_pa_return)

def graph(serie):
    resultat = adfuller(serie)
    rolling_mean = serie.rolling(window = 30).mean()
    rolling_sd = serie.rolling(window=30).std()
    if resultat[1] < 0.05:
        plt.plot(serie,color='blue',label='rendement mensuel')
        plt.plot(rolling_mean,color='red',label = 'rolling mean')
        plt.plot(rolling_sd,color='green', label='rolling standard deviation')
        plt.legend(loc='best')
        plt.title("rendement au cours des 25 dernières années")
        #plt.ylabel("rendement d")
        plt.show(block=False)
    else:
        print("La série n'est pas stationnaire. Rien ne sert de continuer !!!")

graph(or_pa_return)

# Ex de série temp non stationnaire

rolling_mean_bnp = df_stocks['Close']['BNP.PA'].rolling(window=30).mean()
rolling_std_bnp = df_stocks['Close']['BNP.PA'].rolling(window=30).std()

plt.plot(df_stocks['Close']['BNP.PA'], label = "cours mensuel")
plt.plot(rolling_mean_bnp, color='red', label = "rolling mean")
plt.plot(rolling_std_bnp, color = 'black', label = "rolling std")
plt.legend(loc='best')
plt.title("Cours de la BNP à intervalles mensuels sur 25 ans")
## on voit sur ce graphique que la moyenne mobile n'est pas stable dans le temps.
## de même pour l'écart-type


### TESTS

# premier test de prévision pour les 3 ans à venir (rdt mensuel)
model = ARIMA(or_pa_return,order = (2,1,2))
results = model.fit()
print(results.summary())

# résultat graphique 
plt.plot(or_pa_return,color='red',label="rendement mensuel L'Oréal")
plt.plot(results.fittedvalues,color='black',label='résultats du modèle ARIMA')
plt.title("Comparaison entre les résultats du modèle ARIMA et du rdt mensuel")
plt.legend(loc='best')


print("\n", results.summary()) #\n := saut de ligne pour les résultats
print("\n", results.resid.head(40))

fig = results.plot_diagnostics(figsize = (14,8))
plt.show()

fig2 = results.plot_predict(1,210)
plt.show()

# Méthodes pour prédire (plot_predict n'est plus disponible 
# dans from statsmodels.tsa.arima.model import ARIMA)

#utilisons une autre méthode
#docu : https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima.model.ARIMAResults.get_prediction.html

or_pa_return_pred = results.get_prediction(start=0,end=350)
or_pa_return_pred_fit = or_pa_return_pred.fittedvalues() # ne fonctionne pas car pas implémenter
or_pa_return_pred_mean = or_pa_return_pred.predicted_mean
or_pa_return_pred_confint = or_pa_return_pred.conf_int(alpha=0.05)


#choses utiles à faire pour tracer un beau graphique
n_obs = len(or_pa_return) 
n_forecast = 350 - n_obs
print(n_obs)

# représentation graphique des prédictions du rendement sur 3 ans
print(or_pa_return_pred_mean)
fig, ax= plt.subplots(figsize=(14, 6))
ax.plot(range(n_obs),or_pa_return_pred_mean.iloc[:n_obs], 
        color='orange',linewidth=1.2, label='mean ARIMAs model return')
ax.plot(range(n_obs),or_pa_return,
        color='blue',label='rendement l oreal',linewidth=0.7)
ax.set_title("prédiction du rendemend mensuel de loreal sur 3 ans")

if n_forecast > 0:
    ax.plot(range(n_obs-1,350),or_pa_return_pred_mean.iloc[n_obs:], 
            color='red',linewidth=1.5, label='return forcast')
else:
        print("\n graphique à revoir, pas bon \n t'es guez")

ax.legend(loc='best')
plt.show() # Pas très concluant



## ARIMA's model prediction on CAC40 stock

df_cac = yf.download(tickers='^FCHI',start='1990-01-01',end='2025-01-01',auto_adjust=True,
                     interval='1mo')
cac_40 = df_cac['Close']
fig, ax=plt.subplots(figsize=(15,7))
ax.plot(cac_40.index,cac_40,color='black', linewidth=0.9)
ax.set_title('Cours du CAC 40 sur 35 ans')
plt.show()

# On vérifie si la série est stationnaire

def stationnary(serie): 
    resultat = adfuller(serie)
    if resultat[1] > 0.05:
        print("\n la série n'est pas stationnaire mais nous allons remédier à cela")
        diff_serie = serie.diff().dropna
        print("\n la nouvelle série commence par ", diff_serie)
    else:
        print('\n la série est stationnaire')
        return diff_serie
stationnary(cac_40)
#cela ne fonctionne pas comme je veux car je n'arrive pas à créer un data frame diff_serie en 
#dehors de la fonction

resultat = adfuller(cac_40)
print(resultat[1]) #pas stationnaire

#nous allons différencier

cac_40_diff = cac_40.diff().dropna()
cac_40_fit = cac_40_diff.fittedvalues()
resultat2 = adfuller(cac_40_diff)
print(f"\n {resultat2[1]:.67f}")
print(cac_40_diff)

# vérifions la stationnarité par la visualisation graphique
cac_40_rolling_mean = cac_40_diff.rolling(window=30).mean()
cac_40_rolling_std = cac_40_diff.rolling(window=30).std()

if resultat2[1]<0.05:
    fig, ax = plt.subplots(figsize = (14,8))
    ax.plot(cac_40_diff.index, cac_40_diff, color = 'black',linewidth=0.7)
    ax.plot(cac_40_diff.index,cac_40_rolling_mean,color='orange', label='rolling mean of cac40')
    ax.plot(cac_40_diff.index,cac_40_rolling_std,color='blue', label='rolling std of cac40')
    ax.set_title('Cours différencié du cac 40 sur 30')
    ax.legend(loc='best',fontsize=17)
    plt.show()
else:
    print('C est toujours pas bon')
# la série est bien stationnaire à l'oeil et au test 

print(resultat2)

#prédictions du cours du cac40 sur 2 ans (417 mois + 24 mois soit 2 ans de prédiction)
plot_acf(cac_40_diff.dropna(), lags=30)
plot_pacf(cac_40_diff.dropna(), lags=30)
plt.show()

#test
log_cac_40 = np.log(cac_40) - np.log(cac_40.shift(1)) # c'est le rendement et donc pas le but
plt.plot(log_cac_40)







