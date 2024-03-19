# modules imports
import streamlit as st
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import yfinance as yf
from yahooquery import Screener
import requests
from requests_html import HTMLSession
from datetime import datetime
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier

# Helper functions

def tickers_list():
    '''
    Generate 10 most popular cryto currencies tickers
    '''
    session = HTMLSession()
    num_currencies=10
    resp = session.get(f"https://finance.yahoo.com/crypto?offset=0&count={num_currencies}")
    tables = pd.read_html(resp.html.raw_html)               
    df = tables[0].copy()
    cc = df.Symbol.tolist()
    return cc

def get_data(selected_tickers, start_date, end_date):
    '''
    given the selected_tickers, load the corresponding closing price from
    start_date until end_date
    '''
    tickers = yf.Tickers(selected_tickers)
    data = tickers.history(start=start_date,end=end_date,interval='1d')['Close']
    return data

def fed_rate():
    '''
    Loading the US federal reserve interest rate
    and set type as float.
    '''
    # Scrapping the current FED interest rate:
    all_data = pd.read_html('https://www.global-rates.com/en/interest-rates/central-banks/central-bank-america/fed-interest-rate.aspx')
    # FED = [df for df in all_data if df.iloc[0][0] == 'American interest rate (Fed)'][0].iloc[0][1]
    FED = float((all_data[0].iloc[0, 1])[:-2])
    # Formating into float
    return FED #float(FED[:5])/100

def optimized_ratios(data, start_date, end_date, safe):
    '''
    Compute the optimized ratios of selected tickers given:
     - start and end date time-horizon.
     - Risk free interest rate as Safe parameter.
    Return the result as pandas data frame
    '''
    mu = mean_historical_return(data)
    S = CovarianceShrinkage(data).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe(risk_free_rate= safe)
    cleaned_weights = ef.clean_weights()
    # optimized_portfolio = pd.DataFrame([i for i in zip(cleaned_weights.keys(), 
    #                                                    cleaned_weights.values())], 
    #                                                    columns=['Tickers','Ratio'])
    # return optimized_portfolio
    return [round(i, 2) for i in cleaned_weights.values()]

def benchmark_ratios(selected_tickers):
     '''
     Alternative portfolio ratio with equal share of
     the selected tickers.
     Used for performance comparaison.
     Return the results as a pandas data frame
     '''
     return pd.DataFrame([i for i in zip(selected_tickers,
                                         [(1/len(selected_tickers)) for i in (selected_tickers)])])
## Optimized Ratios pieplot
def pie_plot(data):
    '''
    Pie Plot of the calculated ratios of selected tickers
    '''
    pie_fig = alt.Chart(data).mark_arc().encode(
         theta=alt.Theta(field="Ratio", type="quantitative"),
         radius=alt.Radius('Ratio', 
                           scale=alt.Scale(type="sqrt", 
                                           zero=True, 
                                           rangeMin=20)),
         color=alt.Color(field="Tickers", type="nominal"))
    return pie_fig

## func lineplot daily&cumul_daily returns in %
def plot_ret(returns):
    '''
    generate a line plot of calculated returns in %
    '''
    fig = px.line(returns*100).update_layout(
        yaxis_title='%',
        xaxis_title='Date',
        legend_title_text='Returns',
        legend=dict(
        orientation="h", 
        entrywidth=170, 
        yanchor="middle", 
        y=1.3, 
        xanchor="center", 
        x=0.5
    ))
    return fig


# Benchmark Section
# Cumulative Revenues Comparaison to end_date
def h_bar(opt_rslts, ben_rslts):
    '''
     Plot a horizontal bar plot of the investment cumulative gain/loss
     of optimized and benchmark porfolio.
    '''
    bar = pd.DataFrame(zip([opt_rslts, ben_rslts],['Optimized', 'Equally-weighted']), 
                        columns=['Investments Results','Scenarios'])
    return alt.Chart(bar).mark_bar().encode(
        x='Investments Results:Q',
        y='Scenarios:O',
        color=alt.Color(field="Scenarios", type="nominal", legend=None)
    )

def app_exec():
    '''
    main application function
    '''

    #Application Title
    st.set_page_config(page_title = "Crypto Fire!")
    st.title("Crypto-Currency Portfolio Optimizer")

    #Sub_Title_1 # Portfolio Main Parameters
    st.header('Configure your portfolio')
    # Pick The Starting/End Date 
    st.markdown('Define the Optimization time horizon') 
    col1, col2 = st.columns(2)

    with col1:
	    start_date = st.date_input("Start Date",datetime(2020, 1, 1))
	
    with col2:
        end_date = st.date_input("End Date") # with current date as default

    # Selecting Assets
    tickers= tickers_list()
    selected_tickers = st.multiselect("Pick your preferred crypto-currencies ",
                                     tickers,
                                     default=tickers[:2]
                                    )
    
    # Set the Risk Free Rate
    try:
         safe = fed_rate()
    except:
         safe = 0.03
    
    risk_free_rate = st.number_input('Define the Risk Free Rate',
                                     min_value= 0.00,
                                     max_value= None,
                                     value= fed_rate())

    # Initial Investement
    initial_invest = st.number_input('What is your initial investment? (USD)',
                                     min_value= 1.00,
                                     max_value= None,
                                     value= 1000.00)
    
    # Backend variables
    data = get_data(selected_tickers, start_date, end_date)
    returns = data.pct_change()[1:]
    opt_ratio = optimized_ratios(data, start_date, end_date, safe)
    opt_d_ret = pd.Series((opt_ratio * returns).sum(axis= 1), name='Optimized Daily Returns')
    opt_c_ret = pd.Series((1+ opt_d_ret).cumprod()-1, name='Optimized Cumulative Returns')
    ben_ratio = [(1/len(selected_tickers)) for i in selected_tickers]
    ben_d_ret = pd.Series((ben_ratio * returns).sum(axis= 1), name='Equally-weighted Daily Returns')
    ben_c_ret = pd.Series((1+ ben_d_ret).cumprod()-1, name='Equally-weighted Cumulative Return')
    mer_d_ret = pd.concat((opt_d_ret, ben_d_ret), join='inner',axis=1)
    mer_c_ret = pd.concat((opt_c_ret, ben_c_ret), join='inner',axis=1)
    opt_rslts = round(opt_c_ret[-1] * initial_invest, 2)
    ben_rslts = round(ben_c_ret[-1] * initial_invest, 2)

    # Sub_Title_2 # Optimization Results
    if st.button('Optimus Cryptus! :fire:'):
        st.header('Optimization Results')
        col3, col4 = st.columns(2)
        with col3:    
            source = pd.DataFrame(zip(selected_tickers, opt_ratio),
                                columns=['Tickers', 'Ratio'])
            
            st.markdown('Optimized Portfolio')
            st.altair_chart(pie_plot(source),
                            theme = 'streamlit',
                            use_container_width=True)
        with col4:
            source = pd.DataFrame(zip(selected_tickers, ben_ratio),
                                  columns=['Tickers', 'Ratio'])
            
            st.markdown('Equally-weighted Portfolio')
            st.altair_chart(pie_plot(source),
                            theme = 'streamlit',
                            use_container_width=True)
            
        st.markdown('Optimized vs Equally-weighted Portfolio Returns')
        tab1, tab2 = st.tabs(['Daily Returns in %',
                              'Cumulative Daily Returns in %'])
        with tab1:
             st.plotly_chart(plot_ret(mer_d_ret),
                             theme='streamlit',
                             use_container_width=True)

        with tab2:
             st.plotly_chart(plot_ret(mer_c_ret),
                             theme='streamlit',
                             use_container_width=True)

        st.markdown(f'Total Performance of each Portfolio given initial investment of USD {initial_invest}:')
        st.altair_chart(h_bar(opt_rslts, ben_rslts),
                       theme='streamlit',
                       use_container_width=True)

app_exec()
