
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import json

import streamlit as st



# Function to change date to date format
def change_time_format(series):
    return datetime.datetime.strptime(series,"%d/%m/%Y")


from streamlit import caching


st.cache(ttl=60*60*24*2, allow_output_mutation = True)
def get_data():
	# Column name for data in extracted using API
	col_name = ['agebracket', 'contractedfromwhichpatientsuspected', 'currentstatus', 
	            'dateannounced', 'detecteddistrict', 'detectedstate', 
	            'gender', 'nationality', 'notes', 'numcases', 'patientnumber', 
	            'source1', 'source2', 'source3','statecode', 'statepatientnumber', 
	            'statuschangedate', 'typeoftransmission']

	# A temperary data frame for storing data
	temp_data = pd.DataFrame(columns=col_name)

	for i in tqdm(range(1,35)):
	    url = "https://api.covid19india.org/raw_data"+str(i)+".json"
	    response = requests.request("GET", url)

	    class Test(object):
	        def __init__(self, data):
	            self.__dict__ = json.loads(data)

	    data = Test(response.text)

	    data1 = pd.DataFrame(data.raw_data)

	    data1 = data1[data1.statecode=="MP"]
	    
	    data1 = data1[col_name]
	    
	    temp_data = pd.concat([data1, temp_data])
	    
	temp_data.numcases = temp_data.numcases.astype(int)

	temp_data.reset_index(drop=True, inplace=True)

	return temp_data


# Data Cleaning and making it usable
def data_cleaning(temp_data):
	mp_data = pd.pivot_table(temp_data, values='numcases', index=['dateannounced'],
               columns=['currentstatus'], aggfunc=np.sum).reset_index()#["Hospitalized"]


	mp_data.dateannounced = mp_data.dateannounced.apply(change_time_format)
	mp_data.fillna(0, inplace = True)

	mp_data.Hospitalized = mp_data.Hospitalized + mp_data.hospitalized

	mp_data.drop("hospitalized", axis = 1, inplace = True)

	mp_data = mp_data.sort_values(by='dateannounced')#[["dateannounced","Deceased","Hospitalized","Recovered"]]

	mp_data.columns = ['dateannounced', 'Deaths', 'Confirmed', 'Recovered']


	indore_data = temp_data[temp_data.detecteddistrict == 'Indore']
	indore_data = pd.pivot_table(indore_data, values='numcases', index=['dateannounced'],
	               columns=['currentstatus'], aggfunc=np.sum).reset_index()#["Hospitalized"]

	indore_data.dateannounced = indore_data.dateannounced.apply(change_time_format)

	indore_data.fillna(0, inplace = True)

	indore_data = indore_data.sort_values(by='dateannounced')#[["dateannounced","Deceased","Hospitalized","Recovered"]]

	indore_data.columns = ['dateannounced', 'Deaths', 'Confirmed', 'Recovered']

	# Removing from mp_data
	mp_data.set_index("dateannounced", inplace=True)
	mp_data = mp_data.asfreq(freq='d', fill_value=0)
	mp_data.reset_index(inplace = True)


	# Removing incosistency from indore_data
	indore_data.set_index("dateannounced", inplace=True)
	indore_data = indore_data.asfreq(freq='d', fill_value=0)
	indore_data.reset_index(inplace = True)


	mp_data = mp_data[mp_data.dateannounced>='2021-01-01']
	indore_data = indore_data[indore_data.dateannounced>='2021-01-01']

	mp_data['positive'] = mp_data.Confirmed.cumsum()
	indore_data['positive'] = indore_data.Confirmed.cumsum()

	# Combining data for R0
	mp_data['state_city'] = 'MP'
	indore_data['state_city'] = 'INDORE'

	states = pd.concat([mp_data, indore_data])

	states.rename(columns={'dateannounced':'date', 'state_city':'state'}, inplace = True)

	states = states.groupby(['state','date'])['positive'].sum()
	
	return states


import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from scipy import stats as sps
from scipy.interpolate import interp1d

from IPython.display import clear_output

FILTERED_REGION_CODES = ['INDORE', 'MP']


#%config InlineBackend.figure_format = 'retina'






def highest_density_interval(pmf, p=.9, debug=False):
    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)
    
    cumsum = np.cumsum(pmf.values)
    
    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]
    
    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()
    
    # Find the smallest range (highest density)
    try:
        best = (highs - lows).argmin()
    except ValueError:
        best = 0
    
    low = pmf.index[lows[best]]
    high = pmf.index[highs[best]]
    
    return pd.Series([low, high],
                     index=[f'Low_{p*100:.0f}',
                            f'High_{p*100:.0f}'])


def prepare_cases(cases, cutoff=25):
    new_cases = cases.diff()

    smoothed = new_cases.rolling(7,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=2).round()
    
    idx_start = np.searchsorted(smoothed, cutoff)
    
    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]
    
    return original, smoothed


def get_posteriors(sr, sigma=0.15):

    # (1) Calculate Lambda
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

    
    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data = sps.poisson.pmf(sr[1:].values, lam),
        index = r_t_range,
        columns = sr.index[1:])
    
    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range,
                              scale=sigma
                             ).pdf(r_t_range[:, None]) 

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)
    
    # (4) Calculate the initial prior
    #prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 = np.ones_like(r_t_range)/len(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range,
        columns=sr.index,
        data={sr.index[0]: prior0}
    )
    
    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

        #(5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]
        
        #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior
        
        #(5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)
        
        # Execute full Bayes' Rule
        posteriors[current_day] = numerator/denominator
        
        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)
    
    return posteriors, log_likelihood



def plot_rt(result, ax, state_name):
    
    ax.set_title(f"{state_name}")
    
    # Colors
    ABOVE = [1,0,0]
    MIDDLE = [1,1,1]
    BELOW = [0,0,0]
    cmap = ListedColormap(np.r_[
        np.linspace(BELOW,MIDDLE,25),
        np.linspace(MIDDLE,ABOVE,25)
    ])
    color_mapped = lambda y: np.clip(y, .5, 1.5)-.5
    
    index = result['ML'].index.get_level_values('date')
    values = result['ML'].values
    
    # Plot dots and line
    ax.plot(index, values, c='k', zorder=1, alpha=.25)
    ax.scatter(index,
               values,
               s=40,
               lw=.5,
               c=cmap(color_mapped(values)),
               edgecolors='k', zorder=2)
    
    # Aesthetically, extrapolate credible interval by 1 day either side
    lowfn = interp1d(date2num(index),
                     result['Low_90'].values,
                     bounds_error=False,
                     fill_value='extrapolate')
    
    highfn = interp1d(date2num(index),
                      result['High_90'].values,
                      bounds_error=False,
                      fill_value='extrapolate')
    
    extended = pd.date_range(start=pd.Timestamp('2020-12-20'), #this need to change based on data
                             end=index[-1]+pd.Timedelta(days=1))
    
    ax.fill_between(extended,
                    lowfn(date2num(extended)),
                    highfn(date2num(extended)),
                    color='k',
                    alpha=.1,
                    lw=0,
                    zorder=3)

    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25);
    
    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.tick_right()
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0)
    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
    ax.margins(0)
    ax.set_ylim(0.0, 5.0)
    ax.set_xlim(pd.Timestamp('2020-12-20'), result.index.get_level_values('date')[-1]+pd.Timedelta(days=1))
    fig.set_facecolor('w')
    








## ccalling data 

temp_data = get_data()

states = data_cleaning(temp_data)

st.title('Indore & Madhya Pradesh Daily Cases')

line_plot = states.reset_index()

indore_line = line_plot[line_plot.state == 'INDORE']#[['date','positive']]
indore_line['actual_cases'] = indore_line.positive.diff().fillna(indore_line.positive)

mp_line = line_plot[line_plot.state == 'MP']#[['date','positive']]
mp_line['actual_cases'] = mp_line.positive.diff().fillna(mp_line.positive)



updated_data = pd.concat([indore_line, mp_line])
updated_data.rename(columns={'positive':'cumulative_cases'}, inplace = True)
updated_data = updated_data[['state', 'date', 'actual_cases', 'cumulative_cases']]


#Adding data
st.write("Data")
st.write(updated_data) #



st.title("Indore and Madhya Pradesh Daily Cases Plot")

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15,8))

ax[0].plot(indore_line.date, indore_line.actual_cases)
ax[0].set_title('Indore Daily Cases')

ax[1].plot(mp_line.date, mp_line.actual_cases)
ax[1].set_title('Madhya Pradesh Daily Cases')
st.pyplot(fig)



st.markdown('# Real time **R0** Indore and Madhya Pradesh')
# We create an array for every possible value of Rt
R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)

# Gamma is 1/serial interval
# https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article
# https://www.nejm.org/doi/full/10.1056/NEJMoa2001316
GAMMA = 1/7

sigmas = np.linspace(1/20, 1, 20)



targets = states.index.get_level_values('state').isin(FILTERED_REGION_CODES)
states_to_process = states.loc[targets]

results = {}
for state_name, cases in states_to_process.groupby(level='state'):
    
    print(state_name)
    new, smoothed = prepare_cases(cases, cutoff=25)
    
    if len(smoothed) == 0:
        new, smoothed = prepare_cases(cases, cutoff=10)
    
    result = {}
    
    # Holds all posteriors with every given value of sigma
    result['posteriors'] = []
    
    # Holds the log likelihood across all k for each value of sigma
    result['log_likelihoods'] = []
    
    for sigma in sigmas:
        posteriors, log_likelihood = get_posteriors(smoothed, sigma=sigma)
        result['posteriors'].append(posteriors)
        result['log_likelihoods'].append(log_likelihood)
    
    # Store all results keyed off of state name
    results[state_name] = result
    clear_output(wait=True)

print('Done.')



# Each index of this array holds the total of the log likelihoods for
# the corresponding index of the sigmas array.
total_log_likelihoods = np.zeros_like(sigmas)

# Loop through each state's results and add the log likelihoods to the running total.
for state_name, result in results.items():
    total_log_likelihoods += result['log_likelihoods']

# Select the index with the largest log likelihood total
max_likelihood_index = total_log_likelihoods.argmax()

# Select the value that has the highest log likelihood
sigma = sigmas[max_likelihood_index]



final_results = None

for state_name, result in results.items():
    print(state_name)
    posteriors = result['posteriors'][max_likelihood_index]
    #print(posteriors)
    hdis_90 = highest_density_interval(posteriors, p=.9)
    hdis_50 = highest_density_interval(posteriors, p=.5)
    most_likely = posteriors.idxmax().rename('ML')
    result = pd.concat([most_likely, hdis_90, hdis_50], axis=1)
    if final_results is None:
        final_results = result
    else:
        final_results = pd.concat([final_results, result])
    clear_output(wait=True)

print('Done.')








#st.markdown('# Real time **R0** Indore and Madhya Pradesh')
ncols = 1
nrows = 2

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, nrows*5))

for i, (state_name, result) in enumerate(final_results.groupby('state')):
    plot_rt(result.iloc[1:], axes.flat[i], state_name)

fig.tight_layout()
fig.set_facecolor('w')
st.pyplot(fig)

st.write("[Data Source](https://www.covid19india.org/)")
