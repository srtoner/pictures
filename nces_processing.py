#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Opportunity_Insights


@author: stephentoner
"""

# import altair as alt
import pandas as pd
import numpy as np
import math
import dask.dataframe as dd
import os.path as os

import matplotlib.pyplot as plt
import matplotlib

def process_elsi(df, years):
    '''
    
    Parameters
    ----------
    df : DataFrame
        Structured Data in the form of NCES Report:
        First 3 columns are School ID, Name, State
        Followed by County IDs for each year, then stats
    years : int
        Number of years of data returned, used for offsetting

    Returns
    -------
    output : DataFrame
        Cleaned data Frame with years separated from race in long format

    '''
    # Various NA markers
    dash_cross = '‡'
    dash = '–'
    cross = '†'
    
    # Remove NAs
    
    df = df.replace(cross, 0).replace(dash_cross, 0).replace(dash, 0)
    df = df.rename(columns = {
        'School ID - NCES Assigned [Public School] Latest available year' : 'SID',
        'State Name [Public School] Latest available year' : 'State'}
        )
    
    df.iloc[:,3:] = df.iloc[:,3:].apply(pd.to_numeric)
    df.loc['SID'] = df['SID'].apply(str)
  
    
    cols = df.columns
    
    counties = df[cols[:(3 + years)]]
    
    # Get Non-null county nbr
    
    counties['CID'] = counties.iloc[:,3:].apply(max, axis = 1)

    output = counties.iloc[:, [0, 1, 2, -1]]

    output = pd.concat((output, (df[cols[(3+years+1):]])), axis = 1)
    output = output.replace(dash_cross, 0).replace(dash, 0).replace('†', 0)
    output = output.groupby(['SID', 'School Name', 'CID', 'State']).agg('sum').reset_index()

    output = pd.melt(output, id_vars = output.columns[0:4], value_vars = cols[(3+years + 1):])
    temp = (output["variable"]
            .str.split(pat = "\[Public School\]", expand =True)
            .rename(columns ={0 : "Race", 1 : "Year"}))
    temp.Race = temp.Race.astype('category')
    output = pd.concat((output.iloc[:,0:4],temp, output['value']), axis = 1)
    output.State = output.State.astype('category')
    output.Year = output.Year.apply(lambda x: x[:5])
    output.Year = output.Year.apply(pd.to_numeric)
    
    return output

def calc_hhi(df, years):
    df = process_elsi(df, years)[['SID',
                                  'School Name',
                                  'CID',
                                  'State',
                                  'Race',
                                  'Year',
                                  'value']].rename(
                                      columns= {"value" : 'People'})
                
    df = df[df["Race"] != 'Full-Time Equivalent (FTE) Teachers ']
    piv_df = df.pivot(index = ['SID', 
                               'School Name', 
                               'CID', 
                               'State', 
                               'Year'], 
                      columns = ["Race"])
    
    piv_df["Total"] = piv_df.agg('sum', axis = 1).replace(pd.NA, 0)
    pct_df = piv_df.apply(lambda x: x / piv_df['Total'] )

    hhi_df = pct_df.iloc[:, :-1].apply(np.square).replace(pd.NA, 0)
    hhi_df["HHI"] = hhi_df.apply(sum, axis = 1)

    hhi_df.columns = hhi_df.columns.droplevel()

    hhi_df = hhi_df.rename(columns = {"": "HHI"}).reset_index()
    
    return hhi_df



 
if not ((os.exists("nces_cleaned.csv") and os.exists("hhi_data.csv"))):
        
    sd1 = pd.read_csv("ELSI_V1.CSV")
    
    sd2 = pd.read_csv("ELSI_V2.CSV")


    # Unpivot for the HHI Calc
    t1 = process_elsi(sd1, 10)
    t2 = process_elsi(sd2, 5) 

    new1 = t1[['SID', 'School Name', 'CID', 'State', 'Race', 'Year', 'value']]
    new2 = t2[['SID', 'School Name', 'CID', 'State', 'Race', 'Year', 'value']]

    nces = pd.concat((new1, new2), axis = 0)
    nces = nces.rename(columns= {"value" : 'People'})

    pivot_cols = ['SID', 'School Name', 'CID', 'State', 'Year']
    temp = nces[nces["Race"] != 'Full-Time Equivalent (FTE) Teachers ']

    races = temp["Race"].unique()
    piv_df = temp.pivot(index = pivot_cols, columns = ["Race"])
    piv_df["Total"] = piv_df.agg('sum', axis = 1).replace(pd.NA, 0)
    pct_df = piv_df.apply(lambda x: x / piv_df['Total'] )

    hhi_df = pct_df.iloc[:, :-1].apply(np.square).replace(pd.NA, 0)
    hhi_df["HHI"] = hhi_df.apply(sum, axis = 1)

    hhi_df.columns = hhi_df.columns.droplevel()

    hhi_df = hhi_df.rename(columns = {"": "HHI"}).reset_index()
    temp = hhi_df.loc[:, ["SID", "Year", "HHI"]]
    nces.to_csv("nces_cleaned.csv", index = False)
    temp.to_csv("hhi_data.csv", index = False)

nces_dd = pd.read_csv("nces_cleaned.csv")
hhi_info = pd.read_csv("hhi_data.csv")
hhi_info = hhi_info.set_index(keys = ["SID", "Year"])

nces_dd = nces_dd.set_index(keys = ["SID", "Year"])

c1 = nces_dd.shape[0] // 4
c2 = c1 * 2
c3 = c1 * 3

nces_1 = nces_dd.iloc[:c1, :]
nces_2 = nces_dd.iloc[c1:c2, :]
nces_3 = nces_dd.iloc[c2:c3, :]
nces_4 = nces_dd.iloc[c3:, :]

hhi_mean = hhi_info.groupby("SID").agg('mean')

p1 = pd.merge(nces_1, hhi_mean, how = 'left', left_index = True, right_index = True)
p2 = pd.merge(nces_2, hhi_mean, how = 'left', left_index = True, right_index = True)
p3 = pd.merge(nces_3, hhi_mean, how = 'left', left_index = True, right_index = True)
p4 = pd.merge(nces_4, hhi_mean, how = 'left', left_index = True, right_index = True)

total = pd.concat((p1,p2,p3,p4), axis = 0)

total.CID = total.CID.apply(int)
total.CID = total.CID.apply(str)


def leading_zero(string):
    if len(string) < 5:
        return '0' + string
    return string

total.CID = total.CID.apply(leading_zero)

total = total.reset_index()

agg = total.groupby(['CID', 'Race', 'Year', 'State']).agg({'People' : 'sum',
                                                           'HHI' : 'mean'}).reset_index()

total.to_pickle('nces_complete.pkl')



agg.to_pickle('agg_stats_by_cty.pkl')
agg.to_csv('agg_stats_by_cty.csv')
# p1.to_pickle("temp1.pkl")
# p2.to_pickle("temp2.pkl")
# p3.to_pickle("temp3.pkl")
# p4.to_pickle("temp4.pkl")

reduced = agg
reduced = reduced.pivot(index = ['State', 'CID', 'Year', 'HHI'], columns = ['Race'])
# d1 = p1.reset_index()

reduced.columns = reduced.columns.droplevel()
reduced = reduced.reset_index()
# reduced
reduced.to_csv('NCES_Data_Final.csv', index = False)


reduced.to_json('NCES_Data_Final.json')
# d2 = p2.reset_index()
# d3 = p3.reset_index()
# d4 = p4.reset_index()

# agg1 = d1.groupby("CID").agg('mean')

# # p1.to_csv("nces_p1.csv")
# # p2.to_csv("nces_p2.csv")

# p1.to_pickle("nces_p1.pkl")
# p2.to_pickle("nces_p2.pkl")

# p3.to_pickle("nces_p3.pkl")
# p4.to_pickle("nces_p4.pkl")



