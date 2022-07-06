# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 11:19:39 2022

@author: sdabadghao

Title: Customer side analysis

Purpose: This script takes the customer side dataset from datagrip export
that combines the Supply chain from Factset and Compustat financials (including averages)

Initial steps: Clean variables, drop NA's, create compund variables, winsorize
Use playground to find best models for explaining RO'X'

"""

import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
# import statsmodels.formula.api as smf

# from linearmodels import PooledOLS
# import statsmodels.api as sm
from linearmodels.panel import PanelOLS
# from linearmodels import RandomEffects

# Read latest file from datagrip 
df = pd.read_csv(r'C:\Users\sdabadghao\Dropbox (Personal)\Research\CCC in supply chain\Data\DataGrip_exports\SC_with_summary_vars_per_customer_and_pct_rev_21092020.csv')

# Dealing with date column
# Take the object date, convert to datetime and then convert to pandas yyyyqq
df['qtr'] = pd.to_datetime(df['quarter']).dt.to_period('Q')
df['year'] = pd.to_datetime(df['quarter']).dt.to_period('Y')
df['date'] = pd.to_datetime(df['quarter']).dt.date
df['yeardummy'] = df['quarter'].str[:4].astype(int)

# Clean duplicates
df.dropna(subset=['quarter'])
df.drop_duplicates(subset=['c_gvkey', 'quarter'])
df.sort_values(by=['c_gvkey', 'quarter'])

# Generate NAICS2 and NAICS3 variables from the full NAICS code
df['c_naics2'] = df['c_naics'].astype(str).str[:2].astype(int)
df['c_naics3'] = df['c_naics'].astype(str).str[:3].astype(int)

# Flip the sign of c_CCD_ccc for consumer and supplier 
df['c_CCD_ccc'] = -1*df['c_CCD_ccc']
df['s_ave_ccd_ccc'] = -1*df['s_ave_CCD_ccc']

# Generating the day variables
df['c_id'] = df['c_invtq']*90/df['c_cogsq']
df['c_recd'] = df['c_rectq']*90/df['c_saleq']
df['c_apd'] = df['c_apq']*90/df['c_cogsq']

# Generate second order variables for regression
for column in df[['c_CCD_inv', 'c_CCD_ap', 'c_CCD_rec', 'c_CCD_ccc', 's_ave_CCD_ccc', 's_ave_CCD_ap', 's_ave_CCD_inv', 's_ave_CCD_rec']]:
    df[column+'_sq'] = df[column]**2

# Sales growth, Gross margin and log_e
df['c_salesgrowth'] = df['c_saleq'].pct_change()
df['c_gm'] = (df['c_saleq']-df['c_cogsq'])/df['c_saleq']
df['ln_c_gm'] = np.log(df['c_gm'])
df['c_atq_ln'] = np.log(df['c_atq'])

# Winsorizing variables at 1% level
for column in df[['c_id', 'c_recd', 'c_apd', 'c_gm', 's_ave_id', 's_ave_recd', 's_ave_apd', 'c_ROS', 'c_ROA', 'c_ROE', 'c_ROI']]:
    df[column+'_w'] = winsorize(df[column],(0.01,0.01))

# Generating the squeeziness metric
df['squeeze_diff'] = df['c_apd_w'] - df['s_ave_recd_w']
df['squeeze_sq'] = df['squeeze_diff']*df['squeeze_diff']
# If squeeze_dif > 0, then the supplier is being good to the customer. 
# If squeeze_dif < 0, then this supplier gives less time for the customer to pay than their other suppliers

# Drop non-manufacturing industry sectors
df = df[df['c_naics2']>25]
df = df[df['c_naics2']<51]


######################################################
# ## Playground
# indvar = ['c_atq_ln', 'c_CCD_inv', 'c_CCD_inv_sq', 'c_level', 'year', 'c_fqtr']
# paneldata = df[indvar]
# # paneldata = paneldata.set_index(['c_fqtr', 'year'])
# mod = PanelOLS.from_formula('c_ROS_w ~ c_atq_ln c_CCD_inv c_CCD_inv_sq c_level year C(c_fqtr)', df)

# results_2 = smf.ols('c_ROS_w ~ c_atq_ln + c_CCD_inv + c_CCD_inv_sq + c_level', data=df).fit()
# print(results_2.summary())
######################################################



######################################################
#  Set up the panel 
######################################################
panel = ['c_gvkey', 'quarter_id', 'date', 'qtr', 
         'c_ROS_w', 'c_ROE_w', 'c_ROA_w',
         'c_atq_ln', 'ln_c_gm', 'c_salesgrowth',
         'c_CCD_inv', 'c_CCD_inv_sq', 'c_level', 'yeardummy', 'c_fqtr', 
         'c_CCD_rec', 'c_CCD_rec_sq', 'c_CCD_ap', 'c_CCD_ap_sq', 
         'squeeze_diff', 'squeeze_sq']
paneldata = df[panel]

# Need to declare panel variables first - c_gvkey and qtr
paneldata = paneldata.set_index(['c_gvkey','quarter_id'])


######################################################
#  Testing Hypothesis 1
#  Inventory has positive relationship with RO'x'
#  inv_sq has negative relationship indicating concave nature
######################################################

# # Fixed Effects regression
FE1 = PanelOLS(paneldata['c_ROS_w'], paneldata[['c_atq_ln', 
                                                'c_CCD_inv', 'c_CCD_inv_sq', 
                                                'c_level', 
                                                'yeardummy', 'c_fqtr']],
              drop_absorbed=True,
              entity_effects = True,
              time_effects=True
              )

# Result
result1 = FE1.fit(cov_type = 'clustered',
              cluster_entity=True,
              cluster_time=True
              )

print(result1)


######################################################
#  Testing Hypothesis 2
#  Include receivable and payable variables 
######################################################
# # Fixed Effects regression
FE2 = PanelOLS(paneldata['c_ROS_w'], paneldata[['c_atq_ln', 'c_CCD_inv', 'c_CCD_inv_sq', 
                                                'yeardummy', 'c_fqtr',
                                                'c_CCD_rec', 'c_CCD_rec_sq', 
                                                'c_CCD_ap', 'c_CCD_ap_sq',
                                                'squeeze_diff', 'squeeze_sq']],
              drop_absorbed=True,
              entity_effects = True,
              time_effects=True
              )

# Result
result2 = FE2.fit(cov_type = 'clustered',
              cluster_entity=True,
              cluster_time=True
              )

print(result2)


######################################################
#  Testing Hypothesis 3
#   
######################################################
FE3 = PanelOLS(paneldata['c_ROS_w'], paneldata[['c_atq_ln', 'c_salesgrowth', 'ln_c_gm',
                                                'c_CCD_inv', 'c_CCD_inv_sq', 
                                                'c_level', 'yeardummy', 'c_fqtr',
                                                'c_CCD_rec', 'c_CCD_ap', 
                                                'squeeze_diff']],
              drop_absorbed=True,
              entity_effects = True,
              time_effects=True
              )

# Result
result3 = FE3.fit(cov_type = 'clustered',
              cluster_entity=True,
              cluster_time=True
              )

print(result3)