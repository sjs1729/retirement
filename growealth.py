import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from mftool import Mftool
import matplotlib.pyplot as plt
import my_functions as myf
from scipy.optimize import minimize
from scipy import optimize
import random
import math
from fpdf import FPDF
import os
from yahooquery import Ticker
import quantstats as qs
import requests
from bs4 import BeautifulSoup




WIDTH = 297
HEIGHT = 210
np.set_printoptions(precision=2)

st.set_page_config(layout="wide")
risk_score = 0
tday = dt.datetime.today()

col1, col2 = st.sidebar.columns(2)
col1.image('gw_logo.png', width=125)
col2.title("GroWealth Dashboard")

df = pd.read_csv('MF_Perf.csv', index_col=['Scheme_Code'], parse_dates=['Inception Date'])
df_model=df[df['Age']> 3]
#df = df[df['Age']> 0]

df.drop(columns=['Inception Date'], axis=1, inplace=True)

fund_master = pd.read_csv('./Fund_Master.csv')
fund_master = fund_master[fund_master['MoneyControl_Link'] > '0']
g_dashboard_rpt_view = 0



mf = Mftool()

global_fig1 = plt.figure()



option = st.sidebar.selectbox("Which Dashboard?", ('Manage Customer', 'Retirement Planning','Goal Planning',
                                                   'MF Screener', 'MF Ranking','MF Selection','Model Portfolio','Performance Dashboard','Fund Details'), 3)
st.title(option)

@st.cache(suppress_st_warning=True)
def get_all_data():
    mfunds_all = pd.read_csv('mfunds_all.csv', index_col=['Date'])
    return mfunds_all

@st.cache(suppress_st_warning=True)
def get_all_navs_merged():
    df_all_navs = pd.read_csv('all_equity_mf_navs.csv')
    df_all_navs['Date']=pd.to_datetime(df_all_navs['Date'])
    df_all_navs.set_index('Date', inplace=True)
    return df_all_navs

def get_stock_data(ticker, start, end):
    df = Ticker(ticker).history(start=start,end=end)
    df.reset_index(inplace=True)
    df=df[['date','high','low','open','close','volume']]
    df['date']=pd.to_datetime(df['date'])

    df.rename(columns={'date':'Date','high':'High','low':'Low','open':'Open','close':'Close','volume':'Volume'},inplace=True)
    #df['Date']=df['Date'].apply(lambda x: x.date())
    df.set_index('Date',inplace=True)
    
    return df

@st.cache(suppress_st_warning=True)
def get_url_content(mf_url):
    r = requests.get(mf_url)
    return r

@st.cache(suppress_st_warning=True)
def display_table(df):
    #st.write(df)

    headercolor = '#FD8E72'
    odd_rowcolor = '#E5ECF6'
    even_rowcolor = '#D5DCE6'

    fig = go.Figure(data=go.Table(
        columnwidth=[3,3,3,2,2,2,2,2,2,2,2],
        header=dict(values=[j for j in df.columns],
        fill_color='#FD8E72',
        align=['center','center'],
        font_size=14
        ),
        cells=dict(values=[df[k] for k in df.columns],
        fill_color= [[odd_rowcolor,even_rowcolor,odd_rowcolor,even_rowcolor,odd_rowcolor,even_rowcolor,odd_rowcolor,even_rowcolor]*25],
        align='left',
        font_size = 12
        )))
    fig.update_layout(margin=dict(l=1,r=1,b=1,t=1))
    fig.update_layout(height=500)
    fig.update_layout(width=900)


    return fig

@st.cache(suppress_st_warning=True)
def get_fund_data(fund_list, start, end):
    counter=0
    mfunds=pd.DataFrame()
    for funds in fund_list:
        ticker = funds.split("-")[0]
        df_fund = myf.get_scheme_nav(ticker, start, end)
        if counter == 0:
            mfunds = df_fund
        else:
            mfunds = mfunds.merge(df_fund, on='Date')

        counter += 1
    mfunds = mfunds.astype(float)

    return mfunds

@st.cache(suppress_st_warning=True)
def monte_carlo_simulation(port_returns):
    np.random.seed(101)
    num_ports = 20000
    num_funds = len(port_returns.columns)

    all_weights = np.zeros((num_ports, num_funds))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)
    s_weights = []
    my_mcs_progress_bar= st.progress(0)

    for ind in range(num_ports):
        # Create Random Weights
        weights = np.array(np.random.random(num_funds))

        # Rebalance Weights
        weights = weights / np.sum(weights)

        #st.write(weights)
        # Save Weights
        all_weights[ind, :] = weights

        # Expected Return
        ret_arr[ind] = np.sum((port_returns.mean() * weights) * 252)

        # Expected Variance
        vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(port_returns.cov() * 252, weights)))

        # Sharpe Ratio
        sharpe_arr[ind] = ret_arr[ind] / vol_arr[ind]

        txt = 'Weights'
        for j in weights:
            txt = '{}-{}'.format(txt,str(round(j,2)))
        s_weights.append(txt)
        my_mcs_progress_bar.progress((ind+1) / num_ports)

        #st.write(txt)

    #simulation_df = pd.DataFrame([ret_arr,vol_arr,sharpe_arr,s_weights]).transpose()
    #simulation_df.columns = ['Returns','Volatility','Sharpe_Ratio','Weights']

    #st.write(simulation_df.head(50))
    #return simulation_df
        

    return ret_arr, vol_arr, sharpe_arr, s_weights

def get_ret_vol_sr(weights):
    """
    Takes in weights, returns array or return,volatility, sharpe ratio
    """
    weights = np.array(weights)
    ret = np.sum(mfunds_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(mfunds_ret.cov() * 252, weights)))
    sr = ret/vol
    return np.array([ret,vol,sr])

def neg_sharpe(weights):
    return  get_ret_vol_sr(weights)[2] * -1

def check_sum(weights):
    '''
    Returns 0 if sum of weights is 1.0
    '''
    return np.sum(weights) - 1

def limit_volatility(weights):
    return max(get_ret_vol_sr(weights)[1],pct_volatility) - pct_volatility

def minimize_volatility(weights):
    return  get_ret_vol_sr(weights)[1] 

def maximize_return(weights):
    return  get_ret_vol_sr(weights)[0] * -1

def get_bounds(nfunds):

    b=[]
    f_b = []
    def_bounds = (0,1)
    if nfunds == 2:
        c=(0.3,0.7)
    elif nfunds > 2 and nfunds < 7:
        c=(0.1,1/nfunds + 0.3)
    elif nfunds > 6:
        c=(0.05,1/nfunds + 0.25)
        
    for j in range(nfunds):
        b.append(c)
        f_b.append(def_bounds)

    return tuple(b), tuple(f_b)

def get_goals(age,desc,amt,freq):
    goals = []

    if amt > 0 and age <= plan_till:
        n_years_to_goal = age - curr_age
        goal_fut_value = np.power((1 + inflation/100),n_years_to_goal) * amt

        values = age, desc, round(goal_fut_value,0)
        goals.append(values)
        
        if freq > 0:
            for m in range(age + freq, plan_till, freq):
                n_years_to_goal = m - curr_age
                goal_fut_value = np.power((1 + inflation/100),n_years_to_goal) * amt
                values = m, desc, round(goal_fut_value,0)
                goals.append(values)

    return goals

def get_corpus(rate, curr_age, ann_income, retirement_age, corpus, expenses):
    rec = []
    income = 0
    yr_corpus = corpus
    for j in expenses.index:
        if j > curr_age:
            if j < retirement_age:
                income = ann_income
            else:
                income = 0

            yr_corpus = yr_corpus * (1 + rate/100) + income - expenses.loc[j]['Expenses']
        values = j, round(yr_corpus,0)
        rec.append(values)

    df = pd.DataFrame(rec,columns=['Years',f"Corpus-{rate}%"])
    df.set_index('Years', inplace=True)
    return df
        
            
        
def get_optimised_rate(rate, curr_age, ann_income, retirement_age, corpus, expenses, terminal_corpus):
    rec = []
    income = 0
    yr_corpus = corpus
    for j in expenses.index:
        if j > curr_age:
            if j < retirement_age:
                income = ann_income
            else:
                income = 0

            yr_corpus = yr_corpus * (1 + rate/100) + income - expenses.loc[j]['Expenses']

    yr_corpus = yr_corpus - terminal_corpus

    return yr_corpus


def xirr(rate,cash_flow,terminal_value=0):
    
    npv = 0
    for i in cash_flow.index:
        nYears = cash_flow.loc[i,'Num_Days']/365
        pv = cash_flow.loc[i,'Tran_Value']*(pow((1 + rate / 100), nYears)) 
        npv = npv + pv
    
    return  npv+terminal_value

def get_emi(emi, rate, nperiods,target_amt,present_corpus):
    tot_val = present_corpus * pow((1+ rate/100),nperiods/12)
    for per in range(nperiods):
        tot_val = tot_val + emi * pow((1+ rate/1200),nperiods - per)

    return tot_val - target_amt
        


def get_sip_freq(sip_freq):
    n_freq = 21
    if sip_freq == 'Weekly':
        n_freq=5
    elif sip_freq == 'Daily':
        n_freq=1
    elif sip_freq == 'One Time':
        n_freq=100000
    elif sip_freq == 'Fortnightly':
        n_freq=10

    return n_freq

def get_allocation_details(sip_list):

    fund_df = fund_master.set_index('Scheme_Code')
    stock_list = []
    mcap_list  = []
    tot_amount = 0
    for i in range(len(sip_list)):
        
        sc_code= int(sip_list[i][0])
        sc_name= sip_list[i][1]
                
        sc_amt=sip_list[i][2]
        mf_url = fund_df.loc[sc_code]['MoneyControl_Link']
        r = get_url_content(mf_url)
        df_list = pd.read_html(r.text)

        tot_amount = tot_amount + sc_amt

        for i in df_list[3].index:
            s_pct = df_list[3].loc[i]['% of Total Holdings']
            s_pct = float(s_pct.split("%")[0])
            stk_amt  = round(s_pct * sc_amt/100,2)
            stk_name = df_list[3].loc[i]['Stock Invested in']
            stk_sector = df_list[3].loc[i]['Sector']
            m_cap = df_list[3].loc[i]['M-Cap']
            if len(stk_name.split('#  ')) > 1:
                stk_name = stk_name.split('#  ')[1]
            if len(stk_name.split('-  ')) > 1:
                stk_name = stk_name.split('-  ')[1]
        
            values = stk_name,stk_sector,stk_amt,m_cap
            stock_list.append(values)

        x=df_list[3].groupby(['M-Cap'])[['GroupName']].max()
    
        for k in x.index:
    
            if k == 'Large Cap':
                l_cap=x.loc[k]['GroupName'] * sc_amt/100
                values = 'Large Cap', l_cap
                mcap_list.append(values)
            elif k == 'Mid Cap':
                m_cap=x.loc[k]['GroupName'] * sc_amt/100
                values = 'Mid Cap', m_cap
                mcap_list.append(values)    
            elif k == 'Small Cap':
                s_cap=x.loc[k]['GroupName'] * sc_amt/100
                values = 'Small Cap', s_cap
                mcap_list.append(values)
            elif k == 'Other':        
                o_cap=x.loc[k]['GroupName'] * sc_amt/100
                values = 'Uncategorized', o_cap
                mcap_list.append(values)
                
    p_alloc = pd.DataFrame(stock_list, columns=['Stock','Sector','Allocation','M-Cap'])
    p_mcap  = pd.DataFrame(mcap_list, columns=['M-Cap','Amount'])
    p_mcap['Amount'] = round(100 * p_mcap['Amount']/tot_amount,2)


    p_stk=p_alloc.groupby(['Stock']).sum()
    p_stk['Pct %']=round(100*p_stk['Allocation']/p_stk['Allocation'].sum(),2)

    p_sector=p_alloc.groupby(['Sector']).sum()
    p_sector['Pct %']=round(100*p_sector['Allocation']/p_sector['Allocation'].sum(),2)


    return p_stk.sort_values(by=['Pct %'],ascending=False), p_sector.sort_values(by=['Pct %'],ascending=False), p_mcap.groupby(['M-Cap'])[['Amount']].sum()


@st.cache(suppress_st_warning=True)          
def get_sip_transactions(sip_list):

    trans = []
    tday=dt.datetime.today().date()
    for i in range(len(sip_list)):
        start_date=sip_list[i][4]
        end_date = sip_list[i][5]

        #st.write(start_date.month, end_date.day)
        
        sc_code= sip_list[i][0]
        sc_name= sip_list[i][1]
        sc_freq=sip_list[i][3]
        
        sc_amt=sip_list[i][2]

        sc_details = mf.get_scheme_details(sc_code)
        sc_fund_house=sc_details['fund_house']
        sc_category=sc_details['scheme_category']
        counter = 0
        df= myf.get_scheme_nav_between_dates(sc_code, start_date, tday)
        qty = 0
        tran_value = 0.0
        nqty = 0
        for j in df.index:
            unit_price= round(float(df.loc[j,'Nav']),4)
            if counter % sc_freq == 0 and j <= end_date:
                tran_dt = j
                tran_type = 'Buy'

                

                qty=round(sc_amt/unit_price,4)
                tran_value=-1.0 * sc_amt
                #st.write(tran_value)
                nqty = nqty + qty
                #curr_val=float(mf.calculate_balance_units_value(sc_code,qty)['balance_units_value'])
                #print(curr_val)
                values = tran_dt,sc_code,sc_name,sc_fund_house,sc_category,tran_type, tran_value, qty,nqty, unit_price,round((nqty*unit_price),2),(tday - j.date()).days
                trans.append(values)
            else:
                values = j,sc_code,sc_name,sc_fund_house,sc_category,'Hold', 0,0, nqty,unit_price, round((nqty*unit_price),2),0
                trans.append(values)

            counter +=1

    transactions = pd.DataFrame(trans,columns=['Date','Scheme_Code','Scheme_Name','Fund_House','Fund_Category','Tran_Type','Tran_Value','Qty','Net_Qty','Unit_Price', 'Cum_Amount','Num_Days'])
    transactions['Date']=pd.to_datetime(transactions['Date'])
    transactions['Year']=transactions['Date'].apply(lambda x: x.year)
    transactions['Scheme_Type']=transactions['Fund_Category'].apply(lambda x: x.split(" - ")[0])
    transactions['Fund_Category']=transactions['Fund_Category'].apply(lambda x: x.split(" - ")[1])

    return transactions

@st.cache(suppress_st_warning=True)          
def get_portfolio_returns(df_trans):

    transactions = df_trans[df_trans['Tran_Type'] != 'Hold']
    gtot_inv = -1* round(transactions[transactions['Tran_Type']=='Buy']['Tran_Value'].sum(),2)
    gtot_redeem = round(transactions[transactions['Tran_Type']=='Sell']['Tran_Value'].sum(),2)
    #bal_qty = round(df.iloc[-1]['Net Qty'],4)
    #val = mf.calculate_balance_units_value(i,bal_qty)
    cash_flow = transactions[['Tran_Value','Num_Days']]
    gcurr_val = 0.0
    for j in transactions['Scheme_Code'].unique():
        df_1 = transactions[transactions['Scheme_Code']== j]
        gcurr_val = gcurr_val + df_1.iloc[-1]['Cum_Amount']

    tot_xirr=round(optimize.newton(xirr, 10, args=(cash_flow,gcurr_val,)),2)
    


    rec = []
    for i in transactions['Scheme_Code'].unique():
        df= transactions[transactions['Scheme_Code']==i]
        tot_inv = -1* round(df[df['Tran_Type']=='Buy']['Tran_Value'].sum(),2)
        tot_redeem = round(df[df['Tran_Type']=='Sell']['Tran_Value'].sum(),2)
        sc_name = df.iloc[0]['Scheme_Name']
        bal_qty = round(df.iloc[-1]['Net_Qty'],4)
        #val = mf.calculate_balance_units_value(i,bal_qty)
        curr_val = df.iloc[-1]['Cum_Amount']
        cash_flow = df[['Tran_Value','Num_Days']]

        root=round(optimize.newton(xirr, 0, args=(cash_flow,curr_val,)),2)
        #values = sc_name, bal_qty,tot_inv,tot_redeem,round(curr_val,2),root
        values = sc_name, bal_qty,tot_inv,round(curr_val,2),root

        rec.append(values)


    #summary='Total',"",gtot_inv,gtot_redeem,gcurr_val,round(tot_xirr,2)
    summary='Total',"",gtot_inv,gcurr_val,round(tot_xirr,2)

    rec.append(summary)
    df_schemewise = pd.DataFrame(rec, columns=['Scheme','Qty','Purchase','Mkt. Value','XIRR'])

    df_schemewise.set_index('Scheme',inplace=True)
    

    rec=[]
    for i in transactions['Fund_House'].unique():
        df= transactions[transactions['Fund_House']==i]
        tot_inv = -1* round(df[df['Tran_Type']=='Buy']['Tran_Value'].sum(),2)
        tot_redeem = round(df[df['Tran_Type']=='Sell']['Tran_Value'].sum(),2)
        #bal_qty = round(df.iloc[-1]['Net Qty'],4)
        #val = mf.calculate_balance_units_value(i,bal_qty)
        cash_flow = df[['Tran_Value','Num_Days']]

        curr_val = 0.0
        for j in df['Scheme_Code'].unique():
            df_1 = df[df['Scheme_Code']== j]
            curr_val = curr_val + df_1.iloc[-1]['Cum_Amount']

        root=round(optimize.newton(xirr, 0, args=(cash_flow,curr_val,)),2)
        values = i,tot_inv,tot_redeem,round(curr_val,2),root
        rec.append(values)

    summary='Total',gtot_inv,gtot_redeem,gcurr_val,round(tot_xirr,2)
    rec.append(summary)
    df_fundhouse = pd.DataFrame(rec, columns=['Fund House','Purchase','Sell','Mkt. Value','XIRR'])
    df_fundhouse.set_index('Fund House',inplace=True)


    rec=[]
    for i in transactions['Scheme_Type'].unique():
        df= transactions[transactions['Scheme_Type']==i]
        tot_inv = -1* round(df[df['Tran_Type']=='Buy']['Tran_Value'].sum(),2)
        tot_redeem = round(df[df['Tran_Type']=='Sell']['Tran_Value'].sum(),2)
        #bal_qty = round(df.iloc[-1]['Net Qty'],4)
        #val = mf.calculate_balance_units_value(i,bal_qty)
        cash_flow = df[['Tran_Value','Num_Days']]

        curr_val = 0.0
        for j in df['Scheme_Code'].unique():
            df_1 = df[df['Scheme_Code']== j]
            curr_val = curr_val + df_1.iloc[-1]['Cum_Amount']

        root=round(optimize.newton(xirr, 0, args=(cash_flow,curr_val,)),2)
        
        values = i,tot_inv,tot_redeem,round(curr_val,2),root
        rec.append(values)

    summary='Total',gtot_inv,gtot_redeem,gcurr_val,round(tot_xirr,2)
    rec.append(summary)

    df_fundcat = pd.DataFrame(rec, columns=['Scheme_Type','Purchase','Sell','Mkt. Value','XIRR'])
    #df_fundcat.set_index('Scheme_Type',inplace=True)


    rec=[]

    for i in transactions['Fund_Category'].unique():
        df= transactions[transactions['Fund_Category']==i]
        tot_inv = -1* round(df[df['Tran_Type']=='Buy']['Tran_Value'].sum(),2)
        tot_redeem = round(df[df['Tran_Type']=='Sell']['Tran_Value'].sum(),2)
        #bal_qty = round(df.iloc[-1]['Net Qty'],4)
        #val = mf.calculate_balance_units_value(i,bal_qty)
        cash_flow = df[['Tran_Value','Num_Days']]

        curr_val = 0.0
        for j in df['Scheme_Code'].unique():
            df_1 = df[df['Scheme_Code']== j]
            curr_val = curr_val + df_1.iloc[-1]['Cum_Amount']

        root=round(optimize.newton(xirr, 0, args=(cash_flow,curr_val,)),2)
        
        values = i,tot_inv,tot_redeem,round(curr_val,2),root
        rec.append(values)

    summary='Total',gtot_inv,gtot_redeem,gcurr_val,round(tot_xirr,2)
    rec.append(summary)

    df_schemetype = pd.DataFrame(rec, columns=['Fund_Category','Purchase','Sell','Mkt. Value','XIRR'])
    #df_schemetype.set_index('Fund_Category',inplace=True)


    

    rec=[]
    opening_balance = 0.0
    for i in transactions['Year'].unique():
        df= transactions[transactions['Year']==i]
        tot_inv = -1* round(df[df['Tran_Type']=='Buy']['Tran_Value'].sum(),2)
        tot_redeem = round(df[df['Tran_Type']=='Sell']['Tran_Value'].sum(),2)
        #bal_qty = round(df.iloc[-1]['Net Qty'],4)
        #val = mf.calculate_balance_units_value(i,bal_qty)
        cash_flow = df[['Tran_Value','Num_Days']]
        cash_flow.loc[len(cash_flow.index)] = [-1*opening_balance, 365]
        curr_val = 0.0
        for j in df['Scheme_Code'].unique():
            df_1 = df[df['Scheme_Code']== j]
            curr_val = curr_val + df_1.iloc[-1]['Cum_Amount']

        root=round(optimize.newton(xirr, 0, args=(cash_flow,curr_val,)),2)
        
        values = i,opening_balance,tot_inv,round(curr_val,2),root
        rec.append(values)
        opening_balance =  curr_val

    summary='Total',opening_balance,gtot_inv,gcurr_val,round(tot_xirr,2)
    rec.append(summary)

    df_yearwise = pd.DataFrame(rec, columns=['Year','Opening Balance','Purchase','Mkt. Value','XIRR'])
    df_yearwise.set_index('Year')


    return df_schemewise, df_fundhouse, df_fundcat, df_schemetype, df_yearwise


def get_mkt_value_at_xirr(sip_df, rate):
    mkt_price = -1* sip_df['Tran_Value']*pow( 1 + rate/100,   sip_df['Num_Days']/365.0)
    market_value_rate = mkt_price.sum()
    return market_value_rate


df_all = get_all_data()
df_all_navs = get_all_navs_merged()

if option == 'Fund Details':


    mf_list = [f"{fund_master.loc[j]['Scheme_Code']}-{fund_master.loc[j]['Scheme_Name']}" for j in fund_master.index]
    mf_name = st.selectbox("Select Fund",mf_list,0)

    mf_cd = int(mf_name.split("-")[0])
    mf_nm = mf_name.split("-")[1]
    mf_url = fund_master[fund_master['Scheme_Code'] == mf_cd].iloc[0]['MoneyControl_Link']
    r = get_url_content(mf_url)
    df_list = pd.read_html(r.text)
    stk_1=float(df_list[3].iloc[0]['% of Total Holdings'].split('%')[0])
    AUM=df_list[3].iloc[0]['Value(Mn)']*10/stk_1
    AUM=str(AUM)
    st.markdown(f"Fund AUM: Rs. {AUM.split('.')[0]} Cr")
    soup = BeautifulSoup(r.content, 'html.parser')
    inv_blk=soup.find_all(class_="investment_block")
    for i in range(len(inv_blk)):
        st.markdown(inv_blk[i].get_text().strip())
    st.write(df_list[0])
    st.write(df_list[1])
    st.write(df_list[3])

    

if option == 'Performance Dashboard':

    sip_list = []
    report_done = 0

    for sc_code in df.index:
        sip_list_values = '{}-{}'.format(sc_code,df.loc[sc_code]['Scheme Name'])
        sip_list.append(sip_list_values)

    df_cust = pd.read_csv("customer.csv")

    cust_list = [df_cust.loc[j]['Customer_Name'] for j in df_cust.index]



        #st.write(sip_list)
    labl = st.columns((2,4,2,4))

    labl[0].subheader("")
    labl[0].subheader("Customer name")
    cust_name = labl[1].selectbox("",cust_list,0)
    labl[2].subheader("")
    labl[2].subheader("No of SIPs")
    nsips = labl[3].slider("", min_value=1, max_value=10, step=1, value=1)

    df_cust = df_cust[df_cust['Customer_Name']==cust_name]
    #st.write(df_cust)

   
    with st.form(key="sip_form"):
        
        sip_cols = st.columns((10, 6,4,5,5))

        min_start = dt.datetime(2010, 1, 1)
        start = dt.datetime(2016, 1, 1)
        end = dt.datetime.today()

        sip_cols[0].subheader("Scheme")
        sip_cols[1].subheader("Amount")
        sip_cols[2].subheader("Frequency")
        sip_cols[3].subheader("Start Data")
        sip_cols[4].subheader("End Date")


        sip_selection = [sip_cols[0].selectbox("",sip_list, key=f"fund_sel_{col}") for col in range(nsips)]
        sip_amt = [sip_cols[1].number_input("",key=f"sip_amt_{col}",value=0,step=100) for col in range(nsips)]
        sip_freq = [sip_cols[2].selectbox("", ('Monthly','Fortnightly','Weekly','Daily','One Time'), 0,key=f"sip_freq_{col}") for col in range(nsips)]
        sip_start_date = [sip_cols[3].date_input("",key=f"sip_st_date_{col}",value=start, min_value=min_start) for col in range(nsips)]
        sip_end_date = [sip_cols[4].date_input("",key=f"sip_end_date_{col}",value=end, min_value=min_start) for col in range(nsips)]

        



        p_clicked = sip_cols[0].form_submit_button('Portfolio Report')

    if p_clicked:
        sip = []

        for i in range(nsips):

            if sip_amt[i] > 0:
                sip_values = sip_selection[i].split("-")[0],sip_selection[i].split("-")[1],sip_amt[i], get_sip_freq(sip_freq[i]),sip_start_date[i], sip_end_date[i]
                sip.append(sip_values)

        p_stk, p_sector, p_mcap = get_allocation_details(sip)
        st.write(p_stk)
        st.write(p_sector.head(13))
        st.write(p_mcap)

        if len(sip) > 0:
            sip_trans = get_sip_transactions(sip)
           # st.write(sip_trans)

            mkt_price = get_mkt_value_at_xirr(sip_trans[sip_trans['Num_Days'] != 0],6)
            st.write(f"At 6% rate {mkt_price}")
            mkt_price = get_mkt_value_at_xirr(sip_trans[sip_trans['Num_Days'] != 0],8)    
            st.write(f"At 8% rate {mkt_price}")
            mkt_price = get_mkt_value_at_xirr(sip_trans[sip_trans['Num_Days'] != 0],10)
            st.write(f"At 10% rate {mkt_price}")
            mkt_price = get_mkt_value_at_xirr(sip_trans[sip_trans['Num_Days'] != 0],12)    
            st.write(f"At 12% rate {mkt_price}")
            
            min_dt = sip_trans['Date'].min()
            max_dt = sip_trans['Date'].max()

            #st.write(min_dt.strftime("%d-%m-%Y"))

            df_nifty=get_stock_data('^NSEI',min_dt,max_dt)
            #st.write(df_nifty)
            df_nifty['Nifty']=100*df_nifty['Close']/df_nifty.iloc[0]['Close']



            df_scheme, df_fundhouse, df_fund_cat, df_schemetype, df_yearwise = get_portfolio_returns(sip_trans)

            

            rpt_view = st.columns(2)
            #nselect = rpt_view[0].selectbox("Report Type",('Scheme Wise','Fund House Wise','Fund Category Wise','Scheme Type Wise','Year Wise'), g_dashboard_rpt_view)
            #g_dashboard_rpt_view = nselect

            #fig_sch = display_table(df_scheme.reset_index())
            #rpt_view[0].plotly_chart(fig_sch, use_container_width=True)

            
            #fig_sch = display_table(df_yearwise)
            #rpt_view[1].plotly_chart(fig_sch, use_container_width=True)
            
            df_scheme_disp = df_scheme.reset_index()
            df_scheme_disp['Qty']= df_scheme_disp['Qty'].apply(lambda x: str(x))
            rpt_view[0].dataframe(df_scheme_disp)

            df_yearwise['Year']= df_yearwise['Year'].apply(lambda x: str(x))
        
            
            rpt_view[1].write(df_yearwise)


            df_trans_grp =sip_trans.groupby(['Date'])[['Tran_Value','Cum_Amount']].sum()
            #st.write(df_trans_grp)
            
            df_trans_grp['Cum_Amt_pct']=df_trans_grp['Cum_Amount'].pct_change()

            df_trans_grp['Net_Amount']=df_trans_grp['Tran_Value'] + df_trans_grp['Cum_Amount']
            df_trans_grp['RelGrowth']=100*df_trans_grp['Cum_Amount']/df_trans_grp.iloc[0]['Cum_Amount']
            df_trans_grp['Base_Amount']=df_trans_grp['Cum_Amount'].shift(1)
            df_trans_grp['Dly_Returns']=(df_trans_grp['Net_Amount']-df_trans_grp['Base_Amount'])/df_trans_grp['Base_Amount']

            df_trans_grp['Portfolio']=1
            initial=100.0
            for i in df_trans_grp.index:
                x=df_trans_grp.loc[i]['Dly_Returns']
                if math.isnan(x):
                    initial=100.0
                else:
                    initial = initial * (x+1.0)
        
                df_trans_grp.at[i,'Portfolio']=initial

            df_trans_grp['Roll_Max3']= df_trans_grp['Portfolio'].rolling(window=3).max().rolling(window=3).mean()
            df_trans_grp['ChkSum']=(df_trans_grp['Roll_Max3']-df_trans_grp['Portfolio'])/df_trans_grp['Portfolio']

            #df_trans_grp = df_trans_grp.join(df_nifty['NiftyRet'], on='Date', how='inner')
            df_nifty = df_nifty.merge(df_trans_grp, on='Date')
            #st.write(df_nifty)

            df_nifty = df_nifty[(df_nifty['ChkSum'] < 0.25) & (df_nifty['ChkSum'] > -0.25)]
            df_port = df_nifty[['Portfolio']]
            #df_nifty = df_nifty[['Nifty','Portfolio']]

            volatility = round(qs.stats.volatility(df_port['Portfolio'])*100,2)
            sharpe = round(qs.stats.sharpe(df_port['Portfolio'],rf=0.04, periods=252),2)

            #st.write(volatility)
            #st.write(volatality,sharpe)
            #df_nifty = df_nifty.merge(df_port, on='Date')
            

            beta, alpha = qs.stats.greeks(df_nifty['Portfolio'],df_nifty['Nifty'], periods=252)

            maxdd_port  = qs.stats.to_drawdown_series(df_nifty['Portfolio'])
            maxdd_nifty = qs.stats.to_drawdown_series(df_nifty['Nifty'])

            #st.write(maxdd_port)
            #st.write(maxdd_nifty)


            

            #st.write(df_nifty)
            fig = px.line(df_nifty[['Nifty','Portfolio']])
            
            #fig = px.line(df_trans_grp[df_trans_grp['ChkSum'] < 0.4]['100Ret','NiftyRet'])
            #fig = px.line(df_trans_grp[['NiftyRet','100Ret']])

            rpt_view[0].header("")
            
            rpt_view[0].header("")

            fig.update_layout(title_text="Investment Journey",
                          title_x=0.5,
                          title_font_size=20,
                          xaxis_title="",
                          yaxis_title="Value of Rs.100")
            fig.update_layout(showlegend=True)
            fig.update_layout(height=350)
            fig.update_layout(width=550)
            fig.write_image("./images/inv_journey.png")
            rpt_view[0].subheader("")
            #rpt_view[0].header("")
            rpt_view[0].write(fig)


            #fig1 = px.bar(df_yearwise,x=df_yearwise.Year,y=[df_yearwise['Opening Balance'],df_yearwise['Mkt. Value'],df_yearwise.XIRR], title="Year Wise Performance")
            #fig1 = px.bar(df_yearwise,x=df_yearwise.Year,y=[df_yearwise.XIRR], title="Year Wise Performance")

            fig1 = make_subplots(specs=[[{"secondary_y": True}]])

            # Add traces
            fig1.add_trace(
                go.Bar(x=df_yearwise.Year, y=df_yearwise['Opening Balance'], name="Open Bal"),
                secondary_y=False,
            )
                        
            fig1.add_trace(
                go.Bar(x=df_yearwise.Year, y=df_yearwise['Purchase'], name="New Investment"),
                secondary_y=False,
            )
            
            fig1.add_trace(
                go.Bar(x=df_yearwise.Year, y=df_yearwise['Mkt. Value'], name="Mkt Value"),
                secondary_y=False,
            )

            
            fig1.add_trace(
                go.Scatter(x=df_yearwise.Year, y=df_yearwise['XIRR'],name="XIRR %", mode='markers+lines', marker=
                                                                                                       dict(color='LightSkyBlue',
                                                                                                            size=15,
                                                                                                            line=dict(color='MediumPurple',width=1))
                ),
                secondary_y=True,
            )


            # Add figure title
            fig1.update_layout(
                title_text="Year Wise Investments - Returns",
                title_x=0.4,
                title_font_size=25
            )

            # Set x-axis title
            fig1.update_xaxes(title_text="Year",title_font_size=20, showgrid=False)

            # Set y-axes titles
            fig1.update_yaxes(title_text="Amount",title_font_size=20,showgrid=False, secondary_y=False)
            fig1.update_yaxes(title_text="XIRR %",title_font_size=20,showgrid=False, secondary_y=True)

            #fig.show()
            fig1.write_image("./images/ret_yearwise.png")

            rpt_view[1].write(fig1)

            df_schemewise = df_schemetype[df_schemetype['Fund_Category'] != 'Total']
            fig2 = px.pie(df_schemewise,values=df_schemewise['Mkt. Value'], names=df_schemewise.Fund_Category)
            fig2.update_traces(textposition='outside', textinfo='percent+label')
            fig2.update_layout(showlegend=False)
            fig2.update_layout(
                title_text="Actual Fund Allocation by Category",
                title_x=0.5,
                title_font_size=25)

            fig2.write_image("./images/allocation_by_category.png")

            rpt_view[1].write(fig2)
            pdf = FPDF('L')

            pdf.add_page()
            pdf.image("gw_header.png",0,0,WIDTH)
            pdf.ln(20)

            
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(35, 10, "Customer Name:") 
            pdf.set_font('Arial', '', 10)
            pdf.cell(10, 10, df_cust.iloc[0]['Customer_Name'], align='L')

            pdf.ln(5)
            
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(35, 10, "Age:") 
            pdf.set_font('Arial', '', 10)
            pdf.cell(10, 10,str(df_cust.iloc[0]['Age']) , align='L')
            pdf.ln(5)

            pdf.set_font('Arial', 'B', 12)
            pdf.cell(35, 10, "Risk Profile:") 
            pdf.set_font('Arial', '', 10)
            pdf.cell(10, 10, df_cust.iloc[0]['Risk_Profile'], align='L')
            pdf.ln(5)

            pdf.set_font('Arial', 'B', 12)
            pdf.cell(35, 10, "Time Horizon:") 
            pdf.set_font('Arial', '', 10)
            pdf.cell(10, 10, df_cust.iloc[0]['Time_Horizon'], align='L')
            pdf.ln(10)

            pdf.set_font('Arial', 'B', 12)
            pdf.cell(35, 10, "Objective:") 
            pdf.set_font('Arial', '', 8)
            pdf.multi_cell(100, 5, df_cust.iloc[0]['Objective'], align='L')
            pdf.ln(3)
            pdf.cell(35, 10, "")
            pdf.multi_cell(100, 5, df_cust.iloc[0]['Disclaimer'], align='L')

            pdf.image("./images/inv_journey.png",210, 43, WIDTH/3.5)
            
            line_gap = 5
            start_pos_x = 155
            start_pos_y = 30
            pdf.set_xy(start_pos_x,start_pos_y)
            pdf.set_font('Arial', 'BU', 10)
            pdf.cell(40, 10, "Back Testing Results")
            
            start_pos_y = start_pos_y + line_gap
            pdf.set_xy(start_pos_x,start_pos_y)
            pdf.set_font('Arial', 'B', 8)
            pdf.cell(25, 10, f"Test Execution Date:", align='L')
            pdf.set_font('Arial', '', 8)
            pdf.cell(30, 10, f"{dt.datetime.today().strftime('%d-%m-%Y')}", align='C')

            pdf.set_font('Arial', 'B', 8)
            pdf.cell(20, 10, f"Back Test Period:", align='L')
            pdf.set_font('Arial', '', 8)            
            pdf.cell(40, 10, f"{min_dt.strftime('%d-%m-%Y')} - {max_dt.strftime('%d-%m-%Y')}", align='R')
            
            start_pos_y = start_pos_y + 3*line_gap -3
            pdf.set_xy(start_pos_x,start_pos_y-2)
            pdf.set_font('Arial', 'B', 8)
            pdf.cell(30, 10, f"Total Investment :")
            pdf.set_font('Arial', '', 8)
            pdf.cell(10, 10, "Rs. {:,.0f}".format(df_scheme.loc['Total']['Purchase']), align='L')
            start_pos_y = start_pos_y + line_gap
            pdf.set_xy(start_pos_x,start_pos_y-2)
            
            pdf.set_font('Arial', 'B', 8)
            pdf.cell(30, 10, f"Market Value       :")
            pdf.set_font('Arial', '', 8)
            pdf.cell(10, 10, "Rs. {:,.0f}".format(df_scheme.loc['Total']['Mkt. Value']), align='L')
            start_pos_y = start_pos_y + line_gap
            pdf.set_xy(start_pos_x,start_pos_y-2)
            pdf.set_font('Arial', 'B', 8)
            pdf.cell(30, 10, f"XIRR                     :")
            pdf.set_font('Arial', '', 8)
            pdf.cell(10, 10, f"{df_scheme.loc['Total']['XIRR']}%", align='L')
            
            start_pos_y = start_pos_y + line_gap
            pdf.set_xy(start_pos_x,start_pos_y-2)
            pdf.set_font('Arial', 'B', 8)
            pdf.cell(30, 10, "Volatility              :")
            pdf.set_font('Arial', '', 8)
            pdf.cell(10, 10, f"{volatility}%", align='L')
            
            start_pos_y = start_pos_y + line_gap
            pdf.set_xy(start_pos_x,start_pos_y-2)
            pdf.set_font('Arial', 'B', 8)
            pdf.cell(30, 10, f"Sharpe Ratio     :")
            pdf.set_font('Arial', '', 8)
            pdf.cell(10, 10, f"{sharpe}", align='L')

            start_pos_y = start_pos_y + line_gap
            pdf.set_xy(start_pos_x,start_pos_y-2)
            pdf.set_font('Arial', 'B', 8)
            pdf.cell(30, 10, f"Alpha                  :")
            pdf.set_font('Arial', '', 8)
            pdf.cell(10, 10, f"{str(round(alpha,2))}", align='L')

            start_pos_y = start_pos_y + line_gap
            pdf.set_xy(start_pos_x,start_pos_y-2)
            pdf.set_font('Arial', 'B', 8)
            pdf.cell(30, 10, f"Beta                    :")
            pdf.set_font('Arial', '', 8)
            pdf.cell(10, 10, f"{str(round(beta,2))}", align='L')
            
            pdf.ln(53)
            pdf.set_font('Arial', 'B', 10)
            pdf.set_xy(start_pos_x,start_pos_y+2*line_gap-4)
            pdf.cell(60,10,"Fundwise SIP Performance")
            pdf.set_font('Arial', 'B', 7)
            th=pdf.font_size
            
            pdf.set_fill_color(170,190,230)
            pdf.set_xy(start_pos_x,start_pos_y+4*line_gap-6)
            pdf.cell(45,2*th,"Fund Name",border=1,fill=True)
            pdf.cell(12,2*th,"Qty",border=1, fill=True, align='C')
            pdf.cell(15,2*th,"Purchase",border=1, fill=True, align='C')
            pdf.cell(15,2*th,"Mkt Value",border=1, fill=True, align='C')
            pdf.cell(12,2*th,"XIRR %",border=1, fill=True, align='C')
            pdf.cell(25,2*th,"XIRR Contribution",border=1, fill=True, align='C')

  
            pdf.line(WIDTH/2 + 2,33,WIDTH/2+2,200)

            pdf.ln(th)
            cell_spacing = 2*th
            j=1
            pdf.set_font('Arial', 'B', 5)
            for data in df_scheme.index:
                pdf.set_xy(start_pos_x,start_pos_y+(4+j)*line_gap-6)
                pdf.cell(45,cell_spacing,data,border=1)
                pdf.cell(12,cell_spacing,str(df_scheme.loc[data]['Qty']),border=1, align='C')
                pdf.cell(15,cell_spacing,"Rs. {:,.0f}".format(int(df_scheme.loc[data]['Purchase'])),border=1,  align='C')
                pdf.cell(15,cell_spacing,"Rs. {:,.0f}".format(int(df_scheme.loc[data]['Mkt. Value'])),border=1, align='C')
                pdf.cell(12,cell_spacing,str(round(df_scheme.loc[data]['XIRR'],2)),border=1, align='C')
                if data != 'Total':
                    pdf.cell(25,cell_spacing,str(round(df_scheme.loc[data]['XIRR'] * (int(df_scheme.loc[data]['Mkt. Value']) /int(df_scheme.loc['Total']['Mkt. Value'])),2)),border=1, align='C')
                else:
                    pdf.cell(25,cell_spacing,"",border=1, align='C')
                pdf.ln(th)
                j=j+1

            pdf.set_xy(10,85)
            pdf.set_font('Arial', 'B', 11)

            pdf.cell(60,10,"SIP Summary")
            #pdf.cell(60,10,"Portfolio Review")

            pdf.set_font('Arial', 'B', 5)
            pdf.ln(8)
            pdf.cell(50,2*th,"Fund Name",border=1,fill=True)
            pdf.cell(18,2*th,"Fund House",border=1, fill=True)
            pdf.cell(18,2*th,"Fund Type",border=1, fill=True)
            pdf.cell(14,2*th,"Mthly SIP Amt",border=1, fill=True, align='C')
            pdf.cell(35,2*th,"Rational(Risk)",border=1, fill=True)
            
            pdf.set_font('Arial', 'B', 5)
            f_obj = pd.read_csv("./gw_recommended_funds.csv")
            #f_obj['Fund_Code']=f_obj['Fund_Code'].apply(lambda x: int(x))
            fcat_obj = pd.read_csv("./fund_type.csv")
            #st.write(f_obj)

            pie_chart_rec = []
            for sip_rec in sip:
                n_frequency = sip_rec[3]
                if n_frequency == 1:
                    mode = "Daily SIP"
                    mthly_amt = 21 * sip_rec[2]
                elif n_frequency == 5:
                    mode = "Weekly SIP"
                    mthly_amt = 4 * sip_rec[2]
                elif n_frequency == 10:
                    mode = "Fortnightly SIP"
                    mthly_amt = 2 * sip_rec[2]
                elif n_frequency == 21:
                    mode = "Monthly SIP"
                    mthly_amt = 1 * sip_rec[2]
                else:
                    mode = "Lump Sum"
                    mthly_amt = sip_rec[2]
                
                fh_name = sip_trans[sip_trans['Scheme_Name']==sip_rec[1]]['Fund_House'].iloc[0]
                fh_cat = sip_trans[sip_trans['Scheme_Name']==sip_rec[1]]['Fund_Category'].iloc[0]
                #st.write(fh_cat)

                objective_rec = f_obj[f_obj['Fund_Code']==int(sip_rec[0])]
                #st.write(sip_rec[0],len(objective_rec))
                if len(objective_rec) > 0:
                    objective = objective_rec.iloc[0]['Rational_Risk']
                    
                else:
                    objective = fcat_obj[fcat_obj['fund_category']==fh_cat]['fund_objective'].iloc[0]

                if fh_cat == 'Dynamic Asset Allocation or Balanced Advantage':
                    fh_cat = 'Balance Advantage Fund'
                pdf.ln(5)
                pdf.cell(50,2*th,sip_rec[1].replace("Regular Growth",""),border=1)
                pdf.cell(18,2*th,fh_name.replace("Mutual Fund",""),border=1)
                pdf.cell(18,2*th,fh_cat.replace("Fund",""),border=1)
                pdf.cell(14,2*th,"Rs. {:,.0f}".format(mthly_amt),border=1, align='C')
                pdf.cell(35,2*th,objective,border=1)
                values = sip_rec[1],fh_name.replace("Mutual Fund",""),fh_cat.replace("Fund",""),mthly_amt
                pie_chart_rec.append(values)


            pie_chart_df = pd.DataFrame(pie_chart_rec, columns=['scheme_name','fund_house','fund_cat','mthly_amt'])

            fig3 = px.pie(pie_chart_df,values=pie_chart_df['mthly_amt'], names=pie_chart_df.fund_cat)
            fig3.update_traces(textposition='outside', textinfo='percent+label')
            fig3.update_layout(showlegend=False)
            fig3.update_layout(
                title_text="Allocation by Fund Category",
                title_x=0.5,
                title_font_size=25)

            fig3.write_image("./images/planned_allocation_by_category.png")

            rpt_view[0].write(fig3)

            fig4 = px.pie(pie_chart_df,values=pie_chart_df['mthly_amt'], names=pie_chart_df.fund_house)
            fig4.update_traces(textposition='outside', textinfo='percent+label')
            fig4.update_layout(showlegend=False)
            fig4.update_layout(
                title_text="Allocation by Fund House",
                title_x=0.5,
                title_font_size=25)

            fig4.write_image("./images/planned_allocation_by_fundhouse.png")

            rpt_view[0].write(fig4)

           

            pdf.ln(7)
            pdf.set_xy(10,190)
            pdf.set_font('Arial','', 5)
            pdf.set_text_color(0,0,255)
            #pdf.multi_cell(150, 2, f"Disclaimer: The above recommendation is only for the purpose of illustration. \
#The actual investment plan will be based on the customer's risk profile and their existing MF portfolio holdings. \
#The returns is based on historical data and are only representative. Please note, past performance does not guarantee future returns.", align='L')

            #pdf.multi_cell(150, 2, f"Disclaimer: The above recommendation is only for the purpose of illustration.", align='L')

            
            #if len(df_schemewise) > 1:
            #    pdf.image("./images/allocation_by_category.png",155, 155, WIDTH/4.5)
            #    pdf.image("./images/ret_yearwise.png",223, 153, WIDTH/4.5)
            #else:
            #    pdf.image("./images/ret_yearwise.png",175, 145, WIDTH/3.5)

            if len(sip) > 7:

                pdf.image("./images/planned_allocation_by_fundhouse.png",5, 155, 60,45)
                pdf.image("./images/allocation_by_category.png",80, 155, 60,45)
                pdf.rect(10, 155, 135, 45, style = '')


                pdf.image("./images/ret_yearwise.png",175, 155,100,45)
                pdf.rect(155, 155, 124, 45, style = '')
            else:
                
                pdf.image("./images/planned_allocation_by_fundhouse.png",5, 150, 60,45)
                pdf.image("./images/allocation_by_category.png",80, 150, 60,45)
                pdf.rect(10, 145, 135, 55, style = '')


                pdf.image("./images/ret_yearwise.png",175, 145,100,55)
                pdf.rect(155, 145, 124, 55, style = '')

            #pdf.set_xy(10,180)
            #pdf.set_font('Arial','', 5)
            #pdf.set_text_color(255,0,0)
            #pdf.multi_cell(140, 2, f"Disclaimer: {df_cust.iloc[0]['Objective']}", align='L')
            
            pdf.image("gw_footer.png",3,HEIGHT-10,WIDTH-10,8)

            pdf.add_page()
            pdf.image("gw_page2.png",0,0,WIDTH)
            pdf.image("gw_footer.png",3,HEIGHT-10,WIDTH-10,8)
            pdf.ln(10)

            start_pos_x = 85
            start_pos_y = 35
            
            pdf.set_xy(start_pos_x, start_pos_y)
            pdf.set_font('Arial', 'BU', 15)
            pdf.cell(40, 10, f"Top 20 Stocks",align='L')
            pdf.set_xy(195,35)
            pdf.cell(40, 10, f"Top 20 Sectors",align='L')


            pdf.set_text_color(0,0,0)

            pdf.set_fill_color(170,190,230)
            pdf.set_font('Arial', 'B', 9)
            th=pdf.font_size
            start_pos_y = start_pos_y + 12
            pdf.set_xy(start_pos_x,start_pos_y)
            pdf.cell(75,2*th,"Stock Name",border=1,fill=True)
            pdf.cell(25,2*th,"% Allocation",border=1, fill=True, align='C')
            

  
            pdf.ln(th)
            cell_spacing = 2*th
            pdf.set_font('Arial', 'B', 9)
            for data in p_stk.head(20).index:
                start_pos_y= start_pos_y + cell_spacing
                pdf.set_xy(start_pos_x,start_pos_y)
                pdf.cell(75,cell_spacing,data,border=1)
                pdf.cell(25,cell_spacing,str(p_stk.loc[data]['Pct %']),border=1, align='C')
                pdf.ln(2*th)

            start_pos_y= start_pos_y + cell_spacing
            pdf.set_xy(start_pos_x,start_pos_y)
            pdf.cell(75,cell_spacing,"Total",border=1,fill=True)
            pdf.cell(25,cell_spacing,str(round(p_stk['Pct %'].head(20).sum(),2)),border=1, align='C',fill=True)

            start_pos_x = 195
            start_pos_y = 35
            pdf.set_fill_color(170,190,230)
            pdf.set_font('Arial', 'B', 9)
            th=pdf.font_size
            start_pos_y = start_pos_y + 12
            pdf.set_xy(start_pos_x,start_pos_y)
            pdf.cell(70,2*th,"Sector Name",border=1,fill=True)
            pdf.cell(25,2*th,"% Allocation",border=1, fill=True, align='C')
            

  
            pdf.ln(th)
            cell_spacing = 2*th
            pdf.set_font('Arial', 'B', 9)
            for data in p_sector.head(20).index:
                start_pos_y= start_pos_y + cell_spacing
                pdf.set_xy(start_pos_x,start_pos_y)
                pdf.cell(70,cell_spacing,data.replace('\u2013',''),border=1)
                pdf.cell(25,cell_spacing,str(p_sector.loc[data]['Pct %']),border=1, align='C')
                pdf.ln(2*th)

            start_pos_y= start_pos_y + cell_spacing
            pdf.set_xy(start_pos_x,start_pos_y)
            pdf.cell(70,cell_spacing,"Total",border=1,fill=True)
            pdf.cell(25,cell_spacing,str(round(p_sector['Pct %'].head(20).sum(),2)),border=1, align='C',fill=True)

            start_pos_x = 10
            start_pos_y = 75
            line_space = 10
            pdf.set_font('Arial', 'BU', 15)
            pdf.set_text_color(0,0,255)

            pdf.set_xy(start_pos_x,start_pos_y)
            pdf.cell(40, 10, "Asset Allocation",align='L')
            pdf.set_font('Arial', 'BU', 8)
            pdf.set_text_color(0,0,255)
            pdf.cell(2, 10, "",align='L')
            pdf.cell(40, 10, "(Present Snapshot)",align='L')

            start_pos_y = start_pos_y + line_space
            pdf.set_xy(start_pos_x,start_pos_y)
            tot_eqty_pct = round(p_mcap['Amount'].sum(),2)
            pdf.set_text_color(0,0,0)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(43, 12, "Equity - Allocation:")
            pdf.set_font('Arial', '', 12)
            pdf.cell(10, 12, f"{str(tot_eqty_pct)}%", align='L')


            start_pos_y = start_pos_y + line_space
            pdf.set_xy(start_pos_x +5,start_pos_y)
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(30, 12, "No of Stocks :")
            pdf.set_font('Arial', '', 10)
            pdf.cell(10, 12, f"{str(round(p_stk['Pct %'].count()))}", align='L')

            for k in p_mcap.index:
                start_pos_y = start_pos_y + line_space
                pdf.set_xy(start_pos_x +5,start_pos_y)

                pdf.set_text_color(0,0,0)
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(30, 12, f"{k}   :")
                pdf.set_font('Arial', '', 10)
                pdf.cell(10, 12, f"{str(round(p_mcap.loc[k]['Amount']))}%", align='L')



            start_pos_y = start_pos_y + line_space
            pdf.set_xy(start_pos_x,start_pos_y)

            pdf.set_text_color(0,0,0)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(43, 12, "Debt+ - Allocation:")
            pdf.set_font('Arial', '', 12)
            pdf.cell(10, 12, f"{str(round(100-tot_eqty_pct,2))}%", align='L')

            pdf.output(f"./rpt/Report-{df_cust.iloc[0]['Customer_Name']}.pdf",'F')




if option == 'Goal Planning':


    left,centre,right = st.columns((8,1,6))
    goal_type = left.selectbox("Select Goal", ('Marriage', 'Higher Education','Vacation','Buying a Dream Car','Buying Dream Home','Miscellaneous'),1)
    if goal_type == 'Marriage':
        image_path = 'marriage.jpeg'
    elif goal_type == 'Higher Education':
        image_path = 'highereducation.jpeg'
    elif goal_type == 'Vacation':
        image_path = 'vacation.jpeg'
    elif goal_type == 'Buying a Dream Car':
        image_path = 'dreamcar.jpeg'
    elif goal_type == 'Buying Dream Home':
        image_path = 'dreamhome.jpeg'
    else:
        image_path = 'goal.jpeg'

    left.image(image_path, width=500,use_column_width=True   )


    gp_flds = st.columns((4,1,4))

    goal_amount = right.number_input("Amount (Today's Price)", value=0,step=10000)
    years_to_goal = right.slider("Years to Goal?", min_value=1, max_value=30, step=1, value=3)    

    present_corpus = right.number_input("Corpus I Already Have", value=0,step=10000)

    rate = round(right.number_input("Return on Assets", step=0.10),2)
    infl = right.number_input("Inflation", step =0.1)

    adj_amount = goal_amount*pow((1+infl/100),years_to_goal)

    present_value = adj_amount/pow((1+rate/100),years_to_goal)

    tot_mths = 12*years_to_goal        

    mthly_amt=round(optimize.newton(get_emi, 0, tol=0.0000001, args=(rate, tot_mths,adj_amount,present_corpus)),2)

    
    html_text = '<p style="font-family:Courier; color:Blue; font-size: 18px;">Lumpsum Investment Required: ' + "Rs. {:,.2f}</p>".format(present_value - present_corpus)

    left.markdown(html_text, unsafe_allow_html=True)
    html_text = '<p style="font-family:Courier; color:Blue; font-size: 18px;">Monthly SIP Required : ' + "Rs. {:,.2f}</p>".format(mthly_amt)
    left.markdown(html_text, unsafe_allow_html=True)

   


if option == 'Retirement Planning':

    left_1, right_1 = st.columns((1,1))
    curr_age = left_1.slider("Your Current Age?", min_value=30, max_value=100, step=1, value=0)
    yrs_to_retire = right_1.slider("Years to Retire", min_value=0, max_value=30, step=1, value=0)
    plan_till = left_1.slider("Plan Till", min_value=curr_age + yrs_to_retire, max_value=100, step=1, value=90)
    n_goals = right_1.slider("Number of Goals", min_value=0, max_value=10, step=1, value=5)

    with st.form(key="Retirement"):


        
        align = st.columns(3)

        c_annual_income = align[0].number_input("Annual Income", value=0,step=100000)
        c_annual_expense = align[1].number_input("Annual Expense", value=0,step=100000)
        c_corpus = align[2].number_input("Current Corpus", value=0,step=100000)

        cagr = round(align[0].number_input("Return on Assets", step=0.10),2)
        inflation = align[1].number_input("Inflation", step =0.1)
        terminal_corpus = align[2].number_input("Terminal Corpus", value=0,step=100000)
        align[0].subheader("Add Goals")
            
        goal_expander = st.expander("")


        with goal_expander:
            goal_cols = st.columns((2,4,3,2))


            g_year = [goal_cols[0].slider("Age", key=f"g_yr_{col}",min_value=curr_age, max_value=plan_till, step=1, value=curr_age) for col in range(n_goals)]
            g_desc = [goal_cols[1].text_input("Goal Description",key=f"g_desc_{col}") for col in range(n_goals)]
            g_amt  = [goal_cols[2].number_input("Goal Amount (in Today's Price)",key=f"g_amt_{col}", value=0, format="%d", step=50000) for col in range(n_goals)]
            g_freq = [goal_cols[3].number_input("Goal Frequency (once in N years)",key=f"g_freq_{col}", value=0, format="%d", step=1) for col in range(n_goals)]
                
        n_Retire = st.form_submit_button("Retirement Goal")

    if n_Retire:

        exp_data = []
        tot_assets = c_corpus
        start_year = curr_age 
        age_at_retirement = curr_age + yrs_to_retire
        end_year = plan_till + 1
        expense = c_annual_expense
        nyear = 0

        goals = []
        
        for i in range(n_goals):
            if g_amt[i] !=0:
                goals = goals + get_goals(g_year[i], g_desc[i], g_amt[i], g_freq[i])

        goals_df=pd.DataFrame(goals,columns=['Years','Desc','Amount'])
        #st.write(goals_df.groupby(['Years']).sum())
        
        for n in range(start_year, end_year):

            if n==start_year:
                expense_rec = n, 0
            elif n == (start_year + 1):
                expense_rec = n, expense
            else:
                expense = expense * (1 + inflation/100)
                expense_rec = n, round(expense,0)
                
            exp_data.append(expense_rec)

        #st.write(pd.DataFrame(expense_rec))

        df_expense = pd.DataFrame(exp_data,columns=['Years','Expenses'])
        df_expense = df_expense.set_index('Years')

        #st.write(df_expense)

        if len(goals) > 0:
            for key in range(len(goals)):
                #st.write(key,goals[key])
                df_expense.loc[goals[key][0]]['Expenses'] = df_expense.loc[goals[key][0]]['Expenses'] + goals[key][2]

        #st.write(cagr,curr_age, c_annual_income, age_at_retirement, c_corpus)

        df_corpus = get_corpus(cagr,curr_age, c_annual_income, age_at_retirement, c_corpus, df_expense)

        retirement_assets = df_expense.merge(df_corpus, on='Years')
        #st.write(retirement_assets)

        root=round(optimize.newton(get_optimised_rate, 25, tol=0.0000001, args=(curr_age, c_annual_income, age_at_retirement, c_corpus, df_expense,terminal_corpus)),2)

        

        #st.write(root)
        if 0 < root < 25:
            optimised_rate = get_corpus(root,curr_age, c_annual_income, age_at_retirement, c_corpus, df_expense)
            retirement_assets = retirement_assets.merge(optimised_rate, on='Years')

        #st.write(retirement_assets)
        fig = px.line(retirement_assets)
        fig.update_layout(title_text="",
                          title_x=0.5,
                          title_font_size=10,
                          xaxis_title="Age (in Years) ",
                          yaxis_title="Retirement Fund (Rs.)")

        fig.update_layout(margin=dict(l=1,r=1,b=1,t=3))
        yrange = [0 - c_corpus, 4*c_corpus]
        fig.update_yaxes(range=yrange, dtick=5000000,showgrid=False)
        fig.update_xaxes(showgrid=False)

        fig.update_layout(height=800)
        fig.update_layout(width=1200)
        fig.update_layout(legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01
        ))

        st.subheader("Retirement Fund Viability Chart")
        st.write(fig)
        
        #st.write(retirement_assets)
        #st.write(goals)
            
            


if option == 'Model Portfolio':


    with st.form(key="Optimise"):
        c1, gap_1,c2,gap_2, c3, gap_3, c4,gap_4, c5,gap_5, c6,gap_6, c7,gap_7, c8 = st.columns((4,1,4,1,4,1,4,1,4,1,4,1,4,1,4))
    
        c1.text('Large Cap')
        c2.text('Large/Mid Cap')
        c3.text('Mid Cap')
        c4.text('Small Cap')
        c5.text('ELSS')
        c6.text('Flexi Cap')
        c7.text('Sector Focus')
        c8.text('Value/Contra Cap')
    
        num_large_cap = c1.slider("", min_value=0, max_value=6, step=1, value=0)
        num_largemid_cap = c2.slider(" ", min_value=0, max_value=6, step=1, value=0)
        num_mid_cap = c3.slider("  ", min_value=0, max_value=6, step=1, value=0)
        num_small_cap = c4.slider("   ", min_value=0, max_value=6, step=1, value=0)
        num_elss = c5.slider("    ", min_value=0, max_value=6, step=1, value=0)
        num_multi_flexi = c6.slider("     ", min_value=0, max_value=6, step=1, value=0)
        num_sector_focussed = c7.slider("      ", min_value=0, max_value=6, step=1, value=0)
        num_value_contra = c8.slider("       ", min_value=0, max_value=6, step=1, value=0)
    
        n_butn = c1.form_submit_button("Optimised Portfolio")
    
    if n_butn:
        nfunds = num_large_cap + num_largemid_cap + num_mid_cap + num_small_cap + num_elss + num_multi_flexi + num_sector_focussed + num_value_contra
    
        st.write('{} funds selected'.format(nfunds))
    
        large_cap_fund_list = []
        lm_cap_fund_list = []
        mid_cap_fund_list = []
        small_cap_fund_list = []
        elss_fund_list = []
        multi_cap_fund_list = []
        sector_fund_list = []
        value_fund_list = []
    
        if num_large_cap > 0:
            large_cap_fund_list = ['sc_{}'.format(str(col)) for col in df_model[df_model['Scheme_Cat'] == 'Large Cap Fund'].index]
    
        if num_largemid_cap > 0:
            lm_cap_fund_list = ['sc_{}'.format(str(col)) for col in df_model[df_model['Scheme_Cat'] == 'Large & Mid Cap Fund'].index]
    
        if num_mid_cap > 0:
            mid_cap_fund_list = ['sc_{}'.format(str(col)) for col in df_model[df_model['Scheme_Cat'] == 'Mid Cap Fund'].index]
    
        if num_small_cap > 0:
            small_cap_fund_list = ['sc_{}'.format(str(col)) for col in df_model[df_model['Scheme_Cat'] == 'Small Cap Fund'].index]
    
        if num_elss > 0:
            elss_fund_list = ['sc_{}'.format(str(col)) for col in df_model[df_model['Scheme_Cat'] == 'ELSS'].index]
    
        if num_multi_flexi > 0:
            multi_cap_fund_list = ['sc_{}'.format(str(col)) for col in df_model[df_model['Scheme_Cat'] == 'Flexi Cap Fund'].index]
    
        if num_sector_focussed > 0:
            sector_fund_list = ['sc_{}'.format(str(col)) for col in
                                df_model[(df_model['Scheme_Cat'] == 'Focused Fund') | (df_model['Scheme_Cat'] == 'Sectoral/ Thematic')].index]
    
        if num_value_contra > 0:
            value_fund_list = ['sc_{}'.format(str(col)) for col in
                               df_model[(df_model['Scheme_Cat'] == 'Value Fund') | (df_model['Scheme_Cat'] == 'Contra Fund')].index]
    
        all_fund_list = large_cap_fund_list + lm_cap_fund_list + mid_cap_fund_list + small_cap_fund_list + elss_fund_list + multi_cap_fund_list + sector_fund_list + value_fund_list
    
        random_results = []
        my_bar = st.progress(0)
        #st.write(df_all.columns)
        random_size = 2000
        for k in range(random_size):
            rand_cols = random.sample(large_cap_fund_list, num_large_cap)
            rand_cols = rand_cols + random.sample(lm_cap_fund_list, num_largemid_cap)
            rand_cols = rand_cols + random.sample(mid_cap_fund_list, num_mid_cap)
            rand_cols = rand_cols + random.sample(small_cap_fund_list, num_small_cap)
            rand_cols = rand_cols + random.sample(elss_fund_list, num_elss)
            rand_cols = rand_cols + random.sample(multi_cap_fund_list, num_multi_flexi)
            rand_cols = rand_cols + random.sample(sector_fund_list, num_sector_focussed)
            rand_cols = rand_cols + random.sample(value_fund_list, num_value_contra)
    
            mfunds_ret = df_all[rand_cols].pct_change().fillna(0)
    
            rand_cols = [cols.split("_")[1] for cols in rand_cols]
    
            bounds, def_bounds = get_bounds(nfunds)
            init_guess = [round(1 / nfunds, 4) for j in range(nfunds)]
    
            cons_0 = ({'type': 'eq', 'fun': check_sum})
            opt_results = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=def_bounds, constraints=cons_0)
            opt_weights = [round(j, 4) for j in opt_results.x]
            opt_vol_ret_sr = get_ret_vol_sr(opt_weights)
    
            values = rand_cols, opt_weights, nfunds, opt_vol_ret_sr[0], opt_vol_ret_sr[1], opt_vol_ret_sr[2]
    
            random_results.append(values)
            if k > 0:
                my_bar.progress(k / random_size)
    
        random_df = pd.DataFrame(random_results,
                                 columns=['Scheme_Codes', 'Weights', 'Num_Funds', 'Returns', 'Volatility', 'Sharpe Ratio'])
        # random_df.to_csv("random_results.csv")
    
        st.write(random_df)
    
        l_cap = []
        for l_sc in all_fund_list:
            sch_code = l_sc.split("_")[1]
    
            count = 0
            tot_wt = 0.0
            avg_wt = 0.0
            zero_count = 0
            sc_name = df.loc[int(sch_code)]['Scheme Name']
            fund_type = df.loc[int(sch_code)]['Scheme_Cat']
            for r in random_df.index:
                weights = random_df.loc[r]['Weights']
                rand_alloc = random_df.loc[r]['Scheme_Codes']
                if sch_code in rand_alloc:
                    w = weights[rand_alloc.index(sch_code)]
                    if w > 0.1:
                        tot_wt = tot_wt + w
                        count = count + 1
                    else:
                        zero_count += 1
    
            if count > 0:
                avg_wt = tot_wt / (count + zero_count)
            values = sch_code, sc_name, fund_type, zero_count, count, zero_count + count, tot_wt, avg_wt
            l_cap.append(values)
    
        ranked_schemes = pd.DataFrame(l_cap, columns=['Scheme_Code', 'Scheme Name', 'Fund_Type', 'Zero_Count', 'Counts',
                                                      'Total_Count', 'Total_Weights', 'Avg_Weights'])
        ranked_schemes = ranked_schemes[ranked_schemes['Counts'] > 0]
        st.write(ranked_schemes.sort_values(by=['Fund_Type','Avg_Weights'],ascending=False))
                
            
    

            


            
        
    
if option == 'MF Ranking':

    df_eqty_mf = pd.read_csv('mf_equity_category.csv', index_col=['Scheme_Code'])
    
    mf_category_list = [col for col in df_eqty_mf['Scheme_Dtl_Category'].unique()]

    with st.form(key="MF_Ranking"):


        dtl_cat_option = st.selectbox("Select Fund House", mf_category_list)
        min_start = dt.datetime(2010, 1, 1)
        start = dt.datetime(2016, 1, 1)
        end = dt.datetime.today()
        mf_ranking = st.columns(2)
        s_date=mf_ranking[0].date_input("Start Date",value=start, min_value=min_start)
        e_date=mf_ranking[1].date_input("End Date",value=end,min_value=s_date)

        

        clicked = mf_ranking[0].form_submit_button('Submit')

        
    if clicked:

        s_date = dt.datetime(s_date.year, s_date.month, s_date.day)
        e_date = dt.datetime(e_date.year, e_date.month, e_date.day)

        #st.write(type(df_all_navs.index))
        sc_code_category = ["nav{}".format(code) for code in df_eqty_mf[df_eqty_mf['Scheme_Dtl_Category']==dtl_cat_option].index]
        #st.write(len(sc_code_category))
        sc_code_list = []
        k=0
        bounds_1= ((0,1), (0,1))
        init_guess_1 = [0.5,0.5]
        cons_1 = ({'type': 'eq', 'fun': check_sum})
        opt_results_arr = []
        for i in range(len(sc_code_category)):
            for j in range(i,len(sc_code_category)):
                if i != j:
                    #st.write("{}-{}-{}".format(k,sc_code_category[i],sc_code_category[j]))
                    sc_code_list = [sc_code_category[i],sc_code_category[j]]
                    #st.write(sc_code_list)
                    df_cat_sel = df_all_navs[sc_code_list]
                    df_cat_sel.dropna(inplace=True)
                    df_cat_sel = df_cat_sel[(df_cat_sel.index >= s_date) & (df_cat_sel.index <= e_date)]
                    #st.write(df_cat_sel)
                    mfunds_ret = df_cat_sel.pct_change().fillna(0)

                    opt_results = minimize(neg_sharpe, init_guess_1, method='SLSQP', bounds=bounds_1, constraints=cons_1)
                    opt_weights = [round(j, 4) for j in opt_results.x]
                    opt_vol_ret_sr = get_ret_vol_sr(opt_weights)

                    values = sc_code_list, opt_weights, opt_vol_ret_sr[0], opt_vol_ret_sr[1], opt_vol_ret_sr[2]

                    #st.write(values)

                    opt_results_arr.append(values)


        opt_results_df = pd.DataFrame(opt_results_arr, columns=['Scheme_Codes', 'Weights', 'Returns', 'Volatility', 'Sharpe Ratio'])
        #st.write(opt_results_df)

        l_cap = []
        for l_sc in sc_code_category:
            sch_code = l_sc.split("nav")[1]

            count = 0
            tot_wt = 0.0
            avg_wt = 0.0
            zero_count = 0
            sc_name = df.loc[int(sch_code)]['Scheme Name']
            fund_type = df.loc[int(sch_code)]['Scheme_Cat']
            for r in opt_results_df.index:
                weights = opt_results_df.loc[r]['Weights']
                rand_alloc = opt_results_df.loc[r]['Scheme_Codes']
                rand_alloc = [cols.split("nav")[1] for cols in rand_alloc]

                if sch_code in rand_alloc:
                    w = weights[rand_alloc.index(sch_code)]
                    if w > 0.1:
                        tot_wt = tot_wt + w
                        count = count + 1
                    else:
                        zero_count += 1

            if count > 0:
                avg_wt = tot_wt / (count + zero_count)
            values = sch_code, sc_name, fund_type, zero_count, count, zero_count + count, tot_wt, avg_wt
            l_cap.append(values)

        ranked_schemes = pd.DataFrame(l_cap, columns=['Scheme_Code', 'Scheme Name', 'Fund_Type', 'Zero_Count', 'Counts', 'Total_Count', 'Total_Weights', 'Avg_Weights'])
        #ranked_schemes = ranked_schemes[ranked_schemes['Counts'] > 0]
        st.write(ranked_schemes.sort_values(by=['Fund_Type','Avg_Weights'],ascending=False))


if option == 'MF Selection':

    # st.write(df)
    fund_list = []

    with st.form(key="MF_Selection"):
        for sc_code in df.index:
            list_value = '{}-{}'.format(sc_code,df.loc[sc_code]['Scheme Name'])
            fund_list.append(list_value)

        fund_selection = st.multiselect("Select Fund for Portfolio:", fund_list, default=[fund_list[0]])

        min_start = dt.datetime(2015, 1, 1)
        start = dt.datetime(2016, 7, 8)
        end = dt.datetime.today()

        left, centre, right = st.columns(3)
        start_date=left.date_input("Start Date",value=start, min_value=min_start)

        end_date=centre.date_input("End Date",value=end,min_value=start_date)

        pct_volatility = right.number_input("Maximum Volatility %",value=20.00, step=0.05)
        pct_volatility = pct_volatility/100
        st.write(pct_volatility)

        clicked = left.form_submit_button('Optimize')

        nfunds = len(fund_selection)

    if clicked:

        if nfunds > 1:
            mfunds = get_fund_data(fund_selection, start_date, end_date)
            #st.write(mfunds.head())
            mfunds_ret = mfunds.pct_change().fillna(0)
            #mfunds_ret = mfunds.pct_change(1)
            mfunds_normalised = 100*mfunds/mfunds.iloc[0]
            mfunds_normalised['Portfolio'] = mfunds_normalised.mean(axis=1)
            fig = px.line(mfunds_normalised)
            fig.update_layout(margin=dict(l=0, r=1, b=1, t=1))
            fig.update_layout(height=400)
            fig.update_layout(width=900)
            fig.update_layout(legend_title=dict(text="Funds"))
            fig.update_layout(legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ))
            # fig.layout.xaxis.fixedrange = True
            # fig.layout.yaxis.fixedrange = True
            st.header("")
            st.write(fig)
        else:
            st.warning('Please select at least 2 Mutual Funds.')
            st.stop()

    

        #st.write("Optimizing Portfolio...")

        ret_arr, vol_arr, sharpe_arr, weights = monte_carlo_simulation(mfunds_ret)
        frontier_volatility = []
        frontier_y = np.linspace(min(ret_arr),max(ret_arr),40)

        bounds, def_bounds = get_bounds(nfunds)
        init_guess = [round(1/nfunds,4) for j in range(nfunds)]


        cons_0 = ({'type':'eq','fun': check_sum},
                 {'type':'eq','fun': limit_volatility})

        cons = ({'type':'eq','fun': check_sum})

        
        opt_results = minimize(neg_sharpe,init_guess,method='SLSQP',bounds=bounds,constraints=cons)
        opt_weights = [j for j in opt_results.x]

        opt_vol_ret_sr = get_ret_vol_sr(opt_weights)

        #st.write(opt_vol_ret_sr)

        equal_weight_res = get_ret_vol_sr(init_guess)

        #st.write(equal_weight_res)

        for possible_return in frontier_y:
        # function for return
            cons = ({'type':'eq','fun': check_sum},
                    {'type':'eq','fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})
    
            result = minimize(minimize_volatility,init_guess,method='SLSQP',bounds=def_bounds,constraints=cons)
        
    
            frontier_volatility.append(result['fun'])
            

        
        
        
        #mcs_df = monte_carlo_simulation(mfunds_ret)

        #fig2 = px.scatter(x=vol_arr,y=ret_arr,color=sharpe_arr)
        #st.write(fig2)

        fig1 = plt.figure(figsize=(12, 8))
        plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='plasma')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.ylim(min(ret_arr)-0.03,max(ret_arr)+0.03)
        



        plt.plot(frontier_volatility,frontier_y,'g--',linewidth=3)

        plt.scatter(opt_vol_ret_sr[1],opt_vol_ret_sr[0],c='red',s=100,edgecolors='black')
        plt.text(opt_vol_ret_sr[1]+0.001,opt_vol_ret_sr[0], opt_results.x,size=8,bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=2))
        
        plt.scatter(equal_weight_res[1],equal_weight_res[0],c='black',s=100,edgecolors='red')
        plt.text(equal_weight_res[1]+0.001,equal_weight_res[0], init_guess,size=8,bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=2))

        opt_results = minimize(neg_sharpe,init_guess,method='SLSQP',bounds=def_bounds,constraints=cons)
        opt_weights = opt_results.x
        opt_vol_ret_sr = get_ret_vol_sr(opt_weights)

        plt.scatter(opt_vol_ret_sr[1],opt_vol_ret_sr[0],c='green',s=100,edgecolors='black')
        plt.text(opt_vol_ret_sr[1]+0.001,opt_vol_ret_sr[0], opt_results.x,size=8,bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=2))
        
        for k in range(nfunds):
            w = np.zeros(nfunds)
            w[k] = 1
            w_res = get_ret_vol_sr(w)
            plt.scatter(w_res[1],w_res[0],c='magenta',s=100,edgecolors='green')
            plt.text(w_res[1]+0.001,w_res[0], w ,size=8,bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=1))
            

        


        global_fig1 = fig1
        st.write(fig1)


        
        #st.write(weights_arr)

    

if option == 'MF Screener':


    fh_list = [x for x in df['Fund_House'].unique()]

    with st.form(key="MF_Screener"):

        fh_option = st.multiselect("Select Fund House", fh_list)

        screen_layout = st.columns((6,2,6))
        if len(fh_option) == 0:
            df_1 = df
        else:
            df_1 = df[df['Fund_House'] == 'X']
            for fh in fh_option:
                temp = df[df['Fund_House'] == fh]
                temp2 = pd.concat([df_1, temp])
                df_1 = temp2

        
        sch_typ_list = [x for x in df_1['Scheme_Type'].unique()]
        sch_option  = screen_layout[0].multiselect("Select Scheme Type", sch_typ_list)

        screen_layout[1].write(" ")
        if len(sch_option) == 0:
            df_1_5 = df_1
        else:
            df_1_5 = df_1[df_1['Scheme_Type'] == 'X']
            for sch in sch_option:
                df_1_5 = pd.concat([df_1_5, df_1[df_1['Scheme_Type'] == sch]])

        #st.write(df_1_5)

        sch_cat_list = [x for x in df_1_5['Scheme_Cat'].unique()]
        sch_cat_option = screen_layout[2].multiselect("Select Fund Category", sch_cat_list)
        if len(sch_cat_option) > 0:
            df_2 = df_1_5[df_1_5['Scheme_Cat'] == 'X']
            for cat in sch_cat_option:
                df_2 = pd.concat([df_2, df_1_5[df_1_5['Scheme_Cat'] == cat]])
        else:
            df_2 = df_1_5



        my_range = range(0, 10)
        age = screen_layout[0].select_slider("Fund Age Greater Than:", my_range, value=3)
        df_2 = df_2[df_2['Age'] > age]
        df_2 = df_2.sort_values(by=['Volatility'],ascending=True)

        screen_layout[1].write(" ")

        df_2.rename(columns={'Scheme_Cat':'Scheme Category'},inplace=True)
        view_range = ['Basic', 'Extended']
        report_view = screen_layout[2].select_slider("Select Report View:", view_range, value='Basic')

        Basic_View = ['Scheme_Code','Scheme Name', 'Fund_House', 'Scheme Category', 'Age', 'Volatility','Sharpe Ratio', '1Y Ret', '3Y Ret',
                    '5Y Ret', 'Ret Inception',]

        Extended_View = ['Scheme_Code','Scheme Name', 'Fund_House', 'Scheme Category', 'Age', 'Sortino Ratio', 'Beta', 'Alpha', 'R-Squared', 'Pos_Year%','Rel_MaxDD']
        #Benchmark_View = ['Scheme_Code','Scheme Name', 'Fund_House', 'Scheme_Cat', 'Age']


        df_2.reset_index(inplace=True)
        if report_view == 'Basic':
            #st.write(df_2[Returns_View])
            fig = display_table(df_2[Basic_View])
        elif report_view == 'Extended':
            #st.write(df_2[Ratios_View])
            fig = display_table(df_2[Extended_View])
 

        avg_5Yret = round(df_2['5Y Ret'].mean(),2)
        avg_vol   = round(df_2['Volatility'].mean(),2)


        n_clicked = st.form_submit_button("Submit")
    
    if n_clicked:
        table_expander = st.expander("Selected Mutual Funds")
        with table_expander:
            #rpt_radio = st.radio("Report View",('Returns','Ratios','Benchmark'))
            st.subheader("")
            col1,buf1, col2,buf2, col3,buf3 = st.columns((6,1, 8,1, 8,6))
            col1.write(f"Funds Selected: {len(df_2)}")
            col2.write(f"Average  5Y Return:  {avg_5Yret}%")
            col3.write(f"Average Volatility:  {avg_vol}%")

            st.plotly_chart(fig, use_container_width=True)
            #st.plotly_chart(fig)

            #st.write(df_2)

        
            df_chart = df_2[(df_2['3Y Ret'] > 0) & (df_2['5Y Ret'] > 0)].sort_values(by=['3Y Ret'], ascending=False)
            df_chart.fillna(0, inplace=True)

            df_chart['3Y Ret'] = round(df_chart['3Y Ret'], 0)


        

        
        if len(df_chart) > 0:
            fig = px.scatter(df_chart, x=df_chart['Volatility'], y=df_chart['3Y Ret'], color=df_chart['Sharpe Ratio'],
                             size=df_chart['5Y Ret'], size_max=25, hover_name=df_chart['Scheme Name'])

            yrange = [-5, df_chart['3Y Ret'].max()+5]
            # fig.update_xaxes(type=[])
            fig.update_yaxes(range=yrange)
            fig.update_layout(title_text="",
                          title_x=0.5,
                          title_font_size=30,
                          xaxis_title="Volatility %",
                          yaxis_title="3 Year Returns % ")

            fig.update_layout(margin=dict(l=1,r=1,b=1,t=1))
            fig.update_layout(height=400)
            fig.update_layout(width=600)
            fig.update_layout(legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01)
                            )


            perf_expander = st.expander("Performance Chart")
            with perf_expander:
                st.plotly_chart(fig, use_container_width=True)



        

if option == 'Manage Customer':
    cust_df = pd.read_csv("./customer_master.csv", dtype={'pin':'str'})
    max_id = cust_df['cust_id'].max()
    cust_df['pin']=cust_df['pin'].astype(str)
    cust_df.set_index('cust_id',inplace=True)
    #cust_df['dob'] = cust_df['dob'].apply(lambda x: dt.datetime(cust_df['dob'].split('-')[2],cust_df['dob'].split('-')[1],cust_df['dob'].split('-')[0]))

    cust_list = []
    for j in cust_df.index:
        list_values = '{}-{}'.format(j,cust_df.loc[j]['name'])
        cust_list.append(list_values)

    cust_oper = st.radio("Add / Modify / Delete Customers", ('Add', 'Chg', 'Del'))

    cust_id   = ""
    cust_name = ""
    cust_city = ""
    cust_pin  = ""
    cust_pan  = ""
    def_dob  = dt.datetime(1980, 1, 1)
    cust_profile = ""
     

    if cust_oper == 'Chg' or cust_oper == 'Del':
        cust_name = st.selectbox("Select Customer:",cust_list,0)
        cust_id = int(cust_name.split('-')[0])
            
        cust_city = cust_df.loc[cust_id]['city']
        cust_pin = cust_df.loc[cust_id]['pin']
        cust_pan = cust_df.loc[cust_id]['pan']
        #st.write(cust_df.loc[cust_id]['dob'])
        day_mon_year = cust_df.loc[cust_id]['dob'].split('-')
        #st.write(int(day_mon_year[2]),int(day_mon_year[1]),int(day_mon_year[0]))
        cust_dob = dt.datetime(int(day_mon_year[0]),int(day_mon_year[1]),int(day_mon_year[2]))
        cust_profile = int(cust_df.loc[cust_id]['risk_profile'].split('-')[0]) - 1


    with st.form(key="Customer_Master"):

        left, right = st.columns(2)

        if cust_oper == 'Add':
            f_cust_name = left.text_input(label="Customer Name:")
            f_cust_profile = right.selectbox("Risk Profile:",('1-Conservative','2-Cautious','3-Prudent','4-Assertive','5-Aggressive'),3)
            f_cust_city    = left.text_input(label="City")
            f_cust_pin     = right.text_input(label="PIN")
            f_cust_dob     = left.date_input("DOB",value=def_dob)
            f_cust_pan     = right.text_input(label="PAN")
        elif cust_oper == 'Chg':
            left.subheader(f"Customer Id: {cust_name.split('-')[0]}")
            left.subheader(f"Customer Name: {cust_name.split('-')[1]}")
            f_cust_profile = right.selectbox("Risk Profile:",('1-Conservative','2-Cautious','3-Prudent','4-Assertive','5-Aggressive'), cust_profile)
            f_cust_profile = cust_profile
            f_cust_city    = left.text_input(label="City", value=cust_city)
            f_cust_pin     = right.text_input(label="PIN", value=cust_pin)
            f_cust_dob     = left.date_input("DOB",value=cust_dob)
            f_cust_pan     = right.text_input(label="PAN", value=cust_pan)
            
        elif cust_oper == 'Del':
            left.subheader(f"Customer Id: {cust_name.split('-')[0]}")
            right.subheader(f"Customer Name: {cust_name.split('-')[1]}")
            left.subheader(f"City: {cust_city}")
            right.subheader(f"PIN: {cust_pin}")
            left.subheader(f"DOB: {cust_dob.date()}")
            right.subheader(f"PAN: {cust_pan}")
            left.subheader(f"Risk Profile: {cust_df.loc[cust_id]['risk_profile']}")                


        
        if cust_oper == 'Add':
            s_butn = left.form_submit_button('Insert')
        elif cust_oper == 'Chg':
            s_butn = left.form_submit_button('Update')
        elif cust_oper == 'Del':
            s_butn = right.form_submit_button('Delete')




    if s_butn:
        if cust_oper == 'Add':
            #st.write(max_id)
            ins_id = max_id + 1

            cust_df.at[ins_id,'name'] = f_cust_name
            cust_df.at[ins_id,'city'] = f_cust_city
            cust_df.at[ins_id,'pin']  = f_cust_pin
            cust_df.at[ins_id,'pan']  = f_cust_pan
            cust_df.at[ins_id,'dob']  = f_cust_dob
            cust_df.at[ins_id,'risk_profile']  = f_cust_profile
            cust_df.to_csv("./customer_master.csv")
            st.write("Customer Record Inserted")
            f_cust_name    = ""
            f_cust_profile = ""
            f_cust_city    = ""
            f_cust_pin     = ""
            f_cust_dob     = def_dob
            f_cust_pan     = ""

            
        elif cust_oper == 'Chg':
            cust_df.at[cust_id,'name'] = cust_name.split('-')[1]
            cust_df.at[cust_id,'city'] = f_cust_city
            cust_df.at[cust_id,'pin']  = f_cust_pin
            cust_df.at[cust_id,'pan']  = f_cust_pan
            cust_df.at[cust_id,'dob']  = f_cust_dob
            cust_df.at[cust_id,'risk_profile']  = f_cust_profile
            cust_df.to_csv("./customer_master.csv")
            st.write("Customer Record Modified")

        elif cust_oper == 'Del':
            #st.write(cust_id)
            cust_df = cust_df.drop(cust_id)
            cust_df.to_csv("./customer_master.csv")
            st.write("Customer Deleted")

            
        #st.write(f_cust_pin)
        st.write(cust_df)

            
            

        




    

    
            
    


