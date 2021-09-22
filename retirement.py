import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import optimize
#import math




np.set_printoptions(precision=2)

st.set_page_config(layout="wide")
tday = dt.datetime.today()

col1, col2 = st.sidebar.columns(2)
col1.image('gw_logo.png', width=125)
col2.title("GroWealth Dashboard")



option = st.sidebar.selectbox("Which Dashboard?", ( 'Retirement Planning','Goal Planning'), 1)
st.title(option)


def get_goals(age,desc,amt,freq):
    goals = []

    if amt >  0 and age <= plan_till:
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

def get_corpus(rate, curr_age, ann_income, retirement_age, corpus, expenses, fut_income):
    rec = []
    income = 0
    yr_corpus = corpus
    n= len(fut_income)
    for j in expenses.index:
        if j > curr_age:
            if j < retirement_age:
                income = ann_income
            else:
                income = 0

            for k in range(n):
                if j==fut_income[k][0]:
                    income=income + fut_income[k][1]

            yr_corpus = yr_corpus * (1 + rate/100) + income - expenses.loc[j]['Expenses']
        values = j, round(yr_corpus,0)
        rec.append(values)

    df = pd.DataFrame(rec,columns=['Years',f"Corpus-{rate}%"])
    df.set_index('Years', inplace=True)
    return df
        
            
        
def get_optimised_rate(rate, curr_age, ann_income, retirement_age, corpus, expenses, terminal_corpus, fut_income):
    rec = []
    income = 0
    yr_corpus = corpus
    n= len(fut_income)
    for j in expenses.index:
        if j > curr_age:
            if j < retirement_age:
                income = ann_income
            else:
                income = 0

            for k in range(n):
                if j==fut_income[k][0]:
                    income=income + fut_income[k][1]

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
        




if option == 'Goal Planning':


    left,centre,right = st.columns((8,1,6))
    goal_type = left.selectbox("Select Goal", ('Marriage', 'Higher Education','Vacation','Buying a Dream Car','Buying Dream Home','Miscellaneous Goal'),1)
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

    left.image(image_path, width=500, use_column_width=True)


    gp_flds = st.columns((4,1,4))

    goal_amount = right.number_input(f"Cost of {goal_type} (in Today's Price)", value=0,step=10000)
    years_to_goal = right.slider("Years to Goal?", min_value=1, max_value=30, step=1, value=3)    

    present_corpus = right.number_input("Corpus I Already Have", value=0,step=10000)

    rate = round(right.number_input("Return on Assets", step=0.10),2)
    infl = right.number_input("Inflation", step =0.1)

    adj_amount = goal_amount*pow((1+infl/100),years_to_goal)

    present_value = adj_amount/pow((1+rate/100),years_to_goal)

    tot_mths = 12*years_to_goal        

    mthly_amt=round(optimize.newton(get_emi, 0, tol=0.0000001, args=(rate, tot_mths,adj_amount,present_corpus)),2)

    
    html_text = '<p style="font-family:Courier; color:Blue; font-size: 18px;">Onetime Investment Required: ' + "Rs. {:,.2f}</p>".format(present_value - present_corpus)

    left.markdown(html_text, unsafe_allow_html=True)
    html_text = '<p style="font-family:Courier; color:Blue; font-size: 18px;">Monthly SIP Required: ' + "Rs. {:,.2f}</p>".format(mthly_amt)
    left.markdown(html_text, unsafe_allow_html=True)

    

   


if option == 'Retirement Planning':

    st.title("  ")
    user_inputs = st.columns((1,1,1,1))
    curr_age = user_inputs[0].slider("Your Current Age?", min_value=30, max_value=100, step=1, value=0)
    yrs_to_retire = user_inputs[1].slider("Years to Retire", min_value=0, max_value=30, step=1, value=0)
    plan_till = user_inputs[2].slider("Plan Till", min_value=curr_age + yrs_to_retire, max_value=100, step=1, value=90)
    n_goals = user_inputs[3].slider("Number of Goals", min_value=0, max_value=10, step=1, value=5)

    with st.form(key="Retirement"):


        
        align = st.columns(3)

        c_annual_income = align[0].number_input("Annual Income", value=0,step=100000)
        c_annual_expense = align[1].number_input("Annual Expense", value=0,step=100000)
        c_corpus = align[2].number_input("Current Corpus", value=0,step=100000)

        cagr = round(align[0].number_input("Return on Assets", step=0.10),2)
        inflation = align[1].number_input("Inflation", step =0.1)
        terminal_corpus = align[2].number_input("Terminal Corpus", value=0,step=100000)
            
        goal_expander = st.expander("Goals")


        with goal_expander:
            goal_cols = st.columns((2,4,3,2))


            g_year = [goal_cols[0].slider("Age", key=f"g_yr_{col}",min_value=curr_age, max_value=plan_till, step=1, value=curr_age) for col in range(n_goals)]
            g_desc = [goal_cols[1].text_input("Goal Description",key=f"g_desc_{col}") for col in range(n_goals)]
            g_amt  = [goal_cols[2].number_input("Goal Amount (in Today's Price)",key=f"g_amt_{col}", value=0, format="%d", step=50000) for col in range(n_goals)]
            g_freq = [goal_cols[3].number_input("Goal Frequency (once in N years)",key=f"g_freq_{col}", value=0, format="%d", step=1) for col in range(n_goals)]

        fut_income_expander = st.expander("Future Income")
        
        with fut_income_expander:
            fut_income = st.columns((3,5,3))
            f_year = [fut_income[0].slider("Age", key=f"f_yr_{col}",min_value=curr_age, max_value=plan_till, step=1, value=curr_age) for col in range(3)]
            f_desc = [fut_income[1].text_input("Income Description",key=f"f_desc_{col}") for col in range(3)]
            f_amt  = [fut_income[2].number_input("Income Amount (in Today's Price)",key=f"f_amt_{col}", value=0, format="%d", step=50000) for col in range(3)]

                
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
        
        fut_income = []
        for k in range(3):
            if f_amt[k] > 0:
                values = f_year[k], f_amt[k]
                fut_income.append(values)
	    

        df_corpus = get_corpus(cagr,curr_age, c_annual_income, age_at_retirement, c_corpus, df_expense, fut_income)

        retirement_assets = df_expense.merge(df_corpus, on='Years')
        #st.write(retirement_assets)

        root=round(optimize.newton(get_optimised_rate, 25, tol=0.0000001, args=(curr_age, c_annual_income, age_at_retirement, c_corpus, df_expense,terminal_corpus, fut_income)),2)

        

        #st.write(root)
        if 0 < root < 25:
            optimised_rate = get_corpus(root,curr_age, c_annual_income, age_at_retirement, c_corpus, df_expense, fut_income)
            retirement_assets = retirement_assets.merge(optimised_rate, on='Years')
            
        

        retirement_assets = retirement_assets / 10000000

        column_name = retirement_assets.columns[2]
        c_max = retirement_assets[column_name].max()

        c_corpus = c_corpus /10000000
        
        #st.write(retirement_assets)
        fig = px.line(retirement_assets)
        fig.update_layout(title_text="",
                          title_x=0.5,
                          title_font_size=10,
                          xaxis_title="Age (in Years) ",
                          yaxis_title="Retirement Fund (Crores)")

        fig.update_layout(margin=dict(l=1,r=1,b=1,t=3))
        yrange = [0 - c_corpus, min(4*c_corpus,c_max)]
        fig.update_yaxes(range=yrange, dtick=0.5,showgrid=True)
        fig.update_xaxes(showgrid=True)

        fig.update_layout(height=500)
        fig.update_layout(width=950)
        fig.update_layout(legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01
        ))
        header_alignment = st.columns((5,4,4))
        header_alignment[1].subheader("Retirement Planning Chart")
        st.write(fig)
        
            
