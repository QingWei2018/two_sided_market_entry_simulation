# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:16:20 2018

@author: Qing Wei
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import timeit
#pd.options.display.max_columns = None
#pd.option_context('display.float_format', lambda x: '%.5f' % x)
#pd.options.display.float_format = '{:,.5f}'.format
import pyomo.environ as pe
import os


def create_current_matrix(N,C,a1,a2,luck,r):
    market_size=[]
    num=[]
    for n in range(1,101):
        m=1./(n**2)
        #r=1./n
        market_size.append(m)
        num.append(n)
    if luck==1: #good luck
        alpha_k1=np.random.uniform((a1+a2)/2.,a2,5)
    elif luck==-1: #bad luck
        alpha_k1=np.random.uniform(a1,(a2+a1)/2.,5)
    elif luck==0: #中立
        alpha_k1=np.random.uniform(a1,a2,5)
        
    alpha_k2=np.random.uniform(a1,a2,95)
    alpha=np.concatenate((alpha_k1,alpha_k2),axis=0)
    alpha=alpha.tolist()
        
    
    expect_alpha=[(a1+a2)/2]*100
    entry_status=[0]*100
    nan_status=[None]*100
    
    
    df=pd.DataFrame({"market_size":market_size, #市场大小 1/k^2
                 "real_alpha":alpha,
                 "uber_order":entry_status, 
                 "lyft_order":entry_status,
                 "uber_time":nan_status,
                 #"uber_exit_time":nan_status,
                 "lyft_time":nan_status,
                 #"lyft_exit_time":nan_status,
                 "expect_alpha":expect_alpha,
                 "market":num,
                 "uber_share":entry_status,
                 "lyft_share":entry_status,
                 "uber_revenue":entry_status,
                 "lyft_revenue":entry_status})
    df["market_earn"]=df["market_size"]*df["real_alpha"]*N
    df["expect_market_earn"]=df["market_size"]*df["expect_alpha"]*N
    df["entry_cost"]=C/df["market"]
    df["expect_profit"]=df["expect_market_earn"]/r - df["entry_cost"]
    df["real_profit"]=df["market_earn"]/r - df["entry_cost"]
    
    
    df["check"]=df["uber_order"]+df["lyft_order"]
    df=df[["market","market_size","entry_cost","expect_alpha","real_alpha",\
           "expect_market_earn","market_earn","expect_profit","real_profit","uber_order","check", "lyft_order","uber_time","lyft_time","uber_share","lyft_share","uber_revenue","lyft_revenue"]]
    
    return df[df["expect_profit"]>=0]



def create_timing_matrix(K):
    time=[]
    k=[]
    for i in range(0,K):
        time.append("time_"+str(i))
        k.append(i+1)
    df=pd.DataFrame(columns=time )
    dfk=pd.DataFrame({"K":k})
    df2=pd.concat([dfk,df],axis=1)
    df2=df2.fillna(0)
    return df2


def create_states_matrix(uo,l1,K):
    time=[]
    for i in range(0,K+1):
        time.append(i)
    columns=["ut","lt","R_ut","R_lt","rut","rlt","C_ut","C_lt","nu","nl","Nu","Nl"]
    df1=pd.DataFrame(columns=columns)
    df2=pd.DataFrame({"time":time})
    df=pd.concat([df1,df2],axis=1)
    df["ut"][0]=uo
    df["lt"][0]=0
    df["lt"][1]=l1
    df=df.fillna(0.)
    return df

def integer_solver(subset1,u,time):
    model=pe.ConcreteModel()
    cost=subset1["entry_cost"].tolist()
    earn=subset1["earn"].tolist()
    model.M_index=range(len(subset1))
    model.y=pe.Var(model.M_index, initialize=0,within=pe.Binary)


    def _const1(model,i):
        return sum(model.y[i]*cost[i] for i in model.M_index)<=u
    model.const1=pe.Constraint(model.M_index,rule=_const1)

    def _const2(model,i):
        return model.y[i]*cost[i]<=earn[i]*100
    model.const2=pe.Constraint(model.M_index,rule=_const2)



    model.obj=pe.Objective(\
        expr= sum(earn[i]*model.y[i] for i in model.M_index),\
        sense=pe.maximize)

    opt_solver=pe.SolverFactory("glpk")
    result= opt_solver.solve(model,tee=True)
    y=[model.y[i].value for i in model.y]
    index1=subset1.index.tolist()
    entry_market=[i+1 for i in index1]
    dict1=dict(zip(entry_market,y))


    df=pd.DataFrame(list(dict1.items()),columns=["K",time],index=index1)

    return df 


def uber_move(t,p,l1,gamma,current_matrix,timing_matrix,state_matrix):
        
        time="time_%s" %str(t) #时间t for uber
        u=state_matrix["ut"][t] # measures money stock
        l=state_matrix["lt"][t]
        if t>0:
            su=state_matrix["R_ut"][t-1] # measures market occupation stock
            sl=state_matrix["R_lt"][t-1]
            u_ratio=su/(su+sl)
            l_ratio=sl/(sl+su)
        else:
            su=0
            sl=0
            u_ratio=0
            l_ratio=0
        
        uber_share= (p*math.exp(u_ratio)**gamma)/(p*math.exp(u_ratio)**gamma+math.exp(l_ratio)**gamma)
        lyft_share= (math.exp(l_ratio)**gamma)/(p*math.exp(u_ratio)**gamma+math.exp(l_ratio)**gamma)



        # setup choice set for integer programming 
        subset1=current_matrix[current_matrix["uber_order"]==0]

        #making entry decisions
        if len(subset1)>0:
            
            subset1["uber_share"]=subset1["check"].apply(lambda x: 1 if x==0 else uber_share) #when uber_order ==0 check either ==0 or 1
            subset1["earn"]=np.where(subset1["check"]==0,subset1["expect_market_earn"]*subset1["uber_share"], subset1["market_earn"]*subset1["uber_share"])
            
            entry_decision=integer_solver(subset1,u,time)
            enter=entry_decision[entry_decision[time]==1].index.tolist()

        #updating status of other matrixs:

        #timing:
            timing_matrix[time][enter]=1
          

        #current:

            current_matrix["uber_share"][enter]=np.where(current_matrix["check"][enter]==0, 1, uber_share)
            current_matrix["lyft_share"][enter]=np.where(current_matrix["check"][enter]==0, 0, lyft_share)
            current_matrix["uber_time"][enter]=t
            current_matrix["uber_order"][enter]=current_matrix["lyft_order"][enter]+1
            cost=current_matrix["entry_cost"][enter].sum()
            state_matrix["nu"][t]=len(enter) # number of new entries this period
            state_matrix["rut"][t]=current_matrix["uber_revenue"][enter].sum() # profit from this period's entry
        else:
            cost=0


        current_matrix["uber_revenue"]=current_matrix["uber_share"]*current_matrix["market_earn"]
        current_matrix["lyft_revenue"]=current_matrix["lyft_share"]*current_matrix["market_earn"]

        current_matrix["check"]=current_matrix["uber_order"]+current_matrix["lyft_order"]

        

        #state:

        
        state_matrix["nl"][t]=0

        state_matrix["Nu"][t]=state_matrix["nu"][:t+1].sum() #total number of uber entries at time t
        state_matrix["Nl"][t]=state_matrix["nl"][:t+1].sum()

        
        state_matrix["rlt"][t]=0 # no movement no profit for lyft

        state_matrix["R_ut"][t]=current_matrix["uber_revenue"].sum()# revenue from all entered market at time t
        state_matrix["R_lt"][t]=current_matrix["lyft_revenue"].sum()

        state_matrix["C_ut"][t]=cost
        state_matrix["C_lt"][t]=0


        state_matrix["ut"][t+1]= u+ state_matrix["R_ut"][t] - cost

        if t==0:
            state_matrix["lt"][t+1]=l1
        else:
            state_matrix["lt"][t+1]=l + state_matrix["R_lt"][t]
            
        return current_matrix, timing_matrix,state_matrix
        
    
def lyft_move(t,p,gamma,current_matrix,timing_matrix,state_matrix):
        time="time_%s" %str(t) #时间t for lyft
        
            #initialize
        u=state_matrix["ut"][t] # measures money stock
        l=state_matrix["lt"][t]
        su=state_matrix["R_ut"][t-1] # measures market occupation stock
        sl=state_matrix["R_lt"][t-1]
        u_ratio=su/(su+sl)
        l_ratio=sl/(sl+su)
        uber_share= (math.exp(u_ratio)**gamma)/(math.exp(u_ratio)**gamma+p*math.exp(l_ratio)**gamma)
        lyft_share= (p*math.exp(l_ratio)**gamma)/(math.exp(u_ratio)**gamma+p*math.exp(l_ratio)**gamma)



        # setup choice set for integer programming 
        
        subset1=current_matrix[current_matrix["lyft_order"]==0]


        #making entry decisions
        if len(subset1)>0:
            subset1["lyft_share"]=subset1["check"].apply(lambda x: 1 if x==0 else lyft_share) #when uber_order ==0 check either ==0 or 1
            subset1["earn"]=np.where(subset1["check"]==0,subset1["expect_market_earn"]*subset1["lyft_share"], subset1["market_earn"]*subset1["lyft_share"])
            entry_decision=integer_solver(subset1,l,time)
            enter=entry_decision[entry_decision[time]==1].index.tolist()

        #updating status of other matrixs:

        #timing:
            timing_matrix[time][enter]=1


            current_matrix["uber_share"][enter]=np.where(current_matrix["check"][enter]==0, 0, uber_share)
            current_matrix["lyft_share"][enter]=np.where(current_matrix["check"][enter]==0, 1, lyft_share)
            
            current_matrix["lyft_order"][enter]=current_matrix["uber_order"][enter]+1
            current_matrix["lyft_time"][enter]=t
            cost=current_matrix["entry_cost"][enter].sum()
            
            state_matrix["nl"][t]=len(enter) # number of new entries this period
            state_matrix["rlt"][t]=current_matrix["lyft_revenue"][enter].sum() # revenue from this period's entry
        else:
            cost=0


        current_matrix["uber_revenue"]=current_matrix["uber_share"]*current_matrix["market_earn"]
        current_matrix["lyft_revenue"]=current_matrix["lyft_share"]*current_matrix["market_earn"]
        
        current_matrix["check"]=current_matrix["uber_order"]+current_matrix["lyft_order"]
        



        #state:

        
        state_matrix["nu"][t]=0

        state_matrix["Nu"][t]=state_matrix["nu"][:t+1].sum() #total number of uber entries at time t
        state_matrix["Nl"][t]=state_matrix["nl"][:t+1].sum()


        state_matrix["rut"][t]=0 # no movement no profit for uber

        state_matrix["R_ut"][t]=current_matrix["uber_revenue"].sum()  # revenue from all entered market at time t
        state_matrix["R_lt"][t]=current_matrix["lyft_revenue"].sum()

        state_matrix["C_ut"][t]=0
        state_matrix["C_lt"][t]=cost

        state_matrix["ut"][t+1]= u+ state_matrix["R_ut"][t]

        state_matrix["lt"][t+1]= l+ state_matrix["R_lt"][t] - cost
        return current_matrix, timing_matrix,state_matrix
    
    
case1=[20.0,20.0,0.0,1.0]
case2=[20.0,20.0,0.1,1.0]
case3=[20.0,20.0,1.0,0.0]
case4=[20.0,20.0,1.0,1.0]
case5=[10.0,5.0,0.8,1.0]
case6=[10.0,40.0,0.8,1.0]
case7=[20.0,40.0,0.8,1.0]
case8=[40.0,40.0,0.8,1.0]
case9=[20.0,20.0,0.8,1.0]
case10=[20.0,60.0,0.8,1.0]
case11=[10.0,10.0,0.8,1.0]
case12=[10.0,20.0,0.8,1.0]
case13=[20.0,80.0,0.8,1.0]

case_list=[case2,case3,case4]  



start=timeit.default_timer()

for i in case_list:
    u0=i[0]
    l1=i[1]
    p=i[2]
    gamma=i[3]
    
    directory="p_"+str(p)+"_gamma_" +str(gamma) +"_u0_"+str(u0) + "_l1_"+str(l1)

    #os.makedirs(directory)
    for i in range(501,1000):
        i=i+1
        N=20. #citysize
        C=20. #cost of enter a new market
        a1=0. #upper bound of alpha
        a2=1. #lower bound of alpha
        #u0=20. #start fund of uber
        #l1=20. #start fund of lyft
        r=0.01 #depreciation rate
        #p=0. #disadvantage of second mover
        luck=0.
        #gamma=1.


        K=int((N*(a1+a2)/2)/(r*C))  #LAST CITY TO ENTER

        current_matrix=create_current_matrix(N,C,a1,a2,luck,r)
        timing_matrix=create_timing_matrix(K)
        state_matrix=create_states_matrix(u0,l1,K)

        cols = list(timing_matrix.columns.values)
        cols.insert(0,"real_alpha")

        timing_matrix["real_alpha"]=current_matrix["real_alpha"]
        timing_matrix=timing_matrix[cols]
        try:
            t=0
            while t<50:        

                if t%2==0:
                    current_matrix, timing_matrix, state_matrix= uber_move(t,p,l1,gamma,current_matrix,timing_matrix,state_matrix)
                    t=t+1
                if t%2==1:
                    current_matrix, timing_matrix, state_matrix= lyft_move(t,p,gamma,current_matrix,timing_matrix,state_matrix)
                    t=t+1
            current=directory+"/"  + "current_matrix_"+ "_" + str(i)+ ".xlsx"
            timing= directory+"/" +  "timing_matrix_" + "_" + str(i)+ ".xlsx"
            state=  directory+"/"+   "state_matrix_"  + "_" + str(i)+ ".xlsx"

            current_matrix.to_excel(current)
            timing_matrix.to_excel(timing)
            state_matrix.to_excel(state)
        except:
            pass

    stop = timeit.default_timer()

    execution_time = stop - start

    print ("Program Executed in ",execution_time) #It returns time in sec  
