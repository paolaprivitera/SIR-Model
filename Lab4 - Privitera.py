#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Numerical SIR model


# In[2]:


# Libraries
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import t
import random
from numpy.random import rand


# In[3]:


N = 10000 # population
beta = 0.2 # [days^-1] 
infection_period = 14 # day
gamma = 1/infection_period # [day^-1] 

daysOfYear = 365


# In[4]:


# Print
print("Population of size:", N)
print("Infection transmission rate per capita:", beta, "[1/day]")
print("Recovery rate:", gamma, "[1/day]")


# In[5]:


S = np.zeros(daysOfYear) # number of susceptible people
I = np.zeros(daysOfYear) # number of infected people
R = np.zeros(daysOfYear) # number of recovered people

# When the infection starts, there is one infected, N - 1 susceptible people and 0 recovered
S[0] = N - 1
I[0] = 1
R[0] = 0


# In[6]:


R0 = beta/gamma
print("The reproduction number is:", R0)

if R0 > 1:
    print("Since R0 is > 1, the invasion is possible and infection can spread through the population")
else:
    print("Since R0 is < 1, the disease cannot invade the population and the infection will die out over a period of time")


# In[7]:


# Standard numerical models to solve the differential equations
for day in range(daysOfYear-1):
    S[day+1] = S[day] - (beta/N)*S[day]*I[day]
    I[day+1] = I[day] + (beta/N)*S[day]*I[day] - gamma*I[day]
    R[day+1] = R[day] + gamma*I[day]


# In[8]:


def plotSIR(daysOfYear, S, I, R, typeModel):
    x = range(0, daysOfYear)
    y = S
    fig, ax = plt.subplots(figsize=(8,4))
    #plt.xlim(100, 1000000)   

    ax.plot(x, y, label = "Susceptible")

    y2 = I
    ax.plot(x, y2, label = "Infected")

    y3 = R
    ax.plot(x, y3, label = "Recovered")

    plt.legend()
    plt.xlabel("Time [day]")
    plt.ylabel("Number of people")
    
    if typeModel == "numerical":
        plt.title("Numerical SIR model")
    elif typeModel == "simulative":
        plt.title("Simulative SIR model")

    plt.show()


# In[9]:


plotSIR(daysOfYear, S, I, R, "numerical")


# In[10]:


# maximum number of infected people

def getMaxInfected(I):
    max_infected_people = I.max()
    max_infected_people = math.floor(max_infected_people)
    day_max = I.argmax() # it gives the index corresponding to the max value of the numpy array

    return print("The maximum number of infected people is", max_infected_people, "and it occurs at day", day_max)


# In[11]:


getMaxInfected(I)


# In[12]:


# check end of epidemy --> If number of infected becomes < 1 --> there are no more infected
def getEndEpidemy(I):
    end_of_epidemy = False
    for i in range(len(I)):
        if I[i] < 1: # 0.5 ?????
            end_of_epidemy = True
            end_day = i
            break

    if end_of_epidemy == True:
        print("The epidemy ended after", end_day, "days") #end_day+1
    else:
        print("The epidemy did not end")
    return end_day #end_day+1


# In[13]:


getEndEpidemy(I)


# In[14]:


# Simulative SIR model


# In[15]:


# INPUT PARAMETERS
initial_seed = 1305 # to run the simulator under the same conditions
N = 10000 # number of people
runs = 10 # number of runs
confidence_level = 0.95
daysOfYear = 365
beta = 0.2
avg = beta/2
gamma = 1/14


# In[16]:


# Print input parameters
print("INITIAL SETTINGS OF PARAMETERS\n")
print("Initial seed:", initial_seed)
print("Number of runs:", runs)
print("Confidence level:", confidence_level, "%")
print("Number of people:", N)
print("Number of days:", daysOfYear)


# In[17]:


random.seed(a=initial_seed)
np.random.seed(initial_seed)

S_sim = []
I_sim = []
R_sim = []

RT_sim = []
start_sim = []
recovery_sim = []

for run in range(runs):
    people = np.zeros(N) # vector of size people which is initialized at 0
    # it takes value 0 if person i is susceptible
    # value 1 if person i is infected
    # value 2 if person i is recovered
    # with 0 < i < N-1
    
    S = np.zeros(daysOfYear) # number of susceptible people
    I = np.zeros(daysOfYear) # number of infected people
    R = np.zeros(daysOfYear) # number of recovered people  
    
    recovery = np.zeros(N) # for each person, update the number of days of the infection
    start_infection = np.full(N, -1) # array that for each person memorizes the day in which the infection started
    
    
    # When the infection starts, there is one infected, N - 1 susceptible people and 0 recovered
    # Let's suppose that the first person (position 0 of the vector) is infected
    people[0] = 1
    
    # day 0
    # at first day there is one infected
    I[0] = 1
    
    # at first day the susceptible people are
    S[0] = N - 1

    start_infection[0] = 0 # for person 0, the infection starts at day 0

    
    i = 1
    s = N - 1
    r = 0
    
    
    num_days_for_recovery = np.random.geometric(gamma, N) # array of size N that memorizes the duration of the infection for each person
    
    rt_matrix = np.zeros((N, daysOfYear)) # N rows, days columns
    
    for day in range(1, daysOfYear):
        
        num_contacts = np.random.geometric(p=1/(1+beta), size=N) - 1 # geometric distribution with mean beta/2 and support that starts from 0
        # for each person i, it gives the number of people that he meets during the current day
        

        index_person = 0
        for person in people:
            
            if person == 1: # if the person is infected
                
                # update for the current infected the duration of the infection
                recovery[index_person] += 1 
                
                
                # check if the current infected/person is recovered
                if recovery[index_person] >= num_days_for_recovery[index_person]:
                    people[index_person] = 2 # he recovered
                    
                    r += 1 # increment the number of recovered of that day
                    i -= 1 # decrease the number of infected of that day

                else:
                    num_contacts_person = num_contacts[index_person]
                    

                    for c in range(num_contacts_person):
                        person_to_meet = random.randint(0, N-1)

                        state = people[person_to_meet] # see if he is s, i, r 
                        
                        #print("Person to meet", person_to_meet, "state", state)
                        
                        if state == index_person:
                            person_to_meet = random.randint(0, N-1)

                            state = people[person_to_meet]


                        # if the current infected meets a susceptible   
                        if state == 0: # susceptible
                            # save the index of the person that will become infected
                            
                            
                            people[person_to_meet] = 1
                            
                            
                            i += 1 # increment infected 
                            s -= 1 # decrease susceptible
                            

                            rt_matrix[index_person][day] += 1
                            
                            start_infection[person_to_meet] = day

                        # if the current infected meets another infected
                        # nothing happens

                        # if the current infected meets a recovered
                        # nothing happens
   
            
            index_person += 1
        
        
        I[day] = i
        R[day] = r
        S[day] = s
         
        
        #print(I[day])
        #print(S[day])
        #print(R[day])
    
    S_sim.append(S)
    I_sim.append(I)
    R_sim.append(R)
    RT_sim.append(rt_matrix)
    start_sim.append(start_infection)
    recovery_sim.append(num_days_for_recovery)
                


# In[94]:


i = 9
plotSIR(daysOfYear, S_sim[i], I_sim[i], R_sim[i], "simulative")


# In[95]:


getMaxInfected(I_sim[i])


# In[96]:


getEndEpidemy(I_sim[i])


# In[21]:


n = runs
new_runs = []
for i in range(runs):
    if S_sim[i][364] == 9999.0: # degenerate
        n -= 1
    else:
        new_runs.append(i)
        

RT = np.zeros((n, daysOfYear), dtype=float)
#print(len(RT))
#print(RT[0])

index = 0
for run in new_runs:
    # matrix: each row is a person
    rt_matrix = RT_sim[run]
    num_recovery = recovery_sim[run]
    num_start = start_sim[run]
        
    # for each row of the matrix (each person), count the number of people that have been infected by that person (i)
    sums = []
    sums_first_days = []
    for i in range(N):
        sums.append(rt_matrix[i].sum()) #sum all values of the row
        sums_first_days.append(rt_matrix[i].sum())
            
    # N rows, days columns
    # same structure of rt_matrix
    new_rt_matrix = np.zeros((N, daysOfYear))
        
    for i in range(N):
        for j in range(daysOfYear):
            if rt_matrix[i][j] > 0: 
                new_rt_matrix[i][j] = sums[i]
            
                sums[i] = sums[i] - rt_matrix[i][j] # rivolto la matrice
                
    for i in range(N):
        for j in range(daysOfYear):
            if new_rt_matrix[i][j] == 0 and j != 0 and j <= (num_recovery[i] + num_start[i] -1): # fino alla durata dell'infezione
                new_rt_matrix[i][j] = new_rt_matrix[i][j-1]
                    
    for i in range(N):
        num_end = num_recovery[i] + num_start[i] - 1
        for j in range(num_start[i], num_end):
            if sums_first_days[i]>0 and new_rt_matrix[i][j] == 0:
                new_rt_matrix[i][j] = sums_first_days[i]
                    
        
    rt_final = 0
    RT_final = []

    for j in range(daysOfYear):
        tot_infected = 0 # number of infected of that day that can infect other people (provocano secondary infections)
        secondary_infections = 0
        for i in range(N): # per ogni persona, in quel giorno, se la persona è infetta, incremento di 1 il numero degli infetti 
            if new_rt_matrix[i][j] > 0:
            
                tot_infected += 1
                secondary_infections += new_rt_matrix[i][j] # di ogni giorno   
    
        if tot_infected != 0:
            rt_final = secondary_infections/tot_infected
        else:
            rt_final = 0
        
        RT_final.append(rt_final)
    #print(len(RT_final))
    #print(RT_final)
            
    RT[index] = RT_final
    index += 1


# In[22]:


def getConfidenceInterval(rt):
    # 1 - alpha = confidence_level
    # alpha = 1 - confidence_level
    # 1 - alpha/2 = 1 - (1 - confidence_level)/2
    
    q = 1 - (1-confidence_level)/2
    t_student = t.ppf(q, df=n-1)
  
    avg = rt.mean(axis = 0) # average
    stdd = rt.std(axis = 0, ddof=1) # standard deviation
    half_ci = t_student * stdd/math.sqrt(n)
    
    lb = avg - half_ci
    ub = avg + half_ci
    
    return avg, lb, ub


# In[23]:


rt_avg, rt_lb, rt_ub = getConfidenceInterval(RT)


# In[34]:


def plotRT(rt_avg, rt_lb, rt_ub):
    x = range(0, daysOfYear)
    y = rt_avg
    fig, ax = plt.subplots(figsize=(8,4))
    plt.ylim(-1, 10)   
    #plt.xlim(0, 365)
    
    ax.fill_between(x, rt_lb, rt_ub, label = "95% confidence", alpha=0.1)

    ax.plot(x, y, label = "RT")

    plt.legend()
    plt.xlabel("Time [day]")
    plt.ylabel("RT")
    plt.title("RT in function of the day")

    plt.show()


# In[35]:


plotRT(rt_avg, rt_lb, rt_ub)


# In[ ]:




