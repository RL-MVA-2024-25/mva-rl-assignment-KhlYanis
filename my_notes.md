6 state variables :
 - 'T1' : number of healthy type 1 cells (CD4+ T-lymphocytes),
 - 'T1star' : number of infected type 1 cells,
 - 'T2' : number of healthy type 2 cells (macrophages),
 - 'T2star' : number of infected type 2 cells,
 - 'V' : number of free virus particles,
 - 'E' : number of HIV-specific cytotoxic cells (CD8 T-lymphocytes).
These state variables are observed every 5 days (one time step) .


The physician can prescribe two types of drugs :
- Reverse transcriptase inhibitors, which prevent the virus from modifying an infected cell's DNA,
- Protease inhibitors, which prevent the cell replication mechanism.

Four actions : 
 - Prescribe nothing, 
 - Prescribe drug 1 or drug 2,
 - Prescribe both of them

  
Few notes from "[Clinical data based optimal STI strategies for HIV : a reinforcement learning approach]" :

- Structured treatment interruption (STI) --> Patients are cycled on and off drug therapy.
- During interruptions, viral load set points rebound to a high level, consequently activating adaptive immune response.

----------------------------------------------------------------
Test FQI algorithm : (with Random Forest Regressor)

Epsilon-greedy agent : 
- eps = 0.05 : score_agent = 1253411059.5969176/ score_agent_dr = 1777427719.8573072

- eps = 0.1 : score_agent = 881093669.7884636 / score_agent_dr = 799282750.1302704

- eps = 0.2 : score_agent = 315194145.1887892 / score_agent_dr = 187213142.82501233


-------------------------------------
FQI algorithm :  (with XGBRegressor)

Epsilon-greedy agent :
- eps = 0.00 : score_agent = 38405883.87070314 / score_agent_dr = 2331123042.7671223

-eps = 0.01 : score_agent = 373591658.546729 / score_agent_dr = 2188135258.0266795 

- eps = 0.05 : score_agent = 2311786206.8745704 / score_agent_dr = 1519082989.5467958

- eps = 0.1 : score_agent = 2287029283.724952 / score_agent_dr = 988232537.3240012


Few observations : Small values of epsilon tend to lower the performance on the score_agent, while increasing the performance on the score_agent_dr.

Conclusion : Optimal value is epsilon = 0.05 (with XGBRegressor)

Best result obtained with FQI : 3/9 

----------------------------------------------------------------
FQI with incremental (30 samples) data collection : (Agent eps-greedy with epsilon = 0.05 + XGB)

Epsilon-greedy on the exploration strategy :
- eps = 0.15 : score_agent = 7317450267.024633 / score_agent_dr = 4255289969.7059245

- eps = 0.05 : score_agent = 3065469664.357376 / score_agent_dr = 1743873664.3156404

- eps = 0.10 : score_agent = 3680832976.3862076 / score_agent_dr = 3543244976.7507854

Best result obtained with this method : 3/9 

----------------------------------------------------------------
Test DQN (Deep Q-Networks) algorithm :

