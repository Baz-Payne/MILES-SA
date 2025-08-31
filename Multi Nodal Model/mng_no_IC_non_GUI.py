### Multi Nodal Grid (Regional Split) ###



import pandas as pd
import pulp
from datetime import datetime
import time

solverlist = pulp.listSolvers(onlyAvailable=True)
print(solverlist)


def calendar_year_dates(year):
    '''
    Returns the start and end dates of a calendar year

    Parameters
    ----------
    year : int
        The calendar year starting e.g., CY2024 starts in 2024.

    Returns
    -------
    start_date : datetime
        The first day of the calendar year.
    end_date : datetime
        The last day of the calendar year.

    '''
    
    # Calendar year starts on January 1st
    start_date = datetime(year, 1, 1)
    
    # Calendar year ends on midnight of December 31st
    end_date = datetime(year, 12, 31, 23, 59, 59) 
    
    return start_date, end_date



### Read in Demand and Generation Files ###

# Mapping table file path which includes the required file paths and information
file_path = 'Mapping Tables.xlsx'

# Specify the mapping table sheet
sheet_name_mapping = 'Region Files'
sheet_name_IC = 'Interconnectors'
sheet_name_transmission = 'Transmission Limits'
sheet_name_DSP = 'DSP'
sheet_name_inter_demand = 'Interstate Net Demand'

# Read the in the mapping table and the DSP amounts
mapping_table = pd.read_excel(file_path, sheet_name=sheet_name_mapping)
DSP = pd.read_excel(file_path, sheet_name=sheet_name_DSP)

# Get the SA demand file names
sa_regions = mapping_table['Honours Region'].tolist()

# Initialise list with all regions
all_regions = sa_regions

# Initialise the directory that the SA demand and generation files are stored in
demand_dir = 'Demand/'
generation_dir = 'Generation/'

# Read in the demand and generation
demand = {}
generation = {}
solar_generation = {}
wind_generation = {}

for region in sa_regions:
    demand[region] = pd.read_csv(demand_dir + mapping_table.loc[mapping_table['Honours Region'] == region, 'Demand File'].iloc[0])
    demand[region]['Datetime'] = pd.to_datetime(demand[region]['Datetime'])
    demand[region].iloc[:,1] = demand[region].iloc[:,1] / 10**3
    

    # Read in the solar generation of the current SA region
    solar_generation[region] = pd.read_csv(generation_dir + mapping_table.loc[mapping_table['Honours Region'] == region, 'Solar Generation File'].iloc[0])
    solar_generation[region]['Datetime'] = pd.to_datetime(solar_generation[region]['Datetime'])
    solar_generation[region].iloc[:,1] = solar_generation[region].iloc[:,1] / 10**3
    
    # Read in the wind generation of the current SA region
    wind_generation[region] = pd.read_csv(generation_dir + mapping_table.loc[mapping_table['Honours Region'] == region, 'Wind Generation File'].iloc[0])
    wind_generation[region]['Datetime'] = pd.to_datetime(wind_generation[region]['Datetime'])
    wind_generation[region].iloc[:,1] = wind_generation[region].iloc[:,1] / 10**3
    

# Define a matrix that contains the transmission line limits between regions
transmission_matrix = pd.read_excel(file_path, sheet_name=sheet_name_transmission, index_col=0)

# Create a list of the identified net zero years
investigation_period = list(range(2045, 2052))

# Define time interval of AEMO's data
delta_t = 0.5
hours_in_day = int(24/delta_t)


# Create dataframe column headings
column_headings  = ['Type'] + investigation_period

# Create empty dataframes with the net zero years as the column headings for input totals
annual_demand = pd.DataFrame(columns=column_headings)
annual_solar = pd.DataFrame(columns=column_headings)
annual_wind = pd.DataFrame(columns=column_headings)
annual_DSP = pd.DataFrame(columns=column_headings)
annual_dispatchable = pd.DataFrame(columns=column_headings)


# Create empty dataframes with the net zero years as the column headings for output totals
annual_storage = pd.DataFrame(columns=column_headings)
annual_power = pd.DataFrame(columns=column_headings)
annual_curtailment = pd.DataFrame(columns=column_headings)
annual_charge = pd.DataFrame(columns=column_headings)
annual_discharge = pd.DataFrame(columns=column_headings)
annual_equivalent_cyles = pd.DataFrame(columns=column_headings)
annual_SoC_percentage = pd.DataFrame(columns=column_headings)
#annual_min_date = pd.DataFrame(columns=column_headings)
problem_solve_time = pd.DataFrame(columns=column_headings)
total_annual_storage = pd.DataFrame(columns=column_headings)
system_cost = pd.DataFrame(columns=column_headings)
annual_import = pd.DataFrame(columns=column_headings)
annual_export = pd.DataFrame(columns=column_headings)



# Define the types of storages
storage_types = ['shallow', 'medium', 'deep']

# Define the costs of capacity and power
C_e = {storage_types[0]: 0.33, storage_types[1]: 0.209, storage_types[2]: 0.171}
C_p = {storage_types[0]: 0.660, storage_types[1]: 1.672, storage_types[2]: 4.104}

        
# Assume no losses in the charging and discharging
eta_c = {storage_types[0]: 1, storage_types[1]: 1, storage_types[2]: 1}
eta_d = {storage_types[0]: 1, storage_types[1]: 1, storage_types[2]: 1}

# Define the amount of USE
USE_percent = 0.00002   # 0.002%

# Define the minimum SoC percentage allowed for each battery type in each region
E_min = {}
for r in sa_regions:
    E_min[r] = {storage_types[0]: 0, storage_types[1]: 0, storage_types[2]: 0}


# Define temporal constraints of each battery type
duration_constraints = {'shallow': 4, 'medium_min': 4, 'medium_max': 12, 'deep': 12}

M = 1e5     # A large number for Big-M method
k = 1e-8    # Scaling factor to ensure maximum SoC at each time interval
    
    
# Loop through the net renewable years
for year in investigation_period:
    
    # Obtain the start and end dates of the current calendar year
    start_date, end_date = calendar_year_dates(year)
    
    # Define the data for the yearly demand and generation
    D = {r: demand[r].loc[(demand[r]['Datetime'] >= start_date) & (demand[r]['Datetime'] <= end_date), demand[r].columns[:]] for r in sa_regions}
    CY_solar = {r: solar_generation[r].loc[(solar_generation[r]['Datetime'] >= start_date) & (solar_generation[r]['Datetime'] <= end_date), solar_generation[r].columns[:]] for r in sa_regions}
    CY_wind = {r: wind_generation[r].loc[(wind_generation[r]['Datetime'] >= start_date) & (wind_generation[r]['Datetime'] <= end_date), wind_generation[r].columns[:]] for r in sa_regions}
    G = {r: CY_solar[r].copy() for r in sa_regions}  # Copy solar data
    
    for r in sa_regions:
        G[r].iloc[:, 1] += CY_wind[r].iloc[:, 1]
    
    
    # Define the time array and the sampling interval
    T = len(D['Adelaide Metro'].iloc[:,1])
    
    # Initialise variable to keep track of the current day
    day = 1
    
    # DSP capacity
    DSP_capacity = DSP.loc[0, year] / 10**3
    
    # Daily DSP limit
    daily_DSP_limit = (DSP.loc[0, year] / 10 **3) * 2
    
    # Annual DSP limit
    annual_DSP_limit = DSP.loc[1, year] / 10**3
    
    # Annual USE calculation
    USE_limit = sum(D[n].iloc[t, 1] for n in sa_regions for t in range(T)) * delta_t * USE_percent
    
    
    # Define the problem (MNG = Multi Nodal Grid)
    mng = pulp.LpProblem("Minimise_Storage_Capacity_MNG", pulp.LpMinimize)
    
    
    
    # Define the decision variables
    E = {r: {b: pulp.LpVariable.dicts(f"E_{r}_{b}", range(T+1), lowBound=0) for b in storage_types} for r in sa_regions}
    E_max = {r: {b: pulp.LpVariable(f"E_max_{r}_{b}", lowBound=0) for b in storage_types} for r in sa_regions}
    P_max = {r: {b: pulp.LpVariable(f"P_max_{r}_{b}", lowBound=0) for b in storage_types} for r in sa_regions}
    P_chrg = {r: {b: pulp.LpVariable.dicts(f"P_chrg_{r}_{b}", range(T), lowBound=0) for b in storage_types} for r in sa_regions}
    P_disc = {r: {b: pulp.LpVariable.dicts(f"P_disc_{r}_{b}", range(T), lowBound=0) for b in storage_types} for r in sa_regions}
    #alpha = {r: {b: pulp.LpVariable.dicts(f"alpha_{r}_{b}", range(T), cat='Binary') for b in storage_types} for r in sa_regions}
    P_I = {(r1, r2, t): pulp.LpVariable(f"P_I_{r1}_{r2}_{t}", lowBound=0) for r1 in all_regions for r2 in all_regions for t in range(T)}
    P_curt = {r: pulp.LpVariable.dicts(f"P_curtail_{r}", range(T), lowBound=0) for r in sa_regions}
    P_DSP = {r: pulp.LpVariable.dicts(f"P_DSP_{r}", range(T), lowBound=0, upBound=DSP_capacity) for r in sa_regions}
    USE = {r: pulp.LpVariable.dicts(f"USE_{r}", range(T), lowBound=0, upBound=0.3) for r in sa_regions}
    
    
    ### Define the objective function ###
    mng += pulp.lpSum(C_e[b] * E_max[r][b] + C_p[b] * P_max[r][b] for r in sa_regions for b in storage_types) - k * pulp.lpSum(E[r][b][t] for r in sa_regions for b in storage_types for t in range(T))
    
    
    # Define the constraints which must hold over each time interval
    for n in sa_regions:
        for t in range(T):
            
                
            # Supply and demand balance constraint
            mng += (
                G[n].iloc[t, 1]
                + pulp.lpSum(P_disc[n][b][t] for b in storage_types)
                + pulp.lpSum(P_I[j, n, t] for j in all_regions if j != n) # incoming transmission
                - pulp.lpSum(P_I[n, j, t] for j in all_regions if j != n) # outgoing transmission
                - pulp.lpSum(P_chrg[n][b][t] for b in storage_types)
                - P_curt[n][t]
                + P_DSP[n][t]
                - D[n].iloc[t, 1]
                + USE[n][t]
                == 0
            )
            
            
            for b in storage_types:
                
                # Storage dynamics
                mng += E[n][b][t+1] == E[n][b][t] + (eta_c[b] * P_chrg[n][b][t] - P_disc[n][b][t] * (1 / eta_d[b]) ) * delta_t
                
                # Minimum and maximum SoC conditions
                mng += E[n][b][t] >= E_min[n][b]
                mng += E[n][b][t] <= E_max[n][b]
                
                # Define the maximum power of the storage
                mng += P_disc[n][b][t] <= P_max[n][b]
                mng += P_chrg[n][b][t] <= P_max[n][b]
             
            # Ensure that the battery cannot be charging and discharging at the same time
            #mng += P_disc[n][t] <= M * alpha[n][t]
            #mng += P_chrg[n][t] <= M * (1 - alpha[n][t])
            
                
            # Enforce transmission line limits
            for j in sa_regions:
                if n != j:
                    
                    # Check whether a connection between the two regions exists
                    if transmission_matrix.loc[n, j] > 0:
                        
                        # If so, ensure tranmission is constrained
                        mng += P_I[n, j, t] <= transmission_matrix.loc[n, j]
                    else:
                        
                        # Otherwise, there is no connection so ensure no transmission
                        mng += P_I[n, j, t] == 0
        
        
            # Add constraint for daily amount of DSP among all regions
            if n == sa_regions[0]:
                if t % hours_in_day == 0:
                    mng += pulp.lpSum(P_DSP[r][d] for r in sa_regions for d in range((day-1)*hours_in_day,day*hours_in_day))*delta_t <= daily_DSP_limit
                    day += 1
                    
                    
        # Define temporal constraints for each storage type
        if n in sa_regions:
            for b in storage_types:
        
                match b:
                    case 'shallow':
                        mng += E_max[n][b] <= duration_constraints[b] * P_max[n][b]
                    case 'medium':
                        mng += E_max[n][b] >= duration_constraints[b + '_min'] * P_max[n][b]
                        mng +=  E_max[n][b] <= duration_constraints[b + '_max'] * P_max[n][b]
                    case 'deep':
                        mng += E_max[n][b] >= duration_constraints[b] * P_max[n][b]
                      
                # Ensure the initial SoC for each storage type is 100%
                mng += E[n][b][0] == E_max[n][b]
    

    # Constraint to limit DSP amount in each year
    mng += 0.5 * pulp.lpSum(P_DSP[r][t] for r in sa_regions for t in range(T)) <= annual_DSP_limit
    
    # Limit the total USE to a defined percentage of total forecasted demand
    mng += delta_t * pulp.lpSum(USE[r][t] for r in sa_regions for t in range(T)) <= USE_limit
    
    
    # Start the timer
    start_time = time.time()
    
    
    # Solve the problem
    mng.solve(pulp.GUROBI_CMD(msg=False))
    
    # End timer
    elapsed_time = time.time() - start_time
    
    
    # Output results
    print("Status:", pulp.LpStatus[mng.status])
    total_storage = sum(pulp.value(E_max[r][b]) for r in sa_regions for b in storage_types)
    
    print("Minimum required storage capacity:", total_storage)
    
    # Print the elapsed time
    print("--- Solve Time {:.2f} seconds ---".format(elapsed_time))
    

    # Problem Solve time
    problem_solve_time.loc[0, year] = elapsed_time
    problem_solve_time.loc[0, 'Type'] = "Solve Time (s)"
    
    # Calculate total storage requirement
    annual_storage.loc['Total Storage', year] = total_storage
    annual_storage.loc['Total Storage', 'Type'] = "Total Storage Requirement (GWh)"
    
    
    # Total system cost
    #system_cost.loc[0, year] = cost
    #system_cost.loc[0, 'Type'] = 'System Cost ($Millions)'
    
    # Define CY traces as a dictionary as values are functions of regions
    CY_traces = {}
    CY_storage_traces = {}
    transmission_flow = {}
    
    # Create regional traces
    for n in all_regions:
        
        # Create the transmission line flow matrix for the current region
        region_flow = pd.DataFrame()
        region_flow.loc[:, 'Datetime'] = D['Adelaide Metro'].iloc[:,0]
        region_flow.reset_index(drop=True, inplace=True)
        
        # Loop through all regions except for the current region
        for j in all_regions:
            if n != j:
                
                current_flow = pd.DataFrame({
                    j + ' to ' + n: [(pulp.value(P_I[j, n, t]) - pulp.value(P_I[n, j, t])) for t in range(T)]})
                
                
                # Concatenate the current transmission flows to the region flow dataframe
                region_flow = pd.concat([region_flow, current_flow], axis=1)
            
        # Add the current region flows with all other regions to the transmission flow matrix
        transmission_flow[n] = region_flow
        
        
        # If it is current an SA region
        if n in sa_regions:
            
            # Store yearly regional inputs
            annual_demand.loc[n, year] = sum(D[n].iloc[t, 1] for t in range(T)) * delta_t / 10**3
            annual_demand.loc[n, 'Type'] = n + " Demand (TWh)"
            
            annual_solar.loc[n, year] = sum(CY_solar[n].iloc[t, 1] for t in range(T)) * delta_t / 10**3
            annual_solar.loc[n, 'Type'] = n + " Solar Generation (TWh)"
            
            annual_wind.loc[n, year] = sum(CY_wind[n].iloc[t, 1] for t in range(T)) * delta_t / 10**3
            annual_wind.loc[n, 'Type'] = n + " Wind Generation (TWh)"
            
            annual_DSP.loc[n, year] = sum(pulp.value(P_DSP[n][t]) for t in range(T)) * delta_t / 10**3
            annual_DSP.loc[n, 'Type'] = n + " DSP (TWh)"
            
            annual_dispatchable.loc[n, year] = annual_solar.loc[n, year] + annual_wind.loc[n, year] + annual_DSP.loc[n, year]
            annual_dispatchable.loc[n, 'Type'] = n + " Total Generation (TWh)"
            
            
            CY_traces[n] = pd.DataFrame({
                'Datetime': D[n].iloc[:,0],
                'Curtailment (MW)': [pulp.value(P_curt[n][t]) for t in range(T)],
                'DSP (MW)': [pulp.value(P_DSP[n][t]) for t in range(T)]})
            
            CY_curtailment = sum(pulp.value(P_curt[n][t]) for t in range(T)) * delta_t
            annual_curtailment.loc[n, year] = CY_curtailment
            annual_curtailment.loc[n, 'Type'] = n + " Curtailment (GWh)"
            
            CY_storage_traces[n] = {}
            
            # Calculate CY totals for each SA region and each storage type
            for b in storage_types:
                
                
                CY_storage = pulp.value(E_max[n][b])
                CY_charge_discharge_trace = [pulp.value(P_disc[n][b][t] - P_chrg[n][b][t]) for t in range(T)]
                
                
                CY_discharge = sum(x for x in CY_charge_discharge_trace if x > 0) * delta_t
                CY_charge = -sum(x for x in CY_charge_discharge_trace if x < 0) * delta_t
                
                # Account for the zero storage scenario
                if CY_storage == 0:
                    CY_equivalent_cycles = 0
                    CY_SoC_percentage = 0
                else:
                    
                    CY_equivalent_cycles = ((CY_charge/CY_storage) + (CY_discharge/CY_storage)) / 2
                    CY_SoC_percentage = sum(pulp.value(E[n][b][t]) for t in range(T)) / (T * pulp.value(E_max[n][b])) * 100
                    
                 
                
            
                # Add CY totals to the dataframes
                annual_storage.loc[n + " " + b, year] = CY_storage
                annual_storage.loc[n + " " + b, 'Type'] = n + " " + b + " Storage Requirement (GWh)"
                
                annual_power.loc[n + " " + b, year] = pulp.value(P_max[n][b])
                annual_power.loc[n + " " + b, 'Type'] = n + " " + b + " Power Requirement (GW)"
                
                annual_discharge.loc[n + " " + b, year] = CY_discharge
                annual_discharge.loc[n + " " + b, 'Type'] = n + " " + b + " Discharge Amount (GWh)"
                
                annual_charge.loc[n + " " + b, year] = CY_charge
                annual_charge.loc[n + " " + b, 'Type'] = n + " " + b + " Charge Amount (GWh)"
                
                annual_equivalent_cyles.loc[n + " " + b,year] = CY_equivalent_cycles
                annual_equivalent_cyles.loc[n + " " + b, 'Type'] = n + " " + b + " Equivalent Cycles"
                
                annual_SoC_percentage.loc[n + " " + b,year] = CY_SoC_percentage
                annual_SoC_percentage.loc[n + " " + b, 'Type'] = n + " " + b + " Average Annual SoC (%)"
            
            
                # Create the current CY storage traces
                CY_storage_traces[n][b] = pd.DataFrame({
                    'Datetime': D[n].iloc[:,0],
                    b + ' SoC (GWh)': [pulp.value(E[n][b][t]) for t in range(T)],
                    b + ' Charge/Disharge (MW)': CY_charge_discharge_trace})
            
        
     
            
    
    # Combine the annual input and output totals
    annual_input_totals = pd.concat([annual_demand, annual_solar, annual_wind, annual_DSP, annual_dispatchable], ignore_index=True)
    annual_output_totals = pd.concat([annual_storage, total_annual_storage, annual_power, annual_curtailment,
                                      annual_charge, annual_discharge, annual_equivalent_cyles,
                                      annual_SoC_percentage, annual_import, annual_export, problem_solve_time, system_cost],ignore_index=True)
    

    
    
    # Save the current yearly traces
    for n in all_regions:
        
        # Save annual traces to CSV files
        if n in sa_regions:
            for b in storage_types:
                CY_storage_traces[n][b].to_csv(f'Results/{year}_{n}_{b}_Traces.csv', index=False)
            
        CY_traces[n].to_csv(f'Results/{year}_{n}_Traces.csv', index=False)
        transmission_flow[n].to_csv(f'Results/{year}_{n}_Xmission_Traces.csv', index=False)
            
    
# Save the annual input and output totals
with pd.ExcelWriter('Results/Annual Totals No IC.xlsx') as writer:
    annual_input_totals.to_excel(writer, sheet_name='Annual Input Totals', index=False)
    annual_output_totals.to_excel(writer, sheet_name='Annual Output Totals', index=False)
    

    




# https://www.mdpi.com/2071-1050/15/14/11400 Gave the idea of being able to directly curtail generation as a constraint rather than making generation fixed.
# That is, having generation as a dispatchable quantity rather than fixed
"""
            # Battery charge and discharge constraints for each storage type
            for b in storage_types:
                
                # Storage dynamics
                lp += E[b][t+1] == E[b][t] + (eta_c[b] * P_chrg[b][t] - P_disc[b][t] * (1 / eta_d[b]) ) * delta_t
                
                # Minimum and maximum SoC conditions
                lp += E[b][t] >= 0
                lp += E[b][t] <= E_max[b]
                
                # Define the maximum power of the storage
                lp += P_disc[b][t] <= P_max[b]
                lp += P_chrg[b][t] <= P_max[b]
            
                # Ensure that the battery cannot be charging and discharging at the same time
                #lp += P_disc[b][t] <= M * alpha[b][t]
                #lp += P_chrg[b][t] <= M * (1 - alpha[b][t])
         
            
        # Define non time dependent constraints
        
        
       
        for b in storage_types:
    
            # Define temporal constraints for each storage type
    
            # For pypy
            if b == 'shallow':
                lp += E_max[b] <= duration_constraints[b] * P_max[b]
            elif b == 'medium':
                lp += E_max[b] >= duration_constraints['medium_min'] * P_max[b]
                lp +=  E_max[b] <= duration_constraints['medium_max'] * P_max[b]
            elif b == 'deep':
                lp += E_max[b] >= duration_constraints[b] * P_max[b]
            
            match b:
                case 'shallow':
                    lp += E_max[b] <= duration_constraints[b] * P_max[b]
                case 'medium':
                    lp += E_max[b] >= duration_constraints['medium_min'] * P_max[b]
                    lp +=  E_max[b] <= duration_constraints['medium_max'] * P_max[b]
                case 'deep':
                    lp += E_max[b] >= duration_constraints[b] * P_max[b]
        
        # Ensure the initial SoC for each storage type is 100%
        #lp += E[n][0] == E_max[n]
    
        """ 
