### Copper Plate Grid ###

"""
The mixed integer linear program (MILP) is complex and can take many hours/day to run.
To decrease the runtime, remove the binary constraints. While this means the trace data will not be 
accurate, the overall storage requirement and cost will be the same as the MILP.

Results may vary depending on the solver used. The initial run was completed using Gurobi 11.0.3.

Changing other variables such as the charging/discharging efficiencies to 1 can also reduce complexity.
"""

### Note: all trace data is converted to GWs as this decreases computation time ###

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

# Intialise interstate regions
interstate_regions = pd.read_excel(file_path, sheet_name=sheet_name_IC)
interstate_regions = interstate_regions['Interconnectors'].tolist()

# Read in the net demand traces for NSW and Victoria
inter_demand = pd.read_excel(file_path, sheet_name=sheet_name_inter_demand)
net_demand_NSW = pd.read_csv(inter_demand.loc[inter_demand['Interstate Region'] == 'NSW', 'File Name'].iloc[0])
net_demand_NSW['Datetime'] = pd.to_datetime(net_demand_NSW['Datetime'])
net_demand_NSW.iloc[:,1] = net_demand_NSW.iloc[:,1] / 10**3

net_demand_VIC = pd.read_csv(inter_demand.loc[inter_demand['Interstate Region'] == 'VIC', 'File Name'].iloc[0])
net_demand_VIC['Datetime'] = pd.to_datetime(net_demand_VIC['Datetime'])
net_demand_VIC.iloc[:,1] = net_demand_VIC.iloc[:,1] / 10**3


# Initialise the directory that the SA demand and generation files are stored in
demand_dir = '../Data/SA Data/Demand/Copper Plate/'
generation_dir = '../Data/SA Data/Generation/Copper Plate/'

# Read in the demand and generation
demand = {}
generation = {}
solar_generation = {}
wind_generation = {}


# Read in demand and convert to GW
demand = pd.read_csv(demand_dir + mapping_table.loc[mapping_table['Honours Region'] == "SA", 'Demand File'].iloc[0])
demand['Datetime'] = pd.to_datetime(demand['Datetime'])
demand.iloc[:,1] = demand.iloc[:,1] / 10**3

# Read in the solar generation of SA and convert to GW
solar_generation = pd.read_csv(generation_dir + mapping_table.loc[mapping_table['Honours Region'] == "SA", 'Solar Generation File'].iloc[0])
solar_generation['Datetime'] = pd.to_datetime(solar_generation['Datetime'])
solar_generation.iloc[:,1] = solar_generation.iloc[:,1] / 10**3

# Read in the wind generation of SA and convert to GW
wind_generation = pd.read_csv(generation_dir + mapping_table.loc[mapping_table['Honours Region'] == "SA", 'Wind Generation File'].iloc[0])
wind_generation['Datetime'] = pd.to_datetime(wind_generation['Datetime'])
wind_generation.iloc[:,1] = wind_generation.iloc[:,1] / 10**3


# Define a matrix that contains the transmission line limits between regions
transmission_matrix = pd.read_excel(file_path, sheet_name=sheet_name_transmission, index_col=0)
transmission_matrix = transmission_matrix.astype(float)
transmission_matrix = transmission_matrix / 10**3   # Convert to GW


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
total_annual_power = pd.DataFrame(columns=column_headings)
total_annual_storage = pd.DataFrame(columns=column_headings)
problem_solve_time = pd.DataFrame(columns=column_headings)
system_cost = pd.DataFrame(columns=column_headings)
annual_import = pd.DataFrame(columns=column_headings)
annual_export = pd.DataFrame(columns=column_headings)

# Define the types of storages
storage_types = ['shallow', 'medium', 'deep']

# Define the costs of capacity and power. Decreased by a factor of 1e-9 to help with computation time
C_e = {storage_types[0]: 0.33, storage_types[1]: 0.209, storage_types[2]: 0.171}
C_p = {storage_types[0]: 0.66, storage_types[1]: 1.672, storage_types[2]: 4.104}
        
# Assume losses in the charging and discharging
eta_c = {storage_types[0]: 0.92, storage_types[1]: 0.925, storage_types[2]: 0.92}
eta_d = {storage_types[0]: 0.92, storage_types[1]: 0.925, storage_types[2]: 0.92}

# Define USE parameters
USE_percent = 0.00002   # 0.002%
USE_capacity = 0.3      # 300MW

# Define the minimum SoC percentage allowed for each battery type
E_min = {storage_types[0]: 0, storage_types[1]: 0, storage_types[2]: 0}


# Define temporal constraints of each battery type
duration_constraints = {'shallow': 4, 'medium_min': 4, 'medium_max': 12, 'deep': 12}

M = 1e5     # A large number for Big-M method
k = 1e-8    # Scaling factor to ensure maximum SoC at each time interval

    
# Loop through the net renewable years
for year in investigation_period:
    
    start_time = time.time()
    
    # Obtain the start and end dates of the current calendar year
    start_date, end_date = calendar_year_dates(year)
    
    # Define the data for the yearly demand and generation
    D = demand.loc[(demand['Datetime'] >= start_date) & (demand['Datetime'] <= end_date), demand.columns[:]]
    CY_solar = solar_generation.loc[(solar_generation['Datetime'] >= start_date) & (solar_generation['Datetime'] <= end_date), solar_generation.columns[:]]
    CY_wind = wind_generation.loc[(wind_generation['Datetime'] >= start_date) & (wind_generation['Datetime'] <= end_date), wind_generation.columns[:]]
    G = CY_solar.copy()  # Copy solar data
    
    # Create the generation trace which consists of both solar and wind data
    G.iloc[:, 1] += CY_wind.iloc[:, 1]
    
       
    # Define current annual net demand traces for NSW and Victoria
    CY_net_demand_NSW = net_demand_NSW.loc[(net_demand_NSW['Datetime'] >= start_date) & (net_demand_NSW['Datetime'] <= end_date), net_demand_NSW.columns[:]]
    CY_net_demand_VIC = net_demand_VIC.loc[(net_demand_VIC['Datetime'] >= start_date) & (net_demand_VIC['Datetime'] <= end_date), net_demand_VIC.columns[:]]

    # Define the time array and the sampling interval
    T = len(D.iloc[:,1])    
    
    # Initialise variable to keep track of the current day
    day = 1
    
    # DSP capacity
    DSP_capacity = DSP.loc[0, year] / 10**3
    
    # Daily DSP limit
    daily_DSP_limit = (DSP.loc[0, year] / 10 **3) * 2
    
    # Annual DSP limit
    annual_DSP_limit = DSP.loc[1, year] / 10**3
    
    # Annual USE calculation
    USE_limit = sum(D.iloc[t, 1] for t in range(T)) * delta_t * USE_percent
    
    
    # Define the problem (MNG = Multi Nodal Grid)
    cpg = pulp.LpProblem("Minimise_Storage_Capacity_CPG", pulp.LpMinimize)
    
    
    
    # Define the decision variables
    E = {b: pulp.LpVariable.dicts(f"E_{b}", range(T+1), lowBound=0) for b in storage_types}
    E_max = {b: pulp.LpVariable(f"E_max_{b}", lowBound=0) for b in storage_types}
    P_max = {b: pulp.LpVariable(f"P_max_{b}", lowBound=0) for b in storage_types}
    P_chrg = {b: pulp.LpVariable.dicts(f"P_chrg_{b}", range(T), lowBound=0) for b in storage_types}
    P_disc = {b: pulp.LpVariable.dicts(f"P_disc_{b}", range(T), lowBound=0) for b in storage_types}
    alpha = {b: pulp.LpVariable.dicts(f"alpha_{b}", range(T), cat='Binary') for b in storage_types}
    P_impt = {s: pulp.LpVariable.dicts(f"P_impt_{s}", range(T), lowBound=0) for s in interstate_regions}
    P_expt = {s: pulp.LpVariable.dicts(f"P_expt_{s}", range(T), lowBound=0) for s in interstate_regions}
    P_curt = pulp.LpVariable.dicts("P_curtail", range(T), lowBound=0)
    P_DSP = pulp.LpVariable.dicts("P_DSP", range(T), lowBound=0, upBound=DSP_capacity)
    USE = pulp.LpVariable.dicts("USE", range(T), lowBound=0, upBound=USE_capacity)
    
    
    ### Define the objective function ###
    cpg += pulp.lpSum(C_e[b] * E_max[b] + C_p[b] * P_max[b] for b in storage_types) - k * pulp.lpSum(E[b][t] for b in storage_types for t in range(T))
    
    
    # Define the contraints which must hold over each time interval        
    for t in range(T):
        
            
        # Supply and demand balance constraint
        cpg += (
            G.iloc[t, 1]
            + pulp.lpSum(P_disc[b][t] for b in storage_types)
            - pulp.lpSum(P_chrg[b][t] for b in storage_types)
            - P_curt[t]
            + P_DSP[t]
            - D.iloc[t, 1]
            + pulp.lpSum(P_expt[s][t] for s in interstate_regions)
            + USE[t]
            == 0
        )
            
            
        for b in storage_types:
            
            # Storage dynamics
            cpg += E[b][t+1] == E[b][t] + (eta_c[b] * P_chrg[b][t] - P_disc[b][t] * (1 / eta_d[b]) ) * delta_t
            
            # Minimum and maximum SoC conditions
            cpg += E[b][t] >= E_min[b]
            cpg += E[b][t] <= E_max[b]
            
            # Define the maximum power of the storage
            cpg += P_disc[b][t] <= P_max[b]
            cpg += P_chrg[b][t] <= P_max[b]
            
            
            # Ensure that the battery cannot be charging and discharging at the same time
            cpg += P_disc[b][t] <= M * alpha[b][t]
            cpg += P_chrg[b][t] <= M * (1 - alpha[b][t])
                
        
        # Add constraint for daily amount of DSP
        if t % hours_in_day == 0:
            cpg += pulp.lpSum(P_DSP[d] for d in range((day-1)*hours_in_day,day*hours_in_day))*delta_t <= daily_DSP_limit
            day += 1
            
        
        # Check whether there is generation available
        if CY_net_demand_NSW.iloc[t,1] < 0:
            
            # Ensure imports do not exceed the available generation and transmission rating
            cpg += P_expt['PEC'][t] <= -CY_net_demand_NSW.iloc[t,1]
            cpg += P_expt['PEC'][t] <= transmission_matrix.loc['PEC'].max()
            
        # Otherwise, there is no available interstate generation so set export values to 0
        else:
            cpg += P_expt['PEC'][t] == 0
            
        
        # Check whether there is generation available
        if CY_net_demand_VIC.iloc[t,1] < 0:
            
            # Ensure imports do not exceed the available generation and transmission rating
            cpg += P_expt['Murraylink'][t] + P_expt['Heywood'][t] <= -CY_net_demand_VIC.iloc[t,1]
            cpg += P_expt['Murraylink'][t] <= transmission_matrix.loc['Murraylink'].max()
            cpg += P_expt['Heywood'][t] <= transmission_matrix.loc['Heywood'].max()
            
        # Otherwise, there is no available interstate generation so set export values to 0
        else:
            cpg += P_expt['Murraylink'][t] == 0
            cpg += P_expt['Heywood'][t] == 0
    
    
    # Define temporal constraints for each storage type
    for b in storage_types:

        match b:
            case 'shallow':
                cpg += E_max[b] <= duration_constraints[b] * P_max[b]
            case 'medium':
                cpg += E_max[b] >= duration_constraints[b + '_min'] * P_max[b]
                cpg +=  E_max[b] <= duration_constraints[b + '_max'] * P_max[b]
            case 'deep':
                cpg += E_max[b] >= duration_constraints[b] * P_max[b]
              
        # Ensure the initial SoC for each storage type is 100%
        cpg += E[b][0] == E_max[b]


    # Condition to limit DSP amount in each year
    cpg += delta_t * pulp.lpSum(P_DSP[t] for t in range(T)) <= annual_DSP_limit
 
    # Limit the total USE to a defined percentage of total forecasted demand
    cpg += delta_t * pulp.lpSum(USE[t] for t in range(T)) <= USE_limit
    
    
    
    # Record setup time
    setup_time = time.time() - start_time
    
    print("--- Setup Time {:.2f} seconds ---".format(setup_time))

    # Start the timer
    start_time = time.time()
    
    # Solve the problem
    cpg.solve(pulp.GUROBI_CMD(msg=False)) # Change this to your solver of choice


    # End timer
    elapsed_time = time.time() - start_time
    
    
    # Output results
    print("Status:", pulp.LpStatus[cpg.status])
    total_power = sum(pulp.value(P_max[b]) for b in storage_types)
    total_storage = sum(pulp.value(E_max[b]) for b in storage_types)
    print("Storage Requirement: {:.3f} GW / {:.1f} GWh".format(total_power, total_storage))
    
    # Calculate system cost in billions
    cost = sum(C_e[b] * 1e9 * pulp.value(E_max[b]) + C_p[b] * 1e9 * pulp.value(P_max[b]) for b in storage_types) / 10**9
    print("Estimated System Cost: ${:.1f} billion".format(cost))
        
        
    # Print the elapsed time
    print("--- Solve Time {:.2f} seconds ---".format(elapsed_time))
    
    
    ### Store Data ###
    
    # Problem Solve time
    problem_solve_time.loc[0, year] = elapsed_time
    problem_solve_time.loc[0, 'Type'] = "Solve Time (s)"
    
    # System Power and Storage
    total_annual_power.loc['Total Power', year] = total_power
    total_annual_power.loc['Total Power', 'Type'] = "Total Power Requirement (GW)"
    
    total_annual_storage.loc['Total Storage', year] = total_storage
    total_annual_storage.loc['Total Storage', 'Type'] = "Total Storage Requirement (GWh)"
    
    # System cost
    system_cost.loc[0, year] = cost
    system_cost.loc[0, 'Type'] = 'System Cost ($Billions)'
    
    # Define CY traces as a dictionary as values are functions of regions
    CY_traces = {}
    CY_storage_traces = {}
    
        
    # Store yearly regional inputs
    annual_demand.loc[0, year] = sum(D.iloc[t, 1] for t in range(T)) * delta_t / 10**3
    annual_demand.loc[0, 'Type'] = "SA Demand (TWh)"
    
    annual_solar.loc[0, year] = sum(CY_solar.iloc[t, 1] for t in range(T)) * delta_t / 10**3
    annual_solar.loc[0, 'Type'] = "SA Solar Generation (TWh)"
    
    annual_wind.loc[0, year] = sum(CY_wind.iloc[t, 1] for t in range(T)) * delta_t / 10**3
    annual_wind.loc[0, 'Type'] = "SA Wind Generation (TWh)"
    
    annual_DSP.loc[0, year] = sum(pulp.value(P_DSP[t]) for t in range(T)) * delta_t / 10**3
    annual_DSP.loc[0, 'Type'] = "SA DSP (TWh)"
    
    annual_dispatchable.loc[0, year] = annual_solar.iloc[0][year] + annual_wind.iloc[0][year] + annual_DSP.iloc[0][year]
    annual_dispatchable.loc[0, 'Type'] = "SA Total Generation (TWh)"
    
    
    CY_traces = pd.DataFrame({
        'Datetime': D.iloc[:,0],
        'Curtailment (GW)': [pulp.value(P_curt[t]) for t in range(T)],
        'DSP (GW)': [pulp.value(P_DSP[t]) for t in range(T)]})
    
    CY_curtailment = sum(pulp.value(P_curt[t]) for t in range(T)) * delta_t / 10**3
    annual_curtailment.loc[0, year] = CY_curtailment
    annual_curtailment.loc[0, 'Type'] = "SA Curtailment (TWh)"
    
    CY_storage_traces = {}
    
    
    # Calculate CY totals for each SA region and each storage type
    for b in storage_types:
        
        
        CY_storage = pulp.value(E_max[b])
        CY_charge_discharge_trace = [pulp.value(P_disc[b][t] - P_chrg[b][t]) for t in range(T)]
        
        
        CY_discharge = sum(x for x in CY_charge_discharge_trace if x > 0) * delta_t
        CY_charge = -sum(x for x in CY_charge_discharge_trace if x < 0) * delta_t
        
        # Account for the zero storage scenario
        if CY_storage == 0:
            CY_equivalent_cycles = 0
            CY_SoC_percentage = 0
        else:
            
            CY_equivalent_cycles = ((CY_charge/CY_storage) + (CY_discharge/CY_storage)) / 2
            CY_SoC_percentage = sum(pulp.value(E[b][t]) for t in range(T)) / (T * pulp.value(E_max[b])) * 100
            
         
        
    
        # Add CY totals to the dataframes
        annual_storage.loc[0, year] = CY_storage
        annual_storage.loc[0, 'Type'] = "SA " + b + " Storage Requirement (GWh)"
        
        annual_power.loc[0, year] = pulp.value(P_max[b])
        annual_power.loc[0, 'Type'] = "SA " + b + " Power Requirement (GW)"
        
        annual_discharge.loc[0, year] = CY_discharge
        annual_discharge.loc[0, 'Type'] = "SA " + b + " Discharge Amount (GWh)"
        
        annual_charge.loc[0, year] = CY_charge
        annual_charge.loc[0, 'Type'] = "SA " + b + " Charge Amount (GWh)"
        
        annual_equivalent_cyles.loc[0, year] = CY_equivalent_cycles
        annual_equivalent_cyles.loc[0, 'Type'] = "SA " + b + " Equivalent Cycles"
        
        annual_SoC_percentage.loc[0, year] = CY_SoC_percentage
        annual_SoC_percentage.loc[0, 'Type'] = "SA " + b + " Average Annual SoC (%)"
    
    
        # Create the current CY storage traces
        CY_storage_traces[b] = pd.DataFrame({
            'Datetime': D.iloc[:,0],
            b + ' SoC (GWh)': [pulp.value(E[b][t]) for t in range(T)],
            b + ' Charge/Disharge (GW)': CY_charge_discharge_trace})
        

    # Combine the annual input and output totals
    annual_input_totals = pd.concat([annual_demand, annual_solar, annual_wind, annual_DSP, annual_dispatchable], ignore_index=True)
    annual_output_totals = pd.concat([total_annual_power, total_annual_storage, annual_storage, annual_power, annual_curtailment,
                                      annual_charge, annual_discharge, annual_equivalent_cyles,
                                      annual_SoC_percentage, annual_import, annual_export, problem_solve_time, system_cost],ignore_index=True)
    

    
    

    for b in storage_types:
        CY_storage_traces[b].to_csv(f'Results/{year}_{b}_Traces.csv', index=False)
            
    CY_traces.to_csv(f'Results/{year}_Traces.csv', index=False)
            
    
# Save the annual input and output totals
with pd.ExcelWriter('Results/Annual Totals.xlsx') as writer:
    annual_input_totals.to_excel(writer, sheet_name='Annual Input Totals', index=False)
    annual_output_totals.to_excel(writer, sheet_name='Annual Output Totals', index=False)