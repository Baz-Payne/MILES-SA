### Multi Nodal Grid (Region Split + Interconnector) ###



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

# Intialise interstate regions
interstate_regions = pd.read_excel(file_path, sheet_name=sheet_name_IC)
interstate_regions = interstate_regions['Interconnectors'].tolist()

# Initialise list with all regions
all_regions = sa_regions + interstate_regions

# Read in the net demand traces for NSW and Victoria
inter_demand = pd.read_excel(file_path, sheet_name=sheet_name_inter_demand)
net_demand_NSW = pd.read_csv(inter_demand.loc[inter_demand['Interstate Region'] == 'NSW', 'File Name'].iloc[0])
net_demand_NSW['Datetime'] = pd.to_datetime(net_demand_NSW['Datetime'])
net_demand_VIC = pd.read_csv(inter_demand.loc[inter_demand['Interstate Region'] == 'VIC', 'File Name'].iloc[0])
net_demand_VIC['Datetime'] = pd.to_datetime(net_demand_VIC['Datetime'])

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
    
    #generation[region] = pd.read_csv(generation_dir + mapping_table.loc[mapping_table['Honours Region'] == region, 'Generation File'].iloc[0])
    #generation[region]['Datetime'] = pd.to_datetime(generation[region]['Datetime'])

    # Read in the solar generation of the current SA region
    solar_generation[region] = pd.read_csv(generation_dir + mapping_table.loc[mapping_table['Honours Region'] == region, 'Solar Generation File'].iloc[0])
    solar_generation[region]['Datetime'] = pd.to_datetime(solar_generation[region]['Datetime'])
    
    # Read in the wind generation of the current SA region
    wind_generation[region] = pd.read_csv(generation_dir + mapping_table.loc[mapping_table['Honours Region'] == region, 'Wind Generation File'].iloc[0])
    wind_generation[region]['Datetime'] = pd.to_datetime(wind_generation[region]['Datetime'])

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
#C_e = {storage_types[0]: 330 * 1e3,  storage_types[1]: 209 * 1e3, storage_types[2]: 171 * 1e3}
#C_p = {storage_types[0]: 660 * 1e3, storage_types[1]: 1672 * 1e3, storage_types[2]: 4104 * 1e3}
C_e = 1
C_p = 1
        
# Assume no losses in the charging and discharging
#eta_c = {storage_types[0]: 1, storage_types[1]: 1, storage_types[2]: 1}
#eta_d = {storage_types[0]: 1, storage_types[1]: 1, storage_types[2]: 1}
eta_c = 1
eta_d = 1

# Define the minimum SoC percentage allowed for each battery type in each region
E_min = {}
for r in sa_regions:
    E_min[r] = 0
    #E_min[r] = {storage_types[0]: 0, storage_types[1]: 0, storage_types[2]: 0}


# Define temporal constraints of each battery type
duration_constraints = {'shallow': 4, 'medium_min': 4, 'medium_max': 12, 'deep': 12}

M = 1e5  # A large number for Big-M method
k = 1e-8   # Scaling factor to ensure maximum SoC at each time interval
q = 1e-5
    
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
    
    #G = {r: generation[r].loc[(generation[r]['Datetime'] >= start_date) & (generation[r]['Datetime'] <= end_date), generation[r].columns[:]] for r in sa_regions}
    #D = demand['Adelaide Metro'].loc[(demand['Adelaide Metro']['Datetime'] >= start_date) & (demand['Adelaide Metro']['Datetime'] <= end_date), demand['Adelaide Metro'].columns[:]]
    #G = generation['Adelaide Metro'].loc[(generation['Adelaide Metro']['Datetime'] >= start_date) & (generation['Adelaide Metro']['Datetime'] <= end_date), generation['Adelaide Metro'].columns[:]]
      
    # Define current annual net demand traces for NSW and Victoria
    CY_net_demand_NSW = net_demand_NSW.loc[(net_demand_NSW['Datetime'] >= start_date) & (net_demand_NSW['Datetime'] <= end_date), net_demand_NSW.columns[:]]
    CY_net_demand_VIC = net_demand_VIC.loc[(net_demand_VIC['Datetime'] >= start_date) & (net_demand_VIC['Datetime'] <= end_date), net_demand_VIC.columns[:]]

    # Define the time array and the sampling interval
    T = len(D['Adelaide Metro'].iloc[:,1])
    
    # Initialise variable to keep track of the current day
    day = 1
    
    
    # Define the problem (MNG = Multi Nodal Grid)
    mng = pulp.LpProblem("Minimise_Storage_Capacity_MNG", pulp.LpMinimize)
    
    
    
    # Define the decision variables
    # E[r][b][t]
    #E = {r: {b: pulp.LpVariable.dicts(f"E_{r}_{b}", range(T+1), lowBound=0) for b in storage_types} for r in regions}
    
    E = {r: pulp.LpVariable.dicts(f"E_{r}", range(T+1), lowBound=0) for r in sa_regions}
    E_max = {r: pulp.LpVariable(f"E_max_{r}", lowBound=0) for r in sa_regions}
    P_max = {r: pulp.LpVariable(f"P_max_{r}", lowBound=0) for r in sa_regions}
    P_chrg = {r: pulp.LpVariable.dicts(f"P_chrg_{r}", range(T), lowBound=0) for r in sa_regions}
    P_disc = {r: pulp.LpVariable.dicts(f"P_disc_{r}", range(T), lowBound=0) for r in sa_regions}
    alpha = {r: pulp.LpVariable.dicts(f"alpha_{r}", range(T), cat='Binary') for r in sa_regions}
    P_I = {(r1, r2, t): pulp.LpVariable(f"P_I_{r1}_{r2}_{t}", lowBound=0) for r1 in all_regions for r2 in all_regions for t in range(T)}
    P_impt = {s: pulp.LpVariable.dicts(f"P_impt_{s}", range(T), lowBound=0) for s in interstate_regions}
    P_expt = {s: pulp.LpVariable.dicts(f"P_expt_{s}", range(T), lowBound=0) for s in interstate_regions}
    P_curt = {r: pulp.LpVariable.dicts(f"P_curtail_{r}", range(T), lowBound=0) for r in sa_regions}
    P_DSP = {r: pulp.LpVariable.dicts(f"P_DSP_{r}", range(T), lowBound=0, upBound=DSP.loc[0, year]) for r in sa_regions}
    
    
    
    ### Define the objective function ###
    mng += (
        pulp.lpSum(C_e * E_max[r] + C_p * P_max[r] for r in sa_regions)
        - k * pulp.lpSum(E[r][t] for r in sa_regions for t in range(T))
        + q * pulp.lpSum(P_curt[r][t] for r in sa_regions for t in range(T))
    )
    
    
    # Define the constraints which must hold over each time interval
    for n in all_regions:
        for t in range(T):
            
            # if we're at an SA region
            if n in sa_regions:
                
                # Supply and demand balance constraint for SA region
                mng += (
                    G[n].iloc[t, 1]
                    + P_disc[n][t]
                    + pulp.lpSum(P_I[j, n, t] for j in all_regions if j != n) # incoming transmission
                    - pulp.lpSum(P_I[n, j, t] for j in all_regions if j != n) # outgoing transmission
                    - P_chrg[n][t]
                    - P_curt[n][t]
                    + P_DSP[n][t]
                    - D[n].iloc[t, 1]
                    == 0
                )
                
                
                # Storage dynamics
                mng += E[n][t+1] == E[n][t] + (eta_c * P_chrg[n][t] - P_disc[n][t] * (1 / eta_d) ) * delta_t
                
                # Minimum and maximum SoC conditions
                mng += E[n][t] >= E_min[n]
                mng += E[n][t] <= E_max[n]
                
                # Define the maximum power of the storage
                mng += P_disc[n][t] <= P_max[n]
                mng += P_chrg[n][t] <= P_max[n]
                
                
                # Ensure that the battery cannot be charging and discharging at the same time
                #mng += P_disc[n][t] <= M * alpha[n][t]
                #mng += P_chrg[n][t] <= M * (1 - alpha[n][t])
            
            # Otherwise we're at an interstate node
            elif n in interstate_regions:
                
                # Supply and demand balance constraint for interstate region
                mng += (
                    + pulp.lpSum(P_I[j, n, t] for j in all_regions if j != n) # incoming transmission
                    - pulp.lpSum(P_I[n, j, t] for j in all_regions if j != n) # outgoing transmission
                    + P_expt[n][t] # Exports to SA
                    - P_impt[n][t] # Imports interstate
                    == 0
                )
            
                
            # Enforce transmission line limits
            for j in all_regions:
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
                    mng += pulp.lpSum(P_DSP[r][d] for r in sa_regions for d in range((day-1)*hours_in_day,day*hours_in_day))*delta_t <= DSP.loc[0, year] * 2
                    day += 1
               
    

    # Ensure that imports are limited by the availability of interstate generation
    for t in range(T):
                
        # Check whether there is generation available
        if CY_net_demand_NSW.iloc[t,1] < 0:
            
            # Ensure imports do not exceed the available generation
            mng += P_expt['PEC'][t] <= -CY_net_demand_NSW.iloc[t,1]
            
            
        # Otherwise, there is no available interstate generation so set export values to 0
        else:
            mng += P_expt['PEC'][t] == 0
            
        
        # Check whether there is generation available
        if CY_net_demand_VIC.iloc[t,1] < 0:
            
            # Ensure imports do not exceed the available generation
            mng += P_expt['Murraylink'][t] + P_expt['Heywood'][t] <= -CY_net_demand_VIC.iloc[t,1]
            
        # Otherwise, there is no available interstate generation so set export values to 0
        else:
            mng += P_expt['Murraylink'][t] == 0
            mng += P_expt['Heywood'][t] == 0

                
    
    # Condition to limit DSP amount in each year
    #mng += 0.5 * pulp.lpSum(P_DSP[r][t] for r in sa_regions for t in range(T)) == DSP.loc[1, year]
    mng += 0.5 * pulp.lpSum(P_DSP[r][t] for r in sa_regions for t in range(T)) <= DSP.loc[1, year]
 
    
    # Start the timer
    start_time = time.time()
    
    
    # Solve the problem
    mng.solve(pulp.GUROBI_CMD(msg=False))
    
    # End timer
    elapsed_time = time.time() - start_time
    
    
    
    
    
    # Output results
    print("Status:", pulp.LpStatus[mng.status])
    total_storage = sum(pulp.value(E_max[r]) for r in sa_regions) / 10 ** 3
    print("Minimum required storage capacity:", total_storage)
    
    # Print the elapsed time
    print("--- Solve Time {:.2f} seconds ---".format(elapsed_time))
    
        
    
    # Problem Solve time
    problem_solve_time.loc[0, year] = elapsed_time
    problem_solve_time.loc[0, 'Type'] = "Solve Time (s)"
    
    # Calculate total storage requirement
    annual_storage.loc['Total Storage', year] = total_storage
    annual_storage.loc['Total Storage', 'Type'] = "Total Storage Requirement (GWh)"
    
    # Define CY traces as a dictionary as values are functions of regions
    CY_traces = {}
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
                
                if (n in interstate_regions) or (j in interstate_regions):
                    # Create trace for the transmission entering the current node
                    current_flow = pd.DataFrame({
                        j + ' to ' + n: [(pulp.value(P_I[j, n, t]) - pulp.value(P_I[n, j, t])) for t in range(T)]})
                
                else:
                    # Create trace for the transmission entering the current node
                    current_flow = pd.DataFrame({
                        j + ' to ' + n: [pulp.value(P_I[j, n, t]) for t in range(T)]})
                
                # Concatenate the current transmission flows to the region flow dataframe
                region_flow = pd.concat([region_flow, current_flow], axis=1)
            
        # Add the current region flows with all other regions to the transmission flow matrix
        transmission_flow[n] = region_flow
        
        
        # If it is current an SA region
        if n in sa_regions:
            
            # Store yearly regional inputs
            annual_demand.loc[n, year] = sum(D[n].iloc[t, 1] for t in range(T)) * delta_t / 10**6
            annual_demand.loc[n, 'Type'] = n + " Demand (TWh)"
            
            annual_solar.loc[n, year] = sum(CY_solar[n].iloc[t, 1] for t in range(T)) * delta_t / 10**6
            annual_solar.loc[n, 'Type'] = n + " Solar Generation (TWh)"
            
            annual_wind.loc[n, year] = sum(CY_wind[n].iloc[t, 1] for t in range(T)) * delta_t / 10**6
            annual_wind.loc[n, 'Type'] = n + " Wind Generation (TWh)"
            
            annual_DSP.loc[n, year] = sum(pulp.value(P_DSP[n][t]) for t in range(T)) * delta_t / 10**6
            annual_DSP.loc[n, 'Type'] = n + " DSP (TWh)"
            
            annual_dispatchable.loc[n, year] = annual_solar.loc[n, year] + annual_wind.loc[n, year] + annual_DSP.loc[n, year]
            annual_dispatchable.loc[n, 'Type'] = n + " Total Generation (TWh)"
            
            
            
            # Calculate CY totals for each SA region
            CY_storage = pulp.value(E_max[n]) / 10 ** 3
            CY_discharge = sum(pulp.value(P_disc[n][t]) for t in range(T)) * delta_t / 10**3
            CY_charge = sum(pulp.value(P_chrg[n][t]) for t in range(T)) * delta_t / 10**3
            CY_curtailment = sum(pulp.value(P_curt[n][t]) for t in range(T)) * delta_t / 10**3
            CY_equivalent_cycles = ((CY_charge/CY_storage) + (CY_discharge/CY_storage)) / 2
            CY_SoC_percentage = sum(pulp.value(E[n][t]) for t in range(T)) / (T * pulp.value(E_max[n])) * 100
            
            
            
            # Add CY totals to the dataframes
            annual_storage.loc[n, year] = CY_storage
            annual_storage.loc[n, 'Type'] = n + " Storage Requirement (GWh)"
            
            annual_power.loc[n, year] = pulp.value(P_max[n]) / 10**3
            annual_power.loc[n, 'Type'] = n + " Power Requirement (GW)"
            
            annual_discharge.loc[n, year] = CY_discharge
            annual_discharge.loc[n, 'Type'] = n + " Discharge Amount (GWh)"
            
            annual_charge.loc[n, year] = CY_charge
            annual_charge.loc[n, 'Type'] = n + " Charge Amount (GWh)"
            
            annual_curtailment.loc[n, year] = CY_curtailment
            annual_curtailment.loc[n, 'Type'] = n + " Curtailment (GWh)"
            
            annual_equivalent_cyles.loc[n,year] = CY_equivalent_cycles
            annual_equivalent_cyles.loc[n, 'Type'] = n + " Equivalent Cycles"
            
            annual_SoC_percentage.loc[n,year] = CY_SoC_percentage
            annual_SoC_percentage.loc[n, 'Type'] = n + " Average Annual SoC (%)"
            
            
            # Create the current CY traces
            CY_traces[n] = pd.DataFrame({
                'Datetime': D[n].iloc[:,0],
                'SoC (GWh)': [pulp.value(E[n][t]) / 10**3 for t in range(T)],
                'Disharge (MW)': [pulp.value(P_disc[n][t]) for t in range(T)],
                'Charge (MW)': [pulp.value(P_chrg[n][t]) for t in range(T)],
                'Curtailment (MW)': [pulp.value(P_curt[n][t]) for t in range(T)],
                'DSP (MW)': [pulp.value(P_DSP[n][t]) for t in range(T)]})
            
        
            
        # Otherwise its an interstate region
        else:
            
            if n == 'Murraylink':
                
                # Create import/export trace. Positive convention for imports, negative for exports
                import_export_trace = [pulp.value(P_I['Riverland', n, t]) - pulp.value(P_I[n, 'Riverland', t])for t in range(T)]
                
                # Calculate the total number of imports and exports from the trace
                total_imports = sum(x for x in import_export_trace if x > 0) * delta_t / 10 ** 6
                total_exports = -sum(x for x in import_export_trace if x < 0) * delta_t / 10 ** 6   # Negative sign out the front as the exports are negative convention
                
                # Save the annual import and export amounts
                annual_import.loc[n,year] = total_imports
                annual_import.loc[n, 'Type'] = n + " Annual Imports (TWh)"
                
                annual_export.loc[n,year] = total_exports
                annual_export.loc[n, 'Type'] = n + " Annual Exports (TWh)"
                
                
                # Save the trace of the imports and exports
                CY_traces[n] = pd.DataFrame({
                    'Datetime': D['Adelaide Metro'].iloc[:,0],
                    'Imports/(Exports) (MW)': import_export_trace,
                    'Net Demand VIC (MW)': CY_net_demand_VIC.iloc[:,1]})
            
            
            elif n == 'Heywood':
                
                # Create import/export trace. Positive convention for imports, negative for exports
                import_export_trace = [pulp.value(P_I['South East', n, t]) - pulp.value(P_I[n, 'South East', t])for t in range(T)]
                
                # Calculate the total number of imports and exports from the trace
                total_imports = sum(x for x in import_export_trace if x > 0) * delta_t / 10 ** 6
                total_exports = -sum(x for x in import_export_trace if x < 0) * delta_t / 10 ** 6   # Negative sign out the front as the exports are negative convention
                
                # Save the annual import and export amounts
                annual_import.loc[n,year] = total_imports
                annual_import.loc[n, 'Type'] = n + " Annual Imports (TWh)"
                
                annual_export.loc[n,year] = total_exports
                annual_export.loc[n, 'Type'] = n + " Annual Exports (TWh)"
                
                
                # Save the trace of the imports and exports
                CY_traces[n] = pd.DataFrame({
                    'Datetime': D['Adelaide Metro'].iloc[:,0],
                    'Imports/(Exports) (MW)': import_export_trace,
                    'Net Demand VIC (MW)': CY_net_demand_VIC.iloc[:,1]})
                
                
            elif n == 'PEC':
                
                # Create import/export trace. Positive convention for imports, negative for exports
                import_export_trace = [pulp.value(P_I['Mid North', n, t]) - pulp.value(P_I[n, 'Mid North', t])for t in range(T)]
                
                # Calculate the total number of imports and exports from the trace
                total_imports = sum(x for x in import_export_trace if x > 0) * delta_t / 10 ** 6
                total_exports = -sum(x for x in import_export_trace if x < 0) * delta_t / 10 ** 6   # Negative sign out the front as the exports are negative convention
                
                # Save the annual import and export amounts
                annual_import.loc[n,year] = total_imports
                annual_import.loc[n, 'Type'] = n + " Annual Imports (TWh)"
                
                annual_export.loc[n,year] = total_exports
                annual_export.loc[n, 'Type'] = n + " Annual Exports (TWh)"
                
                
                # Save the trace of the imports and exports
                CY_traces[n] = pd.DataFrame({
                    'Datetime': D['Adelaide Metro'].iloc[:,0],
                    'Imports/(Exports) (MW)': import_export_trace,
                    'Net Demand NSW (MW)': CY_net_demand_NSW.iloc[:,1]})
            
    
    # Combine the annual input and output totals
    annual_input_totals = pd.concat([annual_demand, annual_solar, annual_wind, annual_DSP, annual_dispatchable], ignore_index=True)
    annual_output_totals = pd.concat([annual_storage, total_annual_storage, annual_power, annual_curtailment,
                                      annual_charge, annual_discharge, annual_equivalent_cyles,
                                      annual_SoC_percentage, annual_import, annual_export, problem_solve_time],ignore_index=True)
    

    
    
    # Save the current yearly traces
    for n in all_regions:
        
        # Save annual traces to CSV files
        CY_traces[n].to_csv(f'Results/{year}_{n}_Traces.csv', index=False)
        transmission_flow[n].to_csv(f'Results/{year}_{n}_Xmission_Traces.csv', index=False)
            
    
# Save the annual input and output totals
with pd.ExcelWriter('Results/Annual Totals.xlsx') as writer:
    annual_input_totals.to_excel(writer, sheet_name='Annual Input Totals', index=False)
    annual_output_totals.to_excel(writer, sheet_name='Annual Output Totals', index=False)
    
