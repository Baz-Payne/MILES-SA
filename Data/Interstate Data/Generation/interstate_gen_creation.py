import pandas as pd
from datetime import datetime


# Convert half hours to 24 hour EST
half_hours = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48']
time = ['00:00', '00:30', '01:00', '01:30', '02:00', '02:30', '03:00', '03:30', '04:00', '04:30', '05:00', '05:30', '06:00', '06:30', '07:00', '07:30', '08:00', '08:30', '09:00', '09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', '16:00', '16:30', '17:00', '17:30', '18:00', '18:30', '19:00', '19:30', '20:00', '20:30', '21:00', '21:30', '22:00', '22:30', '23:00', '23:30']

# Create a dictionary for renaming columns
rename_dict = {old_name: new_name for old_name, new_name in zip(half_hours, time)}



def financial_year_dates(year):
    '''
    Returns the start and end dates of a financial year

    Parameters
    ----------
    year : int
        The financial year ending, e.g., FY2024-25 ends in 2025.

    Returns
    -------
    start_date : datetime
        The first day of the financial year.
    end_date : datetime
        The last day of the financial year.

    '''
    
    # Financial year starts on July 1st of the previous year
    start_date = datetime(year - 1, 7, 1)
    
    # Financial year ends on 30th June, midnight, of the given year
    end_date = datetime(year, 6, 30, 23, 59)
    
    return start_date, end_date



def update_raw_date_format(AEMO_df, data_column_name):
    '''
    Combines the separate year, month, and day columns into a single datetime column and
    then melts the separate half hour columns into a single datetime column

    Parameters
    ----------
    AEMO_df : pandas dataframe
        The original unformatted dataframe with separate year, month, day, and half hourly columns.
    
    data_column_name : string
        The type of data and unit, e.g., 'Generation (MW)'

    Returns
    -------
    updated_df : pandas dataframe
        The updated dataframe which has a single datetime column.

    '''
    
    # Merge the separate year, month, and day columns
    updated_df = AEMO_df.copy()
    dates = pd.to_datetime(updated_df[['Year', 'Month', 'Day']])
    updated_df.insert(0, 'Date', dates)
    updated_df = updated_df.drop(columns=['Year', 'Month', 'Day'])
    
    # Rename the half hourly columns to their equivalent 24-hour time value
    updated_df = updated_df.rename(columns = rename_dict)
    
    # Convert to date column to string as this makes it easy to melt with the half hours
    updated_df['Date'] = updated_df['Date'].dt.strftime("%Y-%m-%d")
    
    # Unpivot the half hourly columns
    updated_df = pd.melt(updated_df, id_vars=['Date'], var_name='Time', value_name=data_column_name)
    
    # Combine the date and time into a single datetime column
    date_times = pd.to_datetime(updated_df['Date'] + ' ' + updated_df['Time'])
    updated_df.insert(0, 'Datetime', date_times)
    
    # Drop unnecessary columns
    updated_df = updated_df.drop(['Date', 'Time'], axis=1)
    
    # Sort by datetime
    updated_df = updated_df.sort_values(by='Datetime').reset_index(drop=True)
         
    return updated_df



# Important to read
# Generation files are in % of total available capacity (a value of 0 to 1)
# Capacity is the total amount of generation total at a particular site in MW
# Multiplying these two together gives the hh MW generated at each site



### Initial Setup for Generation Combine ###

# Generation mapping tables file
file_path = 'Mapping Tables.xlsx'


# Specify the solar and wind mapping table sheets for NSW and VIC
solar_NSW_sheet_name = 'Solar NSW'
wind_NSW_sheet_name = 'Wind NSW'
solar_VIC_sheet_name = 'Solar VIC'
wind_VIC_sheet_name = 'Wind VIC'


save_files_sheet_name = 'Save File Names'
ref_year_sheet_name = 'Save Names Ref Years'

# Read the in the mapping tables
solar_NSW_mapping_table = pd.read_excel(file_path, sheet_name=solar_NSW_sheet_name)
wind_NSW_mapping_table = pd.read_excel(file_path, sheet_name=wind_NSW_sheet_name)
solar_VIC_mapping_table = pd.read_excel(file_path, sheet_name=solar_VIC_sheet_name)
wind_VIC_mapping_table = pd.read_excel(file_path, sheet_name=wind_VIC_sheet_name)
save_files_mapping_table = pd.read_excel(file_path, sheet_name=save_files_sheet_name)

# Initialise the directory that the wind and solar files are stored in
solar_dir = 'Solar/RAW/'
wind_dir = 'Wind/RAW/'


# Initialise arrays to store the total solar and wind generations
total_NSW_solar = []
total_NSW_wind = []
total_VIC_solar = []
total_VIC_wind = []


# Use one of the files to setup and intialise the total solar and wind generations to zero
total_NSW_solar = pd.read_csv(solar_dir + solar_NSW_mapping_table['File Name'][0])
total_NSW_solar = update_raw_date_format(total_NSW_solar, 'Generation (MW)')


# Remove additional FY2053 as AEMO does not have capacities forecasts for this year
total_NSW_solar = total_NSW_solar[total_NSW_solar['Datetime'] < financial_year_dates(2053)[0]]

# Set all entries to zero
total_NSW_solar.iloc[:, 1:] = 0

# Copy initialisation to wind totals
total_NSW_wind = total_NSW_solar.copy()
total_VIC_solar = total_NSW_solar.copy()
total_VIC_wind = total_NSW_solar.copy()


# Combine into a single mapping table to be iterated through
mapping_tables = [solar_NSW_mapping_table, solar_VIC_mapping_table, wind_NSW_mapping_table, wind_VIC_mapping_table]
totals = [total_NSW_solar, total_VIC_solar, total_NSW_wind, total_VIC_wind]

# Initialise table to store the capacity factors of each generation site
capacity_factors = pd.concat([solar_NSW_mapping_table, solar_VIC_mapping_table, wind_NSW_mapping_table, wind_VIC_mapping_table], ignore_index=True)
capacity_factors.iloc[:, 3:] = 0

# Initialise table to store the annual amount generated at each site
annual_site_generation = capacity_factors.copy()


# Initialise iterator
i = 0

### Combine Solar and Wind Sites ###
for table in mapping_tables:
    
    
    # If this is the first two tables, then we need the solar directory
    if i == 0 or i == 1:
        file_directory = solar_dir
    else:
        file_directory = wind_dir
    
    
    # Loop through each file in the current mapping table
    for site, file in table[['Site Name', 'File Name']].itertuples(index=False):
        
        # Read in the current site/REZ
        current_site = []
        current_site = pd.read_csv(file_directory + file)

        # Update date format
        current_site = update_raw_date_format(current_site, 'Generation (MW)')
        
        # Remove additional FY2053 as it is not being considered
        current_site = current_site[current_site['Datetime'] < financial_year_dates(2053)[0]]
        
        # Extract the site capacity amounts for each FY
        site_capacity = table.loc[(table['Site Name'] == site), table.columns[3:]]
        
        # Reset indexing so that site capacity can be referenced
        site_capacity.reset_index(drop=True, inplace=True)

        # Loop through each financial year and adjust the generated capacity
        for year in range(2025, 2053):
            
            
            ### Calculate Site Generation ###
            # Obtain the start and end dates for the current FY
            start_date, end_date = financial_year_dates(year)
            
            # Locate site capacity for current FY
            FY_capacity = site_capacity[year][0]
            
            # Locate the current FY in the dataframe and mulitple all values by the current FY capacity for the particular site
            current_FY_trace = current_site.loc[(current_site['Datetime'] >= start_date) & (current_site['Datetime'] <= end_date), current_site.columns[1:]]
            
            ### Calculate Site CF ###
            capacity_factors.loc[(capacity_factors['Site Name'] == site), [year]] = current_FY_trace.iloc[:,0].sum() / current_FY_trace.shape[0]

            # Update current trace with capacity
            current_site.loc[(current_site['Datetime'] >= start_date) & (current_site['Datetime'] <= end_date), current_site.columns[1:]] *= FY_capacity
            
            # Calculate total generation
            total_FY = current_site.loc[(current_site['Datetime'] >= start_date) & (current_site['Datetime'] <= end_date), current_site.columns[1:]]
            
            # Multiply by 0.5 here as we are adding over half hour intervals not hours
            hh_FY_total = 0.5 * total_FY.iloc[:, :].sum()
            
            # Calculate the annual generation
            FY_gen_total = hh_FY_total.sum(axis=0)
            
            # Store total generation
            annual_site_generation.loc[(annual_site_generation['Site Name'] == site), [year]] = FY_gen_total / 10**6    # Convert to TWh
            

        # Check which mapping table we are using
        totals[i].iloc[:, 1] = totals[i].iloc[:, 1] + current_site.iloc[:, 1]        
    
    # Update the iterator as we have completed one mapping table
    i += 1




# Loop through each updated generation trace file and save them
for i in range(0,4):
    
    # Save the files
    totals[i].to_csv(save_files_mapping_table.iloc[i,1], index=False)




### Perform Additional Calculations
NSW_solar_annual_generation = site_capacity.copy()  # copied the site_capacity as its column headings contain the financial years
NSW_solar_annual_generation[:] = 0                  # ensure df is zeroed

NSW_wind_annual_generation = NSW_solar_annual_generation.copy()
VIC_solar_annual_generation = NSW_solar_annual_generation.copy()
VIC_wind_annual_generation = NSW_solar_annual_generation.copy()


# Initialise df to store the annual installed capacities of solar and wind
installed_capacities = pd.DataFrame(columns=['Type'] + site_capacity.columns.tolist())  # copied the site_capacity as its column headings contain the financial years
installed_capacities = installed_capacities._append({'Type': 'NSW Solar'}, ignore_index=True)
installed_capacities = installed_capacities._append({'Type': 'NSW Wind'}, ignore_index=True)
installed_capacities = installed_capacities._append({'Type': 'VIC Solar'}, ignore_index=True)
installed_capacities = installed_capacities._append({'Type': 'VIC Wind'}, ignore_index=True)



# Loop through each FY
for year in range(2025, 2053):
    
    start_date, end_date = financial_year_dates(year)
    
    NSW_solar_FY = total_NSW_solar.loc[(total_NSW_solar['Datetime'] >= start_date) & (total_NSW_solar['Datetime'] <= end_date), total_NSW_solar.columns[1:]]
    VIC_solar_FY = total_VIC_solar.loc[(total_VIC_solar['Datetime'] >= start_date) & (total_VIC_solar['Datetime'] <= end_date), total_VIC_solar.columns[1:]]
    NSW_wind_FY = total_NSW_wind.loc[(total_NSW_wind['Datetime'] >= start_date) & (total_NSW_wind['Datetime'] <= end_date), total_NSW_wind.columns[1:]]
    VIC_wind_FY = total_VIC_wind.loc[(total_VIC_wind['Datetime'] >= start_date) & (total_VIC_wind['Datetime'] <= end_date), total_VIC_wind.columns[1:]]
   
    
    # Multiply by 0.5 here as we are adding over half hour intervals not hours
    hh_NSW_solar_FY_total = 0.5 * NSW_solar_FY.iloc[:, :].sum()
    hh_VIC_solar_FY_total = 0.5 * VIC_solar_FY.iloc[:, :].sum()
    hh_NSW_wind_FY_total = 0.5 * NSW_wind_FY.iloc[:, :].sum()
    hh_VIC_wind_FY_total = 0.5 * VIC_wind_FY.iloc[:, :].sum()
    
    # Divide to convert to TWh
    NSW_solar_annual_generation.loc[0, year] = hh_NSW_solar_FY_total.sum(axis=0) / 10**6 
    VIC_solar_annual_generation.loc[0, year] = hh_VIC_solar_FY_total.sum(axis=0) / 10**6
    NSW_wind_annual_generation.loc[0, year] = hh_NSW_wind_FY_total.sum(axis=0) / 10**6
    VIC_wind_annual_generation.loc[0, year] = hh_VIC_wind_FY_total.sum(axis=0) / 10**6
    
    # Calculate the installed solar and wind capacity for the FY in GW. Note that the amount of wind installed does not change between wind medium and wind high
    installed_capacities.loc[(installed_capacities['Type'] == 'NSW Solar'), [year]] = solar_NSW_mapping_table[year].sum() / 10**3
    installed_capacities.loc[(installed_capacities['Type'] == 'NSW Wind'), [year]] = wind_NSW_mapping_table[year].sum() / 10**3
    installed_capacities.loc[(installed_capacities['Type'] == 'VIC Solar'), [year]] = solar_VIC_mapping_table[year].sum() / 10**3
    installed_capacities.loc[(installed_capacities['Type'] == 'VIC Wind'), [year]] = wind_VIC_mapping_table[year].sum() / 10**3
    
  

# Combine solar and wind annual generation into a single df
NSW_solar_annual_generation.insert(0, 'Type', 'NSW Solar')
NSW_wind_annual_generation.insert(0, 'Type', 'NSW Wind')
VIC_solar_annual_generation.insert(0, 'Type', 'VIC Solar')
VIC_wind_annual_generation.insert(0, 'Type', 'VIC Wind')

annual_generation = pd.concat([NSW_solar_annual_generation, NSW_wind_annual_generation, VIC_solar_annual_generation, VIC_wind_annual_generation], ignore_index=True)



# Save the capacity factors and annual generation to the summary spreadsheet
with pd.ExcelWriter('Generation Summary.xlsx') as writer:
    
    # Write each DataFrame to a separate worksheet
    annual_site_generation.to_excel(writer, sheet_name='Annual Site Totals TWh', index=False)
    capacity_factors.to_excel(writer, sheet_name='Site Capacity Factors', index=False)
    installed_capacities.to_excel(writer, sheet_name='Installed Capacities', index=False)
    annual_generation.to_excel(writer, sheet_name='Annual Totals TWh', index=False)