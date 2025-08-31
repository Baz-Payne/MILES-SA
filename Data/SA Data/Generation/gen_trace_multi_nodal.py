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
    date_times = pd.to_datetime(updated_df['Date'] + ' ' + updated_df ['Time'])
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


# Specify the solar, wind medium and wind high mapping table sheets
regional_mapping_sheet_name = 'Region Mapping'
solar_sheet_name = 'Solar'
WM_sheet_name = 'Wind Medium'
WH_sheet_name = 'Wind High'
solar_save_file_sheet_name = 'Solar Save File Names'
wind_save_file_sheet_name = 'Wind Save File Names'

# Read the in the mapping tables
regional_mapping = pd.read_excel(file_path, sheet_name=regional_mapping_sheet_name)
solar_mapping_table = pd.read_excel(file_path, sheet_name=solar_sheet_name)
WM_mapping_table = pd.read_excel(file_path, sheet_name=WM_sheet_name)
WH_mapping_table = pd.read_excel(file_path, sheet_name=WH_sheet_name)
solar_save_file_names = pd.read_excel(file_path, sheet_name=solar_save_file_sheet_name)
wind_save_file_names = pd.read_excel(file_path, sheet_name=wind_save_file_sheet_name)


# Initialise the directory that the wind and solar files are stored in
solar_dir = 'Solar/RAW/'
wind_dir = 'Wind/RAW/'

# Use one of the files to setup and intialise a template file
template_file = pd.read_csv(solar_dir + solar_mapping_table['File Name'][0])
template_file = update_raw_date_format(template_file, 'Generation (MW)')

# Remove additional FY2053 as AEMO does not have capacities forecasts for this year
template_file = template_file[template_file['Datetime'] < financial_year_dates(2053)[0]]

# Set all entries to zero
template_file.iloc[:, 1] = 0

# Create a dataframe of arrays to store regional solar and wind generation traces
solar_traces = {}
for region in solar_save_file_names['Honours Region']:
    solar_traces[region] = template_file.copy()
    
wind_traces = {}
for region in wind_save_file_names['Honours Region']:
    wind_traces[region] = template_file.copy()
    

# Create a dataframe to store the total generation trace
total = template_file.copy()

# Combine solar and wind mapping tables into a single mapping table to be iterated through
mapping_tables = [solar_mapping_table, WM_mapping_table]


# Initialise iterator
i = 0

### Combine solar and wind sites ###
for table in mapping_tables:
    
    
    # If this is the first table, then we need the solar directory
    if i == 0:
        file_directory = solar_dir
    else:
        file_directory = wind_dir
    
    
    # Loop through each file in the current mapping table
    for site, rez, file in table[['Site Name', 'REZ', 'File Name']].itertuples(index=False):

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
            current_site.loc[(current_site['Datetime'] >= start_date) & (current_site['Datetime'] <= end_date), current_site.columns[1:]] *= FY_capacity
            
            # Solar capacity factor calculations
            total_FY = current_site.loc[(current_site['Datetime'] >= start_date) & (current_site['Datetime'] <= end_date), current_site.columns[1:]]
            
            # Multiply by 0.5 here as we are adding over half hour intervals not hours
            hh_FY_total = 0.5 * total_FY.iloc[:, :].sum()
            
            # Calculate the annual generation
            FY_gen_total = hh_FY_total.sum(axis=0)


        # Check which region the current site belongs to then add it to the total
        honours_region = regional_mapping.loc[regional_mapping['REZ'] == rez, 'Honours Region'].iloc[0]

        # Check whether current site is a solar or wind site
        if i == 0:
            
            solar_traces[honours_region].iloc[:, 1] = solar_traces[honours_region].iloc[:, 1] + current_site.iloc[:, 1]        
    
        else:
            
            wind_traces[honours_region].iloc[:, 1] = wind_traces[honours_region].iloc[:, 1] + current_site.iloc[:, 1]
        
        # Add the current trace to the total trace
        total.iloc[:, 1] += current_site.iloc[:, 1]
    
    # Update the iterator as we have completed one mapping table
    i += 1



### Save Combined Files ###
# Save total generation trace
total.to_csv('Copper Plate/TotalRenewableGen.csv', index=False)

# Loop through each solar data frame and save them
for solar_region in solar_save_file_names['Honours Region']:
    
    # Save the files
    save_name = solar_save_file_names.loc[solar_save_file_names['Honours Region'] == solar_region, 'Save File'].iloc[0]
    solar_traces[solar_region].to_csv('Multi Nodal/' + save_name, index=False)
    
# Loop through each wind data frame and save them
for wind_region in wind_save_file_names['Honours Region']:
    
    # Save the files
    save_name = wind_save_file_names.loc[wind_save_file_names['Honours Region'] == wind_region, 'Save File'].iloc[0]
    wind_traces[wind_region].to_csv('Multi Nodal/' + save_name, index=False)
    
