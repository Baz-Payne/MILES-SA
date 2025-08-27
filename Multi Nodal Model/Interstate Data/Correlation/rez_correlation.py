import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt


# Convert half hours to 24 hour EST
half_hours = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48']
time = ['00:00', '00:30', '01:00', '01:30', '02:00', '02:30', '03:00', '03:30', '04:00', '04:30', '05:00', '05:30', '06:00', '06:30', '07:00', '07:30', '08:00', '08:30', '09:00', '09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', '16:00', '16:30', '17:00', '17:30', '18:00', '18:30', '19:00', '19:30', '20:00', '20:30', '21:00', '21:30', '22:00', '22:30', '23:00', '23:30']

# Create a dictionary for renaming columns
rename_dict = {old_name: new_name for old_name, new_name in zip(half_hours, time)}



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



# Intialise dfs to hold the REZ data
solar = pd.DataFrame()
wind = pd.DataFrame()


# Define mapping table
file_path = 'Mapping Tables.xlsx'


# Specify the all wind mapping
all_solar_sheet_name = 'All Solar'

# Loop through each solar REZ
all_solar_table = pd.read_excel(file_path, sheet_name=all_solar_sheet_name)

# Loop through each solar REZ
for site, filename in all_solar_table[['Site Name', 'File Name']].itertuples(index=False):
    
    # Read in the current REZ
    df = pd.read_csv('Solar/RAW/' + filename)
    
    # Update format
    df = update_raw_date_format(df, site)
    
    # Only look at the reference years
    df = df[df['Datetime'] < financial_year_dates(2044)[0]]
    df = df[df['Datetime'] >= financial_year_dates(2031)[0]]

    # Reset index
    df.reset_index(drop=True, inplace=True)

    # Append the current solar REZ to the total solar
    solar[site] = df[site]


# Calculate the correlation over the reference years
solar_correlation_matrix = solar.corr()
    

# Define where the state REZ boundaries lie in terms of rows/columns
solar_state_boundaries = {
    'NSW': (0, 10),
    'SA': (11, 19),
    'VIC': (20, 27)
}


# Define custom labels for the ticks based on the state boundaries
tick_positions = [(start + end) // 2 for start, end in solar_state_boundaries.values()]
tick_labels = list(solar_state_boundaries.keys())


# Generate a heatmap to visualise the correlation matrix
plt.figure(figsize=(36, 18))
ax = sns.heatmap(solar_correlation_matrix,
                 annot=True,
                 cmap='coolwarm',
                 vmin=-1, vmax=1,
                 linewidths=0.5,
                 fmt=".2f",
                 cbar_kws={"shrink": 0.8},
                 annot_kws={"size": 14}
)

# Format the colour bar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)  # Set tick label size for the color bar
cbar.set_label('Correlation', size=20)  # Set label and its size

# Add black line to divide the between the states
for start, end in solar_state_boundaries.values():
    ax.hlines([end], *ax.get_xlim(), color='black', linewidth=2)
    ax.vlines([end], *ax.get_ylim(), color='black', linewidth=2)


# Increase size of tick marks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Create plot title
plt.title('Solar REZ Correlation Matrix Heatmap', fontsize=40)

# Save plot
plt.savefig('Solar Correlation Heatmap.png', dpi=300, bbox_inches='tight')



# Specify the all wind mapping
all_wind_sheet_name = 'All Wind'

# Read the in the mapping tables
all_wind_table = pd.read_excel(file_path, sheet_name=all_wind_sheet_name)

# Loop through each wind REZ
for site, filename in all_wind_table[['Site Name', 'File Name']].itertuples(index=False):
    
    # Read in the current REZ
    df = pd.read_csv('Wind/RAW/' + filename)
    
    # Update format
    df = update_raw_date_format(df, site)
    
    # Only look at the reference years
    df = df[df['Datetime'] < financial_year_dates(2044)[0]]
    df = df[df['Datetime'] >= financial_year_dates(2031)[0]]

    # Reset index
    df.reset_index(drop=True, inplace=True)

    # Append the current solar REZ to the total solar
    wind[site] = df[site]


# Calculate the correlation over the reference years
wind_correlation_matrix = wind.corr()


# Define where the state REZ boundaries lie in terms of rows/columns
wind_state_boundaries = {
    'NSW': (0, 12),
    'SA': (13, 22),
    'VIC': (23, 32)
}


# Define custom labels for the ticks based on the state boundaries
tick_positions = [(start + end) // 2 for start, end in wind_state_boundaries.values()]
tick_labels = list(wind_state_boundaries.keys())


# Generate a heatmap to visualise the correlation matrix
plt.figure(figsize=(36, 18))
ax = sns.heatmap(wind_correlation_matrix,
                 annot=True,
                 cmap='coolwarm',
                 vmin=-1, vmax=1,
                 linewidths=0.5,
                 fmt=".2f",
                 cbar_kws={"shrink": 0.8},
                 annot_kws={"size": 14}
)

# Format the colour bar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)  # Set tick label size for the color bar
cbar.set_label('Correlation', size=20)  # Set label and its size

# Add black line to divide the between the states
for start, end in wind_state_boundaries.values():
    ax.hlines([end], *ax.get_xlim(), color='black', linewidth=2)
    ax.vlines([end], *ax.get_ylim(), color='black', linewidth=2)


# Increase size of tick marks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Create plot title
plt.title('Wind REZ Correlation Matrix Heatmap', fontsize=40)

# Save plot
plt.savefig('Wind Correlation Heatmap.png', dpi=300, bbox_inches='tight')


### Just SA ###
# Specify the all wind mapping
SA_sheet_name = 'SA'
SA = pd.DataFrame()

# Read the in the mapping tables
SA_table = pd.read_excel(file_path, sheet_name=SA_sheet_name)

# Loop through each wind REZ
for site, filename in SA_table[['Site Name', 'File Name']].itertuples(index=False):
    
    # Read in the current REZ
    try:
        # Read in the variables and the problem        
        df = pd.read_csv('Wind/RAW/' + filename)
        
    except Exception:
        
        # Handle any read errors
        df = pd.read_csv('Solar/RAW/' + filename)
    
    
    # Update format
    df = update_raw_date_format(df, site)
    
    # Only look at the reference years
    df = df[df['Datetime'] < financial_year_dates(2044)[0]]
    df = df[df['Datetime'] >= financial_year_dates(2031)[0]]

    # Reset index
    df.reset_index(drop=True, inplace=True)

    # Append the current solar REZ to the total solar
    SA[site] = df[site]


# Calculate the correlation over the reference years
SA_correlation_matrix = SA.corr()


# Define where the state REZ boundaries lie in terms of rows/columns
SA_REZ_boundaries = {
    'Solar': (0, 9),
    'Wind': (10, 22),
}


# Define custom labels for the ticks based on the state boundaries
tick_positions = [(start + end) // 2 for start, end in SA_REZ_boundaries.values()]
tick_labels = list(SA_REZ_boundaries.keys())


# Generate a heatmap to visualise the correlation matrix
plt.figure(figsize=(36, 18))
ax = sns.heatmap(SA_correlation_matrix,
                 annot=True,
                 cmap='coolwarm',
                 vmin=-1, vmax=1,
                 linewidths=0.5,
                 fmt=".2f",
                 cbar_kws={"shrink": 0.8},
                 annot_kws={"size": 14}
)

# Format the colour bar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)  # Set tick label size for the color bar
cbar.set_label('Correlation', size=20)  # Set label and its size

# Add black line to divide the between the states
for start, end in SA_REZ_boundaries.values():
    ax.hlines([end], *ax.get_xlim(), color='black', linewidth=2)
    ax.vlines([end], *ax.get_ylim(), color='black', linewidth=2)


# Increase size of tick marks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Create plot title
plt.title('SA Solar and Wind REZ Correlation Matrix Heatmap', fontsize=40)

# Save plot
plt.savefig('SA Correlation Heatmap.png', dpi=300, bbox_inches='tight')
