import os
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_elevation_histogram(dfts_meta_mp, path_fig):
    total_sites = len(dfts_meta_mp)
    missing_elevation = dfts_meta_mp['elevation'].isna().sum()
    missing_percentage = (missing_elevation / total_sites) * 100

    plt.figure(figsize=(6, 6))
    sns.histplot(dfts_meta_mp['elevation'].dropna(), bins=30, kde=True, color='#6a994e', edgecolor='grey', linewidth=0.5, alpha=0.5)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    textstr = f"N° MPs without elevation: {missing_elevation} ({missing_percentage:.2f}%)"
    plt.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5))
    plt.grid(True, axis='y', linewidth=0.4, alpha=0.5)
    plt.xlabel('Surface Elevation (msnl)', fontsize=16)
    plt.ylabel('Number of MP', fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path_fig), exist_ok=True)
    plt.savefig(path_fig)
    plt.close()


def plot_well_depth_histogram(dfts_meta_mp, path_fig=None):
    total_sites = len(dfts_meta_mp)
    missing_depth = dfts_meta_mp['depth'].isna().sum()
    missing_percentage = (missing_depth / total_sites) * 100

    plt.figure(figsize=(6, 6))
    sns.histplot(dfts_meta_mp['depth'].dropna(), bins=150, kde=True, color='#4682B4', edgecolor='grey', linewidth=0.5, alpha=0.5)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    textstr = f"N° MPs without well depth: {missing_depth} ({missing_percentage:.2f}%)"
    plt.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=13, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5))
    plt.grid(True, axis='y', linewidth=0.4, alpha=0.5)
    plt.xlabel('Well depth (m)', fontsize=16)
    plt.ylabel('Number of MP', fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    if path_fig:
        os.makedirs(os.path.dirname(path_fig), exist_ok=True)
        plt.savefig(path_fig, dpi=300, bbox_inches='tight')
    plt.close()


def plot_screen_length_histogram(dfts_meta_mp, path_fig=None):
    """
    Plot histogram for well screen length with missing value summary.
    """
    total_sites = len(dfts_meta_mp)
    screen_length = pd.to_numeric(dfts_meta_mp['ScreenLength'], errors='coerce')
    screen_length = screen_length[screen_length > 0]

    missing_screen = total_sites - len(screen_length)
    missing_percentage = (missing_screen / total_sites) * 100 if total_sites else 0

    plt.figure(figsize=(6, 6))
    sns.histplot(screen_length, bins=50, kde=True, color='#9b5de5', edgecolor='grey', linewidth=0.5, alpha=0.5)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    textstr = f"MPs without screen length: {missing_screen} ({missing_percentage:.2f}%)"
    plt.text(
        0.95,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', edgecolor='white', alpha=0.5),
    )
    plt.grid(True, axis='y', linewidth=0.4, alpha=0.5)
    plt.xlabel('Screen length (m)', fontsize=16)
    plt.ylabel('Number of MP', fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    if path_fig:
        os.makedirs(os.path.dirname(path_fig), exist_ok=True)
        plt.savefig(path_fig, dpi=300, bbox_inches='tight')
    plt.close()

def plot_num_years_histogram(ts_duration_df, save_path=None): 
    """
    Plot a histogram of the number of observation years in time series, styled to match previous histograms.
    Adds markers for the min, max, mean, 25th, and 75th percentiles.

    Parameters:
    - ts_duration_df: DataFrame with 'id_ts' and 'num_years'.
    - save_path: Optional path to save the histogram plot as a file.

    Returns:
    None
    """
    # Calculate summary statistics and round to 1 decimal place
    min_years = round(ts_duration_df['num_years'].min(), 1)
    max_years = round(ts_duration_df['num_years'].max(), 1)
    mean_years = round(ts_duration_df['num_years'].mean(), 1)
    percentile_25 = round(ts_duration_df['num_years'].quantile(0.25), 1)
    percentile_75 = round(ts_duration_df['num_years'].quantile(0.75), 1)

    # Set up the figure
    plt.figure(figsize=(7, 4))
    sns.histplot(ts_duration_df['num_years'], bins=30, kde=True, color='#4682B4', edgecolor='white', linewidth=0.6, alpha=0.5)

    # Customize the aesthetics to match previous histograms
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.6)

    # Set labels and title with increased font size
    plt.xlabel('Years of Observations', fontsize=12, labelpad=10)
    plt.ylabel('Number of Time Series', fontsize=12, labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Plot summary statistics as vertical lines with labels
    for value, color, label in zip(
        [min_years, max_years, mean_years, percentile_25, percentile_75],
        ['#FF5733', '#33A2FF', '#2ECC71', '#FFC300', '#FFC300'],
        [f'Min: {min_years}', f'Max: {max_years}', f'Mean: {mean_years}', f'25th Percentile: {percentile_25}', f'75th Percentile: {percentile_75}']
    ):
        plt.axvline(value, color=color, linestyle='--', linewidth=1.5, label=label)

    # Add legend with larger font size
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=12)

    # Add a KDE curve for smooth distribution visualization
    sns.kdeplot(ts_duration_df['num_years'], color='darkblue', linewidth=1.5, ax=ax)

    # Tight layout for cleaner output
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_monthly_station_counts(df, time_col='TimeInstant', id_col='id_mp', save_path=None):
    """
    Plot number of stations with observations per month, stacked by country prefix (first two chars of id_mp).
    """
    tmp = df.copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col], errors='coerce')
    tmp = tmp.dropna(subset=[time_col, id_col])
    tmp[time_col] = tmp[time_col].dt.to_period("M").dt.to_timestamp(how="start")
    tmp['country'] = tmp[id_col].astype(str).str[:2]

    counts = (
        tmp.groupby([time_col, 'country'])[id_col]
        .nunique()
        .unstack(fill_value=0)
        .sort_index()
    )

    # Limit to start at 1940
    counts = counts.loc[counts.index >= pd.Timestamp('1940-01-01')]

    if counts.empty:
        print("No data to plot monthly station counts.")
        return

    # Map prefixes to partner/country names (aligned with density plots), grouping some into "Others"
    prefix_map = {
        'AT': 'Austria', 'BE': 'Belgium- Flanders', 'BG': 'Bulgaria', 'CH': 'Switzerland',
        'CY': 'Others', 'CZ': 'Czech Republic', 'DE': 'Germany', 'DK': 'Others',
        'EE': 'Estonia', 'EL': 'Greece', 'ES': 'Spain', 'FI': 'Others', 'FR': 'France',
        'HR': 'Others', 'HU': 'Others', 'IE': 'Others', 'IS': 'Others', 'IT': 'Others',
        'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia', 'MT': 'Others', 'NL': 'Netherlands',
        'NO': 'Norway', 'PL': 'Others', 'PT': 'Portugal', 'RO': 'Others', 'RS': 'Others',
        'SE': 'Sweden', 'SI': 'Others', 'SK': 'Slovakia', 'UK': 'Others', 'FI': 'Others'
    }
    renamed_cols = {c: prefix_map.get(c, c) for c in counts.columns}
    counts = counts.rename(columns=renamed_cols)
    counts = counts.T.groupby(level=0).sum().T

    # Order partners by total observations (descending) so broader bands stay lower and narrower bands remain on top.
    partner_totals = counts.sum(axis=0).sort_values(ascending=False)
    cols = list(partner_totals.index)
    counts = counts[cols]

    # Fixed palette per partner
    fixed_palette = {
        "Germany": "#44AA99",
        "Austria": "#DDCC77",
        "Netherlands": "#6699CC",
        "France": "#332288",
        "Czech Republic": "#CC6677",
        "Spain": "#88CCEE",
        "Switzerland": "#2F80ED",
        "Belgium- Flanders": "#EE7733",
        "Sweden": "#117733",
        "Others": "#BBBBBB",
        "Poland": "#AA4499",
        "Slovenia": "#999933",
        "United Kingdom": "#882255",
        "Denmark": "#66CCEE",
        "Ireland": "#CCBB44",
        "Croatia": "#4477AA",
    }
    default_color = "#b3b3b3"
    palette = [fixed_palette.get(col, default_color) for col in cols]
    fig, ax = plt.subplots(figsize=(12, 5))
    stacked_values = [counts[col].to_numpy() for col in cols]
    ax.stackplot(
        counts.index,
        *stacked_values,
        labels=cols,
        colors=palette,
        alpha=0.95,
        linewidth=0,
    )

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.set_xlabel('')
    ax.set_ylabel('Stations with observations', fontsize=12)
    ax.set_xlim(left=pd.Timestamp('1940-01-01'), right=pd.Timestamp('2025-12-31'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    ax.legend(title='', fontsize=10, ncol=3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()



def plot_stations_by_last_valid_date(last_valid_dates, save_path=None):
    """
    Plots a bar chart showing the number of stations by their last valid observation date.

    Parameters:
    - last_valid_dates: DataFrame with columns ['id_ts', 'last_valid_date'] containing the last valid entry for each station.
    - save_path: Optional path to save the plot as an image.

    Returns:
    None
    """

    # Ensure the last_valid_date column is of datetime type
    last_valid_dates['last_valid_date'] = pd.to_datetime(last_valid_dates['last_valid_date'])

    # Count the number of stations per last valid date
    station_counts = last_valid_dates['last_valid_date'].value_counts().sort_index()
    station_counts = station_counts.reset_index()
    station_counts.columns = ['last_valid_date', 'count']
    station_counts = station_counts.sort_values(by='last_valid_date')

    # Filter the data for the range 2010-2024
    start_date = pd.Timestamp('2010-01-01')
    end_date = pd.Timestamp('2024-12-31')
    station_counts = station_counts[
        (station_counts['last_valid_date'] >= start_date) &
        (station_counts['last_valid_date'] <= end_date)
    ]

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        station_counts['last_valid_date'],
        station_counts['count'],
        color='#4682B4',
        alpha=1,
        width=pd.Timedelta(days=35).days
    )
    
    # Format the x-axis for dates
    ax.xaxis.set_major_locator(mdates.YearLocator(2))  # Major ticks every 2 years
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))   # Minor ticks every year
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format major ticks as years

    # Set labels, title, and ticks
    ax.set_xlabel('Date of Last Observation', fontsize=14)
    ax.set_ylabel('Number of Stations', fontsize=14)
    ax.set_title('Number of Stations by Last Observation Date (2010-2024)', fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    # Adjust layout and save or show the plot
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
