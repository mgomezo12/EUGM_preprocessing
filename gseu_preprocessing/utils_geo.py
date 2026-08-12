import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
import seaborn as sns
import matplotlib.patheffects as path_effects
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


def compute_spatial_density(gdf_points, gdfload):
    if gdf_points.crs != gdfload.crs:
        gdf_points = gdf_points.to_crs(gdfload.crs)
    gdf_joined = gpd.sjoin(gdf_points, gdfload, how='inner', predicate='within')
    country_station_counts = gdf_joined.groupby(['Country', 'partner']).size().reset_index(name='station_count')
    gdfload['area_km2'] = gdfload.to_crs(gdfload.estimate_utm_crs()).area / 1e6
    density_df = country_station_counts.merge(gdfload[['Country', 'partner', 'area_km2']], on=['Country', 'partner'])
    density_df['density_stations_per_100_km2'] = (density_df['station_count'] / density_df['area_km2']) * 100
    total_stations = gdf_points.shape[0]
    total_area_km2 = gdfload['area_km2'].sum()
    total_density = (total_stations / total_area_km2) * 100
    return density_df, total_density


def plot_density_histograms(density_df, save_path=None):
    density_df_country_sorted = density_df.sort_values(by='density_stations_per_100_km2', ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=density_df_country_sorted, x='density_stations_per_100_km2', y='Country', palette='Blues_r')
    plt.xlabel('Density (MP per 100 km²)', fontsize=15)
    plt.ylabel(' ', fontsize=15)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.title(' ')
    plt.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_density_by_country.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_num_ts_map(gdf_points, gdf, dfts_conv, columnnan='nonnan_count2', upper_bound=1200, fontsize=15, ticksize=14, save_path=None):
    gdf_points = gdf_points.copy()
    gdf_points['capped_values'] = gdf_points[columnnan].apply(lambda x: min(x, upper_bound))
    unique_id_ts = f"{dfts_conv['id_ts'].nunique():,}".replace(",", ".")
    unique_id_mp = f"{gdf_points['id_mp'].nunique():,}".replace(",", ".")
    min_value = f"{int(gdf_points[columnnan].min()):,}".replace(",", ".")
    max_value = f"{int(gdf_points[columnnan].max()):,}".replace(",", ".")
    mean_value = f"{int(gdf_points[columnnan].mean()):,}".replace(",", ".")

    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.boundary.plot(ax=ax, color='grey', linewidth=0.5)
    gdf_points.plot(ax=ax, marker='o', column='capped_values', markersize=5, cmap='viridis_r', legend=False, norm=Normalize(vmin=1, vmax=upper_bound))

    cmap = plt.cm.viridis_r
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=1, vmax=upper_bound))
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='2%', pad=0.1)
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', pad=0.02, aspect=30, shrink=0.6)
    cbar.set_label('Number of Observations (monthly resolution)', fontsize=fontsize)
    ticks = list(cbar.get_ticks())
    if upper_bound not in ticks:
        ticks.append(upper_bound)
    cbar.set_ticks(ticks)
    tick_labels = [f"{int(t):,}".replace(",", ".") for t in ticks[:-1]] + [f">{upper_bound:,}".replace(",", ".")]
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=ticksize)

    ax.set_xlim([2500000, 7500000])
    ax.set_ylim([1300000, 5500000])
    stats_text = (
        f'N° Monitoring Points : {unique_id_mp}\n'
        f'N° Time Series : {unique_id_ts}\n'
        f'Min: {min_value} | Max: {max_value}\n'
        f'Mean: {mean_value}'
    )
    ax.text(0.65, 0.95, stats_text, transform=ax.transAxes, fontsize=fontsize, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    ax.grid(False)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_categorical_map_and_histogram_per_column(gdf_points, gdf, column, save_path=None):
    """
    Plots a map and an embedded histogram for a specific categorical column, dynamically selecting a color dictionary.
    """
    if column not in gdf_points.columns:
        print(f"Column '{column}' not found in GeoDataFrame.")
        return

    # Define color mappings for specific columns
    color_dicts = {
        'AquiferMediaType': {
            'porousAndFractured': '#bed9d9',
            'porous': '#74b5cf',            
            'fractured': '#c4ac80',         
            'karsticAndFractured': '#cde0bc', 
            'karstic': '#a1c98c',           
            'unknown': '#b3b6b7',           
            'compound': '#7678ed'           
        },
        'AquiferType': {
            'unconfined': '#0077b6',            
            'confinedSubartesian': '#b08968', 
            'confined': '#ddb892',             
            'confinedArtesian': '#7f5539',    
            'unknown': '#d3d3d3'             
        }
    }

    # Select colors for the current column
    if column in color_dicts:
        color_dict = color_dicts[column]
    else:
        # Default palette for unconfigured columns
        unique_categories = gdf_points[column].unique()
        color_palette = sns.color_palette('tab10', n_colors=len(unique_categories))
        color_dict = dict(zip(unique_categories, color_palette))

    # Handle NaN values by adding a "NaN" category
    gdf_points[column] = gdf_points[column].fillna('NaN')

    # Get unique categories and counts
    category_counts = gdf_points[column].value_counts()

    # Exclude specific categories from the histogram
    filtered_category_counts = category_counts.drop(['Subconfined', 'Confined artesian'], errors='ignore')

    # Map plot setup
    fig, ax_map = plt.subplots(figsize=(12, 10))
    gdf.boundary.plot(ax=ax_map, color='grey', linewidth=0.5)

    # Plot monitoring points on the map
    gdf_points['color'] = gdf_points[column].map(color_dict).fillna('#d3d3d3')  # Default to gray if missing
    gdf_points.plot(
        ax=ax_map,
        marker='o',
        color=gdf_points['color'],
        markersize=5,
        legend=False
    )

    # Set map boundaries for Europe in EPSG:3035
    ax_map.set_xlim([2500000, 7500000])
    ax_map.set_ylim([1300000, 5500000])

    # Add category legend with increased font size
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=cat)
        for cat, color in color_dict.items()
    ]
    ax_map.legend(handles=handles, title=column, loc="lower left", bbox_to_anchor=(0.9, 0),
                  fontsize=16, title_fontsize=18)  # Increased font sizes for legend

    # Remove map frames and ticks
    ax_map.spines['top'].set_visible(False)
    ax_map.spines['right'].set_visible(False)
    ax_map.spines['bottom'].set_visible(False)
    ax_map.spines['left'].set_visible(False)
    ax_map.get_yaxis().set_ticks([])
    ax_map.get_xaxis().set_ticks([])

    # Remove grid
    ax_map.grid(False)

    # Embedded histogram setup with transparent background
    inset_ax = fig.add_axes([0.65, 0.6, 0.2, 0.3], facecolor='none')  # Transparent background
    sns.barplot(
        x=filtered_category_counts.values,
        y=filtered_category_counts.index,
        palette=[color_dict.get(cat, '#d3d3d3') for cat in filtered_category_counts.index],
        ax=inset_ax
    )
    inset_ax.set_xlabel("Number of TS", fontsize=16)  # Increased font size for x-label
    inset_ax.set_ylabel("", fontsize=12)  
    inset_ax.tick_params(axis='both', labelsize=14)  # Increased tick size
    inset_ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
    inset_ax.spines['top'].set_visible(False)
    inset_ax.spines['right'].set_visible(False)

    # Add NaN counts if present
    total_nan = category_counts.get('NaN', 0)
    if total_nan > 0:
        inset_ax.text(
            0.5,
            -0.1,
            f"NaN values: {total_nan}",
            fontsize=14,
            ha='center',
            transform=inset_ax.transAxes
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_{column}.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.close()


def plot_map_booleancolumns(gdf_points, gdfload, column='NRT', save_path=None):
    """
    Plot boolean columns (like NRT) on a map with country statistics.
    """
    if column not in gdf_points.columns:
        print(f"Column '{column}' not found in GeoDataFrame.")
        return

    color_dict = {
        True: '#f95738',
        False: '#7f7f7f'
    }

    gdf_points[column] = gdf_points[column].fillna(False)

    true_count = gdf_points[gdf_points[column] == True]['id_mp'].count()
    false_count = gdf_points[gdf_points[column] == False]['id_mp'].count()

    fig, ax_map = plt.subplots(figsize=(12, 10))
    gdfload.boundary.plot(ax=ax_map, color='grey', linewidth=0.3)

    false_points = gdf_points[gdf_points[column] == False]
    false_points.plot(ax=ax_map, marker='o', color=color_dict[False], markersize=3, alpha=0.1)

    true_points = gdf_points[gdf_points[column] == True]
    true_points.plot(ax=ax_map, marker='o', color=color_dict[True], markersize=13, alpha=1.0)

    ax_map.set_xlim([2500000, 7500000])
    ax_map.set_ylim([1300000, 5500000])

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8,
                   label=f"{label} ({count} points)")
        for label, (color, count) in zip(color_dict.keys(), [(color_dict[True], true_count),
                                                             (color_dict[False], false_count)])
    ]
    ax_map.legend(handles=handles, title=column, loc="lower left", bbox_to_anchor=(0.9, 0),
                  fontsize=16, title_fontsize=18)

    for spine in ax_map.spines.values():
        spine.set_visible(False)
    ax_map.get_yaxis().set_ticks([])
    ax_map.get_xaxis().set_ticks([])
    ax_map.grid(False)

    inset_ax = fig.add_axes([0.55, 0.55, 0.35, 0.25])
    inset_ax.patch.set_facecolor('white')
    inset_ax.patch.set_alpha(0.7)

    df = gdf_points[gdf_points[column] == True].copy()
    df['id_mp'] = df['id_mp'].astype(str)

    joined = gpd.sjoin(df, gdfload[['geometry', 'Country']], how="left", predicate="within")
    joined = joined.drop_duplicates(subset='id_mp')

    joined['prefix'] = joined['id_mp'].str[:2]

    fallback_map = {
        'AT': 'Austria', 'BE': 'Belgium Flanders', 'BG': 'Bulgaria', 'CH': 'Switzerland',
        'CY': 'Cyprus', 'CZ': 'Czech Republic', 'DE': 'Germany', 'DK': 'Denmark',
        'EE': 'Estonia', 'EL': 'Greece', 'ES': 'Spain', 'FI': 'Finland', 'FR': 'France',
        'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland', 'IS': 'Iceland', 'IT': 'Italy',
        'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia', 'MT': 'Malta', 'NL': 'Netherlands',
        'NO': 'Norway', 'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania', 'RS': 'Serbia',
        'SE': 'Sweden', 'SI': 'Slovenia', 'SK': 'Slovakia', 'UK': 'United Kingdom'
    }

    missing_country = joined['Country'].isna()
    joined.loc[missing_country, 'Country'] = joined.loc[missing_country, 'prefix'].map(fallback_map)
    es_mask = joined['prefix'] == 'ES'
    joined.loc[missing_country & es_mask & (joined['relatedParty'] == 'ICGC'), 'Country'] = 'Spain Catalonia'
    joined.loc[missing_country & es_mask & (joined['relatedParty'] != 'ICGC'), 'Country'] = 'Spain'

    country_counts = joined.groupby('Country').size().reset_index(name='count')
    country_counts = country_counts.sort_values('count', ascending=False)

    print("Verified Country counts (based on row count):")
    print(country_counts)

    if len(country_counts) > 0:
        if len(country_counts) > 20:
            country_counts = country_counts.head(20)
            title_suffix = " (Top 20)"
        else:
            title_suffix = ""

        bars = inset_ax.barh(
            country_counts['Country'],
            country_counts['count'],
            color=color_dict[True],
            alpha=0.8
        )

        for bar in bars:
            width = bar.get_width()
            inset_ax.text(
                width + (max(country_counts['count']) * 0.01),
                bar.get_y() + bar.get_height() / 2.,
                f'{int(width)}',
                ha='left',
                va='center',
                fontsize=8,
                color='white',
                fontweight='bold',
                path_effects=[
                    path_effects.withStroke(linewidth=1.5, foreground='gray')
                ]
            )

        inset_ax.set_title(f'N° NRT MP per Country/Region {title_suffix}', fontsize=10)
        inset_ax.tick_params(axis='y', labelsize=8)
        inset_ax.tick_params(axis='x', labelsize=8)

        inset_ax.spines['top'].set_visible(False)
        inset_ax.spines['left'].set_color('gray')
        inset_ax.spines['right'].set_visible(False)
        inset_ax.spines['bottom'].set_color('gray')
        inset_ax.spines['right'].set_linewidth(0.5)
        inset_ax.spines['bottom'].set_linewidth(0.5)

    else:
        inset_ax.text(0.5, 0.5, "No countries with NRT stations found",
                      ha='center', va='center', fontsize=10)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}_{column}.png", dpi=300, bbox_inches='tight', transparent=True)

    plt.close()

    print(f"Total NRT = {true_count}")
    print(f"Rows in joined (True & sjoin) = {len(joined)}")
    print(f"Non-NaN Countries = {joined['Country'].notna().sum()}")
    print(f"Histogram sum = {country_counts['count'].sum()}")


def plot_num_ts_map_range(gdf_points, gdf, ts_duration_df, save_path=None):
    """
    Plots a map showing the number of years of observations per monitoring point with predefined year ranges.
    """
    from matplotlib.colors import ListedColormap
    
    # Remove any NaN or infinite values from num_years column in ts_duration_df
    ts_duration_df = ts_duration_df[ts_duration_df['num_years'].notna() & ts_duration_df['num_years'].apply(lambda x: pd.notnull(x) and x != float('inf'))]

    # Merge gdf_points with time series duration information
    gdf_points = gdf_points.merge(ts_duration_df[['id_ts', 'start_date', 'end_date', 'num_years']], on='id_ts', how='left')
    
    # Remove rows with NaN in num_years after merging
    gdf_points = gdf_points[gdf_points['num_years'].notna()]

    # Define custom bins and labels for categorization
    bins = [0, 10, 20, 30, 40, 100, gdf_points['num_years'].max() + 1]  # Ensuring the last bin captures all values
    labels = ['0-10', '10-20', '20-30', '30-40', '40-100', '>100']
    gdf_points['years_category'] = pd.cut(gdf_points['num_years'], bins=bins, labels=labels, right=False)

    # Custom color mapping using PuBuGn with no white tones
    colors = ['#ddb892', '#b08968', '#bce784', '#5dd39e','#348aa7', '#525174']
    cmap = ListedColormap(colors)

    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 10))
    gdf.boundary.plot(ax=ax, color='grey', linewidth=0.5)

    # Plot monitoring points with color indicating year range
    gdf_points.plot(
        ax=ax,
        marker='o',
        column='years_category',
        markersize=5,
        cmap=cmap,
        legend=True
    )

    # Manually create color bar with category labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=8, label=labels[i]) for i in range(len(labels))]
    ax.legend(handles=handles, title="Years of Observations", loc="lower left", bbox_to_anchor=(0.8, 0.05),fontsize=14, title_fontsize=16)

    # Set map boundaries for Europe in EPSG:3035
    ax.set_xlim([2500000, 7500000])
    ax.set_ylim([1300000, 5500000])

    # Add unique id counts, average years, and start/end dates as text on the map
    unique_id_ts = gdf_points['id_ts'].nunique()
    unique_id_mp = gdf_points['id_mp'].nunique()
    avg_years = gdf_points['num_years'].mean()
    avg_years_text = f"Avg. Years of Data: {avg_years:.1f}"
    earliest_start = gdf_points['start_date'].min().strftime('%Y-%m-%d')
    latest_end = gdf_points['end_date'].max().strftime('%Y-%m-%d')
    
    ax.text(0.65, 0.98, f"N° Monitoring Points (id_mp): {unique_id_mp}\n"
                         f"N° Time Series (id_ts): {unique_id_ts}\n"
                         f"{avg_years_text}\n"
                         f"Start Date: {earliest_start}\n"
                         f"End Date: {latest_end}",
            transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # Remove map frames and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])

    # Remove grid
    ax.grid(False)

    # Save or display the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



def plot_depth_map_continuous(gdf_points: gpd.GeoDataFrame,
                              gdf_boundary: gpd.GeoDataFrame,
                              depth_column: str = 'depth',
                              aquifer_col: str = 'AquiferMediaType',
                              remove_outliers: bool = True,
                              outlier_threshold: int = 99,
                              marker_size: int = 8,
                              save_path=None) -> None:
    """
    Plot a depth map using a continuous color scale and marker shapes per aquifer type.

    Parameters:
    - gdf_points: GeoDataFrame with monitoring points and a depth column.
    - gdf_boundary: GeoDataFrame with European boundaries (EPSG:3035).
    - depth_column: name of depth field (default 'depth').
    - aquifer_col: categorical column for aquifer type; falls back to 'AquiferMed' if missing.
    - remove_outliers: drop extreme depths above percentile threshold.
    - outlier_threshold: percentile for outlier filtering (e.g., 99).
    - marker_size: point size for plotting.
    - save_path: file path to save the figure.
    """
    if depth_column not in gdf_points.columns:
        print(f"Column '{depth_column}' not found; cannot plot depth map.")
        return

    df = gdf_points.copy()
    nan_points = df[df[depth_column].isna()]
    df = df.dropna(subset=[depth_column])
    if df.empty:
        print("No depth data to plot.")
        return

    # Optional outlier removal on high-end depths
    if remove_outliers and len(df) > 10:
        thr = float(np.percentile(df[depth_column], outlier_threshold))
        df = df[df[depth_column] <= thr]
    else:
        thr = float(df[depth_column].max())

    # Cap depths for display and color mapping
    cap_max = 200.0
    df['_depth_capped'] = np.minimum(df[depth_column].astype(float), cap_max)

    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 9))
    if gdf_boundary is not None:
        gdf_boundary.boundary.plot(ax=ax, color='grey', linewidth=0.5)

    # Color normalization and mapper for continuous depth
    vmin = float(df['_depth_capped'].min())
    vmax = cap_max
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis

    # Plot stations without depth info first (behind) in grey
    nan_pct = (len(nan_points) / len(gdf_points) * 100) if len(gdf_points) else 0
    if not nan_points.empty:
        ax.scatter(nan_points.geometry.x, nan_points.geometry.y, color='#b3b3b3', s=marker_size,
                   marker='o', alpha=0.6, edgecolor='none', label=f'Unknown ({nan_pct:.1f}%)')
    # Plot stations with known depth on top
    colors = cmap(norm(df['_depth_capped'].values))
    ax.scatter(df.geometry.x, df.geometry.y, c=colors, s=marker_size, marker='o', alpha=0.85, edgecolor='none')

    # Inset histogram in upper-right, with bar colors matching colormap
    inset_ax = fig.add_axes([0.68, 0.62, 0.28, 0.25])  # [left, bottom, width, height]
    valid_depths = df['_depth_capped'].to_numpy()
    bins = min(40, max(15, int(np.ptp(valid_depths) // 2)))
    counts, bin_edges, patches = inset_ax.hist(valid_depths, bins=bins, alpha=0.9, edgecolor='none')
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for center, patch in zip(bin_centers, patches):
        patch.set_facecolor(cmap(norm(center)))
    inset_ax.set_xlim(vmin, cap_max)
    inset_ax.set_xlabel('Depth (m)', fontsize=12)
    inset_ax.set_ylabel('Number of TS', fontsize=12)
    inset_ax.tick_params(axis='both', labelsize=10)
    # Clean up inset spines
    inset_ax.spines['right'].set_visible(False)
    inset_ax.spines['top'].set_visible(False)
    inset_ax.spines['left'].set_color('gray')
    inset_ax.spines['bottom'].set_color('gray')
    inset_ax.spines['left'].set_linewidth(0.6)
    inset_ax.spines['bottom'].set_linewidth(0.6)

    # Colorbar placed underneath the histogram
    pos = inset_ax.get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - 0.05, pos.width, 0.02])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Depth (m)', fontsize=11)
    # Customize ticks to show >250 m on the right
    ticks = np.linspace(vmin, cap_max, 5)
    cbar.set_ticks(ticks)
    above_cap = (gdf_points[depth_column].astype(float) > cap_max).any()
    if above_cap:
        ticklabels = [f"{int(t)}" for t in ticks[:-1]] + [f">{int(cap_max)}"]
    else:
        ticklabels = [f"{int(t)}" for t in ticks]
    cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=10)

    # Axes styling and bounds
    ax.set_xlim([2500000, 7500000])
    ax.set_ylim([1300000, 5500000])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    ax.grid(False)

    # Station count overlay
    n_stations = df.shape[0]
    stats_text = (
        f"Monitoring Points: {n_stations}\n"
        f"Mean: {df[depth_column].mean():.1f}\n"
        f"Median: {df[depth_column].median():.1f}\n"
        f"Max: {df[depth_column].max():.1f}"
    )
    # Place stats just below the inset histogram, near the right side
    ax.text(0.72, 0.48, stats_text, transform=ax.transAxes, fontsize=14, va='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    if not nan_points.empty:
        ax.legend(loc='lower left', fontsize=11)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_multipoint_map_cdf(
    gdf_points: gpd.GeoDataFrame,
    gdf_boundary: gpd.GeoDataFrame,
    cutoff_m: float = 12.0,
    marker_size: int = 8,
    save_path=None,
) -> None:
    """
    Map of multi-point locations (clusters within cutoff_m) coloured by cluster size,
    single-point locations shown in grey, with a CDF inset of min nearest-neighbour
    distance replacing the usual histogram.

    For clusters that are not at identical coordinates the shallowest point is used
    as the representative location.
    """
    df = gdf_points.copy()
    coords = np.column_stack([df.geometry.x.values, df.geometry.y.values])

    # --- build clusters via union-find on pairs within cutoff_m ---
    tree = cKDTree(coords)
    pairs = tree.query_pairs(cutoff_m)
    parent = list(range(len(df)))

    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        pi, pj = _find(i), _find(j)
        if pi != pj:
            parent[pi] = pj

    df = df.reset_index(drop=True)
    df["_cluster"] = [_find(i) for i in range(len(df))]

    csizes = df.groupby("_cluster").size().rename("_csize")
    df = df.join(csizes, on="_cluster")

    # --- nearest-neighbour distances (for CDF inset) ---
    dists, _ = tree.query(coords, k=2)
    min_dist = dists[:, 1]

    # --- representative point per multi-point cluster (shallowest depth) ---
    multi = df[df["_csize"] > 1].copy()
    depth_col = "depth" if "depth" in multi.columns else None

    def _rep(g):
        if depth_col and g[depth_col].notna().any():
            return g.loc[g[depth_col].idxmin()]   # shallowest
        return g.iloc[0]

    multi_rep = multi.groupby("_cluster", group_keys=False).apply(_rep, include_groups=False).copy()
    # restore cluster id and size dropped by include_groups=False
    if "_cluster" not in multi_rep.columns:
        multi_rep = multi_rep.join(multi[["_cluster", "_csize"]].drop_duplicates("_cluster").set_index("_cluster"), on="_cluster", rsuffix="_r")
        multi_rep["_csize"] = multi_rep["_csize"].fillna(multi_rep.get("_csize_r", np.nan))
    single    = df[df["_csize"] == 1].copy()

    n_single = len(single)
    n_multi  = len(multi_rep)

    # --- colour scale for cluster size ---
    max_size = int(multi_rep["_csize"].max())
    boundaries = np.arange(1.5, max_size + 1.5, 1)   # edges between 2,3,4,...
    cmap_base  = plt.cm.get_cmap("plasma", max_size - 1)
    norm_cat   = BoundaryNorm(boundaries, cmap_base.N)

    # --- figure (same layout as depth_map_hist) ---
    fig, ax = plt.subplots(figsize=(12, 9))
    if gdf_boundary is not None:
        gdf_boundary.boundary.plot(ax=ax, color="grey", linewidth=0.5)

    # single points — grey, behind
    ax.scatter(
        single.geometry.x, single.geometry.y,
        color="#b3b3b3", s=marker_size, alpha=0.5, edgecolor="none",
        label=f"Single-point ({n_single:,})",
    )

    # multi-point representatives — coloured by cluster size, on top
    sc = ax.scatter(
        multi_rep.geometry.x, multi_rep.geometry.y,
        c=multi_rep["_csize"].values,
        cmap=cmap_base, norm=norm_cat,
        s=marker_size * 2.5, alpha=0.9, edgecolor="none",
        label=f"Multi-point locations ({n_multi:,})",
        zorder=3,
    )

    # --- inset: CDF of min nearest-neighbour distance ---
    inset_ax = fig.add_axes([0.68, 0.62, 0.28, 0.25])
    sorted_d = np.sort(min_dist)
    cdf_vals = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    inset_ax.plot(sorted_d, cdf_vals, lw=1.5, color="steelblue")
    inset_ax.axvline(cutoff_m, color="red", lw=1.2, ls="--",
                     label=f"{cutoff_m:.0f} m cutoff")
    frac = float(np.mean(min_dist <= cutoff_m))
    inset_ax.annotate(f"{frac*100:.1f}%", xy=(cutoff_m, frac),
                      xytext=(cutoff_m + 15, frac - 0.07),
                      arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
                      fontsize=8, color="red")
    inset_ax.set_xlim(0, 200)
    inset_ax.set_xlabel("Min dist to neighbour (m)", fontsize=10)
    inset_ax.set_ylabel("Cumulative proportion", fontsize=10)
    inset_ax.tick_params(axis="both", labelsize=9)
    inset_ax.legend(fontsize=8, loc="lower right")
    for spine in ["right", "top"]:
        inset_ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        inset_ax.spines[spine].set_color("gray")
        inset_ax.spines[spine].set_linewidth(0.6)
    inset_ax.grid(True, alpha=0.25)

    # --- discrete colorbar below inset ---
    pos = inset_ax.get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - 0.05, pos.width, 0.02])
    sm  = plt.cm.ScalarMappable(cmap=cmap_base, norm=norm_cat)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Wells per location", fontsize=10)
    tick_vals = list(range(2, max_size + 1))
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([str(v) for v in tick_vals])
    cbar.ax.tick_params(labelsize=9)

    # --- stats text ---
    stats_text = (
        f"Total MPs: {len(df):,}\n"
        f"Single-point: {n_single:,}\n"
        f"Multi-point:  {n_multi:,}\n"
        f"Cutoff: {cutoff_m:.0f} m"
    )
    ax.text(0.72, 0.48, stats_text, transform=ax.transAxes, fontsize=13, va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    ax.legend(loc="lower left", fontsize=10)

    # --- axes styling (matches depth_map_hist) ---
    ax.set_xlim([2500000, 7500000])
    ax.set_ylim([1300000, 5500000])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.grid(False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def verify_country_prefix_alignment(gdf_points: gpd.GeoDataFrame,
                                    gdf_boundary: gpd.GeoDataFrame,
                                    save_path=None) -> None:
    """
    Verify that the country prefix in id_mp matches the country from spatial intersection.
    Generates a map highlighting mismatches and an inset bar with mismatch counts.
    """
    if 'id_mp' not in gdf_points.columns:
        print("Column 'id_mp' not found; cannot verify country prefixes.")
        return

    prefix_map = {
        'AT': 'Austria', 'BE': 'Belgium- Flanders', 'BG': 'Bulgaria', 'CH': 'Switzerland',
        'CY': 'Cyprus', 'CZ': 'Czech Republic', 'DE': 'Germany', 'DK': 'Denmark',
        'EE': 'Estonia', 'EL': 'Greece', 'ES': 'Spain', 'FI': 'Finland', 'FR': 'France',
        'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland', 'IS': 'Iceland', 'IT': 'Italy',
        'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia', 'MT': 'Malta', 'NL': 'Netherlands',
        'NO': 'Norway', 'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania', 'RS': 'Serbia',
        'SE': 'Sweden', 'SI': 'Slovenia', 'SK': 'Slovakia', 'UK': 'United Kingdom'
    }

    df = gdf_points.copy()
    df['prefix'] = df['id_mp'].astype(str).str[:2]

    def expected_country(row):
        if row['prefix'] == 'ES':
            if str(row.get('relatedParty', '')).upper() == 'ICGC':
                return 'Spain Catalonia'
            return 'Spain'
        return prefix_map.get(row['prefix'])

    df['expected_country'] = df.apply(expected_country, axis=1)

    # Spatial join to get actual country
    pts = df
    if pts.crs != gdf_boundary.crs:
        pts = pts.to_crs(gdf_boundary.crs)

    joined = gpd.sjoin(pts, gdf_boundary[['Country', 'geometry']], how='left', predicate='within')
    joined = joined.rename(columns={'Country': 'actual_country'})

    joined['status'] = 'matched'
    joined.loc[joined['actual_country'].isna(), 'status'] = 'no_country'
    joined.loc[
        (joined['expected_country'].notna()) &
        (joined['actual_country'].notna()) &
        (joined['expected_country'] != joined['actual_country']),
        'status'
    ] = 'mismatch'

    status_colors = {
        'matched': '#2ca25f',
        'mismatch': '#d7301f',
        'no_country': '#bdbdbd'
    }

    fig, ax = plt.subplots(figsize=(12, 9))
    if gdf_boundary is not None:
        gdf_boundary.boundary.plot(ax=ax, color='grey', linewidth=0.5)

    for status, color in status_colors.items():
        subset = joined[joined['status'] == status]
        if not subset.empty:
            ax.scatter(subset.geometry.x, subset.geometry.y, s=8, color=color, label=f"{status} ({len(subset)})", alpha=0.8, edgecolor='none')

    ax.set_xlim([2500000, 7500000])
    ax.set_ylim([1300000, 5500000])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    ax.grid(False)

    total_countries = joined['actual_country'].nunique(dropna=True)
    mismatches = joined[joined['status'] == 'mismatch'].shape[0]
    no_country = joined[joined['status'] == 'no_country'].shape[0]
    ax.text(
        0.03, 0.96,
        f"Unique countries (spatial): {total_countries}\n"
        f"Mismatches: {mismatches}\n"
        f"No country: {no_country}",
        transform=ax.transAxes,
        fontsize=12,
        va='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    # Inset bar for mismatch pairs
    mismatched = joined[joined['status'] == 'mismatch']
    inset_ax = fig.add_axes([0.64, 0.62, 0.28, 0.25])
    if not mismatched.empty:
        pair_counts = mismatched.groupby(['prefix', 'actual_country']).size().reset_index(name='count')
        pair_counts = pair_counts.sort_values('count', ascending=False).head(10)
        labels_pairs = pair_counts.apply(lambda r: f"{r['prefix']}→{r['actual_country']}", axis=1)
        inset_ax.barh(labels_pairs, pair_counts['count'], color=status_colors['mismatch'])
        inset_ax.invert_yaxis()
        inset_ax.set_xlabel('Count', fontsize=11)
        inset_ax.tick_params(axis='both', labelsize=9)
    else:
        inset_ax.text(0.5, 0.5, "No mismatches", ha='center', va='center', fontsize=11)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

    inset_ax.spines['right'].set_visible(False)
    inset_ax.spines['top'].set_visible(False)
    inset_ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.6)

    ax.legend(loc='lower left', fontsize=11, frameon=True, facecolor='white', framealpha=0.8)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
def plot_years_map_with_histogram(
    gdf_points: gpd.GeoDataFrame,
    gdf_boundary: gpd.GeoDataFrame,
    ts_duration_df: pd.DataFrame,
    save_path=None,
) -> None:
    """
    Plot a map of number of observation years per monitoring point with an inset histogram.
    """
    if ts_duration_df.empty:
        print("No duration data to plot.")
        return

    df = gdf_points.merge(
        ts_duration_df[['id_mp', 'start_date', 'end_date', 'num_years']],
        on='id_mp',
        how='left'
    )
    df = df[df['num_years'].notna()].copy()
    if df.empty:
        print("No monitoring points with duration data to plot.")
        return

    df['num_years_capped'] = df['num_years'].clip(upper=100)

    bins = [0, 10, 20, 30, 40, 100]
    labels = ['0-10', '10-20', '20-30', '30-40', '40-100']
    colors = ['#ddb892', '#b08968', '#bce784', '#5dd39e', '#348aa7']
    cmap = ListedColormap(colors)

    df['years_category'] = pd.cut(df['num_years_capped'], bins=bins, labels=labels, right=False)
    cat_counts = df['years_category'].value_counts().reindex(labels, fill_value=0)
    raw_perc = (cat_counts / cat_counts.sum() * 100)
    cat_perc_display = []
    for label, perc, count in zip(labels, raw_perc, cat_counts):
        if count > 0:
            cat_perc_display.append(f"{perc:.0f}")
        else:
            cat_perc_display.append("0")

    fig, ax = plt.subplots(figsize=(12, 10))
    if gdf_boundary is not None:
        gdf_boundary.boundary.plot(ax=ax, color='grey', linewidth=0.5)

    # Plot monitoring points with category colors
    df.plot(
        ax=ax,
        marker='o',
        column='years_category',
        markersize=5,
        cmap=cmap,
        legend=False
    )

    handles = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=colors[i],
            markersize=8,
            label=f"{labels[i]} ({cat_perc_display[i]}%)"
        )
        for i in range(len(labels))
    ]
    ax.legend(
        handles=handles,
        title="Years of Observations",
        loc="lower left",
        bbox_to_anchor=(0.8, 0.01),
        fontsize=14,
        title_fontsize=16,
        frameon=True,
        facecolor='white',
        framealpha=0.85,
    )

    # Inset histogram
    inset_ax = fig.add_axes([0.64, 0.62, 0.28, 0.25])
    inset_ax.set_facecolor('white')
    inset_ax.patch.set_alpha(0.82)
    hist_vals, bin_edges, patches = inset_ax.hist(df['num_years_capped'], bins=30, alpha=0.9, edgecolor='none')
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def _color_for_val(v: float) -> str:
        if v < 10:
            return colors[0]
        elif v < 20:
            return colors[1]
        elif v < 30:
            return colors[2]
        elif v < 40:
            return colors[3]
        else:
            return colors[4]

    for center, patch in zip(bin_centers, patches):
        patch.set_facecolor(_color_for_val(center))

    inset_ax.set_xlabel('Years', fontsize=17)
    inset_ax.set_ylabel('Number of TS', fontsize=17)
    inset_ax.tick_params(axis='both', labelsize=15)
    inset_ax.spines['right'].set_visible(False)
    inset_ax.spines['top'].set_visible(False)
    inset_ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.6)

    # Map bounds and styling
    ax.set_xlim([2500000, 7500000])
    ax.set_ylim([1300000, 5500000])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    ax.grid(False)

    # Stats overlay
    unique_id_mp = df['id_mp'].nunique()
    avg_years = df['num_years'].mean()
    median_years = df['num_years'].median()
    min_years = df['num_years'].min()
    max_years = df['num_years'].max()
    earliest_start = df['start_date'].min().strftime('%Y-%m-%d')
    latest_end = df['end_date'].max().strftime('%Y-%m-%d')
    ax.text(
        0.72, 0.55,
        f"Monitoring Points: {unique_id_mp}\n"
        f"Avg. Years: {avg_years:.1f}\n"
        f"Median Years: {median_years:.1f}\n"
        f"Min Years: {min_years:.1f}\n"
        f"Max Years: {max_years:.1f}\n"
        f"Start: {earliest_start}\n"
        f"End: {latest_end}",
        transform=ax.transAxes,
        fontsize=19,
        va='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_start_date_map_with_histogram(
    gdf_points: gpd.GeoDataFrame,
    gdf_boundary: gpd.GeoDataFrame,
    ts_duration_df: pd.DataFrame,
    save_path=None,
) -> None:
    """
    Plot a map of the beginning of observations per monitoring point with an inset histogram.
    """
    if ts_duration_df.empty:
        print("No duration data to plot.")
        return

    df = gdf_points.merge(
        ts_duration_df[['id_mp', 'start_date']],
        on='id_mp',
        how='left'
    )
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df = df[df['start_date'].notna()].copy()
    if df.empty:
        print("No monitoring points with start dates to plot.")
        return

    df['start_year'] = df['start_date'].dt.year
    bins = [1900, 1970, 1980, 1990, 2000, 2010, 2020]
    labels = ['<1970', '1970-1980', '1980-1990', '1990-2000', '2000-2010', '2010-2020']
    start_year_palette = {
        "<1970": "#1B4F72",
        "1970-1980": "#226F9B",
        "1980-1990": "#2A9D8F",
        "1990-2000": "#48CAE4",
        "2000-2010": "#E9D8A6",
        "2010-2020": "#D55E00",
    }
    colors = [
        start_year_palette["<1970"],
        start_year_palette["1970-1980"],
        start_year_palette["1980-1990"],
        start_year_palette["1990-2000"],
        start_year_palette["2000-2010"],
        start_year_palette["2010-2020"],
    ]
    cmap = ListedColormap(colors)

    df['start_category'] = pd.cut(df['start_year'], bins=bins, labels=labels, right=False, include_lowest=True)
    cat_counts = df['start_category'].value_counts().reindex(labels, fill_value=0)
    raw_perc = (cat_counts / cat_counts.sum() * 100)
    cat_perc_display = []
    for label, perc, count in zip(labels, raw_perc, cat_counts):
        if count > 0:
            cat_perc_display.append(f"{perc:.0f}")
        else:
            cat_perc_display.append("0")

    fig, ax = plt.subplots(figsize=(12, 10))
    if gdf_boundary is not None:
        gdf_boundary.boundary.plot(ax=ax, color='grey', linewidth=0.5)

    df.plot(
        ax=ax,
        marker='o',
        column='start_category',
        markersize=5,
        cmap=cmap,
        legend=False
    )

    handles = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=colors[i],
            markersize=8,
            label=f"{labels[i]} ({cat_perc_display[i]}%)"
        )
        for i in range(len(labels))
    ]
    ax.legend(
        handles=handles,
        title="Start Year",
        loc="lower left",
        bbox_to_anchor=(0.8, 0.05),
        fontsize=14,
        title_fontsize=16,
        frameon=True,
        facecolor='white',
        framealpha=0.85,
    )

    inset_ax = fig.add_axes([0.64, 0.62, 0.28, 0.25])
    inset_ax.set_facecolor('white')
    inset_ax.patch.set_alpha(0.82)
    inset_ax.bar(range(len(labels)), cat_counts.values, color=colors, edgecolor='none', width=0.9)
    inset_ax.set_xticks(range(len(labels)))
    inset_ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=13)
    inset_ax.set_xlabel('Start year', fontsize=16)
    inset_ax.set_ylabel('Number of TS', fontsize=16)
    inset_ax.tick_params(axis='y', labelsize=14)
    inset_ax.spines['right'].set_visible(False)
    inset_ax.spines['top'].set_visible(False)
    inset_ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)

    ax.set_xlim([2500000, 7500000])
    ax.set_ylim([1300000, 5500000])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    ax.grid(False)

    n_mp = df['id_mp'].nunique()
    earliest = df['start_date'].min().strftime('%Y-%m-%d')
    latest = df['start_date'].max().strftime('%Y-%m-%d')
    ax.text(
        0.72, 0.46,
        f"Monitoring Points: {n_mp}\n"
        f"Earliest start: {earliest}\n"
        f"Latest start: {latest}",
        transform=ax.transAxes,
        fontsize=19,
        va='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_numeric_map_with_histogram(
    gdf_points: gpd.GeoDataFrame,
    gdf_boundary: gpd.GeoDataFrame,
    column: str,
    label: str,
    cap_max: float,
    cap_min: float | None = None,
    remove_outliers: bool = True,
    outlier_threshold: int = 99,
    marker_size: int = 8,
    cmap_name: str = 'plasma',
    tick_format: str | None = "int",
    stats_decimals: int = 1,
    show_overflow_label: bool = False,
    stats_uncapped: bool = False,
    save_path=None,
) -> None:
    """
    Plot a numeric column on the map with an inset histogram (used for screen length and similar metrics).

    stats_uncapped: if True, the displayed Mean/Median/Min/Max/Monitoring
    Points text is computed from the full data (NaN dropped only) rather
    than the outlier-removed/capped subset used for the map colours and
    histogram. Use this when the 99th-percentile treatment is a
    visualization choice only, not a claim that those values are
    implausible/excluded from the reported statistics (see e.g. the
    signature maps vs. the ScreenLength/depth maps, which do treat the
    top 1% as implausible and want it excluded from stats too).
    """
    if column not in gdf_points.columns:
        print(f"Column '{column}' not found; cannot plot.")
        return

    df = gdf_points.copy()
    df[column] = pd.to_numeric(df[column], errors='coerce')
    nan_points = df[df[column].isna()]
    df = df.dropna(subset=[column])
    if df.empty:
        print(f"No data available to plot for column '{column}'.")
        return
    df_uncapped = df.copy()

    if remove_outliers and len(df) > 10:
        thr = float(np.percentile(df[column], outlier_threshold))
        df = df[df[column] <= thr]
    else:
        thr = float(df[column].max())

    if cap_min is None:
        df['_val_capped'] = np.minimum(df[column].astype(float), cap_max)
    else:
        df['_val_capped'] = np.clip(df[column].astype(float), cap_min, cap_max)

    fig, ax = plt.subplots(figsize=(12, 9))
    if gdf_boundary is not None:
        gdf_boundary.boundary.plot(ax=ax, color='grey', linewidth=0.5)

    vmin = float(cap_min) if cap_min is not None else float(df['_val_capped'].min())
    vmax = float(df['_val_capped'].max())
    if vmin == vmax:
        span = abs(vmin) * 0.1 if abs(vmin) > 0 else 1.0
        vmin -= span * 0.5
        vmax += span * 0.5
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.get_cmap(cmap_name)

    # Plot unknown/NaN points in grey
    nan_pct = (len(nan_points) / len(gdf_points) * 100) if len(gdf_points) else 0
    if not nan_points.empty:
        ax.scatter(nan_points.geometry.x, nan_points.geometry.y, color='#b3b3b3',
                   s=marker_size, marker='o', alpha=0.6, edgecolor='none', label=f'Unknown ({nan_pct:.1f}%)')

    # Plot known values on top of unknown/NaN points
    ax.scatter(df.geometry.x, df.geometry.y, c=cmap(norm(df['_val_capped'].values)),
               s=marker_size, marker='o', alpha=0.85, edgecolor='none')

    inset_ax = fig.add_axes([0.68, 0.62, 0.28, 0.25])
    inset_ax.set_facecolor('white')
    inset_ax.patch.set_alpha(0.82)
    valid_vals = df['_val_capped'].to_numpy()
    bins = min(50, max(15, int(np.ptp(valid_vals) // 2))) if len(valid_vals) > 1 else 10
    counts, bin_edges, patches = inset_ax.hist(valid_vals, bins=bins, alpha=0.9, edgecolor='none')
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for center, patch in zip(bin_centers, patches):
        patch.set_facecolor(cmap(norm(center)))
    inset_ax.set_xlim(vmin, cap_max)
    inset_ax.set_xlabel("")
    inset_ax.set_ylabel('Number of TS', fontsize=16)
    inset_ax.tick_params(axis='both', labelsize=14)
    inset_ax.spines['right'].set_visible(False)
    inset_ax.spines['top'].set_visible(False)
    inset_ax.spines['left'].set_color('gray')
    inset_ax.spines['bottom'].set_color('gray')
    inset_ax.spines['left'].set_linewidth(0.6)
    inset_ax.spines['bottom'].set_linewidth(0.6)

    pos = inset_ax.get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - 0.05, pos.width, 0.02])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label(label, fontsize=15)
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    above_cap = (gdf_points[column].astype(float) > cap_max).any()
    if tick_format is None:
        span = abs(ticks[-1] - ticks[0])
        if span < 5 or max(abs(ticks[0]), abs(ticks[-1])) < 10:
            fmt = lambda t: f"{t:.2f}"
        else:
            fmt = lambda t: f"{int(round(t))}"
    elif tick_format == "int":
        fmt = lambda t: f"{int(round(t))}"
    else:
        fmt = lambda t: f"{t:{tick_format}}"
    if show_overflow_label and above_cap:
        ticklabels = [fmt(t) for t in ticks[:-1]] + [f">{fmt(ticks[-1])}"]
    else:
        ticklabels = [fmt(t) for t in ticks]
    cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlim([2500000, 7500000])
    ax.set_ylim([1300000, 5500000])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    ax.grid(False)

    fmt_s = f".{stats_decimals}f"
    if stats_uncapped:
        n_points = df_uncapped.shape[0]
        stats_values = df_uncapped[column]
        stats_max = stats_values.max()
    else:
        n_points = df.shape[0]
        stats_values = df["_val_capped"]
        stats_max = df[column].max() if show_overflow_label and above_cap else stats_values.max()
    stats_text = (
        f"Monitoring Points: {n_points}\n"
        f"Mean: {stats_values.mean():{fmt_s}}\n"
        f"Median: {stats_values.median():{fmt_s}}\n"
        f"Min: {stats_values.min():{fmt_s}}\n"
        f"Max: {stats_max:{fmt_s}}"
    )
    # Place stats just below the inset histogram, near the right side
    ax.text(0.72, 0.48, stats_text, transform=ax.transAxes, fontsize=14, va='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Legend for unknown points if present
    if not nan_points.empty:
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#b3b3b3',
                               markersize=8, label=f'Unknown ({nan_pct:.1f}%)')]
        ax.legend(handles=handles, loc='lower left', fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
