"""
Plot Kaplan-Meier CDFs with 66% threshold lines and annotations
Shows when each scenario crosses 66% probability for different warming thresholds
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
SSP_NAME_PRETTY = {
    "ssp119": "SSP1–1.9",
    "ssp126": "SSP1–2.6",
    "ssp245": "SSP2–4.5",
    "ssp370": "SSP3–7.0",
    "ssp585": "SSP5–8.5",
}

colors_ssp = {
    'ssp119': '#00a9cf',
    'ssp126': '#1e9583',
    'ssp245': '#4576be',
    'ssp370': '#f11111',
    'ssp585': '#8036a7'
}

# Kaplan-Meier functions
def kaplan_meier(durations, observed):
    """Basic Kaplan-Meier estimator"""
    data = sorted(zip(durations, observed))
    n = len(data)
    at_risk = n
    survival = 1.0
    times, surv_probs = [0], [1.0]
    
    for t, e in data:
        if e == 1:
            survival *= (at_risk - 1) / at_risk
        at_risk -= 1
        times.append(t)
        surv_probs.append(survival)
    
    return times, surv_probs


def km_curve(df_in, threshold, scenario, censor_year=2100):
    """Calculate KM CDF curve"""
    d = df_in[(df_in["threshold"] == threshold) & (df_in["scenario"] == scenario)].copy()
    
    if d.empty:
        return [], []
    
    events = d["first_year"].dropna().astype(int).tolist()
    censored = d[d["first_year"].isna()]
    censored_years = [censor_year] * len(censored)
    
    durations = events + censored_years
    observed = [1] * len(events) + [0] * len(censored_years)
    
    times, surv = kaplan_meier(durations, observed)
    cdf = [1 - s for s in surv]
    
    return times, cdf


def get_cdf_crossing_year(times, cdf, level=0.66):
    """
    Get the year when CDF reaches a given level
    
    Parameters:
    -----------
    times : list
        Time points
    cdf : list
        CDF values
    level : float
        Target CDF level (e.g., 0.66)
        
    Returns:
    --------
    year : int or None
        Year when CDF reaches the level
    """
    if not times or not cdf:
        return None
    
    for t, c in zip(times, cdf):
        if c >= level:
            return int(t)
    
    return None


def plot_cdf_with_annotations(data_file, output_file='cdf_with_66p.png', 
                              thresholds=[1.5, 2.0, 2.5, 3.0],
                              figsize=(18, 14)):
    """
    Plot CDFs with 66% vertical lines and annotations
    
    Parameters:
    -----------
    data_file : str
        Path to CMIP6 data file
    output_file : str
        Output figure filename
    thresholds : list
        List of temperature thresholds to plot
    figsize : tuple
        Figure size
    """
    
    # Load data
    df = pd.read_csv(data_file)
    scenarios = sorted(df["scenario"].unique())
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()
    
    labels_ = ["(a)", "(b)", "(c)", "(d)"]
    
    # Set default font sizes
    plt.rcParams.update({'font.size': 16})
    
    # Store 66% crossing info for each threshold
    crossing_info = {}
    
    for ax_i, threshold in enumerate(thresholds):
        ax = axes[ax_i]
        
        crossing_info[threshold] = {}
        
        # Plot CDF for each scenario
        for scenario in scenarios:
            times, cdf = km_curve(df, threshold, scenario)
            
            if len(times) > 1:
                # Plot CDF
                ax.step(times, cdf, where="post", 
                       label=SSP_NAME_PRETTY[scenario],
                       color=colors_ssp[scenario], 
                       linewidth=2.5)
                
                # Get 66% crossing year
                year_66 = get_cdf_crossing_year(times, cdf, 0.66)
                crossing_info[threshold][scenario] = year_66
        
        # Add horizontal line at 66%
        ax.axhline(0.66, color='black', linestyle='--', linewidth=2, 
                  alpha=0.7, label='66th percentile')
        
        # Add vertical lines and annotations for 66% crossings
        # Sort by year to avoid overlapping annotations
        crossings = [(sc, yr) for sc, yr in crossing_info[threshold].items() if yr is not None]
        crossings.sort(key=lambda x: x[1])
        
        # Stack annotations in lower right corner
        # Base position for the stack
        x_text = 0.85  # Relative to axis (85% from left)
        y_base = 0.15  # Start from bottom
        y_spacing = 0.12  # Vertical spacing between boxes
        
        for i, (scenario, year) in enumerate(crossings):
            # Draw vertical line
            ax.axvline(year, color=colors_ssp[scenario], linestyle=':', 
                      linewidth=2, alpha=0.6)
            
            # Calculate position in stack (from bottom to top)
            y_text = y_base + i * y_spacing
            
            # Add annotation with box, using axis coordinates for text position
            ax.annotate(f'{SSP_NAME_PRETTY[scenario]}: {year}', 
                       xy=(year, 0.66), 
                       xytext=(x_text, y_text),                       xycoords=('data', 'data'),
                       textcoords='axes fraction',
                       color=colors_ssp[scenario],
                       fontsize=12,
                       fontweight='bold',
                       ha='left',
                       va='center',
                       bbox=dict(boxstyle='round,pad=0.4', 
                               facecolor='white', 
                               edgecolor=colors_ssp[scenario],
                               alpha=0.95,
                               linewidth=2),
                       arrowprops=dict(arrowstyle='->', 
                                     color=colors_ssp[scenario],
                                     lw=0,
                                     alpha=1.,
                                     connectionstyle='arc3,rad=0.3'))
        
        # Formatting
        ax.set_xlim(2015, 2100)
        ax.set_ylim(0, 1)
        ax.set_title(f"Threshold = {threshold}°C", fontsize=24, fontweight='bold')
        ax.set_xlabel("Year", fontsize=22)
        ax.set_ylabel("P(crossed)", fontsize=22)
        ax.tick_params(axis='both', labelsize=20)
        ax.grid(True, alpha=0.3)
        
        # Add subplot label
c
    
    # Create legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", 
              bbox_to_anchor=(0.5, 1.00),
              ncol=6, fontsize=20, frameon=True)
    
    # Layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("66th Percentile Crossing Years")
    print("="*80)
    print(f"{'Threshold':<12} {'SSP1-1.9':<10} {'SSP1-2.6':<10} {'SSP2-4.5':<10} {'SSP3-7.0':<10} {'SSP5-8.5':<10}")
    print("-"*80)
    
    for threshold in thresholds:
        row = [f"{threshold}°C"]
        for scenario in ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585']:
            year = crossing_info[threshold].get(scenario)
            row.append(str(year) if year is not None else "—")
        print(f"{row[0]:<12} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10} {row[5]:<10}")
    
    print("="*80)
    
    plt.close()
    
    return crossing_info


def plot_single_threshold_detailed(data_file, threshold, output_file=None,
                                   figsize=(14, 8)):
    """
    Create detailed plot for a single threshold with better annotation placement
    
    Parameters:
    -----------
    data_file : str
        Path to CMIP6 data file
    threshold : float
        Temperature threshold
    output_file : str, optional
        Output filename (auto-generated if None)
    figsize : tuple
        Figure size
    """
    
    if output_file is None:
        output_file = f'cdf_threshold_{threshold}C.png'
    
    # Load data
    df = pd.read_csv(data_file)
    scenarios = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set default font sizes
    plt.rcParams.update({'font.size': 18})
    
    crossing_years = {}
    
    # Plot each scenario
    for scenario in scenarios:
        times, cdf = km_curve(df, threshold, scenario)
        
        if len(times) > 1:
            ax.step(times, cdf, where="post", 
                   label=SSP_NAME_PRETTY[scenario],
                   color=colors_ssp[scenario], 
                   linewidth=3)
            
            # Get 66% crossing
            year_66 = get_cdf_crossing_year(times, cdf, 0.66)
            crossing_years[scenario] = year_66
    
    # Add horizontal line at 66%
    ax.axhline(0.66, color='black', linestyle='--', linewidth=2.5, 
              alpha=0.7, label='66th percentile', zorder=0)
    
    # Add vertical lines and annotations
    crossings = [(sc, yr) for sc, yr in crossing_years.items() if yr is not None]
    crossings.sort(key=lambda x: x[1])
    
    for i, (scenario, year) in enumerate(crossings):
        # Vertical line
        ax.axvline(year, color=colors_ssp[scenario], linestyle=':', 
                  linewidth=2.5, alpha=0.7, zorder=1)
        
        # Annotation with alternating positions
        y_offset = 0.1 * (i % 3) + 0.7
        
        ax.annotate(f'{SSP_NAME_PRETTY[scenario]}\n{year}', 
                   xy=(year, 0.66), 
                   xytext=(year, y_offset),
                   color=colors_ssp[scenario],
                   fontsize=16,
                   fontweight='bold',
                   ha='center',
                   va='bottom',
                   bbox=dict(boxstyle='round,pad=0.5', 
                           facecolor='white', 
                           edgecolor=colors_ssp[scenario],
                           alpha=0.9),
                   arrowprops=dict(arrowstyle='->', 
                                 color=colors_ssp[scenario],
                                 lw=0,
                                 alpha=1.0))
    
    # Formatting
    ax.set_xlim(2015, 2100)
    ax.set_ylim(0, 1)
    ax.set_title(f"Cumulative Probability of Crossing {threshold}°C", 
                fontsize=26, fontweight='bold', pad=20)
    ax.set_xlabel("Year", fontsize=24)
    ax.set_ylabel("Probability of Threshold Crossing", fontsize=24)
    ax.tick_params(axis='both', labelsize=22)
    ax.grid(True, alpha=0.3)
    
    # Legend
    ax.legend(loc='lower right', fontsize=20, frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Single threshold figure saved to: {output_file}")
    plt.close()


if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("CDF Plotting with 66% Annotations")
    print("="*80)
    
    # Check for data files
    data_files = {
        'all': 'cmip6_data_all.csv',
        'filtered': 'cmip6_data_filtered.csv'
    }
    
    # Try to find which file exists
    data_file = None
    data_label = None
    
    for label, fname in data_files.items():
        try:
            pd.read_csv(fname)
            data_file = fname
            data_label = label
            print(f"\nUsing {label} dataset: {fname}")
            break
        except FileNotFoundError:
            # Try with /mnt/user-data/uploads prefix
            try:
                pd.read_csv(f'/mnt/user-data/uploads/{fname}')
                data_file = f'/mnt/user-data/uploads/{fname}'
                data_label = label
                print(f"\nUsing {label} dataset: {data_file}")
                break
            except FileNotFoundError:
                continue
    
    if data_file is None:
        print("\nERROR: Could not find data file!")
        print("Please ensure cmip6_data_all.csv or cmip6_data_filtered.csv is available.")
        sys.exit(1)
    
    # Create main plot with all thresholds
    print("\n1. Creating 2x2 grid plot with all thresholds...")
    crossing_info = plot_cdf_with_annotations(
        data_file,
        output_file=f'figures_new/cdf_with_66p_{data_label}.png',
        thresholds=[1.5, 2.0, 2.5, 3.0],
        figsize=(18, 14)
    )
    
    # Create individual detailed plots for each threshold
    print("\n2. Creating detailed single-threshold plots...")
    for threshold in [1.5, 2.0, 2.5, 3.0]:
        plot_single_threshold_detailed(
            data_file,
            threshold,
            output_file=f'figures_new/cdf_detailed_{threshold}C_{data_label}.png',
            figsize=(14, 8)
        )
    
    print("\n" + "="*80)
    print("All plots created successfully!")
    print("="*80)
