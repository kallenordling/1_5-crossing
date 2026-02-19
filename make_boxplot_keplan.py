"""
Complete self-running script for Kaplan-Meier based boxplots
Generates boxplots for climate threshold crossing years accounting for censored data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Scenario names and colors
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

# Temperature thresholds to analyze
thresholds = [1.5, 2.0, 2.5, 3.0]

# Subplot labels
labels_ = ["(a)", "(b)", "(c)", "(d)"]

# File to read
models = "filtered"  # Change to "filtered" if needed
input_file = f"cmip6_data_{models}.csv"
output_file = f"figures_new/boxplot_km_{models}.png"

# ============================================================================
# KAPLAN-MEIER FUNCTIONS
# ============================================================================

def kaplan_meier(durations, observed):
    """
    Basic Kaplan-Meier survival estimator
    
    Parameters:
    -----------
    durations : list
        Time to event (or censoring)
    observed : list
        1 if event occurred, 0 if censored
        
    Returns:
    --------
    times : list
        Time points
    surv_probs : list
        Survival probability at each time point
    """
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


def km_curve(df_in, threshold, scenario, trend_group=None, censor_year=2100):
    """
    Calculate Kaplan-Meier CDF curve for given conditions
    
    Parameters:
    -----------
    df_in : DataFrame
        Input data with threshold crossing information
    threshold : float
        Temperature threshold
    scenario : str
        Climate scenario
    trend_group : str, optional
        Trend group filter
    censor_year : int
        Year to assign to censored observations
        
    Returns:
    --------
    times : list
        Time points
    cdf : list
        Cumulative distribution function values
    """
    # Filter data
    mask = (df_in["threshold"] == threshold) & (df_in["scenario"] == scenario)
    if trend_group is not None:
        mask = mask & (df_in["trend_group"] == trend_group)
    
    d = df_in[mask].copy()
    
    if d.empty:
        return [], []
    
    # Events (models that crossed threshold)
    events = d["first_year"].dropna().astype(float).tolist()
    
    # Censored (models that didn't cross)
    censored = d[d["first_year"].isna()]
    censored_years = [censor_year] * len(censored)
    
    # Combine
    durations = events + censored_years
    observed = [1] * len(events) + [0] * len(censored_years)
    
    if len(durations) == 0:
        return [], []
    
    # Calculate KM curve
    times, surv = kaplan_meier(durations, observed)
    cdf = [1 - s for s in surv]
    
    return times, cdf


def km_to_samples(df_in, threshold, scenario, trend_group=None, censor_year=2100, 
                  n_samples=1000, year_min=1850, year_max=2100):
    """
    Convert Kaplan-Meier CDF to synthetic samples using inverse transform sampling
    
    This allows us to use KM-estimated distributions with seaborn boxplot
    
    Parameters:
    -----------
    df_in : DataFrame
        Input data
    threshold : float
        Temperature threshold
    scenario : str
        Climate scenario
    trend_group : str, optional
        Trend group filter
    censor_year : int
        Year for censored observations
    n_samples : int
        Number of samples to generate
    year_min, year_max : float
        Valid year bounds
        
    Returns:
    --------
    samples : np.ndarray
        Synthetic samples from the KM distribution
    """
    # Get KM CDF
    times_cdf, cdf = km_curve(df_in, threshold, scenario, trend_group, censor_year)
    
    if len(times_cdf) < 2:
        return np.array([])
    
    # Convert to arrays
    times_cdf = np.array(times_cdf)
    cdf = np.array(cdf)
    
    # Clip to valid range
    valid_mask = (times_cdf >= year_min) & (times_cdf <= year_max)
    if not np.any(valid_mask):
        return np.array([])
    
    first_valid = np.where(valid_mask)[0][0]
    times_cdf = times_cdf[first_valid:]
    cdf = cdf[first_valid:]
    
    # Normalize CDF to [0, 1]
    cdf = cdf - cdf[0]
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    else:
        return np.array([])
    
    # Remove duplicate CDF values for stable interpolation
    times_unique = [times_cdf[0]]
    cdf_unique = [cdf[0]]
    for i in range(1, len(times_cdf)):
        if cdf[i] != cdf_unique[-1]:
            times_unique.append(times_cdf[i])
            cdf_unique.append(cdf[i])
    
    times_cdf = np.array(times_unique)
    cdf = np.array(cdf_unique)
    
    # Inverse transform sampling
    u = np.random.uniform(0, 1, n_samples)
    samples = np.interp(u, cdf, times_cdf)
    
    # Ensure bounds
    samples = np.clip(samples, year_min, year_max)
    
    return samples


def create_km_boxplot_data(df_in, thresholds, scenarios, trend_groups=None, 
                           n_samples=1000, year_min=1850, year_max=2100, verbose=True):
    """
    Create a DataFrame suitable for seaborn boxplot from KM estimates
    
    Parameters:
    -----------
    df_in : DataFrame
        Original data with columns: model, threshold, first_year, scenario, trend_group
    thresholds : list
        List of temperature thresholds
    scenarios : list
        List of climate scenarios
    trend_groups : list, optional
        List of trend group names
    n_samples : int
        Number of synthetic samples per group
    year_min, year_max : float
        Valid year range
    verbose : bool
        Print progress
        
    Returns:
    --------
    DataFrame with columns: threshold, scenario, trend_group, first_year
    """
    rows = []
    
    for threshold in thresholds:
        if verbose:
            print(f"Processing threshold {threshold}°C...")
        
        for scenario in scenarios:
            if trend_groups is not None:
                for trend_group in trend_groups:
                    df_subset = df_in[
                        (df_in['threshold'] == threshold) &
                        (df_in['scenario'] == scenario) &
                        (df_in['trend_group'] == trend_group)
                    ]
                    
                    n_models = len(df_subset)
                    n_crossed = df_subset['first_year'].notna().sum()
                    
                    if n_models > 0:
                        if verbose:
                            print(f"  {scenario} - {trend_group}: {n_crossed}/{n_models} models crossed")
                        
                        samples = km_to_samples(
                            df_subset, threshold, scenario, trend_group, 
                            n_samples=n_samples, year_min=year_min, year_max=year_max
                        )
                        
                        if len(samples) > 0:
                            for sample in samples:
                                rows.append({
                                    'threshold': threshold,
                                    'scenario': scenario,
                                    'trend_group': trend_group,
                                    'first_year': sample
                                })
                        elif verbose:
                            print(f"    WARNING: No samples generated (likely all/none crossed)")
            else:
                df_subset = df_in[
                    (df_in['threshold'] == threshold) &
                    (df_in['scenario'] == scenario)
                ]
                
                if len(df_subset) > 0:
                    samples = km_to_samples(
                        df_subset, threshold, scenario, 
                        n_samples=n_samples, year_min=year_min, year_max=year_max
                    )
                    
                    for sample in samples:
                        rows.append({
                            'threshold': threshold,
                            'scenario': scenario,
                            'first_year': sample
                        })
    
    result = pd.DataFrame(rows)
    if verbose:
        print(f"\n{'='*60}")
        print(f"Generated {len(result)} total samples")
        print(f"{'='*60}\n")
    
    return result


def add_percentile_lines(ax, df, x='trend_group', y='first_year', hue='scenario',
                         order=None, hue_order=None, width=0.8, p=0.66, frac=0.8,
                         color='black', lw=2):
    """
    Add percentile lines to boxplot
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    df : DataFrame
        Data with x, y, and hue columns
    x, y, hue : str
        Column names
    order : list
        Order of x categories
    hue_order : list
        Order of hue categories
    width : float
        Total width of box group
    p : float
        Percentile to plot (0.66 = 66th percentile)
    frac : float
        Fraction of box width for line
    color : str
        Line color
    lw : float
        Line width
    """
    if order is None:
        order = df[x].unique()
    if hue_order is None:
        hue_order = df[hue].unique()
    
    n_hues = len(hue_order)
    box_width = width / n_hues
    
    for i, x_val in enumerate(order):
        for j, hue_val in enumerate(hue_order):
            subset = df[(df[x] == x_val) & (df[hue] == hue_val)]
            
            if len(subset) > 0 and subset[y].notna().sum() > 0:
                percentile_val = subset[y].quantile(p)
                
                # Calculate x position
                offset = (j - (n_hues - 1) / 2) * box_width
                x_pos = i + offset
                
                # Draw line
                line_width = box_width * frac
                ax.plot(
                    [x_pos - line_width/2, x_pos + line_width/2],
                    [percentile_val, percentile_val],
                    color=color,
                    linewidth=lw,
                    solid_capstyle='butt'
                )


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    print("="*60)
    print("Kaplan-Meier Based Boxplot Generator")
    print("="*60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Thresholds: {thresholds}")
    print("="*60 + "\n")
    
    # Create output directory if needed
    Path("figures_new").mkdir(exist_ok=True)
    
    # ========================================================================
    # Step 1: Load and prepare data
    # ========================================================================
    
    print("Loading data...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows, {df['model'].nunique()} unique models\n")
    
    # Compute average trend per model
    print("Computing trend groups...")
    avg_trends = df.groupby("model")["trend"].mean().reset_index()
    
    # Sort models by trend
    avg_trends_sorted = avg_trends.sort_values("trend").reset_index(drop=True)
    
    # Determine group sizes (split into thirds)
    n_models = len(avg_trends_sorted)
    group_size = n_models // 3
    
    # Define trend boundaries
    low_max = avg_trends_sorted["trend"].iloc[group_size - 1]
    med_max = avg_trends_sorted["trend"].iloc[2 * group_size - 1]
    
    print(f"  Total models: {n_models}")
    print(f"  Group size: {group_size}")
    print(f"  Low threshold: <{low_max:.2f} °C/decade")
    print(f"  Medium threshold: {low_max:.2f}-{med_max:.2f} °C/decade")
    print(f"  High threshold: >{med_max:.2f} °C/decade\n")
    
    # Assign group labels (with newlines for original order display)
    avg_trends_sorted["trend_group_display"] = (
        [f"low (<{low_max:.2f}\n °C/decade)"] * group_size +
        [f"medium ({low_max:.2f}-{med_max:.2f}\n °C/decade)"] * group_size +
        [f"high (>{med_max:.2f}\n °C/decade)"] * (n_models - 2 * group_size)
    )
    
    # Also create clean versions (without newlines) for KM functions
    avg_trends_sorted["trend_group"] = (
        [f"low (<{low_max:.2f} °C/decade)"] * group_size +
        [f"medium ({low_max:.2f}-{med_max:.2f} °C/decade)"] * group_size +
        [f"high (>{med_max:.2f} °C/decade)"] * (n_models - 2 * group_size)
    )
    
    # Merge back to original data
    df_plot = df.merge(
        avg_trends_sorted[["model", "trend_group", "trend_group_display"]], 
        on="model"
    )
    
    # Save grouped data
    df_plot.to_csv('model_groups_km.csv', index=False)
    print(f"Saved grouped data to model_groups_km.csv\n")
    
    # ========================================================================
    # Step 2: Generate KM-based samples
    # ========================================================================
    
    # Get unique trend groups (clean versions)
    trend_groups = sorted(df_plot['trend_group'].unique())
    print(f"Trend groups: {trend_groups}\n")
    
    # Get display versions for plotting order - FIXED ORDER: low, medium, high
    # Create explicit order based on the group labels
    order = [
        f"low (<{low_max:.2f}\n °C/decade)",
        f"medium ({low_max:.2f}-{med_max:.2f}\n °C/decade)",
        f"high (>{med_max:.2f}\n °C/decade)"
    ]
    print(f"Display order: {order}\n")
    
    # Generate KM samples
    print("Generating Kaplan-Meier samples...\n")
    df_km_samples = create_km_boxplot_data(
        df_plot, 
        thresholds=thresholds,
        scenarios=df_plot['scenario'].unique(),
        trend_groups=trend_groups,
        n_samples=1000,
        year_min=1850,
        year_max=2100,
        verbose=True
    )
    
    # Add display version of trend_group for plotting
    # Map clean -> display version
    trend_map = df_plot[['trend_group', 'trend_group_display']].drop_duplicates()
    trend_map_dict = dict(zip(trend_map['trend_group'], trend_map['trend_group_display']))
    df_km_samples['trend_group_display'] = df_km_samples['trend_group'].map(trend_map_dict)
    
    # Save KM samples
    df_km_samples.to_csv('km_samples_generated.csv', index=False)
    print(f"Saved KM samples to km_samples_generated.csv\n")
    
    # ========================================================================
    # Step 3: Create boxplots
    # ========================================================================
    
    print("Creating boxplots...\n")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Plot each threshold
    for i, threshold in enumerate(thresholds):
        ax = axes[i]
        print(f"Plotting threshold {threshold}°C...")
        
        # Filter for this threshold
        df_thresh = df_km_samples[df_km_samples['threshold'] == threshold]
        
        if len(df_thresh) == 0:
            print(f"  WARNING: No data for threshold {threshold}°C")
            ax.text(0.5, 0.5, f"No data for {threshold}°C", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Threshold = {threshold}°C")
            continue
        
        # Plot boxplot
        sns.boxplot(
            data=df_thresh,
            x='trend_group_display',
            y='first_year',
            hue='scenario',
            ax=ax,
            order=order,
            palette=colors_ssp,
            legend=False
        )
        
        # Get scenarios present in this threshold
        hue_order_now = [h for h in colors_ssp.keys() 
                        if h in df_thresh['scenario'].unique()]
        
        # Add 66th percentile lines
        add_percentile_lines(
            ax, df_thresh, 
            x='trend_group_display', 
            y='first_year',
            hue='scenario',
            order=order,
            hue_order=hue_order_now,
            width=0.80,
            p=0.66, 
            frac=0.8,
            color="black", 
            lw=3
        )
        
        # Formatting
        ax.set_ylim(1999, 2100)
        ax.set_title(f"Threshold = {threshold}°C", fontsize=20, fontweight='bold')
        ax.set_xlabel("Warming Trend Group", fontsize=18)
        ax.set_ylabel("Year of Threshold Crossing", fontsize=18)
        ax.tick_params(axis='both', labelsize=16)
        if i ==0:

	        ax.axhline(2023, color='black', linestyle='--', linewidth=3, alpha=0.5)
        ax.text(0.02, 0.98, labels_[i], transform=ax.transAxes,
                fontsize=28, fontweight='bold', va='top', ha='left')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels if they have newlines
        ax.tick_params(axis='x', rotation=0)
    
    # Create legend
    legend_elements = [
        Patch(facecolor=colors_ssp[key], label=SSP_NAME_PRETTY[key]) 
        for key in colors_ssp
    ]
    legend_elements.append(
        Line2D([0], [0], color="black", lw=3, label="66th percentile")
    )
    
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(legend_elements),
        fontsize=16,
        frameon=True,
        title="Scenario",
        title_fontsize=18
    )
    
    # Layout adjustment
   # plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"Figure saved to: {output_file}")
    print(f"{'='*60}")
    
    plt.close()
    
    # ========================================================================
    # Step 4: Generate summary statistics
    # ========================================================================
    
    print("\nGenerating summary statistics...\n")
    
    summary_rows = []
    for threshold in thresholds:
        df_thresh = df_km_samples[df_km_samples['threshold'] == threshold]
        
        if len(df_thresh) > 0:
            summary = (
                df_thresh
                .groupby(['trend_group', 'scenario'])['first_year']
                .describe(percentiles=[0.1, 0.25, 0.5, 0.66, 0.75, 0.9])
                .reset_index()
            )
            summary['threshold'] = threshold
            summary_rows.append(summary)
    
    if summary_rows:
        summary_all = pd.concat(summary_rows, ignore_index=True)
        summary_all.to_csv('km_summary_statistics.csv', index=False)
        print(f"Summary statistics saved to: km_summary_statistics.csv")
        print(f"\nSample statistics for threshold 2.0°C:")
        print(summary_all[summary_all['threshold'] == 2.0][
            ['trend_group', 'scenario', 'count', '50%', '66%']
        ].to_string(index=False))
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
