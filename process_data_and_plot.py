import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import gcsfs
import cftime
import time
import glob
import os
from pathlib import Path
import matplotlib.patches as patches
import matplotlib.dates as mdates
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import WhiteKernel
import matplotlib.colors as mcolors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import scipy.stats as stats
from matplotlib.patches import Patch
import itertools
import cmocean
import scipy
from matplotlib.lines import Line2D


def add_percentile_lines(ax, data, x, y, hue=None, order=None, hue_order=None,
                         width=0.80, p=0.66, frac=0.8,
                         color="black", lw=3, linestyle="-", zorder=25):
    """
    Draw a short horizontal line at the `p`-quantile inside each box.
    - width: MUST match the `sns.boxplot(..., width=...)`
    - frac:  fraction of an individual box width to draw (0..1)
    """
    # category order
    cats = order if order is not None else list(pd.Index(data[x]).astype(str).unique())
    if hue is not None:
        hues = hue_order if hue_order is not None else list(pd.Index(data[hue]).astype(str).unique())
    else:
        hues = [None]

    # quantiles
    if hue is None:
        q = data.groupby(x, observed=True)[y].quantile(p)
    else:
        q = data.groupby([x, hue], observed=True)[y].quantile(p)

    n_h = len(hues)
    group_w = float(width)
    box_w   = group_w / n_h if n_h > 0 else group_w
    half_len = (box_w * frac) / 2.0

    for i, cat in enumerate(cats):
        for j, hv in enumerate(hues):
            # center x position (same dodge layout as seaborn)
            if hue is None:
                x_pos = i
                y_q = q.get(cat, np.nan)
            else:
                x_pos = i - group_w/2 + (j + 0.5) * box_w
                y_q = q.get((cat, hv), np.nan)

            if not np.isfinite(y_q):
                continue

            ax.hlines(y=y_q, xmin=x_pos - half_len, xmax=x_pos + half_len,
                      color=color, lw=lw, linestyle=linestyle, zorder=zorder)
def get_median_rows(df):
    median_rows = []
    grouped = df.groupby(['threshold', 'scenario', 'trend_group'])

    for (threshold, scenario, trend_group), subdf in grouped:
        median_trend = subdf['trend'].median()
        idx = (subdf['trend'] - median_trend).abs().idxmin()
        median_rows.append(subdf.loc[idx])
    
    return pd.DataFrame(median_rows)

# =========================
# 1. Define paths and thresholds
# =========================

thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
adjusted_thresholds = [t - 0.88 for t in thresholds]
labels_=["a)","b)","c)","d)"]
models="all"
if models =="all":
	files = {
	    "ssp119": "cmip6/ssp119_allmodels.nc",
	    "ssp126": "cmip6/ssp126_allmodels.nc",
	    "ssp245": "cmip6/ssp245_allmodels.nc",
	    "ssp370": "cmip6/ssp370_allmodels.nc",
	    "ssp585": "cmip6/ssp585_allmodels.nc",
	#    "obs" :   "era5_adjusted.nc"
	}
else:
	files = {
	    "ssp126": "cmip6/ssp126_anom.nc",
	    "ssp245": "cmip6/ssp245_anom.nc",
	    "ssp370": "cmip6/ssp370_anom.nc",
	    "ssp585": "cmip6/ssp585_anom.nc",
	#    "obs" :   "era5_adjusted.nc"
	}

SSP_NAME_PRETTY = {
    "ssp119": "SSP1–1.9",
    "ssp126": "SSP1–2.6",
    "ssp245": "SSP2–4.5",
    "ssp370": "SSP3–7.0",
    "ssp585": "SSP5–8.5",
}	
colors_ssp={'ssp119':'#00a9cf','ssp126':'#1e9583','ssp245':'#4576be','ssp370':'#f11111','ssp585':'#8036a7'}
# =========================
# 2. Time decoding function
# =========================

def decode_time_no_leap(time_var):
    base_date = np.datetime64("1850-01-16T12:00:00")
    return base_date + time_var.astype("timedelta64[h]")

# =========================
# 3. Find first threshold crossing year for each model
# =========================

def GPtrendEarly(y):
    t = np.arange(0, 50).astype(float)
    length_scale = 50.0  # years
    kernel = RBF(length_scale=length_scale, length_scale_bounds="fixed") + WhiteKernel(noise_level=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, alpha=1e-12)
    print(len(t),len(y))
    p50Years = np.polyfit(t, y, deg=1)
    p0, p1 = p50Years[0], p50Years[1]
    gp.fit(t[:, None], y[:, None] - (p0 * t[:, None] + p1))
    # Linear trend for the first 10 years, based on GP model data
    GPdata10years = gp.predict(np.arange(0, 10)[:, None], return_std=False).flatten() + (p0 * np.arange(0, 10) + p1)
    return np.polyfit(np.arange(0, 10), GPdata10years, deg=1)[0]*10


def GPtrendLate(y):
    t = np.arange(0, 50).astype(float)
    length_scale = 50.0  # years
    kernel = RBF(length_scale=length_scale, length_scale_bounds="fixed") + WhiteKernel(noise_level=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, alpha=1e-12)
    p50Years = np.polyfit(t, y, deg=1)
    p0, p1 = p50Years[0], p50Years[1]
    gp.fit(t[:, None], y[:, None] - (p0 * t[:, None] + p1))
    # Linear trend for the first 10 years, based on GP model data
    GPdata10years = gp.predict(np.arange(40, 50)[:, None], return_std=False).flatten() + (p0 * np.arange(40, 50) + p1)
    return np.polyfit(np.arange(40, 50), GPdata10years, deg=1)[0]*10



def find_crossing_years(ds, thresholds):
    results = []
    years = ds.time.dt.year.data#pd.to_datetime(decode_time_no_leap(ds.time.values)).year
    for model in ds.source_id.values:
        tas = ds.sel(source_id=model).tas
        print(len(tas.sel(time=slice('2015-01-01','2065-01-01')).groupby('time.year').mean().values))
        print(tas.sel(time=slice('2015-01-01','2065-01-01')).groupby('time.year').mean().values)
        x=np.squeeze(tas.sel(time=slice('2015-01-01','2065-01-01')).groupby('time.year').mean().values)[0:50]
        trend = GPtrendEarly(x)
        for threshold in thresholds:
            crossed = tas > threshold
            if np.any(crossed):
                first_year = years[np.argmax(crossed.values)]
            else:
                first_year = np.nan
            results.append((model, threshold , first_year,trend))  # restore original label
    return pd.DataFrame(results, columns=["model", "threshold", "first_year",'trend'])

def find_crossing_years_era5(ds, thresholds):
    results = []
    years = pd.to_datetime((ds.valid_time.values)).year
    tas = ds.t2m
    print(years)
    print("YEARS")
    print((tas.sel(valid_time=slice('1975-01-01','2024-01-01')).groupby('valid_time.year').mean().values))

    trend=GPtrendLate(tas.sel(valid_time=slice('1975-01-01','2024-01-01')).groupby('valid_time.year').mean().values)
    for threshold in thresholds:
            crossed = tas > threshold
            if np.any(crossed):
                first_year = years[np.argmax(crossed.values)]
            else:
                first_year = np.nan
            results.append(('ERA5', threshold, first_year,trend))  # restore original label
    return pd.DataFrame(results, columns=["model", "threshold", "first_year",'trend'])


###Prepare data

# =========================
# 4. Loop through files and calculate threshold years
# =========================

combined_results = []
#take only account models which perform well in historical period
model_filter=False
model_data_ = pd.read_csv("model_list_cleaned.csv")
#model_data_ = model_data_[model_data_['scenario'] == 'scenario'][model_data_['part_of'] == 'yes']['model'].unique()
if model_filter:
    models="filtered"
else:
    models="all"
for scenario, filepath in files.items():
    print(model_data_[model_data_['scenario'] == scenario])
    if model_filter:
        model_data__ = model_data_[model_data_['scenario'] == scenario][model_data_['part_of'] == 'yes']['model'].unique()
        ds = xr.open_dataset(filepath).rolling(time=12).mean()
        ds=ds.sel(source_id=ds.source_id.isin(model_data__))
    else:
        model_data__ = model_data_[model_data_['scenario'] == scenario]['model'].unique()
        ds = xr.open_dataset(filepath).rolling(time=12).mean()
        ds=ds.sel(source_id=ds.source_id.isin(model_data__))	    
    print(model_data__,scenario)
    print(ds)
    #input('wait')
    df = find_crossing_years(ds, thresholds)
    df["scenario"] = scenario
    combined_results.append(df)
    print(df)
    
#print(combined_results)
combined_df=pd.concat(combined_results)
print(combined_df)
combined_df.to_csv("cmip6_data.csv", index=False)
# =========================
# 4.1 Loop through observationss and calculate threshold years
# =========================
ds=xr.open_dataset("era5_adjusted.nc")
obs = find_crossing_years_era5(ds, thresholds)
obs.to_csv("era5_data.csv", index=False)
print(obs)
# =========================
# 5. Plot all scenarios in one plot
# =========================
plt.rcParams.update({'font.size': 24})
fig, ax = plt.subplots(figsize=(13, 7))
scenarios = combined_df["scenario"].unique()

for scenario in scenarios:
    scenario_df = combined_df[combined_df["scenario"] == scenario]

    grouped = scenario_df.groupby("threshold").agg(
        mean_year=("first_year", "mean"),
        std_year=("first_year", "std")
    ).reset_index().dropna()

    thresholds_np = grouped["threshold"].to_numpy(dtype=float)
    mean_year_np = grouped["mean_year"].to_numpy(dtype=float)
    std_year_np = grouped["std_year"].to_numpy(dtype=float)

    ax.plot(mean_year_np, thresholds_np, marker="o", label=SSP_NAME_PRETTY[scenario],color=colors_ssp[scenario],linewidth=5)
    ax.fill_betweenx(
        thresholds_np,
        mean_year_np - std_year_np,
        mean_year_np + std_year_np,
        alpha=0.2,color=colors_ssp[scenario]
    )
ax.plot(obs['first_year'],obs['threshold'],'x',markersize=10,color='k',label="ERA5",markeredgewidth=4)
#ax.set_title("Global Warming Threshold Crossing Years (±1 Std Dev)")
ax.set_xlabel("Year")
ax.set_ylabel("Warming Threshold (°C)")
ax.legend(title="Scenario")
ax.grid(True)

plt.savefig("figures/crossing_cs_threshold_"+models+".png")
plt.close()
# =========================
# 6.1 plot PDF's
# =========================
df_model=combined_df
df_model_fixed = df_model.dropna(subset=["first_year"]).copy()
df_model_fixed["first_year"] = df_model_fixed["first_year"].astype(int)

lines = []  # store line handles
labels = []  # store scenario labels (only once)

# Threshold values to visualize
threshold_vals = [1.5, 2.0, 2.5, 3.0]
scenarios = df_model_fixed["scenario"].unique()

from scipy.stats import gaussian_kde

def kde_pdf_bounded_df(x, bounds=None, n=512, bw_adjust=1.0, reflect=True):
    """
    KDE PDF on [L,U] only. Uses boundary reflection and renormalizes so ∫_L^U pdf=1.
    Returns DataFrame(year, pdf) on an even grid within [L,U].
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No data")

    L = x.min() if bounds is None else float(bounds[0])
    U = x.max() if bounds is None else float(bounds[1])
    if not (L < U):
        raise ValueError("Invalid bounds")

    xx = x
    if reflect:
        # rough bandwidth for a window near the edges
        kde_tmp = gaussian_kde(xx)
        kde_tmp.set_bandwidth(kde_tmp.factor * bw_adjust)
        # 1D bandwidth proxy (Scott’s rule scaled by std)
        h = np.sqrt(kde_tmp.covariance.squeeze())
        # reflect points within ~3h of the bounds
        mask_lo = (xx - L) < 3*h
        mask_hi = (U - xx) < 3*h
        xx = np.concatenate([xx, 2*L - xx[mask_lo], 2*U - xx[mask_hi]])

    kde = gaussian_kde(xx)
    kde.set_bandwidth(kde.factor * bw_adjust)

    grid = np.linspace(L, U, n)
    pdf = kde(grid)
    # clip (should already be in-range) and renormalize on [L,U]
    pdf = np.where((grid >= L) & (grid <= U), pdf, 0.0)
    area = np.trapz(pdf, grid)
    if area > 0:
        pdf = pdf / area

    return pd.DataFrame({"year": grid, "pdf": pdf})

def histogram_pdf_df(x, bins="auto", bounds=None):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No data")
    L = x.min() if bounds is None else bounds[0]
    U = x.max() if bounds is None else bounds[1]

    counts, edges = np.histogram(x, bins=bins, range=(L, U), density=True)
    centers = 0.5*(edges[:-1] + edges[1:])
    return pd.DataFrame({"year": centers, "pdf": counts})

# Create 2x2 grid for plotting
fig, axes = plt.subplots(2, 2, figsize=(15, 15),sharex=True,sharey=True)
axes = axes.flatten()
positions=[0.04,0.037,0.034,0.031,0.028]
# Loop over thresholds and plot KDEs for each scenario
for i, threshold in enumerate(threshold_vals):
    ax = axes[i]
    for sce_ind,scenario in enumerate(scenarios):
        subset = df_model_fixed[
            (df_model_fixed["threshold"] == threshold) &
            (df_model_fixed["scenario"] == scenario)
        ]
        if len(subset) > 1:
            
            sns.kdeplot(
                data=subset,bw_adjust=1.5,
                x="first_year",
                label=scenario,
                ax=ax,color=colors_ssp[scenario],linewidth=5
            )
            '''
            sns.histplot(
                data=subset,
                x="first_year",
                label=scenario,
                ax=ax,color=colors_ssp[scenario],linewidth=5,kde=True
            )
            '''
            subset.to_csv("pdf_data/"+scenario+"_"+str(threshold)+"_"+models+".csv")
            df_hist = kde_pdf_bounded_df(subset['first_year'], bw_adjust=1.0)             # discrete, no leakage
            #ax.plot(df_hist["year"], df_hist["pdf"], linestyle="--", label=scenario,color=colors_ssp[scenario],linewidth=5)
            print(df_hist)
            #input('wait')
            if i == 0:  # only collect legend items once
                lines.append(ax.lines[-1])
                labels.append(SSP_NAME_PRETTY[scenario])
            print(subset)    
            median_val = np.median(subset['first_year'])
            median_txt=str(int(median_val))
            ax.axvline(median_val, color=colors_ssp[scenario], linestyle='--',linewidth=3)
            ax.text(2080, positions[sce_ind], f'Median: {median_txt}', rotation=0,
             verticalalignment='bottom', color=colors_ssp[scenario])
            print(positions[sce_ind])
            #input('wait')
    ax.set_title(f"Threshold = {threshold}°C")
    ax.set_xlabel("Year of threshold crossing")
    ax.set_ylabel("Density")
    ax.tick_params(labelleft=True)
    ax.yaxis.set_tick_params(labelleft=True)    #ax.legend()
    ax.grid(True)
    ax.text(0.01, 0.99, labels_[i], transform=ax.transAxes,
        fontsize=24, fontweight='bold', va='top', ha='left')
    if threshold == 1.5:
        ax.axvline(x=2023, color='black', linestyle='--', label='Year 2023',linewidth=5)

fig.legend(
    handles=lines,
    labels=labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=len(labels),
    title="Scenario"
)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig('figures/PDF_'+models+'.png')
plt.close()
# =========================
# 6.1 plot PDF's cumulati
# =========================

def calculateCDF(df):
    df_y = pd.DataFrame({"first_year": df.to_list()})
    x = np.sort(df_y)

    # Empirical CDF (values between 0 and 1)
    cdf =scipy.stats.norm.cdf(x)#np.arange(1, len(x)+1) / len(x)
    print(x)
    print(cdf)
    cdf_df = pd.DataFrame({"year": x, "cdf": cdf})
    print(cdf_df)
    input('wait')
    return cdf_df
    
from scipy.stats import rankdata
from scipy.stats import gaussian_kde
 
def ecdf_scipy(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    r = rankdata(x, method='max')          # right-continuous ECDF
    order = np.argsort(x)
    cdf = x[order], (r[order] / n)    
    return pd.DataFrame({'year': x, 'cdf': cdf})

def kde_cdf_df(samples, grid=None, bw_adjust=1.0):
    samples = np.asarray(samples, float)
    samples = samples[np.isfinite(samples)]
    kde = gaussian_kde(samples)
    kde.set_bandwidth(kde.factor * bw_adjust)

    if grid is None:
        lo, hi = np.percentile(samples, [0.5, 99.5])
        pad = 0.1 * (hi - lo)
        grid = np.linspace(lo - pad, hi + pad, 512)

    # gaussian_kde has an integral method for the CDF
    F = np.array([kde.integrate_box_1d(-np.inf, xi) for xi in grid])
    return pd.DataFrame({'year': grid, 'cdf': F})

def ecdf_scipy_df(years, drop_duplicates=True, method='right'):
    """
    years : 1D array-like
    returns DataFrame with columns ['year', 'cdf'] sorted by year.
    method: 'right' (default, right-continuous), 'left', or 'mid' (average).
    """
    y = np.asarray(years, float)
    y = y[np.isfinite(y)]
    n = y.size
    if n == 0:
        raise ValueError("No valid values for ECDF.")

    m = {'right': 'max', 'left': 'min', 'mid': 'average'}
    ranks = rankdata(y, method=m.get(method, 'max'))  # default right-continuous
    ys = y[order]
    F = ranks[order] / n

    df = pd.DataFrame({'year': ys, 'cdf': F})
    if drop_duplicates:
        # keep one row per unique year with the max CDF at that year (right-continuous)
        df = df.groupby('year', as_index=False, sort=True)['cdf'].max()
    return df
    
df_model=combined_df
df_model_fixed = df_model.dropna(subset=["first_year"]).copy()
df_model_fixed["first_year"] = df_model_fixed["first_year"].astype(int)

# Threshold values to visualize
threshold_vals = [1.5, 2.0, 2.5, 3.0]
scenarios = df_model_fixed["scenario"].unique()

# Create 2x2 grid for plotting
fig, axes = plt.subplots(2, 2, figsize=(15, 15),sharex=True,sharey=True)
axes = axes.flatten()

# Loop over thresholds and plot KDEs for each scenario
for i, threshold in enumerate(threshold_vals):
    ax = axes[i]
    for scenario in scenarios:
        subset = df_model_fixed[
            (df_model_fixed["threshold"] == threshold) &
            (df_model_fixed["scenario"] == scenario)
        ]
        if len(subset) > 1:
            sns.kdeplot(
                data=subset,
                x="first_year",
                label=scenario,
                ax=ax,color=colors_ssp[scenario],linewidth=5,cumulative=True
            )

            print(subset)
            #cdf_df=ecdf_scipy_df(subset['first_year'])
            #print(cdf_df)
            #input('wait')
            #cdf_df.to_csv("cdf_data/"+scenario+"_"+str(threshold)+".csv")
            #ax.plot(cdf_df['year'],cdf_df['cdf'],'--',color=colors_ssp[scenario])
    ax.set_title(f"Threshold = {threshold}°C")
    ax.set_xlabel("Year of threshold crossing")
    ax.set_ylabel("Density")
    ax.tick_params(labelleft=True)
    ax.yaxis.set_tick_params(labelleft=True)
    #ax.legend()
    ax.grid(True)
    ax.text(0.01, 0.99, labels_[i], transform=ax.transAxes,
        fontsize=24, fontweight='bold', va='top', ha='left')
    if threshold == 1.5:
        ax.axvline(x=2023, color='black', linestyle='--', label='Year 2023',linewidth=5)
    print(subset)
    
print(lines)
print(labels)
fig.legend(
    handles=lines,
    labels=labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=len(labels),
    title="Scenario"
)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig('figures/cumulative_'+models+'.png')
plt.close()
#=========================
# 6.2 plot PDF's surface
# =========================
trend_threshold = 0.3




df = pd.read_csv("cmip6_data.csv")

# Step 1: Compute average trend per model
avg_trends = df.groupby("model")["trend"].mean().reset_index()

# Step 2: Sort models by trend
avg_trends_sorted = avg_trends.sort_values("trend").reset_index(drop=True)

# Step 3: Determine group sizes
n_models = len(avg_trends_sorted)
group_size = n_models // 3

# Step 4: Define trend boundaries
low_max = avg_trends_sorted["trend"].iloc[group_size - 1]
med_max = avg_trends_sorted["trend"].iloc[2 * group_size - 1]

# Step 5: Assign human-readable group labels with thresholds
avg_trends_sorted["trend_group"] = (
    [f"low (<{low_max:.2f}\n °C/decade)"] * group_size +
    [f"medium ({low_max:.2f}-{med_max:.2f}\n °C/decade)"] * group_size +
    [f"high (>{med_max:.2f}\n °C/decade)"] * (n_models - 2 * group_size)
)

# Step 6: Merge group labels back to the original dataset
df_grouped_named = df.merge(avg_trends_sorted[["model", "trend_group"]], on="model")
df_plot=df_grouped_named
print(df_grouped_named)
df_grouped_named.to_csv('model_groups.csv')
fig, axes = plt.subplots(2, 2, figsize=(24, 15),sharex=True,sharey=True)
axes = axes.flatten()
print(df_model_fixed)
thresholds = [1.5, 2.0, 2.5, 3.0]
labels_=["a)","b)","c)","d)"]
order=['low (<0.28\n °C/decade)','medium (0.28-0.33\n °C/decade)' ,'high (>0.33\n °C/decade)']
print(df_plot.trend_group.unique())
for i, threshold in enumerate(thresholds):
    ax = axes[i]
    df_thresh = df_plot[df_plot['threshold'] == threshold]
    
    scenarios = df_thresh['scenario'].unique()
    trend_groups = [
        'low (<0.28 °C/decade)',
        'medium (0.28-0.33 °C/decade)',
        'high (>0.33 °C/decade)'
    ]

    # All possible combinations
    all_combos = pd.DataFrame(
        list(itertools.product(scenarios, trend_groups)),
        columns=['scenario', 'trend_group']
    )

    # Merge with original data
    df_filled = (
        all_combos
        .merge(df_thresh, on=['scenario','trend_group'], how='left')
    )
    
    #print(get_median_rows(df_thresh))
    #input('wait')
    print(df_filled)
    sns.boxplot(
        data=df_thresh,
        x='trend_group',
        y='first_year',
        hue='scenario',
        ax=ax,order=order,
        palette=colors_ssp,legend=False
    )

    hue_order_now = [h for h in colors_ssp.keys() if h in df_thresh['scenario'].unique()]
    add_percentile_lines(
    ax, df_thresh, x='trend_group', y='first_year',
    hue='scenario',
    order=order,
    hue_order=hue_order_now,
    width=0.80,        # MUST match the boxplot width above
    p=0.66, frac=0.8,
    color="black", lw=3
    )

    summary_table = (
        df_thresh
        .groupby(['trend_group', 'scenario'])['first_year']
        .describe(percentiles=[0.25, 0.5, 0.75])
        .rename(columns={'50%': 'median', '25%': 'q1', '75%': 'q3'})
        .reset_index()
    ).round(0)#.astype(int)
    numeric_cols = summary_table.select_dtypes(include='number').columns
    summary_table[numeric_cols]  = summary_table[numeric_cols] .fillna(0).replace([float('inf'), float('-inf')], 0).astype(int)


    summary_table[numeric_cols] = summary_table[numeric_cols].round(0).astype(int)
    latex_code = summary_table.to_latex(index=False, caption="Summary statistics of first year of threshold crossing by trend group and scenario. "+str(threshold), label="tab:summary_stats_"+str(threshold), column_format='llrrrrrrrr')
    latex_code = latex_code.replace("\\begin{table}", "\\begin{sidewaystable}")
    latex_code = latex_code.replace("\\end{table}", "\\end{sidewaystable}")
    latex_code = latex_code.replace("\\begin{tabular}", "\\begin{tabular*}{\\textheight}")
    latex_code = latex_code.replace("\\end{tabular}", "\\end{tabular*}")
    # Print or save LaTeX code
    print(threshold)
    print(summary_table)
   # ax.set_xticklabels(order)
    ax.set_ylim(1999,2100)
    ax.set_title(f"Threshold = {threshold}°C")
    ax.set_xlabel("Warming Trend Group")
    ax.set_ylabel("Year of Threshold Crossing")
    ax.axhline(2023, color='black', linestyle='--', linewidth=1)
    ax.text(0.01, 0.99, labels_[i], transform=ax.transAxes,
        fontsize=24, fontweight='bold', va='top', ha='left')
    
#fig.legend(title="Scenario", loc="upper left")
legend_elements = [Patch(facecolor=colors_ssp[key], label=SSP_NAME_PRETTY[key]) for key in colors_ssp]
legend_elements.append(Line2D([0],[0], color="black", lw=3, label="66th percentile"))

fig.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=len(legend_elements),
    title="Scenario"
)
# Layout adjustment
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("figures/boxplot_"+models+".png")

fig, axes = plt.subplots(2, 1, figsize=(24, 15),sharex=True,sharey=True)
axes = axes.flatten()
print(df_model_fixed)
thresholds = [1,1.5]
labels_=["a)","b)"]
order=['low (<=0.28\n °C/decade)','medium (0.28-0.33\n °C/decade)' ,'high (>0.33\n °C/decade)']
print(df_plot.trend_group.unique())
for i, threshold in enumerate(thresholds):
    ax = axes[i]
    df_thresh = df_plot[df_plot['threshold'] == threshold]
    #print(get_median_rows(df_thresh))
    #input('wait')
    sns.boxplot(
        data=df_thresh,
        x='trend_group',
        y='first_year',
        hue='scenario',
        ax=ax,#order=order,
        palette=colors_ssp,legend=False
    )

    summary_table = (
        df_thresh
        .groupby(['trend_group', 'scenario'])['first_year']
        .describe(percentiles=[0.25, 0.5, 0.75])
        .rename(columns={'50%': 'median', '25%': 'q1', '75%': 'q3'})
        .reset_index()
    ).round(0)#.astype(int)
    numeric_cols = summary_table.select_dtypes(include='number').columns
    summary_table[numeric_cols]  = summary_table[numeric_cols] .fillna(0).replace([float('inf'), float('-inf')], 0).astype(int)


    summary_table[numeric_cols] = summary_table[numeric_cols].round(0).astype(int)
    latex_code = summary_table.to_latex(index=False, caption="Summary statistics of first year of threshold crossing by trend group and scenario. "+str(threshold), label="tab:summary_stats_"+str(threshold), column_format='llrrrrrrrr')
    latex_code = latex_code.replace("\\begin{table}", "\\begin{sidewaystable}")
    latex_code = latex_code.replace("\\end{table}", "\\end{sidewaystable}")
    latex_code = latex_code.replace("\\begin{tabular}", "\\begin{tabular*}{\\textheight}")
    latex_code = latex_code.replace("\\end{tabular}", "\\end{tabular*}")
    # Print or save LaTeX code
    print(threshold)
    print(summary_table)
    #ax.set_xticklabels(order)
    ax.set_ylim(1999,2100)
    ax.set_title(f"Threshold = {threshold}°C")
    ax.set_xlabel("Warming Trend Group")
    ax.set_ylabel("Year of Threshold Crossing")
    ax.axhline(2023, color='black', linestyle='--', linewidth=1)
    ax.text(0.01, 0.99, labels_[i], transform=ax.transAxes,
        fontsize=24, fontweight='bold', va='top', ha='left')
    
#fig.legend(title="Scenario", loc="upper left")
legend_elements = [Patch(facecolor=colors_ssp[key], label=SSP_NAME_PRETTY[key]) for key in colors_ssp]
fig.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=len(labels),
    title="Scenario"
)
# Layout adjustment
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("figures/boxplot_"+models+"_1deg.png")
