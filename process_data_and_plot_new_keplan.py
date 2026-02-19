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
from matplotlib.patches import Polygon

def polygon_between_lines(
    x1, y1, x2, y2, *,
    assume_same_x: bool = True,
    sort_by_x: bool = True,
) -> np.ndarray:
    """
    Return Nx2 array of polygon vertices that fills the area between two polylines.

    Parameters
    ----------
    x1, y1 : array-like
        Coordinates of line 1.
    x2, y2 : array-like
        Coordinates of line 2.
    assume_same_x : bool
        If True, requires x1 and x2 to match (after optional sorting) and uses them directly.
        If False, interpolates y2 onto x1 (requires unique, monotonic-ish x2).
    sort_by_x : bool
        If True, sorts each line by x before constructing polygon.

    Returns
    -------
    verts : (N, 2) ndarray
        Polygon vertices in order, closed (first vertex is NOT repeated; Polygon(..., closed=True) closes it).
    """
    x1 = np.asarray(x1, dtype=float).ravel()
    y1 = np.asarray(y1, dtype=float).ravel()
    x2 = np.asarray(x2, dtype=float).ravel()
    y2 = np.asarray(y2, dtype=float).ravel()

    if x1.size != y1.size:
        raise ValueError(f"x1 and y1 must have same length, got {x1.size} and {y1.size}")
    if x2.size != y2.size:
        raise ValueError(f"x2 and y2 must have same length, got {x2.size} and {y2.size}")

    if sort_by_x:
        i1 = np.argsort(x1)
        x1, y1 = x1[i1], y1[i1]
        i2 = np.argsort(x2)
        x2, y2 = x2[i2], y2[i2]

    if assume_same_x:
        if x1.size != x2.size or not np.allclose(x1, x2, rtol=0, atol=0):
            raise ValueError(
                "assume_same_x=True requires x1 and x2 to match. "
                "Set assume_same_x=False to interpolate."
            )
        x = x1
        y_top = y1
        y_bot = y2
    else:
        # Interpolate y2 onto x1 (common x = x1)
        # Requires x2 to be strictly increasing for np.interp (duplicates are bad).
        if np.any(np.diff(x2) <= 0):
            raise ValueError("For interpolation, x2 must be strictly increasing (after sorting).")
        x = x1
        y_top = y1
        y_bot = np.interp(x, x2, y2)

    # Build polygon: go along line1 forward, then line2 backward
    xs = np.concatenate([x, x[::-1]])
    ys = np.concatenate([y_top, y_bot[::-1]])
    verts = np.column_stack([xs, ys])

    # Optional: drop any NaN vertices (simple approach)
    mask = np.isfinite(verts).all(axis=1)
    verts = verts[mask]
    if verts.shape[0] < 3:
        raise ValueError("Not enough finite points to form a polygon.")

    return verts


def fill_between_lines(
    ax,
    x1, y1, x2, y2, *,
    facecolor=None,
    edgecolor=None,
    alpha: float = 0.3,
    linewidth: float = 1.5,
    assume_same_x: bool = False,
    sort_by_x: bool = True,
    zorder: int = 1,
) -> Polygon:
    """
    Add a Polygon patch filling the area between two lines to the given Matplotlib axis.
    Returns the created Polygon.
    """
    verts = polygon_between_lines(
        x1, y1, x2, y2,
        assume_same_x=assume_same_x,
        sort_by_x=sort_by_x,
    )
    poly = Polygon(
        verts,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(poly)
    ax.autoscale_view()
    return poly


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
labels_=["a)","b)","c)","d)",'e)','f)','g)','h)','i)','j)']
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
    #print(len(t),len(y))
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
        #print(len(tas.sel(time=slice('2015-01-01','2065-01-01')).groupby('time.year').mean().values))
        #print(tas.sel(time=slice('2015-01-01','2065-01-01')).groupby('time.year').mean().values)
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
    #print(years)
    #print("YEARS")
    #print((tas.sel(valid_time=slice('1975-01-01','2024-01-01')).groupby('valid_time.year').mean().values))

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
    #print(model_data_[model_data_['scenario'] == scenario])
    if model_filter:
        model_data__ = model_data_[model_data_['scenario'] == scenario][model_data_['part_of'] == 'yes']['model'].unique()
        ds = xr.open_dataset(filepath).rolling(time=12).mean()
        ds=ds.sel(source_id=ds.source_id.isin(model_data__))
    else:
        model_data__ = model_data_[model_data_['scenario'] == scenario]['model'].unique()
        ds = xr.open_dataset(filepath).rolling(time=12).mean()
        ds=ds.sel(source_id=ds.source_id.isin(model_data__))	    
    #print(model_data__,scenario)
    #print(ds)
    #input('wait')
    df = find_crossing_years(ds, thresholds)
    df["scenario"] = scenario
    combined_results.append(df)
    #print(df)
    
#print(combined_results)
combined_df=pd.concat(combined_results)
#print(combined_df)

combined_df.to_csv("cmip6_data_"+models+".csv", index=False)
# =========================
# 4.1 Loop through observationss and calculate threshold years
# =========================
ds=xr.open_dataset("era5_adjusted.nc")
obs = find_crossing_years_era5(ds, thresholds)
obs.to_csv("era5_data.csv", index=False)
#print(obs)
# =========================
# 5. Plot all scenarios in one plot
# =========================
plt.rcParams.update({'font.size': 24})
fig, ax = plt.subplots(2,1,figsize=(20, 20))
scenarios = combined_df["scenario"].unique()
cdf_year_df=pd.read_csv("km_cdf66_years.csv")
thresholds = [1.0,1.5, 2.0, 2.5, 3.0,3.5,4,4.5,5]
for scenario in scenarios:
    scenario_df = combined_df[combined_df["scenario"] == scenario]
    d = cdf_year_df[cdf_year_df["scenario"] == scenario]

    grouped = scenario_df.groupby("threshold").agg(
        mean_year=("first_year", "mean"),
        std_year=("first_year", "std"),
        n_items=("first_year", "count")

    ).reset_index().dropna()

    thresholds_np = grouped["threshold"].to_numpy(dtype=float)
    mean_year_np = grouped["mean_year"].to_numpy(dtype=float)
    std_year_np = grouped["std_year"].to_numpy(dtype=float)
    n_items = grouped["n_items"].to_numpy(dtype=float)
    #ax[0].plot(mean_year_np, thresholds_np, marker="o", label=SSP_NAME_PRETTY[scenario],color=colors_ssp[scenario],linewidth=5)

    # be robust to float column labels (0.1 vs 0.10)
    def col(level):
        if level in wide.columns:
            return level
        # fallback for float representation issues
        for c in wide.columns:
            if np.isclose(c, level):
                return c
        raise KeyError(f"CDF level {level} not found in columns: {list(wide.columns)}")


    wide = (d.pivot(index="threshold", columns="cdf_level", values="t_cdf_year")
            .sort_index())
    c10 = col(0.10)
    c66 = col(0.66)
    c90 = col(0.90)
    y = wide.index.to_numpy(float)

    x10 = wide[c10].to_numpy(float)
    x66 = wide[c66].to_numpy(float)
    x90 = wide[c90].to_numpy(float)

    has10 = np.isfinite(x10)
    has66 = np.isfinite(x66)
    has90 = np.isfinite(x90)

    # Plot 0.66 where available
    # 10%
    #ax[0].plot(x10[has10], y[has10], linestyle=":", marker="o",
    #           color=colors_ssp[scenario], linewidth=3)

    # 66%
    ax[0].plot(x66[has66], y[has66], linestyle="-", marker="o",
               color=colors_ssp[scenario], linewidth=4,
               label=SSP_NAME_PRETTY[scenario])

    # 90%
    #ax[0].plot(x90[has90], y[has90], linestyle="--", marker="o",
    #           color=colors_ssp[scenario], linewidth=3)

    #fill_between_lines(ax[0], x90[has90], y[has90],x10[has10], y[has10], alpha=0.25,facecolor=colors_ssp[scenario])
    x_poly = np.concatenate([x66[has66], x10[has10][::-1]])
    y_poly = np.concatenate([y[has66], y[has10][::-1]])
    # Fill the polygon
    ax[0].fill(x_poly, y_poly, alpha=0.2, fc=colors_ssp[scenario])
    x_poly = np.concatenate([x66[has66], x90[has90][::-1]])
    y_poly = np.concatenate([y[has66], y[has90][::-1]])
    # Fill the polygon
    ax[0].fill(x_poly, y_poly, alpha=0.2, fc=colors_ssp[scenario])
    #ax[0].fill(x90[has90], y[has90],x10[has10], y[has10])

    ax[1].plot(thresholds_np,n_items, marker="o", label=SSP_NAME_PRETTY[scenario],color=colors_ssp[scenario],linewidth=5)

ax[0].plot(obs['first_year'],obs['threshold'],'x',markersize=10,color='k',label="ERA5",markeredgewidth=4)
#ax.set_title("Global Warming Threshold Crossing Years (±1 Std Dev)")
ax[0].set_xlabel("Year")
ax[0].set_ylabel("Warming Threshold (°C)")
ax[1].set_ylabel("Number of models")
ax[1].set_xlabel("Warming Threshold (°C)")
ax[0].legend(title="Scenario")
ax[0].grid(True)
ax[1].grid(True)
ax[0].text(0.02, 0.97, 'a)', transform=ax[0].transAxes,
           fontsize=28, fontweight='bold', va='top', ha='left')
ax[1].text(0.02, 0.97, 'b)', transform=ax[1].transAxes,
           fontsize=28, fontweight='bold', va='top', ha='left')
plt.tight_layout()
plt.savefig("figures_new/crossing_cs_threshold_"+models+"_new.png")
plt.savefig("figures_new/crossing_cs_threshold_"+models+"_new.svg",dpi=100)
plt.close()
''
# =========================
# 6.1 plot PDF's
# =========================
df_model=combined_df
df_model_fixed = df_model.dropna(subset=["first_year"]).copy()
df_model_fixed["first_year"] = df_model_fixed["first_year"].astype(int)

lines = []  # store line handles
labels = []  # store scenario labels (only once)

# Threshold values to visualize
threshold_vals = [1.5, 2.0, 2.5, 3.0,3.5,4,4.5,5]
scenarios = df_model_fixed["scenario"].unique()

from scipy.stats import gaussian_kde

def kaplan_meier(durations, observed):
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

	# --- Apufunktio: CDF-käyrä yhdelle (threshold, scenario) parille ---
def km_curve(df_in, threshold, scenario, censor_year=2100):
		d = df_in[(df_in["threshold"] == threshold) & (df_in["scenario"] == scenario)].copy()
		#print(d)
		#input('wait')
		if d.empty:
			return [], []
		events = d["first_year"].dropna().astype(int).tolist()
		censored = d[d["first_year"].isna()]
		censored_years = [censor_year] * len(censored)
		durations = events + censored_years
		observed = [1]*len(events) + [0]*len(censored_years)
		times, surv = kaplan_meier(durations, observed)
		cdf = [1 - s for s in surv]
		return times, cdf


def naive_curve(df_in, threshold, scenario, t_min=2015, t_max=2100):
		d = df_in[(df_in["threshold"] == threshold) & (df_in["scenario"] == scenario)].copy()
		if d.empty:
			return [], []
		total = len(d)
		# käytetään askelpisteinä kaikki first_year-vuodet + päätepiste
		ev_years = sorted(d["first_year"].dropna().astype(int).unique().tolist())
		t_grid = [t for t in ev_years if t >= t_min] + [t_max]
		if len(t_grid) == 0:
			return [], []
		cdf_vals = []
		for t in t_grid:
			num = (d["first_year"].dropna().astype(int) <= t).sum()
			cdf_vals.append(num / total)
		return t_grid, cdf_vals

def km_pdf(df_in, threshold, scenario, censor_year=2100):
    """
    Calculate PDF from Kaplan-Meier CDF using finite differences.
    
    Returns:
        times: time points (midpoints between CDF steps)
        pdf_vals: probability density at each time point
    """
    # Get the CDF from Kaplan-Meier
    times_cdf, cdf = km_curve(df_in, threshold, scenario, censor_year)
    
    if len(times_cdf) < 2:
        return [], []
    
    # Calculate PDF as derivative of CDF: pdf(t) ≈ (CDF(t) - CDF(t-1)) / (t - t-1)
    times_pdf = []
    pdf_vals = []
    
    for i in range(1, len(times_cdf)):
        dt = times_cdf[i] - times_cdf[i-1]
        if dt > 0:  # avoid division by zero
            dcdf = cdf[i] - cdf[i-1]
            pdf = dcdf / dt
            
            # Use midpoint of interval as the time coordinate
            t_mid = (times_cdf[i] + times_cdf[i-1]) / 2
            
            times_pdf.append(t_mid)
            pdf_vals.append(pdf)
    
    return times_pdf, pdf_vals

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
fig, axes = plt.subplots(2, 4, figsize=(15, 15),sharex=True,sharey=True)
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
            

            subset.to_csv("pdf_data/"+scenario+"_"+str(threshold)+"_"+models+".csv")
            df_hist = kde_pdf_bounded_df(subset['first_year'], bw_adjust=1.0)             # discrete, no leakage
            ax.plot(df_hist["year"], df_hist["pdf"], linestyle="--", label=scenario,color=colors_ssp[scenario],linewidth=5)
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
plt.savefig('figures_new/PDF_'+models+'.png')
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
plt.savefig('figures_new/cumulative_'+models+'.png')
plt.close()
#=========================
# 6.2 plot PDF's surface
# =========================
trend_threshold = 0.3
def km_to_samples(df_in, threshold, scenario, censor_year=2100, n_samples=1000):
    """
    Convert Kaplan-Meier CDF to synthetic samples using inverse transform sampling.
    
    This allows you to use the KM-estimated distribution with seaborn boxplot.
    
    Parameters:
    -----------
    df_in : DataFrame
        Input data with threshold crossings
    threshold : float
        Temperature threshold
    scenario : str
        Climate scenario
    censor_year : int
        Year to use for censored observations
    n_samples : int
        Number of samples to generate
        
    Returns:
    --------
    samples : np.ndarray
        Synthetic samples from the KM distribution
    """
    # Get KM CDF
    times_cdf, cdf = km_curve(df_in, threshold, scenario, censor_year)
    
    if len(times_cdf) < 2:
        return np.array([])
    
    # Inverse transform sampling
    # Generate uniform random numbers
    u = np.random.uniform(0, 1, n_samples)
    
    # For each u, find the corresponding year from the CDF
    samples = np.interp(u, cdf, times_cdf)
    
    return samples


def create_km_boxplot_data(df_in, thresholds, scenarios, trend_groups=None, 
                           n_samples=1000, year_min=1850, year_max=2100, verbose=True):
    """Create boxplot-ready DataFrame from KM estimates"""
    rows = []
    
    for threshold in thresholds:
        if verbose:
            print(f"Processing threshold {threshold}°C...")
        for scenario in scenarios:
            if trend_groups is not None:
                for trend_group in trend_groups:
                    print(trend_group)
                    print(df_in['trend_group'].unique())
                    df_subset = df_in[
                        (df_in['threshold'] == threshold) &
                        (df_in['scenario'] == scenario) &
                        (df_in['trend_group'] == trend_group)
                    ]
                    print( df_subset)
                    if len(df_subset) > 0:
                        samples = km_to_samples(df_subset, threshold, scenario, trend_group, 
                                              n_samples=n_samples, year_min=year_min, year_max=year_max)
                        
                        for sample in samples:
                            rows.append({
                                'threshold': threshold,
                                'scenario': scenario,
                                'trend_group': trend_group,
                                'first_year': sample
                            })
            else:
                df_subset = df_in[
                    (df_in['threshold'] == threshold) &
                    (df_in['scenario'] == scenario)
                ]
                
                if len(df_subset) > 0:
                    samples = km_to_samples(df_subset, threshold, scenario, 
                                          n_samples=n_samples, year_min=year_min, year_max=year_max)
                    
                    for sample in samples:
                        rows.append({
                            'threshold': threshold,
                            'scenario': scenario,
                            'first_year': sample
                        })
    
    return pd.DataFrame(rows)



df = pd.read_csv("cmip6_data_"+models+".csv")

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
    [f"low (<{low_max:.2f} °C/decade)"] * group_size +
    [f"medium ({low_max:.2f}-{med_max:.2f} °C/decade)"] * group_size +
    [f"high (>{med_max:.2f} °C/decade)"] * (n_models - 2 * group_size)
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

# Generate KM-based samples for boxplot
trend_groups = [
    'low (<0.28 °C/decade)',
    'medium (0.28-0.33 °C/decade)',
    'high (>0.33 °C/decade)'
]
print('df_plot')
print(df_plot)
# Create synthetic data from KM distributions
df_km_samples = create_km_boxplot_data(
    df_plot, 
    thresholds=thresholds,
    scenarios=df_plot['scenario'].unique(),
    trend_groups=trend_groups,
    n_samples=1000  # Adjust as needed
)

for i, threshold in enumerate(thresholds):
    ax = axes[i]
    print(threshold, 'threshold')
    print(df_km_samples)
    # Filter for this threshold
    df_thresh = df_km_samples[df_km_samples['threshold'] == threshold]
    
    scenarios = df_thresh['scenario'].unique()
    
    # All possible combinations
    all_combos = pd.DataFrame(
        list(itertools.product(scenarios, trend_groups)),
        columns=['scenario', 'trend_group']
    )
    
    # Merge with KM sample data
    df_filled = (
        all_combos
        .merge(df_thresh, on=['scenario','trend_group'], how='left')
    )
    
    print(df_filled)
    
    # Plot boxplot using KM-derived samples
    sns.boxplot(
        data=df_thresh,
        x='trend_group',
        y='first_year',
        hue='scenario',
        ax=ax,
        order=order,
        palette=colors_ssp,
        legend=False
    )
    
    hue_order_now = [h for h in colors_ssp.keys() if h in df_thresh['scenario'].unique()]
    
    add_percentile_lines(
        ax, df_thresh, x='trend_group', y='first_year',
        hue='scenario',
        order=order,
        hue_order=hue_order_now,
        width=0.80,
        p=0.66, frac=0.8,
        color="black", lw=3
    )
    
    # Summary statistics from KM samples
    summary_table = (
        df_thresh
        .groupby(['trend_group', 'scenario'])['first_year']
        .describe(percentiles=[0.25, 0.5, 0.75])
        .rename(columns={'50%': 'median', '25%': 'q1', '75%': 'q3'})
        .reset_index()
    ).round(0)
    
    numeric_cols = summary_table.select_dtypes(include='number').columns
    summary_table[numeric_cols] = summary_table[numeric_cols].fillna(0).replace([float('inf'), float('-inf')], 0).astype(int)
    
    # ... rest of your LaTeX code ...
    
    ax.set_ylim(1999, 2100)
    ax.set_title(f"Threshold = {threshold}°C")
    ax.set_xlabel("Warming Trend Group")
    ax.set_ylabel("Year of Threshold Crossing")
    if i == 0:
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
plt.savefig("figures_new/boxplot_"+models+".png")

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

