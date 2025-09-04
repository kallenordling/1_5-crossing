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
import matplotlib as mpl

def annotate_box_counts(ax, data, x, y, hue=None, order=None, hue_order=None,
						width=0.8, fmt="{n}", dy=0.02, fontweight="bold"):
	"""
	Add 'n' above each seaborn box.
	- width: same as sns.boxplot(..., width=)
	- dy: vertical offset as fraction of y-range
	"""
	# category and hue levels in the order seaborn uses
	cats = order if order is not None else list(pd.Index(data[x]).astype(str).unique())
	if hue is not None:
		hues = hue_order if hue_order is not None else list(pd.Index(data[hue]).astype(str).unique())
	else:
		hues = [None]

	# counts and max per group (for y placement)
	if hue is None:
		grp = data.groupby(x, observed=True)[y]
		counts = grp.apply(lambda s: s.dropna().size)
		gmax = grp.max()
	else:
		grp = data.groupby([x, hue], observed=True)[y]
		counts = grp.apply(lambda s: s.dropna().size)
		gmax = grp.max()

	ymin, ymax = ax.get_ylim()
	yr = ymax - ymin
	n_h = len(hues)

	for i, cat in enumerate(cats):
		for j, hv in enumerate(hues):
			n = counts.get((cat, hv), counts.get(cat, 0)) if hue is not None else counts.get(cat, 0)
			if n ==0:
				continue
			# x position: approximate seaborn's dodge layout
			if hue is None:
				x_pos = i
			else:
				group_w = width
				box_w = group_w / n_h
				x_pos = i - group_w/2 + (j + 0.5)*box_w
			# y position: just above this group's max
			local_max = (gmax.get((cat, hv), np.nan) if hue is not None else gmax.get(cat, np.nan))
			y_pos = (local_max if np.isfinite(local_max) else ymax) + dy*yr

			ax.text(x_pos, y_pos, fmt.format(n=int(n)), ha="center", va="bottom",
					fontweight=fontweight)

	# give a bit more headroom
	ax.set_ylim(1999, 2105)


def load_pair(ssp, threshold, folder="pdf_data"):
	p_all  = Path(folder) / f"{ssp}_{threshold}.csv"
	p_filt = Path(folder) / f"{ssp}_{threshold}_filtered.csv"  # note: "filtered"

	if not p_all.is_file() or not p_filt.is_file():
		missing = [str(p) for p in (p_all, p_filt) if not p.is_file()]
		print("Missing file(s):", ", ".join(missing))
		return None, None

	all_df  = pd.read_csv(p_all)
	filt_df = pd.read_csv(p_filt)
	return all_df, filt_df
sns.set_theme(style="white")
sns.set_theme(style="whitegrid")
thresholds = [1.5, 2.0, 2.5, 3.0]
labels_=["a)","b)","c)","d)"]
SSP_NAME_PRETTY = {

    "ssp126": "SSP1–2.6",
    "ssp245": "SSP2–4.5",
    "ssp370": "SSP3–7.0",
    "ssp585": "SSP5–8.5",
}

sns.set_theme(context="talk")  # bigger base fonts
mpl.rcParams.update({
    "font.size": 16,          # base size
    "font.weight": "bold",    # make all text bold by default
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})
	
fig, ax = plt.subplots(2,2,figsize=(20,10),sharex=True,sharey=True)
ax=ax.flatten()
order = ["SSP1–2.6", "SSP2–4.5", "SSP3–7.0", "SSP5–8.5"]
palette = sns.color_palette("colorblind", n_colors=2)  # [color for "all", color for "filtered"]



hue_order = ["non-filtered", "filtered"]
hue = "dataset"
palette2 = dict(zip(hue_order, sns.color_palette("colorblind", n_colors=2)))
for i,(ssp,__) in enumerate(SSP_NAME_PRETTY.items()):
	all_df, filt_df = load_pair(ssp, 1.5)
	if all_df is not None:
		dfa = all_df.assign(dataset="non-filtered")
		dff = filt_df.assign(dataset="filtered")
		df = pd.concat([dfa, dff], ignore_index=True)
		df["scenario"] = df["scenario"].map(SSP_NAME_PRETTY).fillna(df["scenario"])	
		print(df)
		bin_w = 0.02  # adjust to taste
		m = np.nanmax(np.abs(df["trend"]))
		bins = np.arange(-m, m + bin_w, bin_w)
		sns.histplot(ax=ax[i], element="step", fill=False, 
			data=df, x="trend", hue="dataset",
			bins=bins, multiple="layer",	# overlay
			 linewidth=5,
			stat="density", common_norm=False			# use 'count' if sample sizes match
		)
		
		
		for g in hue_order:
			s = df.loc[df[hue] == g, "trend"].dropna()
			if s.empty: 
				continue
			mu, sd = s.mean(), s.std(ddof=1)
			ax[i].axvline(mu, color=palette2[g], lw=2, ls="-")
			ax[i].axvspan(mu - sd, mu + sd, color=palette2[g], alpha=0.15, zorder=0)
			
	ax[i].set_title(f"Warming trends {__}")
	#ax[i].get_legend().remove()
	ax[i].set_xlim(0,0.7)
	ax[i].text(0.01, 0.99, labels_[i], transform=ax[i].transAxes,
        fontsize=24, fontweight='bold', va='top', ha='left')
	
legend_elems = [
	Patch(facecolor=palette[0], edgecolor="black", label="All models"),
	Patch(facecolor=palette[1], edgecolor="black", label="Filtered models"),
]
fig.legend(legend_elems, ["All models", "Filtered models"],
	loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
plt.savefig('figures/comp_trend.png')


