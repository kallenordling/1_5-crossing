import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
from scipy import stats
from pathlib import Path

def load_first_years(path: Path) -> np.ndarray:
    """Load and clean the 'first_year' column as a float NumPy array."""
    df = pd.read_csv(path)
    x = pd.to_numeric(df["first_year"], errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError(f"No valid 'first_year' values in: {path}")
    return np.sort(x)

def models_surviving_at_cdf(df_in, threshold, scenario, cdf_level=0.66, censor_year=2100,
                            model_col="model"):
    # subset
    d = df_in[(df_in["threshold"] == threshold) & (df_in["scenario"] == scenario)].copy()
    if d.empty:
        return [], None

    # build KM curve (your functions)
    times, cdf = km_curve(d, threshold, scenario, censor_year=censor_year)
    if not times:
        return [], None

    # find the first time where CDF reaches/exceeds the target level
    t_star = None
    for t, c in zip(times, cdf):
        if c >= cdf_level:
            t_star = t
            break

    # If never reaches 66%, then "still surviving at 66%" is ambiguous;
    # best default: everyone is surviving relative to that unattained level.
    if t_star is None:
        survivors = d[model_col].unique().tolist()
        return survivors, None

    # survivors at t_star: first_year is NaN (censored) or > t_star
    survivors = d[(d["first_year"].isna()) | (d["first_year"] > t_star)][model_col].unique().tolist()
    return survivors, t_star

def ecdf_scipy(x: np.ndarray, xs_eval: np.ndarray) -> np.ndarray:
    """
    Evaluate the ECDF at xs_eval using SciPy.
    Prefer stats.ecdf (SciPy >= 1.11); otherwise approximate via stats.cumfreq.
    """
    if hasattr(stats, "ecdf"):
        res = stats.ecdf(x)                # ECDFResult
        return res.cdf.evaluate(xs_eval)   # array of ECDF values
    else:
        # Approximate ECDF via a fine cumulative histogram
        # Choose many bins so the step function is close to exact
        nbins = max(400, min(4000, int(x.size * 20)))
        h = stats.cumfreq(x, numbins=nbins)
        # Right-edge positions of the bins
        edges = np.linspace(
            h.lowerlimit,
            h.lowerlimit + h.binsize * h.cumcount.size,
            h.cumcount.size + 1
        )
        xr = edges[1:]
        Fr = h.cumcount / x.size

        # Map xs_eval to the stepwise ECDF
        idx = np.searchsorted(xr, xs_eval, side="right") - 1
        idx = np.clip(idx, -1, len(Fr) - 1)
        F = np.w
SSP_NAME_PRETTY = {
    "ssp119": "SSP1–1.9",
    "ssp126": "SSP1–2.6",
    "ssp245": "SSP2–4.5",
    "ssp370": "SSP3–7.0",
    "ssp585": "SSP5–8.5",
}
models=['CanESM5','MIROC6','ACCESS-ESM1-5']

colors_ssp={'ssp119':'#00a9cf','ssp126':'#1e9583','ssp245':'#4576be','ssp370':'#f11111','ssp585':'#8036a7'}
tmp_ = {'cmip6_data_smiles.csv':'solid'}#,'cmip6_data_filtered.csv':'dotted'}
fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()
for data_,style in tmp_.items():

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


	df_all = pd.read_csv(data_)

	for m in models:
		# --- Lataa data ---
		df = df_all[(df_all["model"] == m)]

		# odotetut sarakkeet: model, threshold, first_year, trend, scenario

		# --- Kaplan–Meier (manuaalinen) ---
		# --- Piirto: 4 alikuvaa kynnysarvoille 1.5, 2.0, 2.5, 3.0 ---
		thresholds = [1.0,1.5, 2.0, 2.5, 3.0,3.5,4,4.5,5]
		scenarios = sorted(df["scenario"].unique())
		print(scenarios)


		cdf_dict = {}
		cdf_naive_dict = {}  # naive
		rows = []

		for ax_i, thr in enumerate(thresholds):
			print(thr)
			ax=axes[ax_i]
			for sc in scenarios:
				times, cdf = km_curve(df, thr, sc)
				cdf_dict[(thr, sc)] = list(zip(times, cdf))
				print(df)
				#input('wait')
				if len(times) > 1:
					ax.step(times, cdf, where="post", label=sc, linestyle=style,color=colors_ssp[sc])
				__, t66_year = models_surviving_at_cdf(
					df,
					threshold=thr,
					scenario=sc,
					cdf_level=0.66,
					model_col="member_id"
				)

				__, t95_year = models_surviving_at_cdf(
					df,
					threshold=thr,
					scenario=sc,
					cdf_level=0.95,
					model_col="member_id"
				)

				__, t05_year = models_surviving_at_cdf(
					df,
					threshold=thr,
					scenario=sc,
					cdf_level=0.05,
					model_col="member_id"
				)

				# store one row per surviving model
				rows.append({
					"threshold": thr,
					"scenario": sc,
					"cdf_level": 0.66,
					"t_cdf_year": t66_year,
				})
				rows.append({
					"threshold": thr,
					"scenario": sc,
					"cdf_level": 0.95,
					"t_cdf_year": t95_year,
				})
				rows.append({
					"threshold": thr,
					"scenario": sc,
					"cdf_level": 0.05,
					"t_cdf_year": t05_year,
				})
				# Naive
				#try:
				#	pdf_file= f"pdf_data/{sc}_{thr}.csv"
				#	x_a = load_first_years(pdf_file)
				#	xs = np.linspace(2015, 2100, 100)
				#	cdf = ecdf_scipy(x_a, xs)
				#	ax.plot(xs,cdf,linestyle="-.",color=colors_ssp[sc])
				#except:
				#	print('pdf not found')

			ax.set_xlim(2015, 2100)
			ax.set_ylim(0, 1)
			ax.set_title(f"Threshold {thr} °C")
			ax.set_xlabel("Year")
			ax.set_ylabel("P(crossed)")
			ax.grid(True, alpha=0.3)
		cdf_year_df = pd.DataFrame(rows)
		cdf_year_df.to_csv("km_cdf66_years"+m+".csv", index=False)
		print(cdf_year_df)
		#survivors_df = pd.DataFrame(rows)
		#survivors_df.to_csv("km_survivors_cdf66.csv", index=False)
		# Yhteinen legenda alas
		handles, labels = axes[0].get_legend_handles_labels()
		fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)))
		# plt.suptitle("Ensimmäisen ylityksen CDF kaikille skenaarioille (alkaen 2015)", fontsize=14)
		plt.tight_layout(rect=[0, 0.05, 1, 0.95])
		plt.savefig("figures_new/cdf.png")

		records = []
		for (thr, sc), series in cdf_dict.items():
			for year, val in series:
				records.append({
					"threshold": thr,
					"scenario": sc,
					"year": year,
					"cdf": val
				})

		cdf_df = pd.DataFrame.from_records(records)
		cdf_df.to_csv("keplan_cdf_smiles.csv", index=False)
		print(cdf_df)

