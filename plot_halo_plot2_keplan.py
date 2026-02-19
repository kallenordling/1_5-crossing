import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Arc
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import FuncFormatter, FixedLocator
import matplotlib.colors as colors
from pathlib import Path
import sys
from matplotlib.ticker import FuncFormatter, FixedLocator
from matplotlib.patches import Circle
import cmocean
from scipy import stats
import matplotlib as mpl
import matplotlib.patheffects as pe

# ... [theta_range_for_corner, anchor_text_position, prepare_year_cdf_from_csv stay the same] ...

def theta_range_for_corner(corner: str):
    c = corner.lower()
    if c == "top-left":     return (90, 180)    # 2nd quadrant
    if c == "top-right":    return (0, 90)      # 1st quadrant
    if c == "bottom-left":  return (180, 270)   # 3rd quadrant
    if c == "bottom-right": return (270, 360)   # 4th quadrant
    raise ValueError("corner must be one of: top-left, top-right, bottom-left, bottom-right")

def anchor_text_position(outer_R: float, corner: str, offset: float = 0.15):
    c = corner.lower()
    if c == "top-left":     return (-outer_R - offset,  outer_R + offset, "right", "bottom")
    if c == "top-right":    return ( outer_R + offset,  outer_R + offset, "left",  "bottom")
    if c == "bottom-left":  return (-outer_R - offset, -outer_R - offset, "right", "top")
    if c == "bottom-right": return ( outer_R + offset, -outer_R - offset, "left",  "top")
    return (0,0,"center","center")

def prepare_year_cdf_from_csv(path: str) -> pd.DataFrame:
    """Read CSV and return DataFrame with columns: year(int), cdf(float). No computations."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    # Accept common column names
    year_col = cols.get("year") or cols.get("first_year") or cols.get("yr")
    cdf_col  = cols.get("cdf") or cols.get("cumprob") or cols.get("cum_prob")
    if year_col is None or cdf_col is None:
        raise ValueError(f"CSV must have 'year' and 'cdf' columns (got {list(df.columns)}).")
    out = df[[year_col, cdf_col]].rename(columns={year_col: "year", cdf_col: "cdf"}).dropna()
    out["year"] = out["year"].astype(int)
    out["cdf"]  = out["cdf"].astype(float)
    return out.sort_values("year").reset_index(drop=True)


def plot_quarter_circular_cdf_from_table(
    cdf_df: pd.DataFrame,
    ax,
    *,
    corner: str = "top-left",
    year_min_scale: int | None = None,
    year_max_scale: int | None = None,
    ring_thickness: float = 0.25,
    year_scale: float = 3.0,
    gap: float = 0.03,
    add_colorbar: bool = True,
    title: str | None = None,
    linestyle: str = 'solid',cmap: str = "Wistia"
):
    if not {"year","cdf"}.issubset(cdf_df.columns):
        raise ValueError("cdf_df must have columns ['year','cdf'].")

    cdf_df = cdf_df.sort_values("year").reset_index(drop=True)
    theta1, theta2 = theta_range_for_corner(corner)

    #cmap = plt.get_cmap('YlOrRd')                      # high-contrast sequential
    #norm = colors.PowerNorm(gamma=1.7, vmin=0, vmax=1)
    # Choose number of bins
    n_bins = 20
    bounds = np.linspace(0, 1, n_bins + 1)                 # bin edges in CDF-space
    if cmap == "own":
        # Get the original 'OrRd' colormap
        orig_cmap = mpl.colormaps['plasma']
        # Truncate the 'OrRd' colormap to get the yellow to light red colors
        cmap = truncate_colormap(orig_cmap, 0.4, 0.8)
    else:    
        cmap   = plt.get_cmap(cmap, n_bins)#.reversed()  # or any other
    norm   = colors.BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
    
    ax.set_aspect('equal')

    # Year → radius mapping (now can be pinned globally via year_min/max_scale)
    y0 = int(cdf_df["year"].min()) if year_min_scale is None else int(year_min_scale)
    y_data_max = int(cdf_df["year"].max())
    y1 = y_data_max if year_max_scale is None else int(year_max_scale)
    if y1 <= y0:
        raise ValueError(f"year_max_scale ({y1}) must be > year_min_scale ({y0}).")

    per_year = ring_thickness * year_scale
    def year_to_radius(y: float) -> float:
        return (y - y0) * per_year

    # Annular bands with small gaps
    prev_outer = 0.0
    for i, row in enumerate(cdf_df.itertuples(index=False)):
        year_i = float(row.year)
        cdf_i  = float(row.cdf)
        outer  = year_to_radius(year_i)
        if i == 0:
            band_thickness = max(outer - gap, 0.0)
        else:
            band_thickness = max(outer - prev_outer - gap, 0.0)

        if band_thickness > 0:
            ax.add_patch(Wedge(
                center=(0, 0),
                r=outer,
                theta1=theta1, theta2=theta2,
                width=band_thickness,
                facecolor=cmap(norm(cdf_i)),
                edgecolor='none',
                linewidth=0.0,
                antialiased=False,                 # avoid hairline seams

            ))
        prev_outer = outer

    # Black arc at the chosen outer year
    y_cdf1 = y_data_max
    r_cdf1 = year_to_radius(y_cdf1)
    ax.add_patch(Arc(
        xy=(0, 0),
        width=2*r_cdf1, height=2*r_cdf1,
        theta1=theta1, theta2=theta2,
        edgecolor='black',
        linewidth=10.0,
        linestyle=linestyle,
        capstyle='round',
        zorder=10,              # draw on top of wedges
    ))
    outer_R_global = year_to_radius(2100)
    add_corner_ssp_labels(ax, outer_R_global, per_year)
    # Label dataset's max year
    tx, ty, ha, va = anchor_text_position(r_cdf1, corner)
    #ax.text(tx, ty, f"CDF=1 year: {y_cdf1:.0f}", ha=ha, va=va, fontsize=10)
    # Optional colorbar (kept simple; outside you already add a PowerNorm colorbar)
    if add_colorbar:
        fig = ax.get_figure()
        sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap)
        sm.set_array([])
        cax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
        cb = ColorbarBase(cax, cmap=cmap, norm=Normalize(0,1), orientation='vertical')
        cb.set_label("CDF", rotation=90)
    return cmap,norm
    
def apply_limits(ax, origin_year, show_to_year, per_year, pad_frac=0.05):
    R = (show_to_year - origin_year) * per_year
    pad = max(per_year, R * pad_frac)
    lim = R + pad
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    
    
from matplotlib.ticker import FixedLocator

def set_year_axes_ticks_with_mapper(
    ax,
    year_to_radius,                 # callable: year -> radius, e.g., lambda y: (y - y0) * per_year
    tick_min: int,
    tick_max: int,
    *,
    label_step: int = 10,           # label every N years
    label_base: int | None = None,  # None -> years % label_step == 0; else (year - base) % step == 0
    bold: bool = True,
    mirror_ticks: bool = True,      # show ticks on both + and − axes
    label_both_sides: bool = True,  # <- label negative side too
    x_label: str = "Year",
    y_label: str = "Year",
    end_pad_pts: int = 6,
    tick_length: float = 6.0,
    tick_width: float = 1.5,
    label_fontsize: int = 40,
    x_tick_rotation: float = 45.0,  # <- rotate x tick labels
):
    """Place ticks for every integer year using the provided mapper, label at label_step."""
    # Axes through origin
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Build yearly positions
    years = list(range(int(tick_min), int(tick_max) + 1))
    pos   = [float(year_to_radius(y)) for y in years]

    # Label selection rule
    if label_base is None:
        show_label = lambda y: (y % label_step) == 0
    else:
        show_label = lambda y: ((y - label_base) % label_step) == 0

    labels_pos = [str(y) if show_label(y) else "" for y in years]

    # Mirror ticks to negative side
    if mirror_ticks:
        neg_pos = [-p for p in reversed(pos)]
        xlocs = neg_pos + [0.0] + pos
        ylocs = neg_pos + [0.0] + pos

        labels_neg = [str(y) if show_label(y) else "" for y in years[::-1]]
        if label_both_sides:
            xlabels = labels_neg + [""] + labels_pos
            ylabels = labels_neg + [""] + labels_pos
        else:
            xlabels = [""] * len(neg_pos) + [""] + labels_pos
            ylabels = [""] * len(neg_pos) + [""] + labels_pos
    else:
        xlocs = pos
        ylocs = pos
        xlabels = labels_pos
        ylabels = labels_pos

    # Apply locators and labels
    ax.xaxis.set_major_locator(FixedLocator(xlocs))
    ax.yaxis.set_major_locator(FixedLocator(ylocs))
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    # Tick style
    ax.tick_params(axis='both', which='major', length=tick_length, width=tick_width, zorder=30)

    # Label styling + rotation for x ticks
    if bold:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')
            lbl.set_fontsize(label_fontsize)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(x_tick_rotation)
        lbl.set_horizontalalignment('right')  # helps readability with 45° rotation

    # Axis-end labels placed via the same mapping
    axis_end = float(year_to_radius(tick_max))
    ax.annotate(
        x_label, xy=(axis_end, 0), xytext=(end_pad_pts, 0),
        textcoords="offset points", ha="left", va="center",
        fontweight="bold", fontsize=40
    )
    ax.annotate(
        y_label, xy=(0, axis_end), xytext=(0, end_pad_pts),
        textcoords="offset points", ha="center", va="bottom",
        rotation=0, fontweight="bold", fontsize=40
    )


def load_first_years(path: Path) -> np.ndarray:
    """Load and clean the 'first_year' column as a float NumPy array."""
    df = pd.read_csv(path)
    x = pd.to_numeric(df["first_year"], errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError(f"No valid 'first_year' values in: {path}")
    return np.sort(x)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """
    Function to truncate a colormap.
    Args:
        cmap (matplotlib.colors.Colormap): The original colormap to truncate.
        minval (float): The minimum normalized value to start the new colormap.
        maxval (float): The maximum normalized value to end the new colormap.
        n (int): The number of colors in the new colormap.
    Returns:
        matplotlib.colors.LinearSegmentedColormap: The new, truncated colormap.
    """
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'truncated_' + cmap.name,
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap
# Get the original 'OrRd' colormap
orig_cmap = mpl.colormaps['OrRd']
# Truncate the 'OrRd' colormap to get the yellow to light red colors
new_cmap = truncate_colormap(orig_cmap, 0.4, 0.8)

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
        F = np.where(idx >= 0, Fr[idx], 0.0)
        return F

def add_ring_legend(fig, line_styles, pos=[1.90, 0.12, 0.60, 0.08]):
    """
    Draws three rings outside the main plot as a legend:
    3.0°, 2.5°, 2.0° mapped to line_styles.
    pos: [left, bottom, width, height] in figure fraction.
    """
    ax_leg = fig.add_axes(pos)
    ax_leg.set_xlim(-0.3, 2)
    ax_leg.set_ylim(0, 1)
    ax_leg.set_aspect('equal')
    ax_leg.axis('off')

    entries = [('1.5°', line_styles['1.5']),
               ('2.0°', line_styles['2.0']),
               ('2.5°', line_styles['2.5']),
               ('3.0°', line_styles['3.0'])]
    ys = [0.78, 0.5, 0.1]  # vertical positions (top→bottom)

    xs = [0.10, 0.37, 0.61, 0.88]   # four evenly spaced x positions
    ys = [0.70, 0.70, 0.70, 0.70]   # all on the same row; make different to stagger

    r = 0.12  # ring radius in legend-axes coords

    for (label, ls), x, y in zip(entries, xs, ys):
        ring = Circle((x, y), radius=r, fill=False,
                      edgecolor='black', linewidth=4.0,
                      linestyle=ls, zorder=3)
        ax_leg.add_patch(ring)
        ax_leg.text(x+r*0.15, y - (r + 0.06), label, ha='center', va='center',
                  fontsize=25, fontweight='bold')

    # panel title
    ax_leg.text(0.0, 1.02, "Warming threshold", va='bottom', ha='left',
                fontsize=40, fontweight='bold')

def add_corner_ssp_labels(ax, outer_R, per_year):
    """Label each quadrant with its SSP name outside the ring."""
    labels = {
        "top-left":     "SSP1–2.6",
        "top-right":    "SSP2–4.5",
        "bottom-right": "SSP3–7.0",
        "bottom-left":  "SSP5–8.5",
    }
    # offset in axis units; ~one ring width outside the outer arc
    offset = per_year * -1.
    for corner, text in labels.items():
        x, y, ha, va = anchor_text_position(outer_R-20, corner, offset=offset)
        ax.text(x, y, text,
                ha=ha, va=va, fontsize=34, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7', alpha=0.85))


def plot_quarter_circular_cdf_from_table(
    cdf_df: pd.DataFrame,
    ax,
    *,
    corner: str = "top-left",
    origin_year: int,            # <-- pass the SAME origin you use for ticks
    per_year: float,             # <-- pass the SAME per_year you use for ticks
    year_min_scale: int | None = None,   # kept for compatibility but ignored if given
    year_max_scale: int | None = None,   # kept for compatibility but ignored if given
    gap: float = 0.03,
    add_colorbar: bool = True,
    title: str | None = None,
    linestyle: str = "solid",
    cdf_last: float = 1.0,
    arc_at_cdf: float | None = None,
    arc_at_year: float | None = None,
    interpolate: bool = True,
    cmap: str = "Reds",
):
    """
    Draw quarter-ring CDF bands and a black arc using a SINGLE shared mapping:
        radius = (year - origin_year) * per_year
    """

    # ----- helpers -----
    def theta_range_for_corner(corner_: str):
        c = corner_.lower()
        if c == "top-left":     return (90, 180)
        if c == "top-right":    return (0, 90)
        if c == "bottom-left":  return (180, 270)
        if c == "bottom-right": return (270, 360)
        raise ValueError("corner must be one of: top-left, top-right, bottom-left, bottom-right")

    def year_to_radius(y: float) -> float:
        return (float(y) - float(origin_year)) * float(per_year)

    def year_at_cdf(df_sorted: pd.DataFrame, level: float, do_interp: bool) -> float:
        y = df_sorted["year"].to_numpy(dtype=float)
        c = np.maximum.accumulate(df_sorted["cdf"].to_numpy(dtype=float))
        level = float(np.clip(level, 0.0, 1.0))
        if level <= c[0]:
            return float(y[0])
        for i in range(1, len(c)):
            if c[i] >= level:
                if do_interp and c[i] > c[i-1]:
                    t = (level - c[i-1]) / (c[i] - c[i-1])
                    return float(y[i-1] + t * (y[i] - y[i-1]))
                return float(y[i])
        return float(y[-1])

    # ----- prep data -----
    if not {"year", "cdf"}.issubset(cdf_df.columns):
        raise ValueError("cdf_df must have columns ['year','cdf'].")
    print("CALL DRAWING")
    df = cdf_df.sort_values("year").reset_index(drop=True)
    theta1, theta2 = theta_range_for_corner(corner)

    # color mapping (quantized)
    n_bins = 20
    bounds = np.linspace(0, 1, n_bins + 1)
    if cmap == "own":
        orig_cmap = mpl.colormaps['plasma']
        cmap = truncate_colormap(orig_cmap, 0.4, 0.8)
    else:
        cmap = plt.get_cmap(cmap, n_bins)
    norm = colors.BoundaryNorm(bounds, ncolors=cmap.N, clip=True)

    ax.set_aspect('equal')

    # where to stop plotting (by CDF)
    cdf_last = float(np.clip(cdf_last, 0.0, 1.0))
    y_stop = year_at_cdf(df, cdf_last, interpolate)

    # draw bands with small gaps
    prev_outer = 0.0
    eps = 1e-9
    for i, row in enumerate(df.itertuples(index=False)):
        year_i = float(row.year)
        cdf_i  = float(row.cdf)
        print(year_i,cdf_i,year_i < y_stop - eps,y_stop)
        if year_i < y_stop - eps:
            outer = year_to_radius(year_i)
            band_thickness = max(outer - prev_outer - gap, 0.0) if i > 0 else max(outer - gap, 0.0)
            if band_thickness > 0:
                ax.add_patch(Wedge(
                    center=(0, 0), r=outer, theta1=theta1, theta2=theta2,
                    width=band_thickness, facecolor=cmap(norm(cdf_i)),
                    edgecolor='none', linewidth=0.0, antialiased=False
                ))
            prev_outer = outer
        else:
            break

    # final partial band (if y_stop between rows)
    outer_stop = year_to_radius(y_stop)
    if outer_stop - prev_outer > gap:
        ax.add_patch(Wedge(
            center=(0, 0), r=outer_stop, theta1=theta1, theta2=theta2,
            width=max(outer_stop - prev_outer - gap, 0.0),
            facecolor=cmap(norm(cdf_last)), edgecolor='none',
            linewidth=0.0, antialiased=False
        ))

    # arc placement
    if arc_at_year is not None:
        y_arc = float(arc_at_year)
    elif arc_at_cdf is not None:
        y_arc = year_at_cdf(df, float(arc_at_cdf), interpolate)
    else:
        y_arc = float(y_stop)
    print(y_arc)
    r_arc = year_to_radius(y_arc)
    ax.add_patch(Arc(
        xy=(0, 0), width=2*r_arc, height=2*r_arc,
        theta1=theta1, theta2=theta2, edgecolor='black',
        linewidth=10.0, linestyle=linestyle, capstyle='round', zorder=10
    ))

    # corner labels at fixed outer year (e.g., 2100)
    outer_R_global = year_to_radius(2100)
    add_corner_ssp_labels(ax, outer_R_global, per_year)

    if add_colorbar:
        fig = ax.get_figure()
        sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap)
        sm.set_array([])
        cax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
        cb = ColorbarBase(cax, cmap=cmap, norm=Normalize(0, 1), orientation='vertical')
        cb.set_label("CDF", rotation=90)

    if title:
        ax.set_title(title, fontsize=40, fontweight="bold",pad=60)

    return cmap, norm, {"y_stop": y_stop, "y_arc": y_arc, "cdf_last": cdf_last}

# ---------- MAIN ----------

# Decide global year range first (so all quadrants share the same axis mapping)
exps = ['126','245','370','585']
orders = {
    '126': ['3.0','2.5','2.0','1.5'],
    '245': ['3.0','2.5','2.0','1.5'],
    '370': ['3.0','2.5','2.0','1.5'],
    '585': ['3.0','2.5','2.0','1.5'],
}
all_ymins, all_ymaxs = [], []
for exp in exps:
    for t in orders[exp]:
        df = pd.read_csv(f"cdf_data/ssp{exp}_{t}.csv")
        cols = {c.lower(): c for c in df.columns}
        ycol = cols.get("year") or cols.get("first_year") or cols.get("yr")
        all_ymins.append(int(df[ycol].min()))
        all_ymaxs.append(int(df[ycol].max()))
y0_global = min(all_ymins)
y1_global = max(all_ymaxs)

# Visual scale (keep consistent across calls)
ring_thickness = 0.25
year_scale     = 3.0
per_year       = ring_thickness * year_scale
outer_R_global = (y1_global - y0_global) * per_year

# One source of truth
ring_thickness = 0.25
year_scale     = 3.0
per_year       = ring_thickness * year_scale
origin_year    = y0_global

# Ticks (your updated helper)



fig,ax = plt.subplots(2,1,figsize=(45, 45))

ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
line_styles = {'3.0':'solid','2.5':'dotted','2.0':'dashed','1.5':'dashdot'}
args = sys.argv[1:]
# One source of truth
y0 = y0_global
per_year = ring_thickness * year_scale
year_to_radius = lambda y: (y - y0) * per_year

set_year_axes_ticks_with_mapper(
    ax[0], year_to_radius, tick_min=2020, tick_max=2100,
    label_step=10, label_base=2020, mirror_ticks=True,
    label_both_sides=True, x_tick_rotation=45
)
set_year_axes_ticks_with_mapper(
    ax[1], year_to_radius, tick_min=2020, tick_max=2100,
    label_step=10, label_base=2020, mirror_ticks=True,
    label_both_sides=True, x_tick_rotation=45
)
# Make sure your view limits also use the SAME mapping
R_needed = year_to_radius(2100)
pad = max(per_year, 0.05 * abs(R_needed))
for a in ax:
    a.set_aspect('equal', adjustable='box')
    a.set_xlim(-R_needed - pad, R_needed + pad)
    a.set_ylim(-R_needed - pad, R_needed + pad)
# Limits that match the same mapping
R_needed = (2100 - origin_year) * per_year
for a in ax:
    a.set_aspect('equal', adjustable='box')
    a.set_xlim(-R_needed - per_year, R_needed + per_year)
    a.set_ylim(-R_needed - per_year, R_needed + per_year)
#cdf_last=float(args[0])
cdf_data = pd.read_csv("keplan_cdf.csv")

for exp in exps:
    corner = 'top-left'
    order = orders[exp]
    if exp == "245":
        corner='top-right'
    if exp == "370":
        corner= "bottom-right"
    if exp == "585":
        corner="bottom-left"

    for t in order:
        pdf_file= f"pdf_data/ssp{exp}_{t}_all.csv"
        x_a = load_first_years(pdf_file)
        xs = np.linspace(2015, 2100, 100)
        cdf_tmp = ecdf_scipy(x_a, xs)
        cdf_tmp=pd.DataFrame({'year': xs, 'cdf': cdf_tmp})
        cdf= cdf_data[(cdf_data["scenario"] == "ssp"+exp) & (cdf_data["threshold"] == float(t))]
        sel = cdf.sort_values("year")
        sel_per_year = sel.groupby("year", as_index=False)["cdf"].max()

        print(cdf)
        print(cdf_tmp)
        
        
        cdf_interp = np.interp(
                xs,
                sel_per_year["year"].values,
                sel_per_year["cdf"].values,
                left=0.0,
                right=sel_per_year["cdf"].values[-1],
        )

        # Combine into DataFrame
        interp_df = pd.DataFrame({"year": xs, "cdf": cdf_interp})
        print(interp_df)
        cdf=interp_df
        #input('wait')
        print(exp)
        '''
        cmap,norm=plot_quarter_circular_cdf_from_table(
            cdf, ax,
            corner=corner,
            add_colorbar=False,
            linestyle=line_styles[t],
            year_min_scale=y0_global,
            year_max_scale=y1_global,
            ring_thickness=ring_thickness,
            year_scale=year_scale,
        )
        '''
        print(cdf['cdf'])
        print(cdf['cdf'].max()) 

        if cdf['cdf'].max() > 0.66:
	        cmap,norm,__=plot_quarter_circular_cdf_from_table(cdf, ax[0],  corner=corner,
		             origin_year=origin_year,
		             per_year=per_year,
		             cdf_last=0.66,
		             add_colorbar=False,
		             linestyle=line_styles[t],
		             title="66% Chance (Likely)",
		             cmap="Reds")
        if cdf['cdf'].max() > 0.90:	             
	        cmap,norm,__=plot_quarter_circular_cdf_from_table(cdf, ax[1], corner=corner,
		             origin_year=origin_year,
		             per_year=per_year,
		             cdf_last=0.90,
		             add_colorbar=False,
		             linestyle=line_styles[t],
		             title="90% Chance (Very likely)",
		             cmap="Reds")
        #input('wait')      

# Colorbar with same norm as wedges
#cmap = plt.get_cmap('YlOrRd')
#norm = colors.PowerNorm(gamma=1.7, vmin=0, vmax=1)
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.subplots_adjust(bottom=0.12,wspace=0.15,left=0.01,right=0.98)
#plt.title('CDF limit= '+str(cdf_last))
# Horizontal colorbar at the bottom of the figure
# [left, bottom, width, height] in figure fraction
cax = fig.add_axes([0.25, 0.06, 0.50, 0.03])
cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')

# Bold label and ticks
cb.set_label("CDF", fontweight='bold', labelpad=6,fontsize=40)
for lbl in cb.ax.get_xticklabels():
    lbl.set_fontweight('bold')
    lbl.set_fontsize(30)
# (optional) ensure ticks/label are at the bottom
cb.ax.xaxis.set_ticks_position('bottom')
cb.ax.xaxis.set_label_position('bottom')

# Axes as years, origin at center
origin_year =y0_global

#set_year_axes_range_ticks(ax[0], origin_year, tick_min=2020, tick_max=2100, per_year=per_year, step=10, bold=True, show_origin_label=False)
#set_year_axes_range_ticks(ax[1], origin_year, tick_min=2020, tick_max=2100, per_year=per_year, step=10, bold=True, show_origin_label=False)
# Limits centered at origin, big enough to include 2100
pad = per_year * 0.8
max_span_years = max(origin_year - y0_global, y1_global - origin_year)
origin_year = y0_global
show_to_year = 2100  # or y1_global if you want full data range
#for a in ax:
#    apply_limits(a, origin_year, show_to_year, per_year)


# Limits from global outer radius

# ax.axis('off')  # keep axes ON now
def center_rect(width=3.00, height=0.12):
    # figure fractions
    left  = 0.45 - width/2
    bottom= 0.8#0.5 - height/2
    return [left, bottom, width, height]

# put the legend dead-center in the figure
pos_center = center_rect(width=1.50, height=0.12)
add_ring_legend(fig, line_styles, pos=pos_center)

for a, label in zip(ax, ['a)', 'b)']):
    txt = a.text(
        0.02, 0.98, label,
        transform=a.transAxes,
        ha='left', va='top',
        fontsize=46, fontweight='bold',
        zorder=50, color='#111111'
    )
    # keep readable on any background (optional halo)
    txt.set_path_effects([pe.withStroke(linewidth=4, foreground='white')])

plt.savefig("figures_new/halo__keplan_new.png", bbox_inches="tight")
plt.savefig("figures_new/halo__keplan_new.svg",dpi=180, bbox_inches="tight")
plt.close(fig)
