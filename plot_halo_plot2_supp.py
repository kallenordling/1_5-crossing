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
    
def set_year_axes_range_ticks(ax,
    origin_year,
    tick_min,
    tick_max,
    per_year,
    step=10,
    bold=True,
    show_origin_label=False,
    # NEW:
    label_at_axis_end=True,
    x_label="Year",
    y_label="Year",
    end_pad_pts=6,
    ):
    
    # Spines through origin
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    #ax.xaxis.label.set_color("#E6E6E6")
    #ax.yaxis.label.set_color("#E6E6E6")
    #ax.tick_params(colors="#DADADA")
    ax.tick_params(axis='both', which='both', zorder=30)

    # Tick years and positions (only 2020..2100)
    tick_years = list(range(tick_min, tick_max + 1, step))
    tick_pos = [(y - origin_year) * per_year for y in tick_years]

    ax.xaxis.set_major_locator(FixedLocator(tick_pos))
    ax.yaxis.set_major_locator(FixedLocator(tick_pos))
    ax.tick_params(axis='x', labelrotation=45)   # simple one-liner

    # Labels show the actual year at those positions; nothing elsewhere
    def fmt(v, pos):
        y = int(round(origin_year + v / per_year))
        return str(y) if (tick_min <= y <= tick_max and (y - tick_min) % step == 0) else ""
    ax.xaxis.set_major_formatter(FuncFormatter(fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))

    #ax.set_xlabel("Year", fontweight='bold',fontsize=20)
    #ax.set_ylabel("Year", fontweight='bold')

    if bold:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')
            lbl.set_fontsize(20)
    # Optional: label the origin year (2015) without adding a tick
    if show_origin_label:
        ax.text(0, 0, str(origin_year), ha='left', va='bottom', fontsize=10, fontweight='bold')

    if label_at_axis_end:
        # right/top ends in data coords
        axis_end = (tick_max - origin_year) * per_year

        # x-axis label at right end (slightly offset in points)
        ax.annotate(
            x_label, xy=(axis_end, 0), xytext=(end_pad_pts, 0),
            textcoords="offset points", ha="left", va="center",
            fontweight="bold",fontsize=30
        )

        # y-axis label at top end (slightly offset in points)
        ax.annotate(
            y_label, xy=(0, axis_end), xytext=(0, end_pad_pts),
            textcoords="offset points", ha="center", va="bottom",
            rotation=00, fontweight="bold",fontsize=30
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

def add_ring_legend(fig, line_styles, pos=[0.90, 0.12, 0.60, 0.08]):
    """
    Draws three rings outside the main plot as a legend:
    3.0°, 2.5°, 2.0° mapped to line_styles.
    pos: [left, bottom, width, height] in figure fraction.
    """
    ax_leg = fig.add_axes(pos)
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.set_aspect('equal')
    ax_leg.axis('off')

    entries = [('2.0°', line_styles['2.0']),
               ('2.5°', line_styles['2.5']),
               ('3.0°', line_styles['3.0'])]
    ys = [0.78, 0.5, 0.1]  # vertical positions (top→bottom)

    xs = [0.16, 0.50, 0.83]  # horizontal positions
    y_ring = 0.70            # vertical center for rings
    r = 0.15                 # ring radius in legend-axes coords

    for (label, ls), x in zip(entries, xs):
        ring = Circle((x, y_ring), radius=r, fill=False,
                      edgecolor='black', linewidth=4.0,
                      linestyle=ls, zorder=3)
        ax_leg.add_patch(ring)
        # label under each ring
        ax_leg.text(x, 0.42, label, ha='center', va='center',
                    fontsize=20, fontweight='bold')

    # panel title
    ax_leg.text(0.0, 1.02, "Warming threshold", va='bottom', ha='left',
                fontsize=20, fontweight='bold')

def add_corner_ssp_labels(ax, outer_R, per_year):
    """Label each quadrant with its SSP name outside the ring."""
    labels = {
        "top-left":     "SSP1–2.6",
        "top-right":    "SSP2–4.5",
        "bottom-right": "SSP3–7.0",
        "bottom-left":  "SSP5–8.5",
    }
    # offset in axis units; ~one ring width outside the outer arc
    offset = per_year * -0.3
    for corner, text in labels.items():
        x, y, ha, va = anchor_text_position(outer_R-10, corner, offset=offset)
        ax.text(x, y, text,
                ha=ha, va=va, fontsize=24, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7', alpha=0.85))


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
    linestyle: str = "solid",
    # NEW controls:
    cdf_last: float = 1.0,          # last CDF level to plot bands up to (0..1)
    arc_at_cdf: float | None = None,# if set, draw the arc at the year where CDF reaches this level
    arc_at_year: float | None = None,# if set, draw the arc at this explicit year (overrides arc_at_cdf)
    interpolate: bool = True,cmap: str = "Wistia"        # interpolate year between table rows when targeting a CDF level
):
    """
    Plots annular bands up to a chosen CDF level (cdf_last) and draws a black arc
    either at an explicit year (arc_at_year), a target CDF level (arc_at_cdf),
    or—by default—at the last plotted CDF level.

    Expects cdf_df with columns ['year','cdf'] and cdf non-decreasing with year.
    """
    print(cdf_df)
    if not {"year", "cdf"}.issubset(cdf_df.columns):
        raise ValueError("cdf_df must have columns ['year','cdf'].")

    # --- helpers -------------------------------------------------------------
    def year_at_cdf(df_sorted: pd.DataFrame, level: float, do_interp: bool) -> float:
        """Return the (possibly interpolated) year where CDF first reaches `level`."""
        y = df_sorted["year"].to_numpy(dtype=float)
        c = df_sorted["cdf"].to_numpy(dtype=float)
        # enforce monotonic non-decreasing to be safe
        c = np.maximum.accumulate(c)

        level = float(np.clip(level, 0.0, 1.0))
        if level <= c[0]:
            return float(y[0])
        for i in range(1, len(c)):
            if c[i] >= level:
                if do_interp and c[i] > c[i-1]:
                    # linear interpolation in (year, cdf) space
                    t = (level - c[i-1]) / (c[i] - c[i-1])
                    return float(y[i-1] + t * (y[i] - y[i-1]))
                return float(y[i])
        return float(y[-1])

    # ------------------------------------------------------------------------
    cdf_df = cdf_df.sort_values("year").reset_index(drop=True)
    theta1, theta2 = theta_range_for_corner(corner)

    # Color mapping (kept quantized)
    n_bins = 20
    bounds = np.linspace(0, 1, n_bins + 1)
    if cmap == "own":
        # Get the original 'OrRd' colormap
        orig_cmap = mpl.colormaps['plasma']
        # Truncate the 'OrRd' colormap to get the yellow to light red colors
        cmap = truncate_colormap(orig_cmap, 0.4, 0.8)
    else:    
        cmap   = plt.get_cmap(cmap, n_bins)#.reversed()  # or any other
    norm   = colors.BoundaryNorm(bounds, ncolors=cmap.N, clip=True)

    ax.set_aspect('equal')

    # Year → radius mapping (global pinning via year_min/max_scale)
    y0 = int(cdf_df["year"].min()) if year_min_scale is None else int(year_min_scale)
    y_data_max = float(cdf_df["year"].max())
    y1 = int(y_data_max) if year_max_scale is None else int(year_max_scale)
    if y1 <= y0:
        raise ValueError(f"year_max_scale ({y1}) must be > year_min_scale ({y0}).")

    per_year = ring_thickness * year_scale
    def year_to_radius(y: float) -> float:
        return (float(y) - y0) * per_year

    # Determine where to STOP plotting (last band) by CDF
    cdf_last = float(np.clip(cdf_last, 0.0, 1.0))
    y_stop = year_at_cdf(cdf_df, cdf_last, interpolate)

    # Draw bands up to y_stop
    prev_outer = 0.0
    prev_year  = None
    eps = 1e-9

    for i, row in enumerate(cdf_df.itertuples(index=False)):
        year_i = float(row.year)
        cdf_i  = float(row.cdf)

        if year_i < y_stop - eps:
            outer = year_to_radius(year_i)
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
                    antialiased=False,
                ))
            prev_outer = outer
            prev_year  = year_i
        else:
            break

    # If y_stop falls between table rows, add a final partial band colored at cdf_last
    if prev_year is None:
        prev_outer = 0.0
    outer_stop = year_to_radius(y_stop)
    if outer_stop - prev_outer > gap:
        ax.add_patch(Wedge(
            center=(0, 0),
            r=outer_stop,
            theta1=theta1, theta2=theta2,
            width=max(outer_stop - prev_outer - gap, 0.0),
            facecolor=cmap(norm(cdf_last)),
            edgecolor='none',
            linewidth=0.0,
            antialiased=False,
        ))

    # --- Decide where to draw the black arc ---------------------------------
    if arc_at_year is not None:
        y_arc = float(arc_at_year)
    elif arc_at_cdf is not None:
        y_arc = year_at_cdf(cdf_df, float(arc_at_cdf), interpolate)
    else:
        # default: draw at the same place we stopped plotting (y_stop)
        y_arc = float(y_stop)

    # Clip arc year to plotted scale range
    y_arc = float(np.clip(y_arc, y0, y1))
    r_arc = year_to_radius(y_arc)

    ax.add_patch(Arc(
        xy=(0, 0),
        width=2*r_arc, height=2*r_arc,
        theta1=theta1, theta2=theta2,
        edgecolor='black',
        linewidth=10.0,
        linestyle=linestyle,
        capstyle='round',
        zorder=10,
    ))

    # Keep your SSP corner labels at a fixed global outer radius (e.g., year 2100)
    outer_R_global = year_to_radius(2100)
    add_corner_ssp_labels(ax, outer_R_global, per_year)

    # Optional colorbar
    if add_colorbar:
        fig = ax.get_figure()
        sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap)
        sm.set_array([])
        cax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
        cb = ColorbarBase(cax, cmap=cmap, norm=Normalize(0, 1), orientation='vertical')
        cb.set_label("CDF", rotation=90)

    if title:
        ax.set_title(title,fontsize=20, fontweight="bold")

    # Return a few useful values for callers
    return cmap, norm, {"y_stop": y_stop, "y_arc": y_arc, "cdf_last": cdf_last}

# ---------- MAIN ----------

# Decide global year range first (so all quadrants share the same axis mapping)
exps = ['126','245','370','585']
orders = {
    '126': ['3.0','2.5','2.0'],
    '245': ['3.0','2.5','2.0'],
    '370': ['3.0','2.5','2.0'],
    '585': ['3.0','2.5','2.0'],
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

fig,ax = plt.subplots(2,2,figsize=(45, 30))

for ax_  in ax.ravel():
    ax_.set_aspect('equal')
    #ax[1].set_aspect('equal')
line_styles = {'3.0':'solid','2.5':'dotted','2.0':'dashed'}
args = sys.argv[1:]

cdf_last=float(args[0])
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
        pdf_file= f"pdf_data/ssp{exp}_{t}.csv"
        x_a = load_first_years(pdf_file)
        xs = np.linspace(2015, 2100, 100)
        cdf = ecdf_scipy(x_a, xs)
        cdf3=pd.DataFrame({'year': xs, 'cdf': cdf})
        print(cdf)
        
        
        pdf_file= f"pdf_data/ssp{exp}_{t}_filtered.csv"
        x_a = load_first_years(pdf_file)
        xs = np.linspace(2015, 2100, 100)
        cdf = ecdf_scipy(x_a, xs)
        cdf2=pd.DataFrame({'year': xs, 'cdf': cdf})        
        print(cdf2)
        print(corner)
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
        cmap,norm,__=plot_quarter_circular_cdf_from_table(cdf3, ax[0,0], cdf_last=0.5, corner=corner,
            add_colorbar=False,
            linestyle=line_styles[t],
            year_min_scale=y0_global,
            year_max_scale=y1_global,
            ring_thickness=ring_thickness,
            year_scale=year_scale,title="50% Change",cmap="Reds")
        cmap,norm,__=plot_quarter_circular_cdf_from_table(cdf3, ax[1,0], cdf_last=0.9, corner=corner,
            add_colorbar=False,
            linestyle=line_styles[t],
            year_min_scale=y0_global,
            year_max_scale=y1_global,
            ring_thickness=ring_thickness,
            year_scale=year_scale,title="90% Change",cmap="Reds")
            
        cmap,norm,__=plot_quarter_circular_cdf_from_table(cdf2, ax[0,1], cdf_last=0.5, corner=corner,
            add_colorbar=False,
            linestyle=line_styles[t],
            year_min_scale=y0_global,
            year_max_scale=y1_global,
            ring_thickness=ring_thickness,
            year_scale=year_scale,title="50% Change Filtered",cmap="Reds")
        cmap,norm,__=plot_quarter_circular_cdf_from_table(cdf2, ax[1,1], cdf_last=0.9, corner=corner,
            add_colorbar=False,
            linestyle=line_styles[t],
            year_min_scale=y0_global,
            year_max_scale=y1_global,
            ring_thickness=ring_thickness,
            year_scale=year_scale,title="90% Change Filtered",cmap="Reds")            


# Colorbar with same norm as wedges
#cmap = plt.get_cmap('YlOrRd')
#norm = colors.PowerNorm(gamma=1.7, vmin=0, vmax=1)
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.subplots_adjust(bottom=0.12,wspace=0.01,left=0.01,right=0.98)
#plt.title('CDF limit= '+str(cdf_last))
# Horizontal colorbar at the bottom of the figure
# [left, bottom, width, height] in figure fraction
cax = fig.add_axes([0.15, 0.06, 0.70, 0.03])
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
origin_year = 2015
pad = per_year * 0.8
max_span_years = max(origin_year - y0_global, y1_global - origin_year)
lim = 70#max(max_span_years * per_year, (2100 - origin_year) * per_year) + pad

for ax_ in ax.ravel():
    set_year_axes_range_ticks(ax_, origin_year, tick_min=2020, tick_max=2100, per_year=per_year, step=10, bold=True, show_origin_label=False)
    ax_.set_xlim(-lim, lim)
    ax_.set_ylim(-lim, lim)
#set_year_axes_range_ticks(ax[1], origin_year, tick_min=2020, tick_max=2100, per_year=per_year, step=10, bold=True, show_origin_label=False)
# Limits centered at origin, big enough to include 2100

#ax[0].set_xlim(-lim, lim)
#ax[0].set_ylim(-lim, lim)
#ax[1].set_xlim(-lim, lim)
#ax[1].set_ylim(-lim, lim)

# Limits from global outer radius

# ax.axis('off')  # keep axes ON now
def center_rect(width=1.00, height=0.12):
    # figure fractions
    left  = 0.5 - width/2
    bottom= 0.8#0.5 - height/2
    return [left, bottom, width, height]

# put the legend dead-center in the figure
pos_center = center_rect(width=0.20, height=0.12)
add_ring_legend(fig, line_styles, pos=pos_center)

for a, label in zip(ax.flatten(), ['a)', 'b)','c)','d)']):
    txt = a.text(
        0.02, 0.98, label,
        transform=a.transAxes,
        ha='left', va='top',
        fontsize=36, fontweight='bold',
        zorder=50, color='#111111'
    )
    # keep readable on any background (optional halo)
    txt.set_path_effects([pe.withStroke(linewidth=4, foreground='white')])

plt.savefig("figures/halo_"+str(cdf_last)+"_supp.png", bbox_inches="tight")
plt.close(fig)
