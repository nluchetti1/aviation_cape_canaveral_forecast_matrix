"""
Cape Canaveral Aviation Weather Matrix (LLWS edition)
=====================================================

A chiselled-down sibling of the full LLCC matrix board. This build carries ONLY:

  * Aviation matrix   - PBL Mom Mean, PBL Mom Max, Ceilings, Visibility, LLWS
  * LLWS              - TAF-style low-level wind shear group (WSddd/dddffKT) computed
                        from the BUFKIT profiles and the raw isobaric GRIB columns
  * 10Z Synoptic      - mean flow / regime / anvil flow / Thompson / RH / PWAT /
                        Cizek lightning probability, with monthly climo box plots
  * Calibrated Thunder- HREF CT (1-hr + 4-hr) point column and spatial slider

Everything tied to the Lightning Launch Commit Criteria (isotherm heights, cloud
tops, layer thickness, thick-cloud-layer rule, cumulus standoff probabilities,
HREF lightning density, convective/anvil masking) has been removed.

Model columns: GFS, RAP, HRRR (BUFKIT for the airports, raw GRIB for the pads),
plus ECMWF IFS, RRFS and REFS point-extracted from GRIB for every site.
"""

import datetime
import time
import json
import math
import os
import re
import requests
import concurrent.futures
import threading
import random
import logging
import pygrib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CACHE_DIR = "./workspace_cache"
HISTORY_FILE = "history.json"
STATIONS = ["kdab", "kxmr", "kmlb", "kfpr", "kpbi"]
# MODELS is the full display/column set. BUFKIT_MODELS is the subset that comes from PSU
# BUFKIT + NOMADS grib (the airport soundings and NOMADS pad columns). ECMWF is additive and
# point-extracted from ECMWF Open Data, so it must NOT be swept into the BUFKIT/NOMADS loops.
MODELS = ["gfs", "rap", "hrrr", "ecmwf"]
BUFKIT_MODELS = ["gfs", "rap", "hrrr"]

# ---- PSU BUFKIT politeness ------------------------------------------------------------
# PSU's BUFKIT server throttles hard, and when a burst of parallel requests arrives from a
# single IP (exactly what a GitHub Actions runner looks like) it stops answering altogether:
# every in-flight request read-times-out at the same instant, then subsequent ones can't even
# open a connection. Four things keep us under its radar:
#   1. LOW CONCURRENCY, enforced by a semaphore rather than just a small pool, so the cap
#      holds no matter how the executor is sized.
#   2. A BROWSER USER-AGENT. The default python-requests UA gets dropped on the floor.
#   3. A JITTERED STAGGER before each request, so 15 tasks don't align into a burst.
#   4. EXPONENTIAL BACKOFF WITH JITTER, so retries don't fire in lockstep and re-trigger
#      the same throttle that caused the first failure.
BUFKIT_MAX_CONCURRENCY = 2      # simultaneous connections to PSU. Do not raise casually.
BUFKIT_ATTEMPTS = 4             # total tries per station-model
BUFKIT_CONNECT_TIMEOUT = 12     # seconds to establish the TCP connection
BUFKIT_READ_TIMEOUT = 45        # seconds to receive the body once connected
BUFKIT_STAGGER_S = (0.4, 1.8)   # random pre-request pause, seconds
BUFKIT_BACKOFF_BASE_S = 4.0     # first retry waits ~4 s, then ~8, ~16 (x0.6-1.4 jitter)
BUFKIT_USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64; rv:120.0) "
                     "Gecko/20100101 Firefox/120.0")
# Hard gate shared by every worker thread.
_BUFKIT_GATE = threading.Semaphore(BUFKIT_MAX_CONCURRENCY)

# ---- Low-Level Wind Shear (LLWS) -----------------------------------------------------------
# TAF convention: LLWS is carried in a forecast when the vector wind difference through the
# lowest 2,000 ft AGL reaches 20 kt. The group reads WSddd/dddffKT, where ddd is the height of
# the shear layer top in HUNDREDS of feet AGL, then the wind at that height (direction to the
# nearest 10 deg, speed to the nearest 5 kt).
LLWS_TOP_FT_AGL = 2000.0     # nominal top of the shear layer (ft AGL)
LLWS_SEARCH_FT_AGL = 2100.0  # search a touch deeper so a level just above 2,000 ft still counts
LLWS_TAF_MIN_KT = 20.0       # magnitude at/above which a TAF would carry the WS group

# ---- ECMWF Open Data (IFS HRES 0.25°, CC-BY-4.0) additive global column ----
ECMWF_ENABLED = True
ECMWF_SOURCE = "ecmwf"    # ecmwf-opendata source: ecmwf | aws | azure | google
ECMWF_MAX_FH = 144        # forecast hours to ingest. IFS open-data is 3-hourly out to 144 h (then
                          # 6-hourly, which this step list would silently miss), so 144 is the
                          # natural stop.
ECMWF_LEVELS_HPA = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]

# ---- ECMWF ENS ensemble column for the 10Z panel (IFS ENS 0.25 deg, open data) --------------
# Panel-only, exactly like GEFS: a second global ensemble to read against GEFS at day 3+.
#
# THREE THINGS THE LIVE PROBE ESTABLISHED, all of which shape the settings below:
#   1. The CONTROL member (type="cf") has no pressure-level entries in the open-data tier —
#      a cf request errors with "Cannot find index entries". So this is 10 PERTURBED members
#      (pf 1..10) with no control. For airmass indices that's statistically fine; it just
#      differs from GEFS, which is c00 + p01..p14.
#   2. Every one of t/r/u/v is published on the levels we need, so nothing is approximated.
#   3. The ECMWF portal was FASTER than both cloud mirrors (AWS returned 503 Slow Down at
#      0.2 MB/s vs 2.4 MB/s direct), so this deliberately reuses ECMWF_SOURCE.
ECMWF_ENS_ENABLED = True
ECMWF_ENS_MEMBERS = 10        # perturbed members pf 1..N (no control; see above)
# Capped to match the HRES column. Also the exact limit of the 06/18Z ENS runs, so every
# cycle is usable rather than only the deep 00/12Z ones.
ECMWF_ENS_MAX_FH = 144
# Six levels, no upper set. That drops the 300-150 mb anvil flow for this column (the panel
# renders it as an em dash) and cuts the fetch ~40%. Everything else the panel shows —
# Thompson, PWAT, 700-500 RH, 1000-700 mean flow, regime, Cizek lightning — is unaffected.
ECMWF_ENS_LEVELS_HPA = [1000, 925, 850, 700, 600, 500]
ECMWF_ENS_PARAMS = ["t", "r", "u", "v"]   # gh omitted; heights fall back to barometric
# ENS advances every 6 h while this pipeline runs hourly, so rows are cached against the
# cycle and only refetched when a new one posts (~0.9 GB / 6 min when it does).
ECMWF_ENS_CACHE_ENABLED = True

# Per-member REFS launch-thermo (KXMR panel): pull each ensemble member's full isobaric sounding at
# XMR for the 10Z hours, compute the indices per member, and average the RESULTS — the valid way to
# get an ensemble TI/PWAT (never from the ensemble-MEAN sounding).
REFS_MEMBER_THERMO_ENABLED = True
# Forecast-hour cap for the REFS member sweep. Matches the REFS/RRFS sounding depth.
REFS_MEMBER_WINDOW_FH = 60
# Time-lag the prior cycle's members (+6 h, valid-time aligned) to double the ensemble size
# (5 -> 10 members), which halves the granularity of the per-member index spread.
REFS_MEMBER_TLE = True
# How many prior cycles to fold in when REFS_MEMBER_TLE is on. 1 = the -6 h cycle only (the
# validated sweet spot). Degrades gracefully if an older cycle isn't posted deep enough.
REFS_MEMBER_LAG_CYCLES = 1

# ---- GEFS ensemble column for the 10Z panel (global 0.5 deg, AWS mirror) ---------------------
# Panel-only: GEFS is far coarser than the mesoscale columns and would add nothing to the hourly
# matrix, but it gives a genuine global-ensemble read on the daily airmass out to a week.
# NOTE ON FILES: pgrb2a (the "primary" half-degree file) carries TMP and RH at ONLY 1000/925/850
# mb, so it alone cannot produce K-Index, Lifted Index or 700-500 RH. The mid/upper temperature
# and moisture live in pgrb2b, so BOTH files are byte-ranged and concatenated per member.
GEFS_ENABLED = True
GEFS_AWS_ROOT = "https://noaa-gefs-pds.s3.amazonaws.com"
GEFS_MEMBERS = 15          # c00 control + p01..p14; spread converges quickly for airmass indices
GEFS_MAX_FH = 168          # 3-hourly output; 168 h = 7 forecast days
# GEFS cycles every 6 h but this pipeline runs hourly, so the fetched rows are cached against the
# cycle and only refetched when a new cycle appears (saves ~5 of every 6 runs).
GEFS_CACHE_ENABLED = True
# S3 has no burst limits, but keep a tiny pause as a courtesy / connection-reuse aid.
GEFS_REQUEST_PAUSE_S = 0.05
# GEFS 0.5-deg carries a REDUCED isobaric set (no 975/950/900/800/750/650/550/450/350 mb). Only
# ask for levels it actually publishes; anything else is a wasted lookup.
GEFS_LEVELS_HPA = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100]
# The panel thermo needs only pressure/T/dewpoint/wind - geopotential height is never read, so
# HGT is deliberately NOT fetched (that alone is ~20% of the bytes).
GEFS_VARS = ("TMP", "RH", "UGRD", "VGRD")


STN_COORDS = {
    "kxmr": {"lat": 28.468, "lon": -80.556},
    "kdab": {"lat": 29.180, "lon": -81.058},
    "kmlb": {"lat": 28.103, "lon": -80.645},
    "kfpr": {"lat": 27.498, "lon": -80.373},
    "kpbi": {"lat": 26.683, "lon": -80.095}
}

# Cape Canaveral / KSC launch pads. These are derived from raw model isobaric GRIB2
# (GFS/RAP/HRRR) rather than BUFKIT, since the pads have no dedicated BUFKIT profiles.
# KTTS (KSC Shuttle Landing Facility) and KCOF (Patrick SFB) are airfields handled the
# same GRIB way; they inherit the KXMR calibrated-thunder proxy in the frontend.
LAUNCH_PADS = {
    "lc39a": {"lat": 28.608, "lon": -80.604, "label": "LC-39A (KSC)"},
    "lc39b": {"lat": 28.627, "lon": -80.621, "label": "LC-39B (KSC)"},
    "lc37":  {"lat": 28.532, "lon": -80.565, "label": "LC-37B (CCSFS)"},
    "slc40": {"lat": 28.562, "lon": -80.577, "label": "SLC-40 (CCSFS)"},
    "slc41": {"lat": 28.583, "lon": -80.583, "label": "SLC-41 (CCSFS)"},
    "lc36":  {"lat": 28.470, "lon": -80.538, "label": "LC-36 (CCSFS)"},
    "ktts":  {"lat": 28.615, "lon": -80.695, "label": "KTTS (KSC SLF)"},
    "kcof":  {"lat": 28.235, "lon": -80.610, "label": "KCOF (Patrick SFB)"},
}

# ---- RRFS / REFS configuration -------------------------------------------------
# RRFS (deterministic) and REFS (ensemble) are pre-operational until 2026-08-31 12z.
# We pull from the public AWS Open-Data bucket (no auth, HTTP range-request friendly)
# rather than NOMADS, using each file's .idx sidecar to byte-range only the ~21 isobaric
# levels we need instead of downloading the whole ~40MB CONUS file.
RRFS_ENABLED = True          # master switch for the RRFS deterministic column
REFS_ENABLED = True          # master switch for the REFS ensemble-average column
RRFS_AWS_ROOT = "https://noaa-rrfs-pds.s3.amazonaws.com"
RRFS_CYCLE_HOURS = [0, 6, 12, 18]   # cycles that run to full length
RRFS_MAX_FH = 60             # RRFS/REFS run to 60 h on the extended (00/06/12/18z) cycles
RRFS_LATENCY_H = 4           # approx hours before a cycle's files are complete on AWS

# HRRR pressure-level GRIB2 on AWS (byte-range friendly via .idx, no bot-blocking). HRRR
# only reaches f48 on the 00/06/12/18z "extended" cycles; other cycles stop at f18.
HRRR_AWS_ROOT = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
HRRR_EXTENDED_CYCLES = [0, 6, 12, 18]
HRRR_LATENCY_H = 3

# The exact REFS ensemble-mean filename ordering has drifted across the pre-op feed. We probe
# these candidate patterns (formatted with cycle `c` and forecast-hour ints) once per run and
# cache whichever resolves, so every subsequent hour reuses the confirmed pattern.
REFS_FILENAME_CANDIDATES = [
    "refs.t{c}z.mean.f{f2}.conus.grib2",
    "refs.t{c}z.conus.mean.f{f2}.grib2",
    "refs.t{c}z.mean.f{f3}.conus.grib2",
    "refs.t{c}z.conus.prslev.mean.f{f2}.grib2",
    "rrfsce.t{c}z.conus.mean.f{f2}.grib2",
]
_REFS_RESOLVED_PATTERN = None   # set once we confirm a working pattern this run


# Bounding box — zoomed into the Space Coast launch corridor rather than all of FL
FL_DOMAIN = {"lat_min": 24.5, "lat_max": 31.0, "lon_min": -84.5, "lon_max": -79.0}

# Spatial plot PNGs are written here (relative path, served alongside index.html)
MAPS_DIR = "./maps"

# Global cache for static grid indices to maximize ThreadPool performance
_GRID_INDEX_CACHE = {}

def purge_workspace(cache_dir=CACHE_DIR):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    else:
        for f in os.listdir(cache_dir):
            try:
                os.unlink(os.path.join(cache_dir, f))
            except Exception:
                pass
    return cache_dir


def _collect_map_paths(*containers):
    """Recursively pull every non-empty relative map path out of any mix of nested dicts
    ({row: {thresh: path}}), flat dicts ({row: path}), lists, or bare strings. Used to
    build the 'keep' set for pruning WITHOUT the lossy `{**a, **b}` merge that previously
    let the CT maps clobber the density maps (and ct4 clobber ct1) on shared row keys."""
    paths = set()

    def walk(x):
        if x is None:
            return
        if isinstance(x, str):
            if x:
                paths.add(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple, set)):
            for v in x:
                walk(v)

    for c in containers:
        walk(c)
    return paths


def prune_stale_maps(referenced_paths, maps_dir=MAPS_DIR):
    """Deletes spatial-map PNGs on disk that aren't in `referenced_paths` (the maps/ folder
    only ever needs to hold the latest run's images plus the blank basemap fallback).
    `referenced_paths` is any iterable of relative paths like 'maps/xyz.png'."""
    if not os.path.exists(maps_dir):
        return

    keep = {os.path.basename(p) for p in referenced_paths if p}

    for f in os.listdir(maps_dir):
        if f not in keep:
            try:
                os.unlink(os.path.join(maps_dir, f))
            except Exception:
                pass


def pressure_to_height_ft(pres_hpa):
    return 145366.45 * (1.0 - (pres_hpa / 1013.25) ** 0.190284)


def _fig_to_png_file(fig, filename):
    os.makedirs(MAPS_DIR, exist_ok=True)
    out_path = os.path.join(MAPS_DIR, filename)
    fig.savefig(out_path, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    # Relative path for use directly as an <img src="..."> in index.html
    return f"maps/{filename}"


def generate_blank_basemap():
    """
    Renders a single 'no data available' Florida basemap (coastlines, counties, state
    borders, station markers, no overlay) used as a fallback whenever a given forecast
    hour/threshold has no spatial map yet (download failure, hour outside the HREF
    0-48h window, etc). Overwritten each run; lives at maps/blank_basemap.png.
    """
    try:
        proj = ccrs.PlateCarree()
        states_provinces = cfeature.NaturalEarthFeature(
            category="cultural", name="admin_1_states_provinces_lines",
            scale="50m", facecolor="none"
        )
        counties = cfeature.NaturalEarthFeature(
            category="cultural", name="admin_2_counties",
            scale="10m", facecolor="none"
        )

        fig = plt.figure(figsize=(5.5, 5.8), dpi=120)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent(
            [FL_DOMAIN["lon_min"], FL_DOMAIN["lon_max"], FL_DOMAIN["lat_min"], FL_DOMAIN["lat_max"]],
            crs=proj
        )

        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dbeafe", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f1f5f9", zorder=0)
        ax.add_feature(counties, edgecolor="#cbd5e1", linewidth=0.35, zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#1e293b", linewidth=0.9, zorder=3)
        ax.add_feature(states_provinces, edgecolor="#475569", linewidth=0.8, zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#1e293b", linewidth=0.8, zorder=3)

        for stn_id, coords in STN_COORDS.items():
            ax.plot(
                coords["lon"], coords["lat"], marker="^", markersize=6,
                color="#2563eb", markeredgecolor="white", markeredgewidth=0.8,
                transform=proj, zorder=5
            )
            ax.text(
                coords["lon"] + 0.06, coords["lat"] + 0.05,
                stn_id.upper(), fontsize=6, fontweight="bold", color="#1e3a5f",
                transform=proj, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6, edgecolor="none")
            )

        ax.gridlines(draw_labels=False, linewidth=0.4, color="#94a3b8", alpha=0.5, linestyle="--")
        ax.set_title("No Active Signal", fontsize=9, fontweight="bold", color="#94a3b8")

        out_path = os.path.join(MAPS_DIR, "blank_basemap.png")
        os.makedirs(MAPS_DIR, exist_ok=True)
        fig.savefig(out_path, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        return "maps/blank_basemap.png"
    except Exception as e:
        logging.error(f"Blank basemap generation failed: {e}")
        return None


def _sanitize_grid(grid):
    """masked / NaN / huge-fill (>1e19) / negative values -> 0.0. Returns a float ndarray."""
    try:
        arr = np.ma.filled(np.ma.masked_invalid(np.ma.asarray(grid, dtype=float)), 0.0)
    except (TypeError, ValueError):
        arr = np.zeros_like(np.asarray(grid, dtype=float))
    return np.where((arr > 1e19) | (arr < 0), 0.0, arr)


def _render_ct_domain_map(sub_lons, sub_lats, sub_vals_pct, out_filename, window_label):
    """Render an FL-domain calibrated-thunder map from an ALREADY percent-scaled (0-100)
    subgrid. Kept separate from extraction so the exact same run-level scale drives both the
    table numbers and the map colors. Returns 'maps/<out_filename>' or None."""
    try:
        proj = ccrs.PlateCarree()
        states = cfeature.NaturalEarthFeature(category="cultural",
                    name="admin_1_states_provinces_lines", scale="50m", facecolor="none")
        counties = cfeature.NaturalEarthFeature(category="cultural",
                    name="admin_2_counties", scale="10m", facecolor="none")

        fig = plt.figure(figsize=(5.5, 5.8), dpi=120)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent([FL_DOMAIN["lon_min"], FL_DOMAIN["lon_max"],
                       FL_DOMAIN["lat_min"], FL_DOMAIN["lat_max"]], crs=proj)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dbeafe", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f1f5f9", zorder=0)
        ax.add_feature(counties, edgecolor="#cbd5e1", linewidth=0.35, zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#1e293b", linewidth=0.9, zorder=3)
        ax.add_feature(states, edgecolor="#475569", linewidth=0.8, zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#1e293b", linewidth=0.8, zorder=3)

        masked = np.ma.masked_less_equal(sub_vals_pct, 0.0)
        mesh = ax.pcolormesh(sub_lons, sub_lats, masked, cmap="YlGnBu", vmin=0, vmax=100,
                             shading="auto", transform=proj, zorder=2, alpha=0.85)
        for sid, c in STN_COORDS.items():
            ax.plot(c["lon"], c["lat"], marker="^", markersize=6, color="#b91c1c",
                    markeredgecolor="white", markeredgewidth=0.8, transform=proj, zorder=5)
            ax.text(c["lon"] + 0.06, c["lat"] + 0.05, sid.upper(), fontsize=6,
                    fontweight="bold", color="#7f1d1d", transform=proj, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6, edgecolor="none"))
        ax.gridlines(draw_labels=False, linewidth=0.4, color="#94a3b8", alpha=0.5, linestyle="--")
        ax.set_title(f"HREF Calibrated Thunder ({window_label})", fontsize=9, fontweight="bold", color="#1e293b")
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Probability of Lightning (%)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        return _fig_to_png_file(fig, out_filename)
    except Exception as e:
        logging.error(f"CT domain map render failed: {e}")
        return None


def fetch_calibrated_thunder(window="4hr"):
    """Fetch HREF Calibrated Thunder (HREFCT) for the given accumulation window ('1hr' or
    '4hr') across the 1-48h forecast range. Returns (ct_points, ct_maps):
      ct_points: {stn: {row_key: prob_pct}}
      ct_maps:   {row_key: 'maps/....png' | None}
    The product is a single ML-calibrated probability of >=1 CG flash within 20 km."""
    ct_points = {stn: {} for stn in STATIONS}
    ct_maps = {}

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=3))

    # Robust cycle discovery. SPC HREFCT initializes at 00Z and 12Z. NOMADS frequently throttles
    # a GitHub-Actions IP right after the HREF-lightning download burst that runs just before this,
    # so a single 3-second HEAD is fragile: a throttled probe times out and the run looks "absent"
    # even when it's on disk — and because every probe (today AND the older fallbacks) times out
    # together, the whole product silently drops. Fix: browser User-Agent, longer timeout, retries
    # with backoff, and a newest-first walk across 3 days so a valid fallback is always found.
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
    })

    def _probe_exists(url):
        for attempt in range(3):
            try:
                r = session.head(url, timeout=12, allow_redirects=True)
                if r.status_code == 200:
                    return True
                if r.status_code == 404:
                    return False  # definitively absent — don't burn retries on it
            except Exception:
                pass
            time.sleep(1.5 * (attempt + 1))  # brief backoff to ride out throttling
        return False

    active_cycle = active_date_str = None
    candidates = []
    for days_back in [0, 1, 2]:
        d = (now_utc - datetime.timedelta(days=days_back)).strftime("%Y%m%d")
        for cyc in ["12", "00"]:
            init = datetime.datetime.strptime(f"{d}{cyc}", "%Y%m%d%H").replace(tzinfo=datetime.timezone.utc)
            if init <= now_utc:  # skip cycles that haven't run yet
                candidates.append((init, d, cyc))
    candidates.sort(key=lambda x: x[0], reverse=True)  # newest available cycle first

    for _init, d, cyc in candidates:
        # Probe f004: the first forecast hour valid for BOTH the 1-hr and 4-hr windows (a 4-hr
        # accumulation can't end at f001), so it reliably signals "this cycle exists".
        probe = (f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/spc_post/prod/"
                 f"spc_post.{d}/thunder/spc_post.t{cyc}z.hrefct_{window}.f004.grib2")
        if _probe_exists(probe):
            active_cycle, active_date_str = cyc, d
            break

    if not active_cycle:
        logging.warning(f"No active HREFCT {window} cycle found on NOMADS.")
        return ct_points, ct_maps

    logging.info(f"HREFCT {window}: targeting {active_date_str} {active_cycle}z")
    cycle_init = datetime.datetime.strptime(f"{active_date_str}{active_cycle}", "%Y%m%d%H").replace(tzinfo=datetime.timezone.utc)

    window_label = "1-hr" if window == "1hr" else "4-hr"

    # ---- PHASE A: download every hour, pull RAW (unscaled) FL-domain subgrid + point values.
    # The fraction-vs-percent decision is deliberately NOT made per file. A quiet hour whose
    # entire domain is < 1.0 (a genuine 0.8% field) is indistinguishable from a 0-1 fraction
    # when looked at in isolation — that per-file guess is exactly what turned a real ~1% into
    # a bogus 100%. We instead gather the raw maximum across ALL forecast hours and the whole
    # CONUS grid, then decide the scale ONCE below.
    def _dl_worker(f_hour_int, row_key):
        base = (f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/spc_post/prod/"
                f"spc_post.{active_date_str}/thunder")
        fname = f"spc_post.t{active_cycle}z.hrefct_{window}.f{f_hour_int:03d}.grib2"
        url = f"{base}/{fname}"
        local_path = os.path.join(CACHE_DIR, f"ct_{window}_{fname}")
        try:
            with session.get(url, timeout=10, stream=True) as r:
                if r.status_code != 200:
                    return row_key, None
                with open(local_path, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=8192):
                        fh.write(chunk)

            grbs = pygrib.open(local_path)
            grb = grbs[1]  # single-field product: message 1 is the calibrated probability
            lats, lons = grb.latlons()
            lons_n = np.where(lons > 180, lons - 360.0, lons)
            arr = _sanitize_grid(grb.values)
            grbs.close()

            raw_max = float(arr.max()) if arr.size else 0.0

            # FL-domain subset (indices depend only on the static grid geometry).
            domain_mask = (
                (lats >= FL_DOMAIN["lat_min"]) & (lats <= FL_DOMAIN["lat_max"]) &
                (lons_n >= FL_DOMAIN["lon_min"]) & (lons_n <= FL_DOMAIN["lon_max"])
            )
            ys, xs = np.where(domain_mask)
            sub_lats = sub_lons = sub_vals = None
            if len(ys) > 0:
                y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
                sub_lats = lats[y0:y1 + 1, x0:x1 + 1]
                sub_lons = lons_n[y0:y1 + 1, x0:x1 + 1]
                sub_vals = arr[y0:y1 + 1, x0:x1 + 1]

            # Per-station RAW point value; neighborhood median rejects lone fill artifacts.
            pts = {}
            for stn, c in STN_COORDS.items():
                cache_key = f"ct_{stn}"
                if cache_key in _GRID_INDEX_CACHE:
                    yi, xi = _GRID_INDEX_CACHE[cache_key]
                else:
                    dist = (lats - c["lat"]) ** 2 + (lons_n - c["lon"]) ** 2
                    yi, xi = np.unravel_index(dist.argmin(), dist.shape)
                    _GRID_INDEX_CACHE[cache_key] = (yi, xi)
                yy0 = max(0, yi - 1); yy1 = min(arr.shape[0], yi + 2)
                xx0 = max(0, xi - 1); xx1 = min(arr.shape[1], xi + 2)
                neigh = arr[yy0:yy1, xx0:xx1]
                pts[stn] = float(np.median(neigh)) if neigh.size else float(arr[yi, xi])

            return row_key, {"f": f_hour_int, "raw_max": raw_max, "points": pts,
                             "sub_lats": sub_lats, "sub_lons": sub_lons, "sub_vals": sub_vals}
        except Exception as e:
            logging.debug(f"HREFCT {window} f{f_hour_int:03d} break: {e}")
            return row_key, None
        finally:
            if os.path.exists(local_path):
                try: os.remove(local_path)
                except Exception: pass

    records = {}
    global_raw_max = 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for f_hour_int in range(1, 49):
            valid_dt = cycle_init + datetime.timedelta(hours=f_hour_int)
            row_key = f"{valid_dt.day:02d}/{valid_dt.hour:02d}"
            futures.append(executor.submit(_dl_worker, f_hour_int, row_key))
        for fut in concurrent.futures.as_completed(futures):
            try:
                row_key, rec = fut.result()
            except Exception:
                continue
            if rec is None:
                continue
            records[row_key] = rec
            if rec["raw_max"] > global_raw_max:
                global_raw_max = rec["raw_max"]

    # ---- Decide the scale ONCE for the whole run.
    # If the raw max never exceeds 1.0 across the entire CONUS grid AND all 48 forecast hours,
    # the product is a 0-1 fraction -> x100. Otherwise it is already stored as percent (0-100)
    # and must NOT be rescaled. A true percent field essentially always tops 1% somewhere over
    # 48h, so this cleanly separates the two encodings and kills the per-file 100% misfire.
    run_scale = 100.0 if global_raw_max <= 1.0 else 1.0
    logging.info(f"HREFCT {window}: global raw max={global_raw_max:.4g} -> applying x{run_scale:g}")

    # ---- PHASE B: apply the single scale, populate points, and render maps.
    render_futs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as rex:
        for row_key, rec in records.items():
            for stn in STATIONS:
                raw = rec["points"].get(stn, 0.0)
                ct_points[stn][row_key] = int(round(max(0.0, min(100.0, raw * run_scale))))
            if rec["sub_vals"] is not None:
                sub_pct = np.clip(rec["sub_vals"] * run_scale, 0.0, 100.0)
                out_name = f"ct_{window}_{active_date_str}_{active_cycle}z_f{rec['f']:03d}.png"
                fut = rex.submit(_render_ct_domain_map, rec["sub_lons"], rec["sub_lats"],
                                 sub_pct, out_name, window_label)
                render_futs[fut] = row_key
            else:
                ct_maps[row_key] = None
        for fut in concurrent.futures.as_completed(render_futs):
            rk = render_futs[fut]
            try:
                ct_maps[rk] = fut.result()
            except Exception:
                ct_maps[rk] = None

    n_maps = sum(1 for v in ct_maps.values() if v)
    logging.info(f"HREFCT {window}: points for {len(records)} hours, {n_maps} maps rendered.")
    return ct_points, ct_maps


def _interp_logp(layers, target_p, key):
    """Linear-in-ln(p) interpolation of `key` to target pressure (mb)."""
    below = above = None
    for L in layers:
        if L.get(key) is None:
            continue
        if L["pres"] >= target_p and (below is None or L["pres"] < below["pres"]):
            below = L
        if L["pres"] <= target_p and (above is None or L["pres"] > above["pres"]):
            above = L
    if below is None or above is None:
        return None
    if below["pres"] == above["pres"]:
        return below[key]
    f = (math.log(target_p) - math.log(below["pres"])) / (math.log(above["pres"]) - math.log(below["pres"]))
    return below[key] + f * (above[key] - below[key])


def _sat_vap(tc):
    """Saturation vapor pressure (hPa) over water, Bolton 1980; tc in C."""
    return 6.112 * math.exp(17.67 * tc / (tc + 243.5))


def _mixing_ratio_gkg(td_c, p_mb):
    e = _sat_vap(td_c)
    return 621.97 * e / (p_mb - e)


def _theta_e(tk, tdk, p):
    """Bolton 1980 eq 43 equivalent potential temperature. tk,tdk in K, p in hPa."""
    e = _sat_vap(tdk - 273.15)
    r = 0.62197 * e / (p - e)
    tlcl = 1.0 / (1.0 / (tdk - 56.0) + math.log(tk / tdk) / 800.0) + 56.0
    return tk * (1000.0 / p) ** (0.2854 * (1.0 - 0.28 * r)) * \
        math.exp((3.376 / tlcl - 0.00254) * r * 1000.0 * (1.0 + 0.81 * r))


try:
    import metpy.calc as _mpcalc
    from metpy.units import units as _mpunits
    _HAVE_METPY = True
except Exception:
    _HAVE_METPY = False

_ML_DEPTH_HPA = 100.0  # mixed-layer parcel depth (lowest 100 hPa) for the Lifted Index


def _layer_mean_flow(layers, p_bot, p_top, prefix):
    """Vector-mean wind over [p_top, p_bot] mb: FROM-direction, speed (kt), 8-pt compass regime, and
    the mean u/v components (kept so an ensemble can be averaged in component space). Keys are
    prefixed (e.g. 'mf' -> mf_dir/mf_spd/..., 'av' -> av_dir/...)."""
    us = [L["u"] for L in layers if p_top <= L["pres"] <= p_bot and L.get("u") is not None]
    vs = [L["v"] for L in layers if p_top <= L["pres"] <= p_bot and L.get("v") is not None]
    if not us:
        return {}
    um, vm = sum(us) / len(us), sum(vs) / len(vs)
    frm = math.degrees(math.atan2(-um, -vm)) % 360.0
    compass = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((frm + 22.5) // 45) % 8]
    return {f"{prefix}_dir": round(frm), f"{prefix}_spd": round(math.hypot(um, vm), 1),
            f"{prefix}_regime": compass, f"{prefix}_u": round(um, 3), f"{prefix}_v": round(vm, 3)}


def _thermo_metpy(layers):
    """K-Index, PWAT, and MIXED-LAYER Lifted Index (+ Thompson = KI-LI) via MetPy."""
    P, T, D = [], [], []
    for L in layers:
        if None in (L.get("pres"), L.get("tmpc"), L.get("dwpt")):
            continue
        P.append(L["pres"]); T.append(L["tmpc"]); D.append(min(L["dwpt"], L["tmpc"]))
    if len(P) < 4:
        return {}
    p = np.array(P) * _mpunits.hPa
    Tq = np.array(T) * _mpunits.degC
    Tdq = np.array(D) * _mpunits.degC

    def _scal(q):
        return float(np.atleast_1d(q.magnitude)[0])

    out = {}
    ki = _scal(_mpcalc.k_index(p, Tq, Tdq))
    out["k_index"] = round(ki, 1)
    pw_in = _scal(_mpcalc.precipitable_water(p, Tdq).to("inch"))
    out["pwat_in"] = round(pw_in, 2)
    out["pwat_mm"] = round(pw_in * 25.4, 1)
    # Mixed-layer parcel over the lowest _ML_DEPTH_HPA, lifted to 500 mb
    _, mp_T, mp_Td = _mpcalc.mixed_parcel(p, Tq, Tdq, depth=_ML_DEPTH_HPA * _mpunits.hPa)
    prof = _mpcalc.parcel_profile(p, mp_T, mp_Td).to("degC")
    li = _scal(_mpcalc.lifted_index(p, Tq, prof))
    out["lifted_index"] = round(li, 1)
    out["thompson"] = round(ki - li, 1)
    out["parcel"] = "mixed-layer"
    out["engine"] = "metpy"
    return out


def _thermo_numpy(layers):
    """Fallback for when MetPy isn't installed: KI, PWAT, and a MIXED-LAYER Lifted Index built by
    mixing potential temperature and mixing ratio over the lowest _ML_DEPTH_HPA, then lifting via
    theta-e conservation to 500 mb."""
    out = {}
    t850 = _interp_logp(layers, 850, "tmpc"); td850 = _interp_logp(layers, 850, "dwpt")
    t700 = _interp_logp(layers, 700, "tmpc"); td700 = _interp_logp(layers, 700, "dwpt")
    t500 = _interp_logp(layers, 500, "tmpc")
    ki = None
    if None not in (t850, td850, t700, td700, t500):
        ki = (t850 - t500) + td850 - (t700 - td700)
        out["k_index"] = round(ki, 1)
    li = None
    p_sfc = layers[0]["pres"]
    ml = [L for L in layers if L.get("tmpc") is not None and L.get("dwpt") is not None
          and (p_sfc - L["pres"]) <= _ML_DEPTH_HPA]
    if t500 is not None and ml:
        theta_ml = sum((L["tmpc"] + 273.15) * (1000.0 / L["pres"]) ** 0.2854 for L in ml) / len(ml)
        w_ml = sum(_mixing_ratio_gkg(L["dwpt"], L["pres"]) / 1000.0 for L in ml) / len(ml)
        ml_tk = theta_ml * (p_sfc / 1000.0) ** 0.2854
        e = max(w_ml * p_sfc / (0.62197 + w_ml), 1e-6)
        ml_tdc = 243.5 * math.log(e / 6.112) / (17.67 - math.log(e / 6.112))
        thetae_parcel = _theta_e(ml_tk, ml_tdc + 273.15, p_sfc)
        lo, hi = 200.0, 320.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _theta_e(mid, mid, 500.0) < thetae_parcel:
                lo = mid
            else:
                hi = mid
        li = t500 - (0.5 * (lo + hi) - 273.15)
        out["lifted_index"] = round(li, 1)
    if ki is not None and li is not None:
        out["thompson"] = round(ki - li, 1)
    wl = [L for L in layers if L.get("dwpt") is not None]
    if len(wl) >= 3:
        tot = 0.0
        for i in range(len(wl) - 1):
            p1, p2 = wl[i]["pres"], wl[i + 1]["pres"]
            w1 = _mixing_ratio_gkg(wl[i]["dwpt"], p1) / 1000.0
            w2 = _mixing_ratio_gkg(wl[i + 1]["dwpt"], p2) / 1000.0
            tot += 0.5 * (w1 + w2) * ((p1 - p2) * 100.0) / 9.81
        out["pwat_mm"] = round(tot, 1)
        out["pwat_in"] = round(tot / 25.4, 2)
    out["parcel"] = "mixed-layer"
    out["engine"] = "numpy"
    return out


def _rh_of_layer(l):
    """Relative humidity (%) for a layer: native GRIB 'rh' when present, else Magnus-derived from
    T/Td (BUFKIT path). Mirrors the _layer_rh helper used for cloud decks."""
    rh = l.get("rh")
    if rh is not None:
        return rh
    t, td = l.get("tmpc"), l.get("dwpt")
    if t is None or td is None:
        return None
    a, b = 17.625, 243.04
    es = lambda x: math.exp((a * x) / (b + x))
    return max(1.0, min(100.0, 100.0 * es(min(td, t)) / es(t)))


def _layer_mean_rh(layers, p_bot, p_top):
    """Mean RH (%) through [p_top, p_bot] mb."""
    vals = [_rh_of_layer(l) for l in layers if p_top <= l["pres"] <= p_bot]
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


# ---- Cizek lightning-probability random forest (KSC/CCSFS, 10Z sounding) --------------------
# Source model: cyclonecizek/LightningProbabilityTool, RFC_model_limited_depth_10Z_updated.sav
# (RandomForestClassifier, 500 trees, max_depth 6, scikit-learn 1.3.2).
#
# That pickle requires numpy<2 and cannot be loaded alongside this pipeline's numpy 2.x, so the
# forest was exported to a plain .npz of per-tree arrays and is evaluated here in pure numpy. The
# extraction was verified to reproduce sklearn's predict_proba EXACTLY (max abs diff 0.0) over a
# validation grid, so this is a re-implementation of the same model, not an approximation.
#
# Features, in this exact order (model.feature_names_in_):
#   Thompson_Index                        -- K-Index minus Lifted Index
#   1000-700mb_Average_U-Wind_Component   -- NOTE: the upstream tool defines this as
#                                            speed_kt * cos(deg2rad(270 - direction)), i.e. a
#                                            WESTERLY-POSITIVE component in KNOTS off the
#                                            meteorological FROM-direction. This is NOT the
#                                            standard math-convention u, so it is rebuilt with the
#                                            upstream formula rather than reusing mf_u.
#   700-500mb_Average_RH                  -- percent
RF_LIGHTNING_ENABLED = True
RF_LIGHTNING_NPZ = "rf_lightning_10Z.npz"
_RF_LTG = None
_RF_LTG_TRIED = False


def _rf_lightning_load():
    """Load the exported forest once. Returns the arrays dict, or None if unavailable (in which
    case the column simply shows '-' — a missing model must never fail the run)."""
    global _RF_LTG, _RF_LTG_TRIED
    if _RF_LTG_TRIED:
        return _RF_LTG
    _RF_LTG_TRIED = True
    if not RF_LIGHTNING_ENABLED:
        return None
    try:
        path = RF_LIGHTNING_NPZ
        if not os.path.exists(path):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RF_LIGHTNING_NPZ)
        d = np.load(path, allow_pickle=True)
        _RF_LTG = {k: d[k] for k in d.files}
        ntrees = len(_RF_LTG["tree_offsets"]) - 1
        logging.info(f"Cizek lightning RF loaded: {ntrees} trees, "
                     f"{int(_RF_LTG['tree_offsets'][-1])} nodes.")
    except Exception as e:
        logging.warning(f"Cizek lightning RF unavailable ({e}); probability column will be blank.")
        _RF_LTG = None
    return _RF_LTG


def rf_lightning_u_wind(mf_dir, mf_spd):
    """Upstream tool's '1000-700mb Average U-Wind Component': westerly-positive knots."""
    if mf_dir is None or mf_spd is None:
        return None
    return mf_spd * math.cos(math.radians(270.0 - mf_dir))


def rf_lightning_prob(thompson, u_wind, rh_700_500):
    """P(lightning) in percent for one 10Z environment, or None if a feature is missing."""
    r = _rf_lightning_load()
    if r is None or thompson is None or u_wind is None or rh_700_500 is None:
        return None
    try:
        X = np.array([[float(thompson), float(u_wind), float(rh_700_500)]], dtype=float)
        cl, cr = r["children_left"], r["children_right"]
        fe, th, va, off = r["feature"], r["threshold"], r["value"], r["tree_offsets"]
        acc = np.zeros((1, va.shape[1]), dtype=float)
        for t in range(len(off) - 1):
            a, b = int(off[t]), int(off[t + 1])
            tcl, tcr, tfe, tth = cl[a:b], cr[a:b], fe[a:b], th[a:b]
            node = 0
            while tcl[node] != -1:                     # walk to a leaf
                node = tcl[node] if X[0, tfe[node]] <= tth[node] else tcr[node]
            acc += va[a:b][node]
        acc /= (len(off) - 1)
        c1 = int(np.where(r["classes"] == 1.0)[0][0])
        return round(float(acc[0, c1]) * 100.0, 1)
    except Exception as e:
        logging.debug(f"Cizek lightning RF predict failed: {e}")
        return None


def compute_launch_thermo(profile_layers):
    """1000-700 mb mean flow (+ regime), K-Index, MIXED-LAYER Lifted Index, Thompson Index (KI-LI),
    and PWAT. Uses MetPy when available and falls back to an equivalent numpy implementation.
    NOTE: this is now called on demand for the launch-thermo panel only (not per profile), since the
    MetPy path is ~17 ms/profile. Returns {} if the profile is too thin."""
    try:
        layers = sorted([L for L in profile_layers if L.get("pres") is not None],
                        key=lambda x: -x["pres"])
        if len(layers) < 4:
            return {}
        out = dict(_layer_mean_flow(layers, 1000.0, 700.0, "mf"))
        out.update(_layer_mean_flow(layers, 300.0, 150.0, "av"))
        rh75 = _layer_mean_rh(layers, 700.0, 500.0)   # Cizek RF feature 3
        if rh75 is not None:
            out["rh_700_500"] = rh75
        core = _thermo_metpy(layers) if _HAVE_METPY else {}
        if not core:
            core = _thermo_numpy(layers)
        out.update(core)
        return out
    except Exception:
        return {}


def compute_profile_variables(profile_layers):
    """
    Given a list of profile layers (each a dict with pres/hght/tmpc/dwpt/depr/sknt/u/v),
    compute the aviation variable set: mixed-layer momentum (mean + max), ceiling,
    visibility, and TAF-style low-level wind shear. Shared by the BUFKIT station path
    and the raw-GRIB (pad / RRFS / REFS / ECMWF) path so the math stays identical.
    Returns the per-hour data dict, or None if the profile is unusable.
    """
    if not profile_layers:
        return None
    profile_layers = sorted(profile_layers, key=lambda x: x["pres"], reverse=True)

    sfc_hght = profile_layers[0]["hght"]

    def _interp_uv_agl(target_agl_ft):
        """Linearly interpolate the u/v wind components to a target height (ft AGL).
        Returns (None, None) when the target is not bracketed by two levels that both
        carry a wind vector — deliberately strict, so a coarse column can never silently
        substitute a stratospheric level for a 2,000 ft one."""
        for i in range(len(profile_layers) - 1):
            l1, l2 = profile_layers[i], profile_layers[i + 1]
            if None in (l1.get("u"), l1.get("v"), l2.get("u"), l2.get("v")):
                continue
            agl1 = l1["hght"] - sfc_hght
            agl2 = l2["hght"] - sfc_hght
            if (agl1 <= target_agl_ft <= agl2) or (agl2 <= target_agl_ft <= agl1):
                if agl1 == agl2:
                    return l1["u"], l1["v"]
                f = (target_agl_ft - agl1) / (agl2 - agl1)
                return l1["u"] + f * (l2["u"] - l1["u"]), l1["v"] + f * (l2["v"] - l1["v"])
        return None, None

    def calc_llws():
        """TAF-style Low-Level Wind Shear through the lowest ~2,000 ft AGL.

        LLWS is a VECTOR difference, not a speed difference: a wind that backs 90 degrees
        while holding 15 kt is ~21 kt of shear even though the speed never changed. So each
        level's wind is taken as its u/v components and every pair of levels in the layer is
        differenced, keeping the largest magnitude (the classic BUFKIT/TAF hand method — the
        strongest shear is often between two levels aloft rather than surface-to-top).

        Coarse isobaric columns (GRIB: RRFS/REFS/HRRR/ECMWF/pads) carry only two or three
        levels below 2,000 ft, so an interpolated 2,000 ft AGL level is always spliced in.
        That keeps every model column evaluated over the same nominal depth, though a
        mandatory-level column still can't resolve a shallow nocturnal jet the way the
        ~40-level BUFKIT soundings can.

        Returns (magnitude_kt, top_height_ft_agl, from_direction_deg, speed_kt) or None.
        """
        cands = []
        for L in profile_layers:
            if L.get("u") is None or L.get("v") is None:
                continue
            agl = L["hght"] - sfc_hght
            if -50.0 <= agl <= LLWS_SEARCH_FT_AGL:
                cands.append((max(0.0, agl), L["u"], L["v"]))
        u2k, v2k = _interp_uv_agl(LLWS_TOP_FT_AGL)
        if u2k is not None and v2k is not None:
            cands.append((LLWS_TOP_FT_AGL, u2k, v2k))
        if len(cands) < 2:
            return None
        cands.sort(key=lambda c: c[0])

        best_mag, best_top = 0.0, None
        for i in range(len(cands) - 1):
            for j in range(i + 1, len(cands)):
                mag = math.hypot(cands[j][1] - cands[i][1], cands[j][2] - cands[i][2])
                if mag > best_mag:
                    best_mag, best_top = mag, cands[j]
        if best_top is None:
            return None
        h_agl, u_top, v_top = best_top
        # Recover the meteorological FROM-direction and speed at the shear-layer top.
        drct = math.degrees(math.atan2(-u_top, -v_top)) % 360.0
        spd = math.hypot(u_top, v_top)
        return best_mag, h_agl, drct, spd

    def _layer_rh(layer):
        """Relative humidity (%) for a layer. Uses stored 'rh' if present (raw GRIB paths),
        otherwise derives it from temperature/dewpoint (BUFKIT path) via the Magnus formula."""
        rh = layer.get("rh")
        if rh is not None:
            return rh
        t = layer.get("tmpc")
        td = layer.get("dwpt")
        if t is None or td is None:
            return None
        a, b = 17.625, 243.04
        try:
            gt = (a * t) / (b + t)
            gd = (a * td) / (b + td)
            return 100.0 * math.exp(gd - gt)
        except Exception:
            return None

    def _group_cloud_layers(is_cloud_fn):
        """Walk the profile bottom-up, grouping contiguous 'in cloud' levels into decks.
        is_cloud_fn(layer) -> bool decides membership. To avoid a coarse (mandatory-level)
        GRIB column merging widely-separated moist levels into one impossibly-thick deck,
        two consecutive in-cloud levels are only joined if the vertical gap between them is
        at most MAX_LEVEL_GAP_FT; a larger jump breaks the deck (we can't confirm the air
        between two sparse levels is actually cloudy)."""
        MAX_LEVEL_GAP_FT = 5000.0
        decks = []
        active = None
        prev_hght = None
        for layer in profile_layers:
            if is_cloud_fn(layer):
                if active is None:
                    active = {"base": layer["hght"], "top": layer["hght"]}
                elif (layer["hght"] - prev_hght) <= MAX_LEVEL_GAP_FT:
                    active["top"] = layer["hght"]
                else:
                    # Gap too large to trust as a single deck; close current, start new.
                    decks.append(active)
                    active = {"base": layer["hght"], "top": layer["hght"]}
                prev_hght = layer["hght"]
            elif active is not None:
                decks.append(active)
                active = None
                prev_hght = None
        if active:
            decks.append(active)
        return decks

    # Ceiling uses the RH >= 95% criterion: a discrete "is there a solid deck here" test
    # that avoids over-calling MVFR.
    RH_CLOUD_THRESHOLD = 95.0
    ceiling_decks = _group_cloud_layers(
        lambda l: (_layer_rh(l) is not None and _layer_rh(l) >= RH_CLOUD_THRESHOLD)
    )

    # --- Mixed-layer momentum (BUFKIT-style) -------------------------------------
    # Both PBL Mom Mean (transport-style mean wind through the mixed layer) and PBL Mom Max
    # (gust/mixing potential = strongest wind within the mixed layer) are evaluated over the
    # DIAGNOSED mixed-layer depth, not a fixed 850 hPa slab. The mixed-layer top is found by
    # walking up from the surface until potential temperature (theta) rises more than
    # THETA_DELTA_K above the surface value — the classic well-mixed-layer criterion.
    def _theta_k(layer):
        t_k = layer["tmpc"] + 273.15
        return t_k * (1000.0 / layer["pres"]) ** 0.286

    THETA_DELTA_K = 1.5  # K above surface theta that marks the mixed-layer top
    MIN_ML_TOP_FT = 1000.0   # floor so a strong nocturnal inversion still yields a usable layer
    MAX_ML_TOP_FT = 12000.0  # ceiling guard against runaway deep-convective profiles

    sfc_theta = _theta_k(profile_layers[0])
    ml_top_ft = None
    for layer in profile_layers[1:]:
        if _theta_k(layer) - sfc_theta > THETA_DELTA_K:
            ml_top_ft = layer["hght"]
            break
    if ml_top_ft is None:
        ml_top_ft = profile_layers[-1]["hght"]
    # Clamp the diagnosed top into a sane AGL band
    ml_top_ft = max(sfc_hght + MIN_ML_TOP_FT, min(ml_top_ft, sfc_hght + MAX_ML_TOP_FT))

    ml_winds = [l["sknt"] for l in profile_layers if l["hght"] <= ml_top_ft]
    if not ml_winds:                       # degenerate guard: at least use the surface layer
        ml_winds = [profile_layers[0]["sknt"]]
    mean_wind = sum(ml_winds) / len(ml_winds)
    max_pbl = max(ml_winds)

    sfc_depr = profile_layers[0]["depr"] if profile_layers else 10.0
    vis = 0.25 if sfc_depr <= 0.5 else (1.0 if sfc_depr <= 1.0 else (3.0 if sfc_depr <= 2.0 else 10.0))
    valid_ceilings = [c for c in ceiling_decks if c["base"] >= 100.0]
    ceiling_val = round(valid_ceilings[0]["base"]) if valid_ceilings else 24000.0

    out = {
        "mom_mean": round(mean_wind, 1),
        "mom_max": round(max_pbl, 1),
        "vis": vis,
        "ceiling": ceiling_val,
        "_layers": profile_layers,
    }

    llws = calc_llws()
    if llws is not None:
        mag, h_agl, drct, spd = llws
        out["llws"] = round(mag, 1)
        out["llws_hft"] = round(h_agl)
        out["llws_dir"] = round(drct) % 360 or 360
        out["llws_spd"] = round(spd, 1)
        if mag >= LLWS_TAF_MIN_KT:
            # TAF group: WSddd/dddffKT — height in hundreds of ft AGL (capped at the nominal
            # 2,000 ft layer top), direction to the nearest 10 deg, speed to the nearest 5 kt.
            hund = max(1, min(int(round(LLWS_TOP_FT_AGL / 100.0)), int(round(h_agl / 100.0))))
            d10 = int(round(drct / 10.0) * 10) % 360 or 360
            s5 = int(round(spd / 5.0) * 5)
            out["llws_taf"] = f"WS{hund:03d}/{d10:03d}{s5:02d}KT"
        else:
            out["llws_taf"] = None
    else:
        out["llws"] = None
        out["llws_taf"] = None

    return out
def parse_time_series_bufkit(bufkit_text):
    hourly_data = {}
    blocks = bufkit_text.split("STID = ")

    for block in blocks:
        if not block.strip():
            continue
        time_match = re.search(r"TIME\s*=\s*(\d{6})/(\d{4})", block)
        if not time_match:
            continue

        date_part, time_part = time_match.groups()
        try:
            valid_hour_key = f"{int(date_part[4:6]):02d}/{int(time_part[0:2]):02d}"
        except (ValueError, IndexError):
            continue

        lines = block.splitlines()
        profile_layers = []
        pres_idx, tmpc_idx, dwpt_idx, sknt_idx, drct_idx = 0, 1, 3, 5, 4
        header_names = []
        in_profile = False

        for line in lines:
            cleaned = line.strip()
            if not cleaned:
                continue
            if "PRES" in cleaned or "TMPC" in cleaned or "SKNT" in cleaned:
                in_profile = True
                header_names.extend(cleaned.split())
                try:
                    if "PRES" in header_names: pres_idx = header_names.index("PRES")
                    if "TMPC" in header_names: tmpc_idx = header_names.index("TMPC")
                    if "DWPT" in header_names: dwpt_idx = header_names.index("DWPT")
                    if "SKNT" in header_names: sknt_idx = header_names.index("SKNT")
                    if "DRCT" in header_names: drct_idx = header_names.index("DRCT")
                except ValueError:
                    pass
                continue

            if in_profile:
                if "STID" in cleaned or "STNM" in cleaned:
                    break
                parts = cleaned.split()
                if len(parts) > max(pres_idx, tmpc_idx, dwpt_idx, sknt_idx, drct_idx):
                    try:
                        if not parts[0].replace(".", "", 1).replace("-", "", 1).isdigit():
                            continue
                        pres = float(parts[pres_idx])
                        tmpc = float(parts[tmpc_idx])
                        dwpt = float(parts[dwpt_idx])
                        sknt = float(parts[sknt_idx])
                        try:
                            drct = float(parts[drct_idx])
                        except (ValueError, IndexError):
                            drct = None
                        if 100.0 <= pres <= 1050.0:
                            # Meteorological wind vector components (u: east+, v: north+).
                            # "FROM" direction convention -> components point opposite the heading.
                            if drct is not None and 0.0 <= drct <= 360.0:
                                u_comp = -sknt * math.sin(math.radians(drct))
                                v_comp = -sknt * math.cos(math.radians(drct))
                            else:
                                u_comp, v_comp = None, None
                            profile_layers.append({
                                "pres": pres,
                                "hght": pressure_to_height_ft(pres),
                                "tmpc": tmpc,
                                "dwpt": dwpt,
                                "depr": tmpc - dwpt,
                                "sknt": sknt,
                                "drct": drct,
                                "u": u_comp,
                                "v": v_comp,
                            })
                    except (ValueError, IndexError):
                        continue

        if not profile_layers:
            continue
        profile_layers.sort(key=lambda x: x["pres"], reverse=True)

        result = compute_profile_variables(profile_layers)
        if result is not None:
            hourly_data[valid_hour_key] = result
    return hourly_data


def fetch_station_model(session, stn, model):
    """Pull one station-model BUFKIT profile from PSU, politely.

    Returns (stn, model, hourly_data). An empty dict means the fetch failed or the file
    wasn't posted; run_pipeline will try to carry the previous run's column forward rather
    than render a blank column.
    """
    download_id = "xmr" if stn == "kxmr" else stn
    model_prefix = "gfs3" if model == "gfs" else model
    # https, not http: PSU redirects anyway, and the redirect costs an extra round trip
    # against a server that is already rate-limiting us.
    url = (f"https://www.meteo.psu.edu/bufkit/data/{model.upper()}/latest/"
           f"{model_prefix}_{download_id}.buf")
    headers = {
        "User-Agent": BUFKIT_USER_AGENT,
        "Accept": "text/plain,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    last = "no attempt made"
    with _BUFKIT_GATE:
        # Stagger inside the gate so the two permitted slots don't fire simultaneously.
        time.sleep(random.uniform(*BUFKIT_STAGGER_S))
        for attempt in range(BUFKIT_ATTEMPTS):
            try:
                r = session.get(url, headers=headers,
                                timeout=(BUFKIT_CONNECT_TIMEOUT, BUFKIT_READ_TIMEOUT))
                if r.status_code == 200:
                    body = r.text
                    if body and "STID" in body:
                        data = parse_time_series_bufkit(body)
                        if data:
                            return stn, model, data
                        last = f"200 OK ({len(body)} B) but no parseable profiles"
                    else:
                        # A throttle page or truncated body can still come back 200.
                        last = f"200 OK but body is not BUFKIT ({len(body or '')} B)"
                elif r.status_code == 404:
                    # Genuinely not posted (some models skip some sites). Don't burn retries.
                    logging.warning(f"BUFKIT {stn}/{model}: 404, file not posted.")
                    return stn, model, {}
                elif r.status_code in (403, 429, 500, 502, 503, 504):
                    # These are exactly the throttle codes worth backing off on — the old
                    # code `break`-ed here and never retried them.
                    last = f"HTTP {r.status_code} (throttled)"
                else:
                    last = f"HTTP {r.status_code}"
            except Exception as e:
                last = f"{type(e).__name__}"

            if attempt < BUFKIT_ATTEMPTS - 1:
                wait = BUFKIT_BACKOFF_BASE_S * (2 ** attempt) * random.uniform(0.6, 1.4)
                logging.info(f"BUFKIT {stn}/{model}: {last}; retry "
                             f"{attempt + 2}/{BUFKIT_ATTEMPTS} in {wait:.1f}s")
                time.sleep(wait)

    logging.error(f"BUFKIT {stn}/{model} failed after {BUFKIT_ATTEMPTS} attempts ({last}).")
    return stn, model, {}


def _row_is_future(row_key, now_utc):
    """True when a 'DD/HH' row key is at or after the current hour (same wrap-safe rule
    run_pipeline uses to trim the live BUFKIT rows)."""
    try:
        d, h = map(int, row_key.split("/"))
    except Exception:
        return True
    if d < now_utc.day and now_utc.day - d < 25:
        return False
    if d == now_utc.day and h < now_utc.hour:
        return False
    return True


def _prior_run_station_data():
    """Newest stored run's station block from history.json, as (data, timestamp)."""
    try:
        with open(HISTORY_FILE, "r") as f:
            payload = json.load(f)
        runs = payload.get("runs", []) if isinstance(payload, dict) else payload
        for r in (runs or []):
            d = (r or {}).get("data") or {}
            if d:
                return d, r.get("timestamp")
    except Exception:
        pass
    return {}, None


def carry_forward_missing(sounding_matrix, models_to_check=None):
    """When PSU throttles us out of an entire airport column, reuse the newest stored rows
    for that (station, model) instead of rendering a blank column. Only EMPTY columns are
    filled — a partial fetch is never overwritten — and only forecast hours still in the
    future are carried, so nothing rots into the past. Each carried profile is tagged with
    the run it came from, which the frontend surfaces as a stale marker.

    A carried BUFKIT column is a genuinely older forecast, not a nowcast: treat it as the
    last known good run, and note that RRFS/REFS/ECMWF in the same row are current."""
    prior, ts = _prior_run_station_data()
    if not prior:
        return 0
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    filled = 0
    for stn, mdls in sounding_matrix.items():
        for mdl in (models_to_check or list(mdls.keys())):
            if mdls.get(mdl):
                continue  # this column fetched fine
            old = ((prior.get(stn) or {}).get(mdl)) or {}
            carried = {}
            for rk, prof in old.items():
                if isinstance(prof, dict) and _row_is_future(rk, now_utc):
                    p = dict(prof)
                    p["stale"] = p.get("stale") or ts or "a previous run"
                    carried[rk] = p
            if carried:
                mdls[mdl] = carried
                filled += 1
                logging.warning(f"Carried forward {stn}/{mdl}: {len(carried)} future hours "
                                f"from {ts} (BUFKIT fetch returned nothing this run).")
    if filled:
        logging.warning(f"{filled} BUFKIT column(s) carried forward and flagged stale.")
    return filled

# ---------------------------------------------------------------------------
# Launch-pad soundings derived from raw isobaric GRIB2 (GFS / RAP / HRRR)
# ---------------------------------------------------------------------------

# Isobaric levels to request from the NOMADS GRIB filter, in hPa. GFS carries the
# full mandatory+standard set; RAP/HRRR carry 25 hPa spacing but we request the same
# nominal list and just use whatever comes back.
PAD_LEVELS_HPA = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600,
                  550, 500, 450, 400, 350, 300, 250, 200, 150, 100]


def _rh_to_dewpoint_c(temp_c, rh_pct):
    """Magnus-formula dewpoint (°C) from temperature (°C) and relative humidity (%)."""
    if rh_pct is None or rh_pct <= 0:
        return temp_c - 30.0  # very dry fallback
    rh = max(1.0, min(100.0, rh_pct))
    a, b = 17.625, 243.04
    gamma = math.log(rh / 100.0) + (a * temp_c) / (b + temp_c)
    return (b * gamma) / (a - gamma)


def _nomads_grib_url(model, date_str, cycle, f_hour_int):
    """Build a NOMADS GRIB-filter URL that subsets to isobaric T/RH/HGT/UGRD/VGRD +
    surface pressure over a small Cape Canaveral bounding box (keeps downloads tiny)."""
    lev_params = "".join(f"&lev_{lv}_mb=on" for lv in PAD_LEVELS_HPA)
    var_params = "&var_TMP=on&var_RH=on&var_HGT=on&var_UGRD=on&var_VGRD=on&var_PRES=on"
    region = "&subregion=&leftlon=-81.2&rightlon=-80.0&toplat=29.2&bottomlat=28.0"

    if model == "hrrr":
        # Use the pressure-level HRRR filter (filter_hrrr_2d.pl is SURFACE fields only and
        # cannot serve the wrfprs 3D isobaric file we need for a sounding).
        base = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_sub.pl"
        f_name = f"hrrr.t{cycle}z.wrfprsf{f_hour_int:02d}.grib2"
        dir_part = f"&dir=%2Fhrrr.{date_str}%2Fconus"
    elif model == "rap":
        base = "https://nomads.ncep.noaa.gov/cgi-bin/filter_rap.pl"
        f_name = f"rap.t{cycle}z.awp130pgrbf{f_hour_int:02d}.grib2"
        dir_part = f"&dir=%2Frap.{date_str}"
    else:  # gfs
        base = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        f_name = f"gfs.t{cycle}z.pgrb2.0p25.f{f_hour_int:03d}"
        dir_part = f"&dir=%2Fgfs.{date_str}%2F{cycle}%2Fatmos"

    return f"{base}?file={f_name}{lev_params}{var_params}{region}{dir_part}"


def _grib_levels_to_layers(levels):
    """Convert a {pressure_hPa: {field: value}} dict (fields decoded from raw isobaric GRIB2:
    t, rh, hgt, u, v) into the profile_layers schema consumed by compute_profile_variables().
    Shared by the launch-pad NOMADS path and the ECMWF Open Data path so the math is identical."""
    layers = []
    for pres, f in levels.items():
        if "t" not in f:
            continue
        tmpc = f["t"] - 273.15 if f["t"] > 100 else f["t"]  # K -> C guard
        rh = f.get("rh")
        dwpt = _rh_to_dewpoint_c(tmpc, rh)
        u = f.get("u")
        v = f.get("v")
        sknt = math.hypot(u, v) * 1.943844 if (u is not None and v is not None) else 0.0
        # GRIB geopotential height (gpm) -> feet if present, else barometric fallback.
        hght_ft = f["hgt"] * 3.280839895 if "hgt" in f else pressure_to_height_ft(pres)
        layers.append({
            "pres": pres,
            "hght": hght_ft,
            "tmpc": tmpc,
            "dwpt": dwpt,
            "depr": tmpc - dwpt,
            "rh": rh,  # native GRIB RH (%), used directly for RH>=95% cloud detection
            "sknt": sknt,
            "drct": None,
            "u": u * 1.943844 if u is not None else None,  # m/s -> kt
            "v": v * 1.943844 if v is not None else None,
        })
    return layers


def build_pad_profiles_from_grib(filepath, pad_coords, debug=False):
    """
    Extract a vertical column at each pad's nearest grid cell from a raw isobaric
    GRIB2 file and assemble profile_layers dicts (matching the BUFKIT schema) so the
    shared compute_profile_variables() can run on them.
    Returns {pad_id: profile_layers_list}. When debug=True, logs a summary of every
    distinct (shortName, typeOfLevel) seen and how many isobaric fields matched, so a
    first live run reveals exactly what NOMADS returned vs what the parser expects.
    """
    per_pad_levels = {pid: {} for pid in pad_coords}
    seen_short_types = {}   # (shortName, typeOfLevel) -> count      [debug]
    matched_counts = {"t": 0, "rh": 0, "hgt": 0, "u": 0, "v": 0}   # [debug]
    isobaric_levels_seen = set()                                    # [debug]
    total_msgs = 0                                                  # [debug]
    try:
        grbs = pygrib.open(filepath)
        # Cache lat/lon grid + nearest-cell index per pad from the first message.
        grid_lats, grid_lons = None, None
        pad_ij = {}

        for grb in grbs:
            total_msgs += 1
            try:
                level = grb.level
                short = getattr(grb, "shortName", "")
                type_lvl = getattr(grb, "typeOfLevel", "")
            except Exception:
                continue

            if debug:
                key = (short, type_lvl)
                seen_short_types[key] = seen_short_types.get(key, 0) + 1

            if type_lvl != "isobaricInhPa" or level not in PAD_LEVELS_HPA:
                continue

            if debug:
                isobaric_levels_seen.add(level)

            if grid_lats is None:
                grid_lats, grid_lons = grb.latlons()
                glons = np.where(grid_lons > 180, grid_lons - 360.0, grid_lons)
                for pid, c in pad_coords.items():
                    dist = (grid_lats - c["lat"]) ** 2 + (glons - c["lon"]) ** 2
                    pad_ij[pid] = np.unravel_index(np.argmin(dist), dist.shape)

            vals = grb.values
            field = None
            if short in ("t", "TMP"): field = "t"
            elif short in ("r", "RH"): field = "rh"
            elif short in ("gh", "HGT"): field = "hgt"
            elif short in ("u", "UGRD", "10u"): field = "u"
            elif short in ("v", "VGRD", "10v"): field = "v"
            if field is None:
                continue

            if debug:
                matched_counts[field] += 1

            for pid, (iy, ix) in pad_ij.items():
                per_pad_levels[pid].setdefault(level, {})[field] = float(vals[iy, ix])

        grbs.close()
    except Exception as e:
        logging.error(f"Pad GRIB parse failed for {filepath}: {e}")
        return {}

    if debug:
        logging.info(f"[PAD DEBUG] {os.path.basename(filepath)}: {total_msgs} total GRIB messages")
        logging.info(f"[PAD DEBUG]   distinct (shortName, typeOfLevel) seen: "
                     + ", ".join(f"{k[0]}/{k[1]}={v}" for k, v in sorted(seen_short_types.items())))
        logging.info(f"[PAD DEBUG]   isobaric levels matched (hPa): {sorted(isobaric_levels_seen, reverse=True)}")
        logging.info(f"[PAD DEBUG]   fields matched to parser: {matched_counts}")
        if sum(matched_counts.values()) == 0:
            logging.warning("[PAD DEBUG]   >>> ZERO fields matched. shortNames above don't match the "
                            "parser's expected set (t/r/gh/u/v). Update the field mapping to match.")

    pad_profiles = {}
    for pid, levels in per_pad_levels.items():
        layers = _grib_levels_to_layers(levels)
        if layers:
            pad_profiles[pid] = layers
    return pad_profiles


def fetch_pad_model(session, model, date_str, cycle, f_hour_int, row_key, debug=False):
    """Download one raw GRIB2 subset and build pad profiles/variables for a single
    model forecast hour. Returns (row_key, model, {pad_id: variables_dict}).
    When debug=True, logs the request URL, HTTP status, and downloaded byte size."""
    url = _nomads_grib_url(model, date_str, cycle, f_hour_int)
    local_path = os.path.join(CACHE_DIR, f"pad_{model}_{cycle}z_f{f_hour_int:03d}.grib2")
    out = {}
    try:
        with session.get(url, timeout=25, stream=True) as r:
            if debug:
                logging.info(f"[PAD DEBUG] {model.upper()} f{f_hour_int:03d} HTTP {r.status_code}")
                logging.info(f"[PAD DEBUG]   URL: {url}")
            if r.status_code != 200:
                if debug:
                    logging.warning(f"[PAD DEBUG]   >>> Non-200 status. Check the NOMADS filter path/"
                                    f"filename for {model.upper()}. First 300 chars of body:")
                    try:
                        logging.warning(f"[PAD DEBUG]   {r.text[:300]}")
                    except Exception:
                        pass
                return row_key, model, out
            with open(local_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=16384):
                    fh.write(chunk)

        if debug:
            sz = os.path.getsize(local_path) if os.path.exists(local_path) else 0
            logging.info(f"[PAD DEBUG]   downloaded {sz} bytes")

        pad_profiles = build_pad_profiles_from_grib(local_path, LAUNCH_PADS, debug=debug)
        for pid, layers in pad_profiles.items():
            result = compute_profile_variables(layers)
            if result is not None:
                out[pid] = result
        if debug:
            sample_pid = next(iter(pad_profiles), None)
            n_layers = len(pad_profiles[sample_pid]) if sample_pid else 0
            logging.info(f"[PAD DEBUG]   built {len(pad_profiles)} pad profiles, "
                         f"~{n_layers} levels each, {len(out)} produced variable sets")
    except Exception as e:
        logging.debug(f"Pad fetch break {model} f{f_hour_int:03d}: {e}")
        if debug:
            logging.warning(f"[PAD DEBUG]   >>> Exception during {model.upper()} f{f_hour_int:03d}: {e}")
    finally:
        if os.path.exists(local_path):
            try: os.remove(local_path)
            except Exception: pass
    return row_key, model, out


def determine_model_cycle(session, model):
    """Find the most recent available cycle for a given model on NOMADS by probing
    directory listings for the last few candidate cycles."""
    now = datetime.datetime.now(datetime.timezone.utc)
    if model == "gfs":
        cycle_hours, latency_h = [0, 6, 12, 18], 5
    else:  # rap, hrrr are hourly
        cycle_hours, latency_h = list(range(24)), 2

    for back in range(0, 30):
        cand = now - datetime.timedelta(hours=back)
        if cand.hour not in cycle_hours:
            continue
        if (now - cand).total_seconds() / 3600.0 < latency_h:
            continue
        date_str = cand.strftime("%Y%m%d")
        cycle = f"{cand.hour:02d}"
        # Probe one representative file
        probe_url = _nomads_grib_url(model, date_str, cycle, 1)
        try:
            resp = session.head(probe_url, timeout=8)
            if resp.status_code == 200:
                return date_str, cycle
            resp = session.get(probe_url, timeout=8, stream=True)
            if resp.status_code == 200:
                resp.close()
                return date_str, cycle
        except Exception:
            continue
    return None, None


def fetch_all_pad_soundings():
    """Build the pad sounding matrix {pad_id: {model: {row_key: variables}}} from NOMADS.
    GFS and RAP are pulled here via the NOMADS grib-filter; HRRR is intentionally skipped
    (its NOMADS filter probe was unreliable) and instead sourced from AWS in the RRFS pass."""
    pad_matrix = {pid: {m: {} for m in MODELS} for pid in LAUNCH_PADS}
    nomads_models = [m for m in BUFKIT_MODELS if m != "hrrr"]  # gfs, rap (HRRR + ECMWF fetched elsewhere)

    with requests.Session() as session:
        for model in nomads_models:
            date_str, cycle = determine_model_cycle(session, model)
            if not cycle:
                logging.warning(f"No available {model.upper()} cycle found for pad soundings.")
                continue
            cycle_init = datetime.datetime.strptime(f"{date_str}{cycle}", "%Y%m%d%H").replace(tzinfo=datetime.timezone.utc)

            # All three are requested hourly across the 48h window. GFS carries hourly
            # native output through f120 on NOMADS (3-hourly only kicks in after f120),
            # so within 48h we get a full hourly series that matches the BUFKIT airports
            # and avoids sparse every-third-row gaps in the merged table.
            max_fh = 48
            step = 1
            f_hours = list(range(step, max_fh + 1, step))

            logging.info(f"Fetching {model.upper()} pad columns: {date_str} {cycle}z, {len(f_hours)} hours")
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                futures = []
                for idx, fh in enumerate(f_hours):
                    valid_dt = cycle_init + datetime.timedelta(hours=fh)
                    row_key = f"{valid_dt.day:02d}/{valid_dt.hour:02d}"
                    # Emit verbose diagnostics only on the first forecast hour of each
                    # model so the log shows exactly what NOMADS returned without spam.
                    dbg = (idx == 0)
                    futures.append(executor.submit(
                        fetch_pad_model, session, model, date_str, cycle, fh, row_key, dbg
                    ))
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        row_key, mdl, pad_vals = fut.result()
                        for pid, vars_dict in pad_vals.items():
                            pad_matrix[pid][mdl][row_key] = vars_dict
                    except Exception:
                        pass

            # Per-model summary: how many forecast hours produced usable pad data.
            sample_pad = next(iter(LAUNCH_PADS))
            hours_ok = len(pad_matrix[sample_pad].get(model, {}))
            if hours_ok == 0:
                logging.warning(f"[PAD DEBUG] {model.upper()} produced ZERO usable pad-hours — "
                                f"see the [PAD DEBUG] lines above for HTTP status / shortName mismatch.")
            else:
                logging.info(f"{model.upper()} pad soundings: {hours_ok}/{len(f_hours)} forecast hours produced data.")

    return pad_matrix


# ---------------------------------------------------------------------------
# RRFS (deterministic) + REFS (ensemble mean) pad columns via AWS Open Data
# ---------------------------------------------------------------------------

def _parse_grib_idx(idx_text):
    """Parse a GRIB2 .idx sidecar into a list of (msg_num, byte_start, shortName, level).
    Each idx line looks like: '1:0:d=2026070100:REFC:entire atmosphere:...'
    We only need the byte offsets so we can range-request specific messages."""
    entries = []
    lines = [ln for ln in idx_text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        parts = ln.split(":")
        if len(parts) < 5:
            continue
        try:
            msg_num = int(parts[0])
            byte_start = int(parts[1])
        except ValueError:
            continue
        short = parts[3].strip()
        level = parts[4].strip()
        # Byte end = start of next message - 1 (or EOF for the last message)
        byte_end = None
        if i + 1 < len(lines):
            nxt = lines[i + 1].split(":")
            try:
                byte_end = int(nxt[1]) - 1
            except (ValueError, IndexError):
                byte_end = None
        entries.append({"msg": msg_num, "start": byte_start, "end": byte_end,
                        "short": short, "level": level})
    return entries


def _range_download_grib(session, grib_url, idx_entries, wanted_levels_hpa, debug=False):
    """Given parsed idx entries, byte-range download only the isobaric TMP/RH/HGT/UGRD/VGRD
    messages at the wanted levels and concatenate them into a local temp GRIB2 file."""
    # Match idx level strings like "500 mb" and variable names.
    wanted_vars = ("TMP", "RH", "HGT", "UGRD", "VGRD")
    wanted_level_strs = {f"{lv} mb" for lv in wanted_levels_hpa}

    ranges = []
    for e in idx_entries:
        if e["short"] not in wanted_vars:
            continue
        if e["level"] not in wanted_level_strs:
            continue
        if e["end"] is None:
            ranges.append((e["start"], ""))  # open-ended to EOF
        else:
            ranges.append((e["start"], e["end"]))

    if not ranges:
        if debug:
            logging.warning("[RRFS DEBUG]   idx parsed but no matching isobaric TMP/RH/HGT/U/V "
                            "messages at wanted levels — check idx var/level naming.")
        return None

    local_path = os.path.join(CACHE_DIR, f"rrfs_col_{abs(hash(grib_url)) % 10_000_000}.grib2")
    try:
        with open(local_path, "wb") as fh:
            # Group into a single multi-range request where possible; fall back to per-range.
            for start, end in ranges:
                hdr = {"Range": f"bytes={start}-{end}"}
                r = session.get(grib_url, headers=hdr, timeout=25)
                if r.status_code in (200, 206):
                    fh.write(r.content)
        if os.path.getsize(local_path) == 0:
            os.remove(local_path)
            return None
        return local_path
    except Exception as e:
        if debug:
            logging.warning(f"[RRFS DEBUG]   range download failed: {e}")
        if os.path.exists(local_path):
            try: os.remove(local_path)
            except Exception: pass
        return None


def _rrfs_determine_cycle(session, model_kind):
    """Find the most recent RRFS/REFS cycle available on AWS by probing .idx existence.
    model_kind: 'rrfs' (deterministic) or 'refs' (ensemble). Returns (date_str, cycle).
    For REFS, also resolves and caches which filename pattern actually carries isobaric
    temperature data (guards against picking a precip-only product like 'avrg')."""
    global _REFS_RESOLVED_PATTERN
    now = datetime.datetime.now(datetime.timezone.utc)

    # HRRR only reaches f48 on the 00/06/12/18z extended cycles; restrict to those so we
    # never pick an odd-hour cycle that stops at f18.
    if model_kind == "hrrr":
        cycle_hours, latency_h = HRRR_EXTENDED_CYCLES, HRRR_LATENCY_H
    else:
        cycle_hours, latency_h = RRFS_CYCLE_HOURS, RRFS_LATENCY_H

    for back in range(0, 36):
        cand = now - datetime.timedelta(hours=back)
        if cand.hour not in cycle_hours:
            continue
        if (now - cand).total_seconds() / 3600.0 < latency_h:
            continue
        date_str = cand.strftime("%Y%m%d")
        cycle = f"{cand.hour:02d}"

        if model_kind == "refs":
            # Try each candidate filename; accept the first whose idx contains isobaric TMP.
            # Probe several forecast hours (some ensemble products don't emit f01), so we
            # don't reject a valid pattern just because its earliest hour is missing.
            base = f"{RRFS_AWS_ROOT}/rrfs_a/refs.{date_str}/{cycle}/enspost"
            resolved = False
            for pat in REFS_FILENAME_CANDIDATES:
                for probe_fh in (1, 6, 8, 12):
                    fn = pat.format(c=cycle, f2=f"{probe_fh:02d}", f3=f"{probe_fh:03d}")
                    probe = f"{base}/{fn}.idx"
                    try:
                        r = session.get(probe, timeout=10)
                        if r.status_code == 200 and "TMP" in r.text and "mb" in r.text:
                            _REFS_RESOLVED_PATTERN = pat
                            logging.info(f"[RRFS DEBUG] REFS resolved filename pattern: {pat} "
                                         f"(confirmed at f{probe_fh:02d})")
                            resolved = True
                            break
                    except Exception:
                        continue
                if resolved:
                    return date_str, cycle
            continue  # this cycle had no working REFS mean file; try older cycle
        else:
            probe = _rrfs_grib_url(model_kind, date_str, cycle, 1) + ".idx"
            try:
                r = session.get(probe, timeout=10)
                if r.status_code == 200 and len(r.text) > 50:
                    return date_str, cycle
            except Exception:
                continue
    return None, None


def _rrfs_grib_url(model_kind, date_str, cycle, f_hour_int):
    """Build the AWS S3 URL for an RRFS deterministic, REFS ensemble-mean, or HRRR
    pressure-level file. For REFS, uses the module-cached resolved filename pattern."""
    if model_kind == "refs":
        pat = _REFS_RESOLVED_PATTERN or REFS_FILENAME_CANDIDATES[0]
        f_name = pat.format(c=cycle, f2=f"{f_hour_int:02d}", f3=f"{f_hour_int:03d}")
        return f"{RRFS_AWS_ROOT}/rrfs_a/refs.{date_str}/{cycle}/enspost/{f_name}"
    elif model_kind == "hrrr":
        f_name = f"hrrr.t{cycle}z.wrfprsf{f_hour_int:02d}.grib2"
        return f"{HRRR_AWS_ROOT}/hrrr.{date_str}/conus/{f_name}"
    else:
        f_name = f"rrfs.t{cycle}z.prslev.3km.f{f_hour_int:03d}.conus.grib2"
        return f"{RRFS_AWS_ROOT}/rrfs_public/rrfs.{date_str}/{cycle}/{f_name}"


def fetch_rrfs_pad_hour(session, model_kind, date_str, cycle, f_hour_int, row_key, all_coords, debug=False):
    """Fetch one RRFS/REFS forecast hour from AWS via idx byte-range, extract columns at
    every site in all_coords (pads + airports), and compute variables.
    Returns (row_key, model_kind, {site_id: variables})."""
    grib_url = _rrfs_grib_url(model_kind, date_str, cycle, f_hour_int)
    idx_url = grib_url + ".idx"
    out = {}
    local_path = None
    try:
        idx_resp = session.get(idx_url, timeout=15)
        if debug:
            logging.info(f"[RRFS DEBUG] {model_kind.upper()} f{f_hour_int:03d} idx HTTP {idx_resp.status_code}")
            logging.info(f"[RRFS DEBUG]   idx URL: {idx_url}")
        if idx_resp.status_code != 200:
            if debug:
                logging.warning(f"[RRFS DEBUG]   >>> idx not found. Check {model_kind.upper()} "
                                f"AWS path/filename. GRIB URL was: {grib_url}")
            return row_key, model_kind, out

        idx_entries = _parse_grib_idx(idx_resp.text)
        if debug:
            uniq_vars = sorted({e["short"] for e in idx_entries})
            logging.info(f"[RRFS DEBUG]   idx has {len(idx_entries)} messages; distinct vars: {uniq_vars[:25]}")

        local_path = _range_download_grib(session, grib_url, idx_entries, PAD_LEVELS_HPA, debug=debug)
        if not local_path:
            return row_key, model_kind, out

        if debug:
            sz = os.path.getsize(local_path)
            logging.info(f"[RRFS DEBUG]   range-downloaded {sz} bytes of isobaric fields")

        site_profiles = build_pad_profiles_from_grib(local_path, all_coords, debug=debug)
        for sid, layers in site_profiles.items():
            result = compute_profile_variables(layers)
            if result is not None:
                out[sid] = result
    except Exception as e:
        logging.debug(f"RRFS fetch break {model_kind} f{f_hour_int:03d}: {e}")
        if debug:
            logging.warning(f"[RRFS DEBUG]   >>> exception: {e}")
    finally:
        if local_path and os.path.exists(local_path):
            try: os.remove(local_path)
            except Exception: pass
    return row_key, model_kind, out


def fetch_all_rrfs_refs_soundings(include_hrrr=True):
    """Build {site_id: {'rrfs'|'refs'|'hrrr': {row_key: variables}}} from AWS, for BOTH the
    launch pads and the BUFKIT airport points (airports have no BUFKIT RRFS/REFS profiles).
    HRRR is pulled here too (via the same idx byte-range path) because the NOMADS grib-filter
    probe for HRRR was unreliable; its results replace the failed NOMADS HRRR pad column."""
    kinds = []
    if RRFS_ENABLED: kinds.append("rrfs")
    if REFS_ENABLED: kinds.append("refs")
    if include_hrrr: kinds.append("hrrr")

    # Combined site set: launch pads + the 5 airport stations, all point-extracted.
    all_coords = {}
    for pid, c in LAUNCH_PADS.items():
        all_coords[pid] = {"lat": c["lat"], "lon": c["lon"]}
    for sid, c in STN_COORDS.items():
        all_coords[sid] = {"lat": c["lat"], "lon": c["lon"]}

    if not kinds:
        return {sid: {} for sid in all_coords}

    matrix = {sid: {k: {} for k in kinds} for sid in all_coords}

    with requests.Session() as session:
        for kind in kinds:
            date_str, cycle = _rrfs_determine_cycle(session, kind)
            if not cycle:
                logging.warning(f"No available {kind.upper()} cycle found on AWS.")
                continue
            cycle_init = datetime.datetime.strptime(f"{date_str}{cycle}", "%Y%m%d%H").replace(tzinfo=datetime.timezone.utc)
            # RRFS/REFS/HRRR all provide hourly forecast output; request the full window and let
            # any missing hour 404 on its .idx probe (so exact availability is never hardcoded).
            # HRRR only reaches f48 even on extended cycles, so don't chase f49-60 for it.
            kind_max_fh = 48 if kind == "hrrr" else RRFS_MAX_FH
            f_hours = list(range(1, kind_max_fh + 1))

            logging.info(f"Fetching {kind.upper()} columns from AWS ({len(all_coords)} sites): {date_str} {cycle}z, {len(f_hours)} hours")
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for idx, fh in enumerate(f_hours):
                    valid_dt = cycle_init + datetime.timedelta(hours=fh)
                    row_key = f"{valid_dt.day:02d}/{valid_dt.hour:02d}"
                    dbg = (idx == 0)  # verbose only on first hour
                    futures.append(executor.submit(
                        fetch_rrfs_pad_hour, session, kind, date_str, cycle, fh, row_key, all_coords, dbg
                    ))
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        row_key, mk, site_vals = fut.result()
                        for sid, vd in site_vals.items():
                            matrix[sid][mk][row_key] = vd
                    except Exception:
                        pass

            sample = next(iter(all_coords))
            hours_ok = len(matrix[sample].get(kind, {}))
            if hours_ok == 0:
                logging.warning(f"[RRFS DEBUG] {kind.upper()} produced ZERO usable site-hours — "
                                f"see [RRFS DEBUG] lines above for the idx/URL mismatch.")
            else:
                logging.info(f"{kind.upper()} soundings: {hours_ok}/{len(f_hours)} forecast hours produced data.")

    return matrix


def fetch_all_ecmwf_soundings():
    """Add an ECMWF IFS (HRES 0.25°) column, point-extracted for every launch pad + airport,
    from the free ECMWF Open Data distribution (CC-BY-4.0). One multi-step GRIB2 file is
    pulled via the ecmwf-opendata client (byte-range subset of pressure-level t/gh/r/u/v),
    then grouped by valid time and run through the shared compute_profile_variables().

    Vertical resolution is coarse (12 tropospheric levels) vs the CAMs, so isotherm heights,
    PBL winds and shear are solid while the moisture-based LLCC fields (ceiling, cloud top,
    layer thickness) are advisory. Returns {site_id: {row_key: variables}} (empty on any
    failure — the column simply won't appear, the rest of the dashboard is unaffected)."""
    if not ECMWF_ENABLED:
        return {}
    try:
        from ecmwf.opendata import Client
    except Exception as e:
        logging.warning(f"ecmwf-opendata not installed; skipping ECMWF column ({e}).")
        return {}

    all_coords = {}
    for pid, c in LAUNCH_PADS.items():
        all_coords[pid] = {"lat": c["lat"], "lon": c["lon"]}
    for sid, c in STN_COORDS.items():
        all_coords[sid] = {"lat": c["lat"], "lon": c["lon"]}

    steps = list(range(0, ECMWF_MAX_FH + 1, 3))  # IFS open-data cadence is 3-hourly
    target = os.path.join(CACHE_DIR, "ecmwf_ifs_pl.grib2")
    if os.path.exists(target):
        try: os.remove(target)
        except Exception: pass

    try:
        client = Client(source=ECMWF_SOURCE)
        result = client.retrieve(
            type="fc",
            step=steps,
            levtype="pl",
            levelist=ECMWF_LEVELS_HPA,
            param=["t", "gh", "r", "u", "v"],
            target=target,
        )
        init_dt = getattr(result, "datetime", None)
        size_kib = os.path.getsize(target) // 1024 if os.path.exists(target) else 0
        logging.info(f"ECMWF IFS: retrieved {size_kib} KiB, init {init_dt}, {len(steps)} steps.")
    except Exception as e:
        logging.error(f"ECMWF Open Data retrieve failed, skipping column: {e}")
        if os.path.exists(target):
            try: os.remove(target)
            except Exception: pass
        return {}

    # Parse the multi-step file: group messages by VALID time -> row_key, per site per level.
    per = {}                                   # row_key -> sid -> {level: {field: val}}
    seen = {}                                  # (shortName, typeOfLevel) -> count  [debug]
    matched = {"t": 0, "rh": 0, "hgt": 0, "u": 0, "v": 0}
    decode_errors = 0
    try:
        grbs = pygrib.open(target)
        grid_lats = grid_lons = None
        site_ij = {}
        for grb in grbs:
            try:
                type_lvl = getattr(grb, "typeOfLevel", "")
                short = getattr(grb, "shortName", "")
            except Exception:
                continue
            seen[(short, type_lvl)] = seen.get((short, type_lvl), 0) + 1
            if type_lvl != "isobaricInhPa":
                continue
            level = grb.level
            if level not in ECMWF_LEVELS_HPA:
                continue
            field = None
            if short in ("t", "TMP"): field = "t"
            elif short in ("r", "RH"): field = "rh"
            elif short in ("gh", "HGT"): field = "hgt"
            elif short in ("u", "UGRD"): field = "u"
            elif short in ("v", "VGRD"): field = "v"
            if field is None:
                continue
            try:
                vd = grb.validDate  # datetime of the valid time
            except Exception:
                continue
            row_key = f"{vd.day:02d}/{vd.hour:02d}"
            if grid_lats is None:
                grid_lats, grid_lons = grb.latlons()
                glons = np.where(grid_lons > 180, grid_lons - 360.0, grid_lons)
                for sid, c in all_coords.items():
                    dist = (grid_lats - c["lat"]) ** 2 + (glons - c["lon"]) ** 2
                    site_ij[sid] = np.unravel_index(np.argmin(dist), dist.shape)
            try:
                vals = grb.values
            except Exception as e:
                # CCSDS decode failure surfaces here if eccodes lacks aec/libaec support.
                decode_errors += 1
                if decode_errors <= 3:
                    logging.error(f"ECMWF GRIB value decode failed ({short}@{level}): {e}")
                continue
            matched[field] += 1
            for sid, (iy, ix) in site_ij.items():
                per.setdefault(row_key, {}).setdefault(sid, {}).setdefault(level, {})[field] = float(vals[iy, ix])
        grbs.close()
    except Exception as e:
        logging.error(f"ECMWF GRIB parse failed: {e}")
        return {}
    finally:
        if os.path.exists(target):
            try: os.remove(target)
            except Exception: pass

    logging.info("[ECMWF DEBUG] shortName/typeOfLevel seen: "
                 + ", ".join(f"{k[0]}/{k[1]}={v}" for k, v in sorted(seen.items())))
    logging.info(f"[ECMWF DEBUG] fields matched to parser: {matched}"
                 + (f" | {decode_errors} value-decode errors" if decode_errors else ""))
    if sum(matched.values()) == 0:
        logging.warning("[ECMWF DEBUG] ZERO isobaric fields matched. If shortNames above look right "
                        "but decode errors are nonzero, eccodes likely lacks CCSDS/aec (libaec) support.")
        return {}

    # Build profiles + run the shared variable computation (same engine as pads/BUFKIT).
    matrix = {sid: {} for sid in all_coords}
    for row_key, sites in per.items():
        for sid, levels in sites.items():
            layers = _grib_levels_to_layers(levels)
            vars_dict = compute_profile_variables(layers) if layers else None
            if vars_dict:
                # ECMWF's coarse 12-level grid + upper-level humidity reported relative to ICE
                # make the moisture-based cloud detection unreliable — it can false-flag ~46 kft
                # "cloud tops" from near-tropopause ice-saturation, and can't resolve low decks
                # (no levels between 1000 and 925 hPa). Blank those fields so the column shows
                # "-" instead of misleading values; isotherms, PBL winds and shear stay valid.
                for mk in ("ceiling", "cloud_top", "cloud_thick", "thick_layer", "thick_layer_ft"):
                    vars_dict[mk] = None
                matrix[sid][row_key] = vars_dict

    n = sum(len(v) for v in matrix.values())
    logging.info(f"ECMWF IFS soundings: {n} site-hours across {len(all_coords)} sites, {len(per)} valid times.")
    return matrix

# ---------------------------------------------------------------------------------------------
# Launch-thermo climatology (KXMR), assessed at 10Z. Both PWAT and Thompson are now full 15-point
# monthly distributions from the XMR period of record; _climo_percentile interpolates the value's
# rank within them. Percentile ranks (fractions -> 0-100):
CLIMO_PCTL_POINTS_15 = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]

# PWAT: monthly percentile distribution (inches) for XMR / Cape Kennedy.
PWAT_PCTL_POINTS = CLIMO_PCTL_POINTS_15
PWAT_CLIMO_XMR = {
     1: [0.093, 0.171, 0.27, 0.338, 0.466, 0.577, 0.699, 0.808, 0.948, 1.083, 1.229, 1.432, 1.582, 1.757, 2.834],  # Jan
     2: [0.091, 0.19, 0.297, 0.375, 0.53, 0.676, 0.782, 0.907, 1.032, 1.154, 1.275, 1.401, 1.539, 1.726, 2.026],  # Feb
     3: [0.096, 0.217, 0.324, 0.421, 0.566, 0.676, 0.791, 0.889, 0.992, 1.118, 1.236, 1.412, 1.551, 1.779, 1.962],  # Mar
     4: [0.131, 0.252, 0.431, 0.518, 0.679, 0.779, 0.892, 1.003, 1.098, 1.238, 1.36, 1.552, 1.681, 1.852, 2.236],  # Apr
     5: [0.179, 0.422, 0.671, 0.809, 0.961, 1.097, 1.192, 1.293, 1.39, 1.502, 1.603, 1.774, 1.904, 2.143, 2.715],  # May
     6: [0.0, 0.768, 1.096, 1.287, 1.471, 1.585, 1.66, 1.748, 1.836, 1.922, 2.004, 2.13, 2.207, 2.435, 2.669],  # Jun
     7: [0.142, 1.016, 1.307, 1.444, 1.637, 1.716, 1.806, 1.895, 1.957, 2.025, 2.093, 2.191, 2.264, 2.447, 2.987],  # Jul
     8: [0.011, 0.751, 1.225, 1.456, 1.677, 1.787, 1.875, 1.943, 2.0, 2.067, 2.158, 2.249, 2.317, 2.491, 2.762],  # Aug
     9: [0.038, 0.743, 1.137, 1.257, 1.472, 1.635, 1.761, 1.844, 1.926, 2.022, 2.113, 2.218, 2.314, 2.478, 2.901],  # Sep
    10: [0.277, 0.399, 0.638, 0.773, 0.975, 1.11, 1.262, 1.403, 1.575, 1.729, 1.906, 2.075, 2.264, 2.475, 2.787],  # Oct
    11: [0.065, 0.325, 0.471, 0.577, 0.732, 0.861, 0.965, 1.093, 1.204, 1.337, 1.485, 1.68, 1.827, 2.075, 2.352],  # Nov
    12: [0.132, 0.217, 0.307, 0.394, 0.577, 0.739, 0.847, 0.967, 1.094, 1.211, 1.362, 1.523, 1.689, 1.917, 2.873],  # Dec
}

# Thompson Index (K − LI): monthly percentile distribution for XMR (coworker-supplied climatology).
THOMPSON_PCTL_POINTS = CLIMO_PCTL_POINTS_15
THOMPSON_CLIMO_XMR = {
     1: [-104.6, -74.3, -55.2, -48.2, -34.7, -25.6, -17.9, -9.0, -1.9, 5.0, 16.6, 25.3, 30.5, 36.8, 40.9],  # Jan
     2: [-88.3, -69.4, -53.5, -41.8, -27.6, -19.0, -11.4, -3.1, 3.9, 11.1, 19.9, 27.2, 32.0, 37.4, 42.2],  # Feb
     3: [-73.6, -61.5, -47.6, -37.1, -22.9, -15.8, -8.4, -2.0, 4.3, 10.8, 19.8, 29.1, 34.8, 39.2, 45.9],  # Mar
     4: [-60.8, -51.1, -34.9, -26.5, -15.9, -8.0, -2.0, 3.4, 10.7, 17.9, 25.6, 32.1, 36.2, 41.3, 45.1],  # Apr
     5: [-50.3, -31.4, -17.1, -8.8, 0.6, 7.1, 12.8, 18.7, 23.8, 28.4, 32.3, 35.9, 38.7, 42.4, 47.3],  # May
     6: [-27.3, -6.3, 7.6, 14.9, 23.0, 27.2, 30.3, 32.2, 33.8, 35.8, 37.5, 39.5, 41.4, 44.3, 48.6],  # Jun
     7: [-15.4, 6.2, 16.3, 20.5, 26.7, 29.6, 31.7, 33.2, 34.8, 36.1, 37.6, 39.7, 41.6, 44.6, 51.6],  # Jul
     8: [-15.5, 0.2, 14.0, 21.6, 27.6, 30.4, 32.2, 33.9, 35.0, 36.6, 38.4, 40.4, 41.7, 44.9, 51.2],  # Aug
     9: [-32.3, -13.8, 2.4, 9.2, 21.8, 27.2, 30.4, 32.4, 34.6, 36.2, 37.8, 39.6, 41.2, 44.1, 46.7],  # Sep
    10: [-68.6, -50.5, -29.3, -18.7, -7.9, 1.0, 9.4, 17.0, 23.4, 30.2, 34.3, 38.4, 40.0, 44.9, 54.2],  # Oct
    11: [-73.9, -55.5, -40.4, -31.5, -19.5, -11.3, -4.3, 0.9, 8.2, 16.4, 24.2, 31.3, 35.1, 41.3, 45.4],  # Nov
    12: [-87.2, -67.5, -52.9, -42.0, -28.4, -17.5, -10.3, -3.4, 3.2, 11.2, 18.8, 27.7, 31.4, 37.4, 45.1],  # Dec
}

# When a model has no sounding valid exactly at the assessment hour (10Z) on a given day — common
# for short-range RAP/HRRR depending on cycle timing — accept the nearest hour within this many
# hours instead of dropping the day. Exact 10Z always wins when present.
ASSESS_HOUR_TOL = 2

# How many panel snapshots to retain for its DPROG/DT. The cron is hourly but the models are not:
# GFS/ECMWF/RRFS/REFS cycle every 6 h, so consecutive hourly snapshots of those columns are
# IDENTICAL and stepping run-by-run tells you nothing. The frontend therefore steps by TIME
# (-6 h / -12 h for 6-hourly models, -1 h for hourly RAP/HRRR) and picks the nearest stored
# snapshot, so this needs ~19 h of depth (3 x 6 h back) plus margin for missed runs.
LAUNCH_THERMO_HISTORY_RUNS = 21

# 700-500 mb mean RH: monthly percentile distribution (%) for XMR.
RH75_PCTL_POINTS = CLIMO_PCTL_POINTS_15
RH75_CLIMO_XMR = {
     1: [0.9, 2.1, 5.5, 7.4, 8.3, 11.2, 15.8, 20.9, 25.7, 33.5, 46.1, 69.7, 80.4, 94.3, 99.6],  # Jan
     2: [1.1, 2.4, 6.1, 7.5, 9.2, 13.3, 18.3, 23.5, 29.4, 37.4, 49.9, 66.5, 81.4, 95.3, 98.9],  # Feb
     3: [1.6, 2.7, 6.0, 7.5, 9.1, 13.5, 17.3, 22.0, 27.5, 33.7, 43.7, 62.6, 76.0, 92.2, 98.8],  # Mar
     4: [1.8, 3.0, 6.0, 7.7, 10.1, 13.4, 17.3, 23.7, 28.8, 35.5, 44.5, 60.7, 73.3, 93.8, 99.7],  # Apr
     5: [1.8, 3.5, 7.8, 8.5, 13.2, 17.6, 23.7, 30.5, 35.7, 43.2, 52.9, 68.6, 79.8, 92.6, 97.0],  # May
     6: [4.1, 8.3, 13.8, 21.0, 29.8, 38.6, 45.9, 53.7, 60.5, 68.0, 75.3, 83.9, 89.7, 94.8, 97.4],  # Jun
     7: [8.8, 11.4, 19.1, 26.8, 35.9, 43.3, 50.9, 57.4, 63.3, 69.2, 76.3, 82.2, 87.4, 94.1, 97.6],  # Jul
     8: [3.7, 9.0, 19.3, 26.2, 37.5, 45.5, 51.6, 57.4, 63.4, 68.1, 73.6, 81.3, 86.3, 94.2, 100.0],  # Aug
     9: [3.9, 6.2, 10.8, 15.3, 24.5, 34.5, 43.3, 52.0, 59.9, 66.6, 73.8, 82.9, 88.1, 95.1, 99.6],  # Sep
    10: [1.7, 3.0, 6.3, 8.3, 10.8, 14.4, 19.6, 26.4, 33.3, 44.0, 57.2, 75.3, 87.7, 94.5, 99.1],  # Oct
    11: [1.4, 2.5, 5.7, 7.8, 8.9, 11.3, 15.0, 19.9, 24.9, 31.1, 41.4, 64.6, 76.5, 92.9, 99.1],  # Nov
    12: [1.2, 3.2, 5.4, 7.5, 8.7, 11.7, 15.2, 19.3, 25.7, 32.0, 43.0, 60.5, 77.6, 94.3, 98.6],  # Dec
}


def _climo_percentile(value, breaks, points):
    """Interpolated percentile rank of `value` within monthly breakpoint values at `points`."""
    if value is None or not breaks or not points or len(breaks) != len(points):
        return None
    if value <= breaks[0]:
        return points[0]
    if value >= breaks[-1]:
        return points[-1]
    for i in range(len(breaks) - 1):
        if breaks[i] <= value <= breaks[i + 1]:
            if breaks[i + 1] == breaks[i]:
                return points[i]
            f = (value - breaks[i]) / (breaks[i + 1] - breaks[i])
            return round(points[i] + f * (points[i + 1] - points[i]))
    return None


def _valid_day_fields(dd, now):
    """From a forecast day-of-month and 'now', reconstruct (weekday, 'Mon DD', sort key, month, year),
    wrapping into next month when the day has already passed this month."""
    month, year = now.month, now.year
    if dd < now.day - 5:
        month += 1
        if month > 12:
            month, year = 1, year + 1
    try:
        vdate = datetime.date(year, month, dd)
        return (vdate.strftime("%A"), vdate.strftime("%b %d"),
                f"{year:04d}{month:02d}{dd:02d}", month, year)
    except Exception:
        return (f"{dd:02d}", f"{dd:02d}", f"{dd:02d}", now.month, now.year)


def fetch_gefs_member_thermo(site="kxmr", assess_hour=10, cache=None):
    """GEFS ensemble launch-thermo for the 10Z panel: pull each member's isobaric sounding at the
    site for the forecast hours nearest the assessment hour, compute the indices PER MEMBER, and
    average the results via _ensemble_thermo_row.

    Returns (rows, cycle_key). `cache` may be a previous {"cycle":..., "rows":...}; if the newest
    posted cycle matches it the cached rows are returned untouched, which skips the whole fetch on
    the ~5 of every 6 hourly runs where GEFS has not advanced."""
    if not GEFS_ENABLED:
        return {}, None
    sc = STN_COORDS.get(site)
    if not sc:
        return {}, None
    coords = {site: {"lat": sc["lat"], "lon": sc["lon"]}}

    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=3))
    session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"})

    members = ["c00"] + [f"p{i:02d}" for i in range(1, max(1, GEFS_MEMBERS))]

    # SOURCE: AWS S3, not the NOMADS grib_filter CGI. The filter would be far cheaper in bytes
    # (server-side cropping), but by the time this runs the pipeline has already made hundreds of
    # NOMADS requests for the GFS/RAP pad columns, HREF and HREFCT - so the filter answers every
    # GEFS probe with "302 Over Rate Limit" and the column silently vanishes. S3 has no such limit.
    #
    # The tradeoff is bandwidth: a 0.5-deg message is global (~200 KB), so byte-ranging pulls whole
    # fields to read one point. Trimmed to 4 vars x 11 levels and cached against the 6-hourly cycle
    # that is ~0.9 GB per cycle-change, i.e. ~150 MB/run amortized - roughly a tenth of what the
    # hourly ECMWF fetch already costs.
    def _url(d, cc, mem, fh, ab):
        sub = "pgrb2ap5" if ab == "a" else "pgrb2bp5"
        return (f"{GEFS_AWS_ROOT}/gefs.{d}/{cc}/atmos/{sub}/"
                f"ge{mem}.t{cc}z.pgrb2{ab}.0p50.f{fh:03d}")

    _lvl_re = re.compile(r"^(\d+)\s*mb$")

    def _wanted(entries):
        """Pick the TMP/RH/UGRD/VGRD messages at GEFS_LEVELS_HPA out of a parsed .idx."""
        want = []
        for e in entries:
            if e["short"] not in GEFS_VARS:
                continue
            m = _lvl_re.match((e.get("level") or "").strip())
            if m and int(m.group(1)) in GEFS_LEVELS_HPA:
                want.append(e)
        return want

    def _merge(entries, gap=4096):
        """Merge byte ranges that are adjacent or nearly so. GRIB messages for one variable sit
        contiguously in the file, so this collapses ~44 requests into a handful without pulling
        materially more data."""
        rngs = sorted(((e["start"], e["end"]) for e in entries), key=lambda x: x[0])
        out = []
        for s, e in rngs:
            if out and out[-1][1] is not None and s - out[-1][1] <= gap:
                out[-1] = (out[-1][0], e if e is not None else None)
            else:
                out.append((s, e))
        return out

    def _idx_ok(d, cc, mem, fh, ab="a"):
        """A cycle exists if its .idx is served (a few KB of text, no data transfer)."""
        try:
            r = session.get(_url(d, cc, mem, fh, ab) + ".idx", timeout=15)
            return r.status_code == 200 and "TMP" in r.text
        except Exception:
            return False

    # newest posted GEFS cycle (00/06/12/18Z), probing back up to 24 h
    now = datetime.datetime.now(datetime.timezone.utc)
    date_str = cycle = None
    cyc_dt = None
    for back in range(0, 25):
        cand = now - datetime.timedelta(hours=back)
        if cand.hour not in (0, 6, 12, 18):
            continue
        d, cc = cand.strftime("%Y%m%d"), f"{cand.hour:02d}"
        if _idx_ok(d, cc, "c00", 3):
            date_str, cycle = d, cc
            cyc_dt = cand.replace(minute=0, second=0, microsecond=0)
            break
    if not cycle:
        logging.warning("GEFS thermo: no cycle .idx found on AWS after probing 24 h back; "
                        "GEFS omitted from panel.")
        return {}, None

    cycle_key = f"{date_str}{cycle}"
    if GEFS_CACHE_ENABLED and isinstance(cache, dict) and cache.get("cycle") == cycle_key and cache.get("rows"):
        logging.info(f"GEFS thermo: cycle {date_str} {cycle}z unchanged - reusing {len(cache['rows'])} cached rows.")
        return cache["rows"], cycle_key

    # GEFS is 3-hourly, so an exact 10Z valid time never exists off a 00/06/12/18Z cycle. Take the
    # step nearest the assessment hour on each forecast day, within ASSESS_HOUR_TOL.
    best_by_day = {}
    for fh in range(3, GEFS_MAX_FH + 1, 3):
        v = cyc_dt + datetime.timedelta(hours=fh)
        diff = abs(v.hour - assess_hour)
        if diff > ASSESS_HOUR_TOL:
            continue
        key = v.strftime("%Y%m%d")
        if key not in best_by_day or diff < best_by_day[key][0]:
            best_by_day[key] = (diff, fh, v)
    picks = sorted(best_by_day.values(), key=lambda x: x[1])
    if not picks:
        return {}, cycle_key

    def _grab(d, cc, mem, fh, ab, out_fh):
        """Byte-range the wanted messages out of one pgrb2a/b file into `out_fh`. Returns bytes
        written. GRIB2 files are concatenated messages, so a and b can share one local file."""
        try:
            ir = session.get(_url(d, cc, mem, fh, ab) + ".idx", timeout=20)
            if ir.status_code != 200:
                return 0
            want = _wanted(_parse_grib_idx(ir.text))
            if not want:
                return 0
            n = 0
            for s, e in _merge(want):
                rng = f"bytes={s}-{'' if e is None else e}"
                rr = session.get(_url(d, cc, mem, fh, ab), headers={"Range": rng}, timeout=60)
                if rr.status_code in (200, 206) and rr.content:
                    out_fh.write(rr.content)
                    n += len(rr.content)
            return n
        except Exception as exc:
            logging.debug(f"GEFS {mem} f{fh:03d} pgrb2{ab}: {exc}")
            return 0

    out = {}
    probed = False
    for (_diff, fh, valid) in picks:
        rk = f"{valid.day:02d}/{valid.hour:02d}"
        per = []
        for mi, mem in enumerate(members):
            if mi:
                time.sleep(GEFS_REQUEST_PAUSE_S)   # brief courtesy pause; S3 has no burst limit
            local = os.path.join(CACHE_DIR, f"gefs_{mem}_{cycle}z_f{fh:03d}.grib2")
            try:
                with open(local, "wb") as fhandle:
                    n_a = _grab(date_str, cycle, mem, fh, "a", fhandle)
                    n_b = _grab(date_str, cycle, mem, fh, "b", fhandle)
                if not probed:
                    # One-time check: pgrb2a carries TMP/RH only at 1000/925/850, so a zero-byte
                    # b file would silently blank Thompson and 700-500 RH for the whole GEFS
                    # column. Log both sizes so that failure mode is visible immediately.
                    logging.info(f"[GEFS PROBE] {mem} f{fh:03d}: pgrb2a {n_a/1024:.0f} KB + "
                                 f"pgrb2b {n_b/1024:.0f} KB via S3 byte-range "
                                 f"(mid/upper TMP+RH come from the b file).")
                    probed = True
                if (n_a + n_b) == 0:
                    continue
                prof = build_pad_profiles_from_grib(local, coords, debug=False).get(site)
                if not prof:
                    continue
                th = compute_launch_thermo(prof)
                if th:
                    per.append(th)
            except Exception as e:
                logging.debug(f"GEFS thermo {mem} f{fh:03d}: {e}")
            finally:
                if os.path.exists(local):
                    try:
                        os.remove(local)
                    except Exception:
                        pass
        if per:
            out[rk] = _ensemble_thermo_row(per)

    got = max((r.get("n", 0) for r in out.values()), default=0)
    logging.info(f"GEFS thermo: cycle {date_str} {cycle}z, {got}/{len(members)} members returned, "
                 f"{len(out)} rows near {assess_hour}Z at {site.upper()} (index-of-member mean).")
    # Don't cache a badly degraded fetch — caching is keyed to the 6-hourly cycle, so a partial
    # result would be frozen in for hours. Returning cycle_key=None forces a retry next run.
    if out and got < max(2, len(members) // 2):
        logging.warning(f"GEFS thermo: only {got}/{len(members)} members returned - not caching, "
                        f"will refetch next run.")
        return out, None
    return out, cycle_key


def _ensemble_thermo_row(per):
    """Collapse a list of per-member compute_launch_thermo() dicts into one ensemble row.

    Scalars (Thompson, PWAT, RH) are averaged directly; winds are averaged in u/v COMPONENT space
    and re-derived to direction/speed so opposing directions don't average to nonsense. The
    lightning RF is run on EACH member's own environment and the probabilities averaged (with the
    member spread kept) — never on a mean sounding, whose moisture structure is smeared.
    Shared by the REFS and GEFS ensemble columns."""
    def _avg(key):
        v = [t[key] for t in per if t.get(key) is not None]
        return sum(v) / len(v) if v else None

    COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    row = {"n": len(per)}
    for u_key, v_key, pfx in (("mf_u", "mf_v", "mf"), ("av_u", "av_v", "av")):
        uu, vv = _avg(u_key), _avg(v_key)
        if uu is None or vv is None:
            continue
        frm = math.degrees(math.atan2(-uu, -vv)) % 360.0
        row[f"{pfx}_dir"] = round(frm)
        row[f"{pfx}_spd"] = round(math.hypot(uu, vv), 1)
        row[f"{pfx}_regime"] = COMPASS[int((frm + 22.5) // 45) % 8]
    ti, pw, rh = _avg("thompson"), _avg("pwat_in"), _avg("rh_700_500")
    if ti is not None:
        row["thompson"] = round(ti, 1)
    if pw is not None:
        row["pwat_in"] = round(pw, 2)
    if rh is not None:
        row["rh_700_500"] = round(rh, 1)
    member_p = []
    for t in per:
        p = rf_lightning_prob(t.get("thompson"),
                              rf_lightning_u_wind(t.get("mf_dir"), t.get("mf_spd")),
                              t.get("rh_700_500"))
        if p is not None:
            member_p.append(p)
    if member_p:
        row["ltg"] = round(sum(member_p) / len(member_p), 1)
        row["ltg_min"] = round(min(member_p), 1)
        row["ltg_max"] = round(max(member_p), 1)
        row["ltg_n"] = len(member_p)
    return row


def fetch_refs_member_thermo(site="kxmr", assess_hour=10):
    """Meteorologically valid REFS launch-thermo: pull EACH RRFS ensemble member's isobaric sounding
    at the site for the forecast hours that land on the assessment hour (10Z), compute the indices
    per member with compute_launch_thermo, and average the RESULTS (TI/PWAT as scalar means; mean
    flow in u/v component space). This is the correct ensemble number — never the index of the mean
    sounding. Returns {row_key: {mf_dir, mf_spd, mf_regime, thompson, pwat_in, n}} (empty on failure,
    so the panel simply omits REFS)."""
    if not REFS_MEMBER_THERMO_ENABLED:
        return {}
    sc = STN_COORDS.get(site)
    if not sc:
        return {}
    coords = {site: {"lat": sc["lat"], "lon": sc["lon"]}}

    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=3))
    session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"})

    tmpl = "rrfs_a/rrfsens.{d}/{cc}/{mem}/rrfs.t{cc}z.{mem}.prslev.3km.f{fh:03d}.conus.grib2"

    def _idx_ok(d, cc, mem, fh):
        try:
            r = session.get(f"{RRFS_AWS_ROOT}/{tmpl.format(d=d, cc=cc, mem=mem, fh=fh)}.idx", timeout=12)
            return r.status_code == 200 and len(r.text) > 50
        except Exception:
            return False

    # newest rrfsens cycle whose m001 prslev reaches at least the first assess-hour valid time
    now = datetime.datetime.now(datetime.timezone.utc)
    date_str = cycle = None
    cyc_dt = None
    for back in range(0, 48):
        cand = now - datetime.timedelta(hours=back)
        if cand.hour not in (0, 6, 12, 18):
            continue
        d, cc = cand.strftime("%Y%m%d"), f"{cand.hour:02d}"
        first_fh = ((assess_hour - cand.hour) % 24) or 24
        if _idx_ok(d, cc, "m001", first_fh):
            date_str, cycle, cyc_dt = d, cc, cand.replace(minute=0, second=0, microsecond=0)
            break
    if not cycle:
        logging.warning("REFS member thermo: no rrfsens prslev cycle found; REFS omitted from panel.")
        return {}

    members = []
    for n in range(1, 21):
        mem = f"m{n:03d}"
        first_fh = ((assess_hour - cyc_dt.hour) % 24) or 24
        if _idx_ok(date_str, cycle, mem, first_fh):
            members.append(mem)
        elif members:
            break
    if not members:
        return {}

    fhs = [fh for fh in range(1, REFS_MEMBER_WINDOW_FH + 1)
           if (cyc_dt + datetime.timedelta(hours=fh)).hour == assess_hour]
    if not fhs:
        return {}

    # --- time-lagged ensemble: fold in the prior cycle's members, valid-time aligned ---------
    # Same principle as the Cumulus echo-top NMEP: pairing this cycle's members with the -6 h
    # cycle's members at the SAME valid time doubles the sample (5 -> 10). For daily airmass
    # indices a 6 h older forecast is a legitimate additional draw on the same airmass, so this is
    # more defensible here than it would be for convective placement.
    sources = [(date_str, cycle, cyc_dt, members)]
    if REFS_MEMBER_TLE:
        lag_dt = cyc_dt - datetime.timedelta(hours=6 * REFS_MEMBER_LAG_CYCLES)
        ld, lcc = lag_dt.strftime("%Y%m%d"), f"{lag_dt.hour:02d}"
        lag_first_fh = int((cyc_dt + datetime.timedelta(hours=fhs[0]) - lag_dt).total_seconds() // 3600)
        if lag_first_fh <= REFS_MEMBER_WINDOW_FH and _idx_ok(ld, lcc, "m001", lag_first_fh):
            lag_members = []
            for n in range(1, 21):
                mem = f"m{n:03d}"
                if _idx_ok(ld, lcc, mem, lag_first_fh):
                    lag_members.append(mem)
                elif lag_members:
                    break
            if lag_members:
                sources.append((ld, lcc, lag_dt, lag_members))

    out = {}
    for fh in fhs:
        valid = cyc_dt + datetime.timedelta(hours=fh)
        rk = f"{valid.day:02d}/{valid.hour:02d}"
        per = []
        for (s_date, s_cycle, s_dt, s_members) in sources:
            s_fh = int((valid - s_dt).total_seconds() // 3600)
            if s_fh < 1 or s_fh > REFS_MEMBER_WINDOW_FH:
                continue
            for mem in s_members:
                grib_url = f"{RRFS_AWS_ROOT}/{tmpl.format(d=s_date, cc=s_cycle, mem=mem, fh=s_fh)}"
                local = None
                try:
                    ir = session.get(grib_url + ".idx", timeout=15)
                    if ir.status_code != 200:
                        continue
                    local = _range_download_grib(session, grib_url, _parse_grib_idx(ir.text),
                                                 PAD_LEVELS_HPA, debug=False)
                    if not local:
                        continue
                    prof = build_pad_profiles_from_grib(local, coords, debug=False).get(site)
                    if not prof:
                        continue
                    th = compute_launch_thermo(prof)
                    if th:
                        per.append(th)
                except Exception as e:
                    logging.debug(f"REFS member thermo {s_cycle}z {mem} f{s_fh:03d}: {e}")
                finally:
                    if local and os.path.exists(local):
                        try:
                            os.remove(local)
                        except Exception:
                            pass
        if not per:
            continue
        out[rk] = _ensemble_thermo_row(per)

    src_txt = " + ".join(f"{s[1]}z({len(s[3])})" for s in sources)
    n_total = sum(len(s[3]) for s in sources)
    logging.info(f"REFS member thermo: cycle {date_str} {cycle}z, {n_total}-member TLE [{src_txt}], "
                 f"{len(out)} x {assess_hour}Z rows at {site.upper()} (index-of-member mean).")
    return out



def _ecmwf_ens_profiles_by_member(filepath, lat, lon, levels=None):
    """Split one multi-member ENS GRIB file into per-member point profiles.

    A single ENS file holds every requested member for a step. build_pad_profiles_from_grib()
    has no notion of members, so it would let each member overwrite the last and return only
    the final one's column. This walks the file grouping by perturbationNumber, then hands
    each member's level dict to the same _grib_levels_to_layers() everything else uses, so
    the profile schema stays identical.

    Returns {member_number: profile_layers}.
    """
    allowed = set(levels if levels is not None else ECMWF_ENS_LEVELS_HPA)
    per_member = {}
    iy = ix = None
    try:
        grbs = pygrib.open(filepath)
        for grb in grbs:
            try:
                if getattr(grb, "typeOfLevel", "") != "isobaricInhPa":
                    continue
                level = grb.level
                # Filter to the levels we actually REQUESTED, not the wider pad set. If the
                # server ever returns a stray upper level, letting it through would hand
                # compute_launch_thermo a single-level "anvil flow" — a plausible-looking
                # number derived from one level, which is worse than the honest em dash.
                if level not in allowed:
                    continue
                short = getattr(grb, "shortName", "")
                mem = getattr(grb, "perturbationNumber", None)
            except Exception:
                continue
            if short in ("t", "TMP"):
                field = "t"
            elif short in ("r", "RH"):
                field = "rh"
            elif short in ("gh", "HGT"):
                field = "hgt"
            elif short in ("u", "UGRD"):
                field = "u"
            elif short in ("v", "VGRD"):
                field = "v"
            else:
                continue
            if mem is None:
                continue
            if iy is None:
                # Nearest grid cell, resolved once from the first usable message.
                glats, glons = grb.latlons()
                gl = np.where(glons > 180, glons - 360.0, glons)
                d = (glats - lat) ** 2 + (gl - lon) ** 2
                iy, ix = np.unravel_index(np.argmin(d), d.shape)
            per_member.setdefault(mem, {}).setdefault(level, {})[field] = float(grb.values[iy, ix])
        grbs.close()
    except Exception as e:
        logging.error(f"ECMWF ENS GRIB parse failed for {os.path.basename(filepath)}: {e}")
        return {}

    out = {}
    for mem, levels in per_member.items():
        layers = _grib_levels_to_layers(levels)
        if layers:
            out[mem] = layers
    return out


def fetch_ecmwf_ens_member_thermo(site="kxmr", assess_hour=10, cache=None):
    """ECMWF ENS column for the 10Z panel: pull each perturbed member's sounding at the site,
    compute the indices PER MEMBER, and average the RESULTS via _ensemble_thermo_row — never
    from an ensemble-mean sounding, whose moisture structure is smeared.

    One retrieve per forecast step covering all members at once (10 requests would pay the
    index lookup ten times over for the same bytes). Returns (rows, cycle_key); `cache` may be
    a previous {"cycle":..., "rows":...} and is returned untouched when the cycle hasn't moved.
    """
    if not ECMWF_ENS_ENABLED:
        return {}, None
    sc = STN_COORDS.get(site)
    if not sc:
        return {}, None
    try:
        from ecmwf.opendata import Client
    except Exception as e:
        logging.warning(f"ecmwf-opendata not installed; skipping ECMWF ENS column ({e}).")
        return {}, None

    client = Client(source=ECMWF_SOURCE)
    members = list(range(1, max(1, ECMWF_ENS_MEMBERS) + 1))

    # Which cycle is newest? latest() probes the index rather than guessing at latency.
    try:
        init_dt = client.latest(stream="enfo", type="pf", levtype="pl",
                                param="t", number=1)
    except Exception as e:
        logging.error(f"ECMWF ENS: could not resolve latest cycle ({e}); skipping column.")
        return {}, None
    if init_dt is None:
        return {}, None
    cycle_key = init_dt.strftime("%Y%m%d%H")

    if ECMWF_ENS_CACHE_ENABLED and isinstance(cache, dict) \
            and cache.get("cycle") == cycle_key and cache.get("rows"):
        logging.info(f"ECMWF ENS: cycle {cycle_key} unchanged, reusing {len(cache['rows'])} cached rows.")
        return cache["rows"], cycle_key

    # ENS is 3-hourly to 144 h. Take the step nearest the assessment hour on each forecast
    # day, within ASSESS_HOUR_TOL — same rule the GEFS column uses.
    best_by_day = {}
    for fh in range(3, ECMWF_ENS_MAX_FH + 1, 3):
        v = init_dt + datetime.timedelta(hours=fh)
        diff = abs(v.hour - assess_hour)
        if diff > ASSESS_HOUR_TOL:
            continue
        key = v.strftime("%Y%m%d")
        if key not in best_by_day or diff < best_by_day[key][0]:
            best_by_day[key] = (diff, fh, v)
    picks = sorted(best_by_day.values(), key=lambda x: x[1])
    if not picks:
        logging.warning("ECMWF ENS: no forecast step landed near the assessment hour.")
        return {}, cycle_key

    out = {}
    t_start = time.time()
    total_mb = 0.0
    for (_diff, fh, valid) in picks:
        rk = f"{valid.day:02d}/{valid.hour:02d}"
        local = os.path.join(CACHE_DIR, f"ecens_{cycle_key}_f{fh:03d}.grib2")
        try:
            client.retrieve(
                stream="enfo", type="pf", number=members, step=fh,
                levtype="pl", levelist=ECMWF_ENS_LEVELS_HPA,
                param=ECMWF_ENS_PARAMS, target=local,
            )
            total_mb += os.path.getsize(local) / 1e6
            profiles = _ecmwf_ens_profiles_by_member(local, sc["lat"], sc["lon"])
            per = []
            for _mem, layers in sorted(profiles.items()):
                th = compute_launch_thermo(layers)
                if th:
                    per.append(th)
            if per:
                out[rk] = _ensemble_thermo_row(per)
        except Exception as e:
            logging.warning(f"ECMWF ENS f{fh:03d} ({rk}): {type(e).__name__}: {e}")
        finally:
            if os.path.exists(local):
                try:
                    os.remove(local)
                except Exception:
                    pass

    got = max((r.get("n", 0) for r in out.values()), default=0)
    logging.info(f"ECMWF ENS thermo: cycle {cycle_key}, {got}/{len(members)} members, "
                 f"{len(out)} rows near {assess_hour}Z at {site.upper()} "
                 f"({total_mb:.0f} MB in {time.time() - t_start:.0f}s, index-of-member mean).")

    # Never cache a badly degraded fetch — the cache is keyed to a 6-hourly cycle, so a
    # partial result would be frozen in for hours. cycle_key=None forces a retry next run.
    if out and got < max(2, len(members) // 2):
        logging.warning(f"ECMWF ENS: only {got}/{len(members)} members returned — not caching.")
        return out, None
    return out, cycle_key

def build_launch_thermo(combined_data, site="kxmr", assess_hour=10, refs_member_rows=None,
                        gefs_member_rows=None, ecens_member_rows=None):
    """Assemble the launch-thermo panel: for each model that has a KXMR sounding, one row per
    forecast day at the assessment hour (10Z), with mean flow, regime, Thompson Index (+percentile),
    and PWAT (+percentile). Returns {"site","hour","models":[...],"by_model":{model:[rows]}}."""
    now = datetime.datetime.now(datetime.timezone.utc)
    site_models = combined_data.get(site, {}) or {}
    by_model = {}
    for model, rows in site_models.items():
        if not isinstance(rows, dict):
            continue
        # Skip the REFS ensemble-MEAN sounding: airmass indices (moisture-driven PWAT, and KI/LI/
        # TI) are not meaningful from a mean sounding — averaging RH across members destroys the
        # moisture structure (PWAT collapses, KI goes dry). RRFS deterministic is the hi-res stand-in.
        if model == "refs":
            continue
        day_rows = []
        # Gather candidate profiles per forecast day within ASSESS_HOUR_TOL of the assessment hour,
        # then keep the one nearest to it (exact 10Z wins). Short-range models (RAP/HRRR) frequently
        # skip exactly 10Z depending on cycle timing, so an exact-only match dropped them entirely.
        day_cands = {}
        for row_key, prof in rows.items():
            if not isinstance(prof, dict):
                continue
            try:
                dd, hh = map(int, row_key.split("/"))
            except Exception:
                continue
            diff = abs(hh - assess_hour)
            if diff > ASSESS_HOUR_TOL or not prof.get("_layers"):
                continue
            day_cands.setdefault(dd, []).append((diff, hh, prof))
        for dd, cands in day_cands.items():
            cands.sort(key=lambda c: (c[0], c[1]))
            for diff, hh, prof in cands:
                th = compute_launch_thermo(prof["_layers"])  # MetPy (mixed-layer LI), on demand
                if not th:
                    continue
                day_label, date_str, sort_key, month, _yr = _valid_day_fields(dd, now)
                ti = th.get("thompson")
                pwat = th.get("pwat_in")
                u_ltg = rf_lightning_u_wind(th.get("mf_dir"), th.get("mf_spd"))
                rh_ltg = th.get("rh_700_500")
                ltg = rf_lightning_prob(ti, u_ltg, rh_ltg)
                day_rows.append({
                    "day": day_label,
                    "date": date_str,
                    "sort": sort_key,
                    "vhh": hh,
                    "month": month,
                    "ltg": ltg,
                    "ltg_u": None if u_ltg is None else round(u_ltg, 2),
                    "ltg_rh": rh_ltg,
                    "rh_pct": _climo_percentile(rh_ltg, RH75_CLIMO_XMR.get(month), RH75_PCTL_POINTS),
                    "mf_dir": th.get("mf_dir"),
                    "mf_spd": th.get("mf_spd"),
                    "regime": th.get("mf_regime"),
                    "av_dir": th.get("av_dir"),
                    "av_spd": th.get("av_spd"),
                    "ti": ti,
                    "ti_pct": _climo_percentile(ti, THOMPSON_CLIMO_XMR.get(month), THOMPSON_PCTL_POINTS),
                    "pwat": pwat,
                    "pwat_pct": _climo_percentile(pwat, PWAT_CLIMO_XMR.get(month), PWAT_PCTL_POINTS),
                    "engine": th.get("engine"),
                })
                break
        if day_rows:
            day_rows.sort(key=lambda r: r["sort"])
            by_model[model] = day_rows

    # Ensemble columns (REFS, GEFS) come in pre-averaged from their member fetches: per-member
    # indices averaged, never the index of a mean sounding. Same row shape as the deterministic
    # models, plus the member count and the lightning spread.
    def _add_ensemble(model_name, member_rows):
        if not member_rows:
            return
        rows_out = []
        for row_key, r in member_rows.items():
            try:
                dd, hh = map(int, row_key.split("/"))
            except Exception:
                continue
            day_label, date_str, sort_key, month, _yr = _valid_day_fields(dd, now)
            ti, pwat = r.get("thompson"), r.get("pwat_in")
            rows_out.append({
                "day": day_label,
                "date": date_str,
                "sort": sort_key,
                "vhh": hh,
                "month": month,
                "ltg": r.get("ltg"),
                "ltg_min": r.get("ltg_min"),
                "ltg_max": r.get("ltg_max"),
                "ltg_n": r.get("ltg_n"),
                "ltg_u": (None if r.get("mf_dir") is None else
                          round(rf_lightning_u_wind(r.get("mf_dir"), r.get("mf_spd")), 2)),
                "ltg_rh": r.get("rh_700_500"),
                "rh_pct": _climo_percentile(r.get("rh_700_500"), RH75_CLIMO_XMR.get(month), RH75_PCTL_POINTS),
                "ltg_members": r.get("n"),
                "mf_dir": r.get("mf_dir"),
                "mf_spd": r.get("mf_spd"),
                "regime": r.get("mf_regime"),
                "av_dir": r.get("av_dir"),
                "av_spd": r.get("av_spd"),
                "ti": ti,
                "ti_pct": _climo_percentile(ti, THOMPSON_CLIMO_XMR.get(month), THOMPSON_PCTL_POINTS),
                "pwat": pwat,
                "pwat_pct": _climo_percentile(pwat, PWAT_CLIMO_XMR.get(month), PWAT_PCTL_POINTS),
                "engine": f"metpy\u00b7{r.get('n', 0)}-mem",
            })
        if rows_out:
            rows_out.sort(key=lambda x: x["sort"])
            by_model[model_name] = rows_out

    _add_ensemble("refs", refs_member_rows)
    _add_ensemble("gefs", gefs_member_rows)
    _add_ensemble("ecens", ecens_member_rows)

    # order models: put the ones with the most rows first, stable-ish preferred order
    pref = ["gfs", "ecmwf", "gefs", "ecens", "rrfs", "refs", "rap", "hrrr"]
    models = sorted(by_model.keys(), key=lambda m: (pref.index(m) if m in pref else 99, m))

    # Dump the EXACT feature values fed to the Cizek RF so they can be typed straight into the
    # upstream Streamlit tool and compared. Feature order matches model.feature_names_in_.
    try:
        logging.info("=" * 78)
        logging.info(f"CIZEK LIGHTNING RF INPUTS — {site.upper()} {assess_hour:02d}Z "
                     f"(Thompson_Index, 1000-700mb_Average_U-Wind_Component[kt], 700-500mb_Average_RH[%])")
        logging.info(f"  {'model':6s} {'valid':11s} {'hr':>4s} {'Thompson':>9s} {'U-wind':>8s} {'RH':>7s} {'P(ltg)':>8s}")
        for m in models:
            for r in by_model[m]:
                hr = f"{r.get('vhh'):02d}Z" if r.get("vhh") is not None else "--"
                ti = "n/a" if r.get("ti") is None else f"{r['ti']:9.1f}"
                uu = "     n/a" if r.get("ltg_u") is None else f"{r['ltg_u']:8.2f}"
                rh = "    n/a" if r.get("ltg_rh") is None else f"{r['ltg_rh']:7.1f}"
                pp = "     n/a" if r.get("ltg") is None else f"{r['ltg']:7.1f}%"
                note = ""
                if m == "refs" and r.get("ltg_members"):
                    note = (f"   [P is the mean of {r['ltg_members']} per-member probabilities; "
                            f"features shown are member means and will NOT reproduce it exactly]")
                logging.info(f"  {m:6s} {r['date']:11s} {hr:>4s} {ti} {uu} {rh} {pp}{note}")
        logging.info("=" * 78)
    except Exception as e:
        logging.debug(f"Cizek RF input dump failed: {e}")

    return {"site": site.upper(), "hour": assess_hour, "models": models, "by_model": by_model}

def generate_aviation_dashboard(stations, models, current_sounding_matrix, time_rows, pad_matrix=None):
    # HREF Calibrated Thunder (HREFCT): ML-calibrated probability of >=1 CG flash within
    # 20 km. Fetch both the 1-hour and 4-hour windows. The 4-hour field also drives the
    # calibrated-thunder spatial slider; 1-hour supplies its own table column + maps.
    try:
        ct1_points, ct1_maps = fetch_calibrated_thunder(window="1hr")
    except Exception as e:
        logging.error(f"HREFCT 1hr fetch failed: {e}")
        ct1_points, ct1_maps = {stn: {} for stn in STATIONS}, {}
    try:
        ct4_points, ct4_maps = fetch_calibrated_thunder(window="4hr")
    except Exception as e:
        logging.error(f"HREFCT 4hr fetch failed: {e}")
        ct4_points, ct4_maps = {stn: {} for stn in STATIONS}, {}

    history_runs = []
    prior_thermo_runs = []
    prior_gefs_cache = None
    prior_ecens_cache = None
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                existing = json.load(f)
            # Tolerate the legacy flat-array history.json format.
            history_runs = existing.get("runs", []) if isinstance(existing, dict) else existing
            if isinstance(existing, dict):
                prior_thermo_runs = existing.get("launch_thermo_runs", []) or []
                prior_gefs_cache = existing.get("gefs_cache") or None
                prior_ecens_cache = existing.get("ecens_cache") or None
        except Exception:
            history_runs = []

    # Merge launch-pad (raw-GRIB) soundings into the same station-keyed data block so the
    # frontend treats them identically to the BUFKIT stations (just extra dropdown entries).
    combined_data = dict(current_sounding_matrix)
    if pad_matrix:
        for pid, model_data in pad_matrix.items():
            combined_data[pid] = model_data

    current_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    current_entry = {
        "timestamp": current_timestamp,
        "data": combined_data,
        # Calibrated-thunder point probabilities (1hr + 4hr) participate in dprog/dt history.
        "ct1_points": ct1_points,
        "ct4_points": ct4_points,
    }

    if not history_runs or history_runs[0]["timestamp"] != current_timestamp:
        history_runs.insert(0, current_entry)
    history_runs = history_runs[:5]

    # The spatial PNG maps are NOT part of dprog/dt history — they always reflect only the
    # latest run and get fully overwritten (and pruned) each pipeline pass.
    blank_basemap_path = generate_blank_basemap()
    href_maps_latest = {
        "timestamp": current_timestamp,
        "ct1_maps": ct1_maps,
        "ct4_maps": ct4_maps,
        "blank_map": blank_basemap_path,
    }

    # Launch-thermo panel (KXMR, 10Z per day) — latest run only, like the maps.
    refs_member_rows = {}
    if REFS_MEMBER_THERMO_ENABLED:
        try:
            refs_member_rows = fetch_refs_member_thermo(site="kxmr", assess_hour=10)
        except Exception as e:
            logging.error(f"REFS member thermo fetch failed: {e}")
            refs_member_rows = {}
    try:
        gefs_member_rows, gefs_cycle_key = fetch_gefs_member_thermo(
            site="kxmr", assess_hour=10, cache=prior_gefs_cache)
    except Exception as e:
        logging.error(f"GEFS member thermo fetch failed: {e}")
        gefs_member_rows, gefs_cycle_key = {}, None
    try:
        ecens_member_rows, ecens_cycle_key = fetch_ecmwf_ens_member_thermo(
            site="kxmr", assess_hour=10, cache=prior_ecens_cache)
    except Exception as e:
        logging.error(f"ECMWF ENS member thermo fetch failed: {e}")
        ecens_member_rows, ecens_cycle_key = {}, None
    try:
        launch_thermo = build_launch_thermo(combined_data, site="kxmr", assess_hour=10,
                                            refs_member_rows=refs_member_rows,
                                            gefs_member_rows=gefs_member_rows,
                                            ecens_member_rows=ecens_member_rows)
        logging.info(f"Launch thermo: {len(launch_thermo['models'])} models, "
                     f"rows/model={ {m: len(launch_thermo['by_model'][m]) for m in launch_thermo['models']} }")
    except Exception as e:
        logging.error(f"Launch thermo build failed: {e}")
        launch_thermo = {"site": "KXMR", "hour": 10, "models": [], "by_model": {}}

    # Strip the raw sounding layers stashed for on-demand thermo — they must NOT bloat history.json.
    for _sid, _models in combined_data.items():
        if not isinstance(_models, dict):
            continue
        for _mk, _rows in _models.items():
            if not isinstance(_rows, dict):
                continue
            for _prof in _rows.values():
                if isinstance(_prof, dict):
                    _prof.pop("_layers", None)

    # Panel DPROG/DT: unlike the spatial maps, the 10Z panel DOES keep run history so trends in
    # Thompson/PWAT/lightning can be eyeballed run over run. Current + 3 back.
    thermo_runs = [r for r in prior_thermo_runs
                   if isinstance(r, dict) and r.get("timestamp") != current_timestamp]
    thermo_runs.insert(0, {"timestamp": current_timestamp, "thermo": launch_thermo})
    thermo_runs = thermo_runs[:LAUNCH_THERMO_HISTORY_RUNS]

    payload = {
        "runs": history_runs,
        "href_maps_latest": href_maps_latest,
        "launch_thermo": launch_thermo,
        "launch_thermo_runs": thermo_runs,
        # GEFS cycles 6-hourly while this runs hourly; cache the rows so the fetch is skipped
        # until a new cycle posts.
        "gefs_cache": ({"cycle": gefs_cycle_key, "rows": gefs_member_rows}
                       if (GEFS_CACHE_ENABLED and gefs_cycle_key and gefs_member_rows) else None),
        "ecens_cache": ({"cycle": ecens_cycle_key, "rows": ecens_member_rows}
                        if (ECMWF_ENS_CACHE_ENABLED and ecens_cycle_key and ecens_member_rows) else None),
        # Monthly percentile distributions, shipped so the panel can draw box-and-whisker plots
        # from exactly the same numbers the percentile badges use.
        "climo": {
            "points": CLIMO_PCTL_POINTS_15,
            "thompson": THOMPSON_CLIMO_XMR,
            "pwat": PWAT_CLIMO_XMR,
            "rh75": RH75_CLIMO_XMR,
        },
    }

    with open(HISTORY_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    # Keep every current-run PNG (1-hr CT, 4-hr CT, blank basemap); prune the rest.
    referenced_paths = _collect_map_paths(ct1_maps, ct4_maps, blank_basemap_path)
    prune_stale_maps(referenced_paths)
    logging.info("Aviation matrix completely compiled and written to history.json.")


def run_pipeline():
    logging.info("Starting complete structural iteration run...")
    purge_workspace()
    sounding_matrix = {stn: {mdl: {} for mdl in MODELS} for stn in STATIONS}

    temp_time_rows_set = set()
    with requests.Session() as session:
        # Our own backoff handles retries; let urllib3 fail fast rather than silently
        # stacking a second retry layer on top of an already-throttled host.
        session.mount("https://", requests.adapters.HTTPAdapter(
            pool_connections=BUFKIT_MAX_CONCURRENCY,
            pool_maxsize=BUFKIT_MAX_CONCURRENCY,
            max_retries=0))
        # The pool is sized to the semaphore so we don't park a dozen threads on a gate
        # they can't pass; _BUFKIT_GATE is the hard guarantee either way.
        with concurrent.futures.ThreadPoolExecutor(max_workers=BUFKIT_MAX_CONCURRENCY) as executor:
            futures = [
                executor.submit(fetch_station_model, session, s, m)
                for s in STATIONS
                for m in BUFKIT_MODELS
            ]
            for future in concurrent.futures.as_completed(futures):
                stn, model, data = future.result()
                if data:
                    sounding_matrix[stn][model] = data
                    temp_time_rows_set.update(data.keys())

    ok = sum(1 for s in STATIONS for m in BUFKIT_MODELS if sounding_matrix[s].get(m))
    total = len(STATIONS) * len(BUFKIT_MODELS)
    logging.info(f"BUFKIT: {ok}/{total} station-model columns fetched.")
    if ok < total:
        # PSU throttling should degrade the board, not blank it.
        carry_forward_missing(sounding_matrix, models_to_check=BUFKIT_MODELS)
        for s in STATIONS:
            for m in BUFKIT_MODELS:
                temp_time_rows_set.update((sounding_matrix[s].get(m) or {}).keys())

    time_rows = sorted(list(temp_time_rows_set))
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    trimmed_rows = []
    for row in time_rows:
        try:
            d_part, h_part = map(int, row.split("/"))
            if d_part < now_utc.day and now_utc.day - d_part < 25:
                continue
            if d_part == now_utc.day and h_part < now_utc.hour:
                continue
            trimmed_rows.append(row)
        except Exception:
            trimmed_rows.append(row)
    if trimmed_rows:
        time_rows = trimmed_rows

    # Fetch launch-pad soundings from raw GRIB2 (additive; independent of BUFKIT stations).
    try:
        pad_matrix = fetch_all_pad_soundings()
        pad_hours = sum(len(m.get("hrrr", {})) for m in pad_matrix.values())
        logging.info(f"Launch-pad soundings assembled ({pad_hours} HRRR pad-hours across {len(pad_matrix)} pads).")
    except Exception as e:
        logging.error(f"Launch-pad sounding fetch failed, continuing without pads: {e}")
        pad_matrix = None

    # Fetch RRFS + REFS + HRRR columns from AWS (single idx-based pass). RRFS/REFS are
    # point-extracted for BOTH pads and airports; AWS HRRR is applied to PADS ONLY (the
    # airports already have superior BUFKIT HRRR soundings, so we don't overwrite those).
    if RRFS_ENABLED or REFS_ENABLED:
        try:
            aws_matrix = fetch_all_rrfs_refs_soundings(include_hrrr=True)
            if pad_matrix is None:
                pad_matrix = {pid: {} for pid in LAUNCH_PADS}
            for sid, kinds in aws_matrix.items():
                is_pad = sid in LAUNCH_PADS
                target = pad_matrix if is_pad else sounding_matrix
                target.setdefault(sid, {})
                for kind, rows in kinds.items():
                    if not rows:
                        continue
                    # AWS HRRR only fills pad columns; airports retain BUFKIT HRRR.
                    if kind == "hrrr" and not is_pad:
                        continue
                    target[sid][kind] = rows
            r_hours = sum(len(k.get("rrfs", {})) for k in aws_matrix.values())
            e_hours = sum(len(k.get("refs", {})) for k in aws_matrix.values())
            h_hours = sum(len(aws_matrix[p].get("hrrr", {})) for p in LAUNCH_PADS if p in aws_matrix)
            logging.info(f"AWS columns merged (RRFS {r_hours}, REFS {e_hours} site-hours; HRRR {h_hours} pad-hours).")
        except Exception as e:
            logging.error(f"AWS RRFS/REFS/HRRR fetch failed, continuing without them: {e}")

    # Fetch the ECMWF IFS column (additive; point-extracted for pads + airports) from ECMWF
    # Open Data. Merged under the "ecmwf" key exactly like the AWS columns; total isolation via
    # try/except so any ECMWF outage or missing dependency leaves the rest of the run intact.
    if ECMWF_ENABLED:
        try:
            ecmwf_matrix = fetch_all_ecmwf_soundings()
            if ecmwf_matrix:
                if pad_matrix is None:
                    pad_matrix = {pid: {} for pid in LAUNCH_PADS}
                for sid, rows in ecmwf_matrix.items():
                    if not rows:
                        continue
                    is_pad = sid in LAUNCH_PADS
                    target = pad_matrix if is_pad else sounding_matrix
                    target.setdefault(sid, {})
                    target[sid]["ecmwf"] = rows
                merged = sum(len(r) for r in ecmwf_matrix.values())
                logging.info(f"ECMWF column merged ({merged} site-hours).")
        except Exception as e:
            logging.error(f"ECMWF fetch failed, continuing without it: {e}")

    generate_aviation_dashboard(STATIONS, MODELS, sounding_matrix, time_rows, pad_matrix=pad_matrix)


if __name__ == "__main__":
    run_pipeline()
