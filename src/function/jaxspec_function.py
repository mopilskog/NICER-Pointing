# ------------------------------ #
        # Python's packages
        
from astropy.io import fits
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.time import Time
from termcolor import colored
from typing import Dict
from tqdm import tqdm
from jaxspec.data.util import fakeit_for_multiple_parameters
from typing import List

import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  registers 'science' / related styles with matplotlib
import jax.numpy as jnp
import catalog_information as dict_cat
from function.log_function import log, RESULT

# Silence matplotlib's texmanager INFO log about "No LaTeX-compatible font found
# for the serif fontfamily in rcParams". The rendering still uses Computer Modern
# via LaTeX itself; the message is just rcParams-introspection noise.
logging.getLogger("matplotlib.texmanager").setLevel(logging.WARNING)

# ------------------------------ #

"""
This module provides functions for astronomical data analysis and spectral modeling. It leverages the astropy library for handling astronomical data tables and FITS files, and uses jaxspec for spectral modeling.

Functions:
- modeling_source_spectra: Generates model spectra for sources in a nearby sources table using specified instrument and model parameters.
- total_plot_spectra: Creates and saves the plots of the simulated spectra.
- write_pha_file: Writes the simulated spectra (summed and per source) and the source information to one PHA FITS file.

The module integrates with astropy.io, astropy.table, termcolor, tqdm, and jaxspec, among others, for its functionality. It is designed to facilitate the cross-catalog matching of astronomical sources, the generation and analysis of spectral data, and the efficient presentation and storage of the results.

Note:
- The module is intended for use in the field of astrophysics, particularly in the analysis of spectral data from various astronomical sources.
- It requires a proper understanding of astronomical data formats, spectral analysis techniques, and the specific instruments and models involved in the spectral data generation.
"""


def source_flux_and_name(nearby_sources_table: Table, index, key):
    """Observed flux (erg/cm²/s) and IAU-name column for a source. Column names
    come from the central catalog config; "merged" uses the pipeline's uniform
    columns since it is not a single catalog."""
    if key == "merged":
        flux_col, name_col = "flux", "iau_name"
    else:
        flux_col = dict_cat.dictionary_catalog[key]["flux_obs"]
        name_col = dict_cat.dictionary_coord[key]["iauname"]
    target_flux = nearby_sources_table[flux_col][index] * (u.erg / u.cm**2 / u.s)
    return target_flux, name_col


def norm_for_alpha_samples(model, nh_over_1e22, target_flux, alpha_samples):
    """Normalization per photon-index sample, so every sample reproduces the
    source's observed flux. The absorbed power-law energy flux is linear in
    norm at fixed alpha and N_H, so norm(alpha) = target_flux / G(alpha), where
    G(alpha) is the 0.3-10 keV energy flux at norm=1. G is evaluated only on the
    unique alpha values (the constant-alpha branches then cost one model call)."""
    energies = jnp.geomspace(0.3, 10, 100)
    uniq, inverse = np.unique(alpha_samples, return_inverse=True)
    norm_uniq = np.empty(len(uniq), dtype=float)
    for i, alpha in enumerate(uniq):
        param = {"tbabs_1_nh": nh_over_1e22, "powerlaw_1_alpha": float(alpha),
                 "powerlaw_1_norm": 1.0}
        g = np.sum(model.energy_flux(param, energies[:-1], energies[1:], n_points=100)) \
            * (u.keV / (u.s * u.cm**2))
        norm_uniq[i] = (target_flux / g.to(u.erg / u.cm**2 / u.s)).to_value(u.dimensionless_unscaled)
    return norm_uniq[inverse]

def modeling_source_spectra(nearby_sources_table: Table, obsconfig, model, key) -> List:
    """
    Generates model spectra for astronomical sources using specified instrument and model parameters.

    This function processes each source in the provided nearby sources table, applying the specified spectral model. 
    It takes into account factors like vignetting and adjusts parameters such as 'N_H' (hydrogen column density) 
    and 'alpha' (spectral index) based on the source data. The function is primarily used in astronomical data analysis 
    to simulate observed spectra.

    Args:
        nearby_sources_table (Table): An astropy Table containing data of nearby astronomical sources.
        obsconfig: An ObsConfiguration object containing the data  of the folding model, which is the link between the unfolded and folded spectra.
        model: The spectral model to be applied for spectra generation.
        key (str): Catalog key (e.g. 'XMM', 'merged'), used to find the flux and name columns.

    Returns:
        List: One array of simulated spectra per source (one row per photon-index sample).

    Note:
        - The function expects 'vignetting_factor', 'Nh', and 'Photon Index' columns in the nearby sources table.
        - Spectra are generated using the 'fakeit_for_multiple_parameters' function, based on the model and instrument.
        - The function is tailored for astrophysical spectral analysis and may require specific data formats.
    """
    log.info(f"\n{colored('Modeling spectra...', 'yellow', attrs=['underline'])}")
    total_spectra = []
    size = 100

    for index, vignet_factor in tqdm(enumerate(nearby_sources_table["vignetting_factor"])):
        target_flux, iauname = source_flux_and_name(nearby_sources_table, index=index, key=key)

        pho_idx = float(nearby_sources_table["Photon Index"][index])
        pho_idx_err = float(nearby_sources_table["Photon Index Error"][index])

        # Bad fit: when the photon-index uncertainty is large (>2), the
        # underlying curve_fit didn't converge and the value is unreliable.
        # Use the catalog-class default (1.7) with no error propagation.
        if pho_idx_err > 2.0:
            log.debug(colored(
                f"  Bad fit for {iauname} (PhI={pho_idx:.3f} ± {pho_idx_err:.3f}); "
                f"using default alpha=1.7, no sampling.",
                'yellow'
            ))
            alpha_samples = np.full(size, 1.7)
        elif pho_idx_err > 0.0:
            # Sample the photon index from N(alpha, perr). Negative (hard) alpha
            # is allowed: with per-sample normalization the spectrum stays at the
            # observed flux, so it does not blow up at high energy.
            alpha_samples = np.random.normal(pho_idx, pho_idx_err, size)
        else:
            # No error info available (e.g. failed fit path, user-added source).
            alpha_samples = np.full(size, pho_idx if pho_idx > 0.0 else 1.7)

        nh_over_1e22 = nearby_sources_table["Nh"][index] / 1e22
        # One norm per sample so all 100 spectra reproduce the observed flux;
        # the 16-84% spread then reflects photon-index shape only, not a flux drift.
        norm_samples = norm_for_alpha_samples(model, nh_over_1e22, target_flux, alpha_samples)

        # jaxspec >= 0.2 expects flat parameter dicts keyed "<component>_<param>".
        parameters = {
            "tbabs_1_nh": np.full(size, nh_over_1e22),
            "powerlaw_1_alpha": alpha_samples,
            "powerlaw_1_norm": norm_samples,
        }

        spectra = fakeit_for_multiple_parameters(obsconfig, model, parameters) * vignet_factor

        total_spectra.append(spectra)

    return total_spectra


def variability_ratios(nearby_sources_table, n_sources):
    """Per-source flux scaling factors for the variability envelope.

    Spectra scale linearly with the normalization at fixed photon index and
    N_H, so multiplying a source's simulated spectra by flux_min/flux and
    flux_max/flux gives its low- and high-state spectra without re-simulating.
    A source gets an envelope if its fluxes changed between observations
    (var_ratio > 1) or XMM flagged it as variable within an observation
    (short_term_var). Without variability columns (single-catalog modes) every
    factor is 1.

    Returns:
        var_flags (bool array), ratio_lower (float array), ratio_upper (float array).
    """
    var_flags = np.zeros(n_sources, dtype=bool)
    ratio_lower = np.ones(n_sources)
    ratio_upper = np.ones(n_sources)
    colnames = set(getattr(nearby_sources_table, "colnames", []))
    if nearby_sources_table is None or not {"var_ratio", "flux_min", "flux_max", "flux"} <= colnames:
        return var_flags, ratio_lower, ratio_upper

    vr_col = np.array(nearby_sources_table["var_ratio"], dtype=float)
    fmin_col = np.array(nearby_sources_table["flux_min"], dtype=float)
    fmax_col = np.array(nearby_sources_table["flux_max"], dtype=float)
    f_col = np.array(nearby_sources_table["flux"], dtype=float)
    stv_col = (np.array(nearby_sources_table["short_term_var"], dtype=bool)
               if "short_term_var" in colnames else np.zeros(n_sources, dtype=bool))
    for i in range(n_sources):
        if (vr_col[i] > 1.0 or stv_col[i]) and f_col[i] > 0.0:
            var_flags[i] = True
            ratio_lower[i] = fmin_col[i] / f_col[i]
            ratio_upper[i] = fmax_col[i] / f_col[i]
    return var_flags, ratio_lower, ratio_upper


def summed_variability_spectra(total_spectra, ratio_lower, ratio_upper):
    """Summed spectra with every source at its low (resp. high) flux state."""
    summed_lower = 0.0
    summed_upper = 0.0
    for i in range(len(total_spectra)):
        summed_lower = summed_lower + total_spectra[i] * ratio_lower[i]
        summed_upper = summed_upper + total_spectra[i] * ratio_upper[i]
    return summed_lower, summed_upper


def total_plot_spectra(total_spectra: List, obsconfig, simulation_data: Dict, catalog_name: str) -> None:
    """
    Generates and saves a plot of spectral modeling data and returns a summary of the spectral data.

    This function processes spectral data from a specified catalog and creates a series of plots to visualize the spectral modeling. 
    It handles both regular and variable spectra, plotting median spectra, sum of spectra, and a summation with variability error. 
    The plot is saved in a directory specified in the `simulation_data` dictionary.

    Args:
        total_spectra (List): A list containing spectra from nearby sources.
        obsconfig: An ObsConfiguration object containing the data  of the folding model, which is the link between the unfolded and folded spectra.
        simulation_data (Dict): Dictionary containing simulation parameters and paths.
        catalog_name (str): Name of the catalog used for spectral modeling.

    Important:
        The function generates three subplots:
            1. Spectra from Nearby Sources: Plots the median spectra for each source.
            2. Sum of Spectra: Shows the sum of all spectra.
            3. Spectrum Summed with Variability Sources Error: Includes error bars representing variability.

    The plot uses logarithmic scaling on both axes, with the x-axis representing energy in keV and the y-axis representing counts.
    The plots are saved as PNG files. The numbers behind them are written by write_pha_file.

    Note:
        - The file is named based on the catalog name and the object of interest from the simulation data.
        - Assumes `instrument` has an attribute `out_energies` for energy values.
        - The function is designed for astrophysical spectral analysis and visualization.
        
    """
    
    object_data = simulation_data["object_data"]
    os_dictionary = simulation_data["os_dictionary"]
    # X-axis spans the full simulated range. The jaxspec simulation runs over
    # NICER's RMF EBOUNDS (~0.01-15 keV), so capping the display short of 15
    # would just hide existing data. Lower bound 0.2 keV trims the noise floor
    # below NICER's documented sensitive band.
    # X-axis spans the same range used in the master thesis (0.3-20 keV) so
    # the new pipeline outputs are directly comparable to the thesis figures.
    # The jaxspec simulation itself runs across NICER's RMF EBOUNDS
    # (~0.01-15 keV); above 15 keV the data is implicit zero.
    graph_data = {"min_lim_x": 0.3, "max_lim_x": 2e1,
                  "min_lim_y": 5e-3, "max_lim_y": 1e4}

    energies = obsconfig.out_energies[0]

    # Index-wise sum across sources: spectrum_summed has shape (size, n_energy);
    # row i is the total at iteration i (sum of every source's i-th realization).
    spectrum_summed = 0.0
    for item in range(len(total_spectra)):
        spectrum_summed += total_spectra[item]

    var_flags, ratio_lower, ratio_upper = variability_ratios(
        simulation_data.get("nearby_sources_table"), len(total_spectra))
    any_variable = bool(np.any(var_flags))
    spectrum_summed_lower, spectrum_summed_upper = summed_variability_spectra(
        total_spectra, ratio_lower, ratio_upper)

    # Source colors come from matplotlib's tab20 + tab20b + tab20c qualitative
    # palettes interleaved — 60 perceptually distinct colors with strong
    # contrast between adjacent source indices (the interleave alternates
    # between three palette families). Cycles for >60 sources. Catalog
    # identity is conveyed by the legend label and per-source plot title;
    # neighborhood-plot rings still use the per-catalog hue scheme.
    from matplotlib.lines import Line2D
    import matplotlib.cm as mcm
    nearby_table = simulation_data.get("nearby_sources_table")
    has_origin = (
        nearby_table is not None
        and "origin_catalog" in getattr(nearby_table, "colnames", [])
    )

    n_sources = max(len(total_spectra), 1)
    t20 = list(mcm.get_cmap("tab20").colors)
    t20b = list(mcm.get_cmap("tab20b").colors)
    t20c = list(mcm.get_cmap("tab20c").colors)
    palette = [c for triple in zip(t20, t20b, t20c) for c in triple]  # 60 colors
    source_colors = [palette[i % len(palette)] for i in range(n_sources)]

    if has_origin:
        origins = list(nearby_table["origin_catalog"])
        iaunames = [str(n).strip() for n in nearby_table["iau_name"]]
    else:
        origins = None
        iaunames = None

    # Use scienceplots' "science" style for LaTeX-formatted publication-quality
    # plots. Scoped to this function so other matplotlib output in the project
    # is unaffected.
    with plt.style.context(['science', 'retro']):
        # Combined two-panel diagnostic plot (top-5 brightest + remaining on the
        # left, total summed spectrum on the right). The single-panel
        # title-free counterparts of each panel are saved separately further
        # below so they can be included individually in LaTeX/PDF reports.
        figure_1, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
        figure_1.text(0.5, 0.04, 'Energy [keV]', ha='center', va='center', fontsize=16)
        figure_1.text(0.085, 0.5, 'Counts', ha='center', va='center', rotation='vertical', fontsize=16)

        for ax in axes:
            ax.set_xlim([graph_data["min_lim_x"], graph_data["max_lim_x"]])
            ax.set_ylim([graph_data["min_lim_y"], graph_data["max_lim_y"]])
            ax.loglog()

        # Left panel: only the N brightest sources drawn individually (each as
        # median + 16-84% band), with all other sources summed into one
        # auxiliary curve. Prevents over-crowding at 50+ sources while keeping
        # the dominant contributors visible. Sorting is by post-vignetting
        # count_rate (already on nearby_table after c_f.count_rates ran).
        # Colors here use a pastel rainbow generated independently of the
        # tab20-family per-source palette; the latter is still used elsewhere.
        ax0 = axes[0]
        N_BRIGHTEST = 5
        CATALOG_ORDER = ["XMM", "Swift", "eRASS1", "Chandra", "user"]
        from matplotlib.colors import hsv_to_rgb
        count_rates_arr = np.array([
            float(nearby_table["count_rate"][i]) for i in range(len(total_spectra))
        ])
        order = np.argsort(-count_rates_arr)
        brightest_idx = list(order[:N_BRIGHTEST])
        other_idx = list(order[N_BRIGHTEST:])

        # Top-N brightest sources use matplotlib's Pastel2 qualitative colormap
        # — 8 distinct pastel hues. Cycles for n_top > 8.
        pastel2 = list(plt.get_cmap("Pastel2").colors)
        brightest_colors = [
            pastel2[r % len(pastel2)]
            for r in range(len(brightest_idx))
        ]

        for rank, i in enumerate(brightest_idx):
            spectra = total_spectra[i]
            median = np.median(spectra, axis=0)
            p16 = np.percentile(spectra, 16, axis=0)
            p84 = np.percentile(spectra, 84, axis=0)
            color = brightest_colors[rank]
            label = (
                f"{origins[i]}: {iaunames[i]}" if has_origin
                else (iaunames[i] if iaunames else f"source {i}")
            )
            ax0.fill_between(energies, p16, p84, step="post", color=color, alpha=0.40)
            ax0.step(energies, median, where="post", color=color, lw=1.5, label=label)

        if other_idx:
            others_summed = 0
            for i in other_idx:
                others_summed = others_summed + total_spectra[i]
            om = np.median(others_summed, axis=0)
            o16 = np.percentile(others_summed, 16, axis=0)
            o84 = np.percentile(others_summed, 84, axis=0)
            ax0.fill_between(energies, o16, o84, step="post", color='lightcoral', alpha=0.40)
            ax0.step(
                energies, om, where="post", color='lightcoral', lw=1.5,
                label=f"sum of {len(other_idx)} remaining sources",
            )

        ax0.legend(loc='upper right', fontsize=8)

        # Right panel: total summed spectrum as median + 16-84% band. Same
        # color scheme as the per-source plots so the figures are visually
        # consistent. The variability bracket is intentionally not overlaid
        # here — it lives in the dedicated `merged_sum_with_variability_*`
        # plot so this diagnostic two-panel view stays uncluttered.
        ax1 = axes[1]
        sum_median = np.median(spectrum_summed, axis=0)
        sum_p16 = np.percentile(spectrum_summed, 16, axis=0)
        sum_p84 = np.percentile(spectrum_summed, 84, axis=0)
        ax1.fill_between(
            energies, sum_p16, sum_p84, step="post",
            color='lightskyblue', alpha=0.4, label="16-84% envelope",
        )
        ax1.step(
            energies, sum_median, where="post",
            color='steelblue', linewidth=1.3, label="median",
        )
        ax1.legend(loc='upper right', fontsize=9)

        key = simulation_data["os_dictionary"]["catalog_key"]
        img_path = os.path.join(os_dictionary['img'], f"{key}_spectral_modeling_close_to_{object_data['object_name']}.png".replace(" ", "_")).replace("\\", "/")
        plt.savefig(img_path, bbox_inches='tight')
        plt.show()

        # Standalone, title-free single-panel version of the left panel
        # (top-5 brightest sources + remaining-sum auxiliary curve). Same
        # plotting logic as ax0 above, just on its own axes so it can be
        # included individually in LaTeX without needing the right panel.
        fig_top5, ax_top5 = plt.subplots(1, 1, figsize=(8, 7))
        ax_top5.set_xlim([graph_data["min_lim_x"], graph_data["max_lim_x"]])
        ax_top5.set_ylim([graph_data["min_lim_y"], graph_data["max_lim_y"]])
        ax_top5.loglog()
        for rank, i in enumerate(brightest_idx):
            spectra = total_spectra[i]
            median = np.median(spectra, axis=0)
            p16 = np.percentile(spectra, 16, axis=0)
            p84 = np.percentile(spectra, 84, axis=0)
            color = brightest_colors[rank]
            label = (
                f"{origins[i]}: {iaunames[i]}" if has_origin
                else (iaunames[i] if iaunames else f"source {i}")
            )
            ax_top5.fill_between(energies, p16, p84, step="post", color=color, alpha=0.40)
            ax_top5.step(energies, median, where="post", color=color, lw=1.5, label=label)
        if other_idx:
            others_summed = 0
            for i in other_idx:
                others_summed = others_summed + total_spectra[i]
            om = np.median(others_summed, axis=0)
            o16 = np.percentile(others_summed, 16, axis=0)
            o84 = np.percentile(others_summed, 84, axis=0)
            ax_top5.fill_between(energies, o16, o84, step="post", color='lightcoral', alpha=0.40)
            ax_top5.step(
                energies, om, where="post", color='lightcoral', lw=1.5,
                label=f"sum of {len(other_idx)} remaining sources",
            )
        ax_top5.set_xlabel("Energy [keV]")
        ax_top5.set_ylabel("Counts")
        ax_top5.legend(loc='upper right', fontsize=9)
        top5_img_path = os.path.join(
            os_dictionary['img'],
            f"{key}_top5_close_to_{object_data['object_name']}.png".replace(" ", "_"),
        ).replace("\\", "/")
        plt.savefig(top5_img_path, bbox_inches='tight')
        plt.close(fig_top5)

        # Merged-mode only: one plot per source (pink curve-spaghetti) plus a
        # dedicated sum plot (blue curve-spaghetti). Each per-source plot is
        # annotated with the source name, the catalog the data came from, and
        # which catalogs detected the source according to Master_source_cone.
        if has_origin:
            per_source_dir = os.path.join(
                os_dictionary['img'], 'per_source'
            ).replace("\\", "/")
            if not os.path.exists(per_source_dir):
                os.makedirs(per_source_dir)

            # Load master cone to determine which catalogs detected each MS_ID.
            ms_id_to_detected = {}
            modeling_file_path = simulation_data["os_dictionary"]["modeling_file_path"]
            master_cone_path = os.path.join(
                modeling_file_path, "Pointings", "Master_source_cone.fits"
            ).replace("\\", "/")
            if os.path.exists(master_cone_path):
                with fits.open(master_cone_path, memmap=True) as hdul:
                    mc = Table(hdul[1].data)
                for row in mc:
                    detected = []
                    for c in CATALOG_ORDER:
                        if c == "user":
                            continue
                        v = str(row[c]).strip() if c in mc.colnames else ""
                        if v and v != "--":
                            detected.append(c)
                    ms_id_to_detected[int(row["MS_ID"])] = detected

            ms_ids = list(nearby_table["MS_ID"])

            # One plot per source. For variable sources (var_ratio > 1) overlay
            # a wider dotted envelope built from the source's observed flux
            # extrema; the inner solid+band still shows the photon-index
            # uncertainty at the nominal flux.
            for i, spectra in enumerate(total_spectra):
                fig_s, ax_s = plt.subplots(1, 1, figsize=(8, 6))
                ax_s.set_xlim([graph_data["min_lim_x"], graph_data["max_lim_x"]])
                ax_s.set_ylim([graph_data["min_lim_y"], graph_data["max_lim_y"]])
                ax_s.loglog()
                # Median + 16-84% percentile band across the photon-index
                # samples for this source. Cleaner than per-curve spaghetti at
                # this scale and conveys the same uncertainty information.
                median = np.median(spectra, axis=0)
                p16 = np.percentile(spectra, 16, axis=0)
                p84 = np.percentile(spectra, 84, axis=0)
                if var_flags[i]:
                    # Outer envelope combining flux variability + photon-index
                    # uncertainty: 16th percentile at the low-flux state up to
                    # the 84th percentile at the high-flux state. Because
                    # ratio_lower <= 1 <= ratio_upper, this envelope is
                    # guaranteed to be at least as wide as the inner blue
                    # 16-84% band at the nominal flux (so orange always
                    # encloses blue, as a "combined plausible range" should).
                    src_low = np.percentile(spectra * ratio_lower[i], 16, axis=0)
                    src_up = np.percentile(spectra * ratio_upper[i], 84, axis=0)
                    ax_s.fill_between(
                        energies, src_low, src_up, step="post",
                        color='bisque', alpha=0.4,
                        label="variability range",
                    )
                ax_s.fill_between(
                    energies, p16, p84, step="post",
                    color='lightskyblue', alpha=0.4,
                    label="16-84% envelope",
                )
                ax_s.step(
                    energies, median, where="post",
                    color='steelblue', linewidth=1.3,
                    label="median",
                )
                ax_s.set_xlabel("Energy [keV]")
                ax_s.set_ylabel("Counts")
                ax_s.legend(loc='upper right', fontsize=9)

                # No in-figure title -- the source name and var_ratio are
                # encoded in the saved filename, and any report that includes
                # this plot puts the identification in the LaTeX caption.

                safe_name = iaunames[i].replace(" ", "_").replace("/", "_")
                source_img_path = os.path.join(
                    per_source_dir,
                    f"{safe_name}.png",
                ).replace("\\", "/")
                plt.savefig(source_img_path, bbox_inches='tight')
                plt.close(fig_s)

            # Standalone total-sum plot. Median + 16-84% band — same colors as
            # the per-source plots so the figures are visually consistent.
            # Variability bracket NOT shown here — see the dedicated
            # `*_sum_with_variability_*.png` sibling for the with-variability
            # view. Keeping the two as separate files lets the thesis show
            # them side-by-side ("without variability" vs "with variability").
            fig_sum, ax_sum = plt.subplots(1, 1, figsize=(8, 7))
            ax_sum.set_xlim([graph_data["min_lim_x"], graph_data["max_lim_x"]])
            ax_sum.set_ylim([graph_data["min_lim_y"], graph_data["max_lim_y"]])
            ax_sum.loglog()
            sum_median_s = np.median(spectrum_summed, axis=0)
            sum_p16_s = np.percentile(spectrum_summed, 16, axis=0)
            sum_p84_s = np.percentile(spectrum_summed, 84, axis=0)
            ax_sum.fill_between(
                energies, sum_p16_s, sum_p84_s, step="post",
                color='lightskyblue', alpha=0.4, label="16-84% envelope",
            )
            ax_sum.step(
                energies, sum_median_s, where="post",
                color='steelblue', linewidth=1.3, label="median",
            )
            ax_sum.set_xlabel("Energy [keV]")
            ax_sum.set_ylabel("Counts")
            ax_sum.legend(loc='upper right', fontsize=9)
            sum_title_origins = ', '.join(sorted(
                set(origins),
                key=lambda c: CATALOG_ORDER.index(c) if c in CATALOG_ORDER else 99,
            ))
            sum_img_path = os.path.join(
                os_dictionary['img'],
                f"{key}_sum_close_to_{object_data['object_name']}.png".replace(" ", "_"),
            ).replace("\\", "/")
            plt.savefig(sum_img_path, bbox_inches='tight')
            plt.close(fig_sum)

            # Variability-aware companion of the standalone sum. Same baseline
            # (median + 16-84% photon-index band) plus the light-orange
            # variability range (linear-scaled sum using min/max observed flux
            # for each source flagged var_ratio > 1). Provided as a separate
            # file so the with/without comparison is a clean two-figure pair.
            if any_variable:
                fig_sv, ax_sv = plt.subplots(1, 1, figsize=(8, 7))
                ax_sv.set_xlim([graph_data["min_lim_x"], graph_data["max_lim_x"]])
                ax_sv.set_ylim([graph_data["min_lim_y"], graph_data["max_lim_y"]])
                ax_sv.loglog()
                # Outer envelope combining flux variability + photon-index
                # uncertainty: 16th percentile of the low-state summed
                # spectrum up to the 84th percentile of the high-state summed
                # spectrum. Guaranteed to enclose the inner blue 16-84% band
                # at the nominal flux.
                sum_low_s = np.percentile(spectrum_summed_lower, 16, axis=0)
                sum_up_s = np.percentile(spectrum_summed_upper, 84, axis=0)
                ax_sv.fill_between(
                    energies, sum_low_s, sum_up_s, step="post",
                    color='bisque', alpha=0.4,
                    label="variability range",
                )
                ax_sv.fill_between(
                    energies, sum_p16_s, sum_p84_s, step="post",
                    color='lightskyblue', alpha=0.4, label="16-84% envelope",
                )
                ax_sv.step(
                    energies, sum_median_s, where="post",
                    color='steelblue', linewidth=1.3, label="median",
                )
                ax_sv.set_xlabel("Energy [keV]")
                ax_sv.set_ylabel("Counts")
                ax_sv.legend(loc='upper right', fontsize=9)
                sum_var_img_path = os.path.join(
                    os_dictionary['img'],
                    f"{key}_sum_with_variability_close_to_{object_data['object_name']}.png".replace(" ", "_"),
                ).replace("\\", "/")
                plt.savefig(sum_var_img_path, bbox_inches='tight')
                plt.close(fig_sv)

            # Per-catalog vs total summed spectra. Two panels: left shows each
            # origin catalog's collective spectrum (median + 16-84% band) using
            # matplotlib's Pastel2 qualitative colormap; right shows the total
            # summed spectrum in the same blue used elsewhere for the FoV
            # total. Pastel2 indices in use:
            #   0=green, 1=peach/orange, 2=blue, 3=pink, 5=yellow
            pastel2 = list(plt.get_cmap("Pastel2").colors)
            CATALOG_PASTEL2_COLOR = {
                "XMM":     pastel2[0],  # pale green
                "Chandra": pastel2[3],  # pale pink
                "Swift":   pastel2[2],  # pale blue
                "eRASS1":  pastel2[1],  # pale peach / orange
                "user":    pastel2[5],  # pale yellow
            }
            unique_origins_ordered = [
                c for c in CATALOG_ORDER if c in set(origins)
            ]

            fig_cat, cat_axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
            fig_cat.text(0.5, 0.04, 'Energy [keV]', ha='center', va='center', fontsize=14)
            fig_cat.text(0.085, 0.5, 'Counts', ha='center', va='center',
                         rotation='vertical', fontsize=14)
            for ax in cat_axes:
                ax.set_xlim([graph_data["min_lim_x"], graph_data["max_lim_x"]])
                ax.set_ylim([graph_data["min_lim_y"], graph_data["max_lim_y"]])
                ax.loglog()

            cat_ax_left = cat_axes[0]
            for cat in unique_origins_ordered:
                cat_indices = [i for i, o in enumerate(origins) if o == cat]
                if not cat_indices:
                    continue
                cat_summed = 0
                for i in cat_indices:
                    cat_summed = cat_summed + total_spectra[i]
                cm = np.median(cat_summed, axis=0)
                c16 = np.percentile(cat_summed, 16, axis=0)
                c84 = np.percentile(cat_summed, 84, axis=0)
                color = CATALOG_PASTEL2_COLOR.get(cat, pastel2[0])
                cat_ax_left.fill_between(
                    energies, c16, c84, step="post",
                    color=color, alpha=0.40,
                )
                cat_ax_left.step(
                    energies, cm, where="post",
                    color=color, linewidth=1.5,
                    label=f"{cat} ({len(cat_indices)} sources)",
                )
            cat_ax_left.legend(
                loc='upper right', fontsize=9,
                title="median + 16-84% envelope",
                title_fontsize=8,
            )

            # Right panel: total summed spectrum in lightskyblue / steelblue,
            # matching the right-panel style of the merged spectra plot. No
            # variability overlay — that view lives in
            # `*_sum_with_variability_*.png`.
            cat_ax_right = cat_axes[1]
            tm = np.median(spectrum_summed, axis=0)
            t16 = np.percentile(spectrum_summed, 16, axis=0)
            t84 = np.percentile(spectrum_summed, 84, axis=0)
            cat_ax_right.fill_between(
                energies, t16, t84, step="post",
                color='lightskyblue', alpha=0.4,
                label="16-84% envelope",
            )
            cat_ax_right.step(
                energies, tm, where="post",
                color='steelblue', linewidth=1.3,
                label="median",
            )
            cat_ax_right.legend(loc='upper right', fontsize=9)

            cat_img_path = os.path.join(
                os_dictionary['img'],
                f"{key}_per_catalog_sum_close_to_{object_data['object_name']}.png".replace(" ", "_"),
            ).replace("\\", "/")
            plt.savefig(cat_img_path, bbox_inches='tight')
            plt.close(fig_cat)

            # Standalone, title-free single-panel version of the per-origin
            # sum (left panel of fig_cat). Saved separately so it can be
            # included individually in LaTeX without the right (total)
            # panel next to it.
            fig_origin, ax_origin = plt.subplots(1, 1, figsize=(8, 7))
            ax_origin.set_xlim([graph_data["min_lim_x"], graph_data["max_lim_x"]])
            ax_origin.set_ylim([graph_data["min_lim_y"], graph_data["max_lim_y"]])
            ax_origin.loglog()
            for cat in unique_origins_ordered:
                cat_indices = [i for i, o in enumerate(origins) if o == cat]
                if not cat_indices:
                    continue
                cat_summed = 0
                for i in cat_indices:
                    cat_summed = cat_summed + total_spectra[i]
                cm = np.median(cat_summed, axis=0)
                c16 = np.percentile(cat_summed, 16, axis=0)
                c84 = np.percentile(cat_summed, 84, axis=0)
                color = CATALOG_PASTEL2_COLOR.get(cat, pastel2[0])
                ax_origin.fill_between(
                    energies, c16, c84, step="post",
                    color=color, alpha=0.40,
                )
                ax_origin.step(
                    energies, cm, where="post",
                    color=color, linewidth=1.5,
                    label=f"{cat} ({len(cat_indices)} sources)",
                )
            ax_origin.set_xlabel("Energy [keV]")
            ax_origin.set_ylabel("Counts")
            ax_origin.legend(
                loc='upper right', fontsize=9,
                title="median + 16-84% envelope",
                title_fontsize=8,
            )
            origin_img_path = os.path.join(
                os_dictionary['img'],
                f"{key}_per_origin_close_to_{object_data['object_name']}.png".replace(" ", "_"),
            ).replace("\\", "/")
            plt.savefig(origin_img_path, bbox_inches='tight')
            plt.close(fig_origin)


def _source_columns(nearby_sources_table: Table, key: str):
    """Name, RA and Dec column names of the nearby-sources table for this mode."""
    if key == "merged":
        return "iau_name", "RA", "DEC"
    coord = dict_cat.dictionary_coord[key]
    return coord["iauname"], coord["right_ascension"], coord["declination"]


def write_pha_file(total_spectra: List, obsconfig, simulation_data: Dict, key: str,
                   pointing_ra: float, pointing_dec: float, radius_arcmin: float,
                   exposure: float, rmf_path: str, arf_path: str) -> str:
    """
    Writes the simulated nearby-source spectra to one OGIP PHA FITS file.

    The file holds the same numbers as the summed-spectrum figures, so a user
    can reuse them in their own analysis (e.g. as a background spectrum in
    XSPEC: ``backgrnd 1 <file>.pha``). It has three extensions:

    1. ``SPECTRUM`` (OGIP type I): the summed spectrum of all nearby sources per
       PI channel (0-1500, matching the NICER RMF).

       - ``COUNTS``: median of the simulated summed spectra (the plotted median).
       - ``STAT_ERR``: (P84 - P16) / 2.
       - ``COUNTS_P16``, ``COUNTS_P84``: 16-84 % band without variability
         (photon-index uncertainty only, blue band in the figures).
       - ``COUNTS_VAR_LOW``, ``COUNTS_VAR_HIGH``: band with variability
         (16th percentile with every variable source at its lowest flux, 84th
         percentile at its highest flux, orange band in the figures).
       - ``E_MIN``, ``E_MAX``: channel energy bounds in keV.

    2. ``SOURCES``: one row per nearby source (position, off-axis angle,
       vignetting, catalog, flux, photon index, N_H, count rate, variability).

    3. ``SOURCE_SPECTRA``: the same five spectra as in ``SPECTRUM`` for each
       source, in the same row order as ``SOURCES``.

    Without variability information (single-catalog modes) the variability
    columns equal the 16-84 % band and the header keyword VARINFO is False.

    Args:
        total_spectra (List): Simulated spectra per source, from modeling_source_spectra.
        obsconfig: jaxspec ObsConfiguration used for the simulation.
        simulation_data (Dict): Simulation parameters, paths and the nearby-sources table.
        key (str): Catalog key ('merged', 'XMM', ...).
        pointing_ra, pointing_dec (float): Pointing used for the vignetting, in degrees.
        radius_arcmin (float): Search radius around the target, in arcmin.
        exposure (float): Exposure time in seconds.
        rmf_path, arf_path (str): NICER response files used for the simulation.

    Returns:
        str: Path of the written file.
    """
    from catalog_class.MultiCatalogPipeline import K_FVAR

    object_data = simulation_data["object_data"]
    os_dictionary = simulation_data["os_dictionary"]
    table = simulation_data["nearby_sources_table"]
    n_sources = len(total_spectra)

    # Simulated channels -> PI channels of the RMF. jaxspec drops PI 0
    # (0-0.01 keV), so channels are matched on their lower energy bound.
    with fits.open(rmf_path) as hdul:
        ebounds = hdul["EBOUNDS"].data
        pi_channel = np.array(ebounds["CHANNEL"], dtype=np.int32)
        e_min = np.array(ebounds["E_MIN"], dtype=float)
        e_max = np.array(ebounds["E_MAX"], dtype=float)
    sim_e_min = np.asarray(obsconfig.out_energies[0], dtype=float)
    sim_to_pi = np.array([np.argmin(np.abs(e_min - e)) for e in sim_e_min])

    def to_pi(spectrum):
        full = np.zeros(len(pi_channel))
        full[sim_to_pi] = spectrum
        return full

    def five_spectra(spectra, ratio_low, ratio_high):
        """median, P16, P84, variability low, variability high (PI channels)."""
        return [to_pi(np.median(spectra, axis=0)),
                to_pi(np.percentile(spectra, 16, axis=0)),
                to_pi(np.percentile(spectra, 84, axis=0)),
                to_pi(np.percentile(spectra * ratio_low, 16, axis=0)),
                to_pi(np.percentile(spectra * ratio_high, 84, axis=0))]

    var_flags, ratio_lower, ratio_upper = variability_ratios(table, n_sources)
    var_info = {"var_ratio", "flux_min", "flux_max"} <= set(table.colnames)

    spectrum_summed = 0.0
    for spectra in total_spectra:
        spectrum_summed = spectrum_summed + spectra
    summed_lower, summed_upper = summed_variability_spectra(total_spectra, ratio_lower, ratio_upper)
    median, p16, p84 = five_spectra(spectrum_summed, 1.0, 1.0)[:3]
    var_low = to_pi(np.percentile(summed_lower, 16, axis=0))
    var_high = to_pi(np.percentile(summed_upper, 84, axis=0))

    # ---------- SPECTRUM ---------- #
    spectrum_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="CHANNEL", format="J", array=pi_channel),
        fits.Column(name="COUNTS", format="D", unit="count", array=median),
        fits.Column(name="STAT_ERR", format="D", unit="count", array=(p84 - p16) / 2),
        fits.Column(name="COUNTS_P16", format="D", unit="count", array=p16),
        fits.Column(name="COUNTS_P84", format="D", unit="count", array=p84),
        fits.Column(name="COUNTS_VAR_LOW", format="D", unit="count", array=var_low),
        fits.Column(name="COUNTS_VAR_HIGH", format="D", unit="count", array=var_high),
        fits.Column(name="E_MIN", format="D", unit="keV", array=e_min),
        fits.Column(name="E_MAX", format="D", unit="keV", array=e_max),
    ], name="SPECTRUM")
    h = spectrum_hdu.header
    for card in [
        ("TELESCOP", "NICER", "Mission"),
        ("INSTRUME", "XTI", "Instrument"),
        ("FILTER", "NONE", "Filter"),
        ("OBJECT", object_data["object_name"], "Target of the pointing"),
        ("EXPOSURE", float(exposure), "[s] Simulated exposure time"),
        ("AREASCAL", 1.0, "Area scaling factor"),
        ("BACKSCAL", 1.0, "Background scaling factor"),
        ("CORRSCAL", 0.0, "Correction scaling factor"),
        ("BACKFILE", "NONE", "Background file"),
        ("CORRFILE", "NONE", "Correction file"),
        ("RESPFILE", os.path.basename(rmf_path), "Response matrix used"),
        ("ANCRFILE", os.path.basename(arf_path), "Ancillary response used"),
        ("HDUCLASS", "OGIP", "Format conforms to OGIP standard"),
        ("HDUCLAS1", "SPECTRUM", "PHA dataset"),
        ("HDUCLAS2", "BKG", "Simulated contribution of nearby sources"),
        ("HDUCLAS3", "COUNT", "PHA data stored as counts"),
        ("HDUVERS", "1.2.1", "OGIP version"),
        ("CHANTYPE", "PI", "Channel type"),
        ("DETCHANS", len(pi_channel), "Number of channels"),
        ("POISSERR", False, "Errors are given in STAT_ERR"),
        ("SYS_ERR", 0, "No systematic error"),
        ("QUALITY", 0, "All channels good"),
        ("GROUPING", 0, "No grouping"),
        ("RA_OBJ", float(object_data["object_position"].ra.deg), "[deg] Target RA"),
        ("DEC_OBJ", float(object_data["object_position"].dec.deg), "[deg] Target Dec"),
        ("RA_PNT", float(pointing_ra), "[deg] Pointing used for the vignetting"),
        ("DEC_PNT", float(pointing_dec), "[deg] Pointing used for the vignetting"),
        ("CATMODE", key, "Catalog mode (merged or single catalog)"),
        ("RADIUS", float(radius_arcmin), "[arcmin] Search radius around the target"),
        ("NSOURCES", n_sources, "Number of nearby sources summed"),
        ("NVAR", int(np.sum(var_flags)), "Number of sources with a variability band"),
        ("VARINFO", bool(var_info), "Variability information available"),
        ("K_FVAR", float(K_FVAR), "XMM FVAR scaling of the variability band"),
        ("NSAMPLES", int(np.shape(total_spectra[0])[0]) if n_sources else 0,
         "Simulations per source (photon-index samples)"),
        ("CREATOR", "NICER-Pointing", "Software that created this file"),
        ("DATE", Time.now().isot, "File creation date (UTC)"),
    ]:
        h[card[0]] = (card[1], card[2])
    h.add_comment("COUNTS = median of the summed simulated spectra (as in the figures).")
    h.add_comment("Without variability: COUNTS_P16 - COUNTS_P84 (photon-index uncertainty).")
    h.add_comment("With variability: COUNTS_VAR_LOW - COUNTS_VAR_HIGH (all variable")
    h.add_comment("sources at their lowest / highest observed flux).")

    # ---------- SOURCES ---------- #
    name_col, ra_col, dec_col = _source_columns(table, key)
    flux_col = "flux" if key == "merged" else dict_cat.dictionary_catalog[key]["flux_obs"]
    names = np.array([str(n).strip() for n in table[name_col]])
    ra = np.array(table[ra_col], dtype=float)
    dec = np.array(table[dec_col], dtype=float)
    pointing = SkyCoord(ra=pointing_ra * u.deg, dec=pointing_dec * u.deg)
    off_axis = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).separation(pointing).arcmin
    flux = np.array(table[flux_col], dtype=float)
    pho_index = np.array(table["Photon Index"], dtype=float)
    pho_index_err = np.array(table["Photon Index Error"], dtype=float)
    catalog = (np.array([str(c) for c in table["origin_catalog"]])
               if "origin_catalog" in table.colnames else np.full(n_sources, key))
    nan = np.full(n_sources, np.nan)
    var_ratio = np.array(table["var_ratio"], dtype=float) if var_info else nan
    flux_min = np.array(table["flux_min"], dtype=float) if var_info else nan
    flux_max = np.array(table["flux_max"], dtype=float) if var_info else nan
    short_term_var = (np.array(table["short_term_var"], dtype=bool)
                      if "short_term_var" in table.colnames else np.zeros(n_sources, dtype=bool))
    name_len = max([len(n) for n in names] + [1])
    cat_len = max([len(c) for c in catalog] + [1])

    sources_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="NAME", format=f"{name_len}A", array=names),
        fits.Column(name="CATALOG", format=f"{cat_len}A", array=catalog),
        fits.Column(name="RA", format="D", unit="deg", array=ra),
        fits.Column(name="DEC", format="D", unit="deg", array=dec),
        fits.Column(name="OFFAXIS", format="D", unit="arcmin", array=off_axis),
        fits.Column(name="VIGNETTING", format="D", array=np.array(table["vignetting_factor"], dtype=float)),
        fits.Column(name="FLUX", format="D", unit="erg/cm2/s", array=flux),
        fits.Column(name="PHO_INDEX", format="D", array=pho_index),
        fits.Column(name="PHO_INDEX_ERR", format="D", array=pho_index_err),
        fits.Column(name="DEFAULT_GAMMA", format="L",
                    array=(pho_index == 1.7) & (pho_index_err == 0.0)),
        fits.Column(name="NH", format="D", unit="cm-2", array=np.array(table["Nh"], dtype=float)),
        fits.Column(name="COUNT_RATE", format="D", unit="count/s",
                    array=np.array(table["count_rate"], dtype=float)),
        fits.Column(name="VAR_FLAG", format="L", array=var_flags),
        fits.Column(name="VAR_RATIO", format="D", array=var_ratio),
        fits.Column(name="FLUX_MIN", format="D", unit="erg/cm2/s", array=flux_min),
        fits.Column(name="FLUX_MAX", format="D", unit="erg/cm2/s", array=flux_max),
        fits.Column(name="SHORT_TERM_VAR", format="L", array=short_term_var),
    ], name="SOURCES")
    sources_hdu.header.add_comment("DEFAULT_GAMMA: photon index set to 1.7 (error 0) after a bad or failed fit.")
    sources_hdu.header.add_comment("VAR_FLAG: source has a variability band (VAR_RATIO > 1 or SHORT_TERM_VAR).")
    sources_hdu.header.add_comment("SHORT_TERM_VAR: XMM VAR_FLAG (variability within one observation).")

    # ---------- SOURCE_SPECTRA ---------- #
    per_source = np.array([five_spectra(total_spectra[i], ratio_lower[i], ratio_upper[i])
                           for i in range(n_sources)]).reshape(n_sources, 5, len(pi_channel))
    vec = f"{len(pi_channel)}D"
    source_spectra_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="NAME", format=f"{name_len}A", array=names),
        fits.Column(name="COUNTS", format=vec, unit="count", array=per_source[:, 0]),
        fits.Column(name="COUNTS_P16", format=vec, unit="count", array=per_source[:, 1]),
        fits.Column(name="COUNTS_P84", format=vec, unit="count", array=per_source[:, 2]),
        fits.Column(name="COUNTS_VAR_LOW", format=vec, unit="count", array=per_source[:, 3]),
        fits.Column(name="COUNTS_VAR_HIGH", format=vec, unit="count", array=per_source[:, 4]),
    ], name="SOURCE_SPECTRA")
    source_spectra_hdu.header.add_comment("One row per source (same order as SOURCES), one value per PI channel.")

    pha_path = os.path.join(
        os_dictionary["catalog_directory"],
        f"{key}_nearby_sources_{object_data['object_name']}.pha".replace(" ", "_"),
    ).replace("\\", "/")
    fits.HDUList([fits.PrimaryHDU(), spectrum_hdu, sources_hdu, source_spectra_hdu]).writeto(pha_path, overwrite=True)
    log.log(RESULT, f"\nSpectrum file of the nearby sources has been created in {colored(pha_path, 'blue')}")
    return pha_path
