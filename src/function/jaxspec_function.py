# ------------------------------ #
        # Python's packages
        
from astropy.io import fits
from astropy import units as u
from astropy.table import Table
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

# Silence matplotlib's texmanager INFO log about "No LaTeX-compatible font found
# for the serif fontfamily in rcParams". The rendering still uses Computer Modern
# via LaTeX itself; the message is just rcParams-introspection noise.
logging.getLogger("matplotlib.texmanager").setLevel(logging.WARNING)

# ------------------------------ #

"""
This module provides functions for astronomical data analysis and spectral modeling. It leverages the astropy library for handling astronomical data tables and FITS files, and uses jaxspec for spectral modeling.

Functions:
- cross_catalog_index: Determines the indices in a nearby sources table that correspond to sources found in a master source cone file.
- modeling_source_spectra: Generates model spectra for sources in a nearby sources table using specified instrument and model parameters.
- total_plot_spectra: Creates and saves a plot of spectral modeling data from a specified catalog and returns spectral data.
- write_txt_file: Writes the spectral modeling data into a formatted text file.

The module integrates with astropy.io, astropy.table, termcolor, tqdm, and jaxspec, among others, for its functionality. It is designed to facilitate the cross-catalog matching of astronomical sources, the generation and analysis of spectral data, and the efficient presentation and storage of the results.

Note:
- The module is intended for use in the field of astrophysics, particularly in the analysis of spectral data from various astronomical sources.
- It requires a proper understanding of astronomical data formats, spectral analysis techniques, and the specific instruments and models involved in the spectral data generation.
"""


def cross_catalog_index(output_name: str, key: str, iauname: str, nearby_sources_table: Table) -> List:
    """
    Identifies indices in a nearby sources table corresponding to sources from a master source cone file.

    This function is designed for astronomical data analysis, particularly for cross-catalog matching. It reads 
    a master source cone FITS file located at the specified directory, extracts source names, and then matches 
    these names with those in the provided nearby sources table to determine corresponding indices.

    Args:
        output_name (str): Directory path where the master source cone file is located.
        key (str): Key representing the catalog (e.g., 'CS_Chandra' or 'Chandra').
        iauname (str): Column name in the nearby sources table representing the IAU name of sources.
        nearby_sources_table (Table): An astropy Table object containing data about nearby sources.

    Returns:
        List[int]: A list of indices from the nearby sources table. These indices correspond to the sources 
                   found in the master source cone file.

    Note:
        - The function is specific to astronomical data analysis and is used for matching sources across different catalogs.
        - It assumes the master source cone file is in the FITS format and contains a key column that matches the 'key' argument.
    """
    
    # Merged mode: variability cross-matching is deferred (decision #6). Return
    # an empty list so modeling_source_spectra populates an empty
    # total_var_spectra; the spectrum plot no longer uses it after Commit B.
    if key == "merged":
        return []

    master_source_cone_path = os.path.join(output_name, "Master_source_cone.fits").replace("\\", "/")
    with fits.open(master_source_cone_path) as data:
        master_source_cone = Table(data[1].data)

    if key == "CS_Chandra":
        key = "Chandra"

    msc_name = [name for name in master_source_cone[key] if name != ""]
    var_index_in_nearby_sources_table = []
    for name in msc_name:
        a = nearby_sources_table[iauname]
        b = (nearby_sources_table[iauname] == name)
        if name in nearby_sources_table[iauname]:
            index_in_table = list(nearby_sources_table[iauname]).index(name)
            var_index_in_nearby_sources_table.append(index_in_table)

    return var_index_in_nearby_sources_table

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

def modeling_source_spectra(nearby_sources_table: Table, obsconfig, model, var_index, key) -> List:
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
        var_index (List[int]): Indices of variable sources in the nearby sources table.

    Returns:
        Tuple[List, List]: A tuple containing two lists: 
                            - The first list is the total spectra for all sources.
                            - The second list is the total variable spectra for variable sources.

    Note:
        - The function expects 'vignetting_factor', 'Nh', and 'Photon Index' columns in the nearby sources table.
        - Spectra are generated using the 'fakeit_for_multiple_parameters' function, based on the model and instrument.
        - The function is tailored for astrophysical spectral analysis and may require specific data formats.
    """
    print(f"\n{colored('Modeling spectra...', 'yellow', attrs=['underline'])}")
    total_spectra = []
    total_var_spectra = []
    size = 100

    for index, vignet_factor in tqdm(enumerate(nearby_sources_table["vignetting_factor"])):
        target_flux, iauname = source_flux_and_name(nearby_sources_table, index=index, key=key)

        pho_idx = float(nearby_sources_table["Photon Index"][index])
        pho_idx_err = float(nearby_sources_table["Photon Index Error"][index])

        # Bad fit: when the photon-index uncertainty is large (>2), the
        # underlying curve_fit didn't converge and the value is unreliable.
        # Use the catalog-class default (1.7) with no error propagation.
        if pho_idx_err > 2.0:
            print(colored(
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

        if index in var_index:
            total_var_spectra.append(spectra)

        total_spectra.append(spectra)

    return total_spectra, total_var_spectra


def total_plot_spectra(total_spectra: List, total_var_spectra: List, obsconfig, simulation_data: Dict, catalog_name: str) -> Dict:
    """
    Generates and saves a plot of spectral modeling data and returns a summary of the spectral data.

    This function processes spectral data from a specified catalog and creates a series of plots to visualize the spectral modeling. 
    It handles both regular and variable spectra, plotting median spectra, sum of spectra, and a summation with variability error. 
    The plot is saved in a directory specified in the `simulation_data` dictionary.

    Args:
        total_spectra (List): A list containing spectra from nearby sources.
        total_var_spectra (List): A list containing variability spectra data.
        obsconfig: An ObsConfiguration object containing the data  of the folding model, which is the link between the unfolded and folded spectra.
        simulation_data (Dict): Dictionary containing simulation parameters and paths.
        catalog_name (str): Name of the catalog used for spectral modeling.

    Returns:
        Dict: A dictionary containing energy values, median counts, and their upper and lower limits.

    Important:
        The function generates three subplots:
            1. Spectra from Nearby Sources: Plots the median spectra for each source.
            2. Sum of Spectra: Shows the sum of all spectra.
            3. Spectrum Summed with Variability Sources Error: Includes error bars representing variability.

    The plot uses logarithmic scaling on both axes, with the x-axis representing energy in keV and the y-axis representing counts.
    The final plot is saved as a PNG file and the function returns a dictionary with energy, counts, and their limits.

    Note:
        - The file is named based on the catalog name and the object of interest from the simulation data.
        - Assumes `instrument` has an attribute `out_energies` for energy values.
        - The function is designed for astrophysical spectral analysis and visualization.
        
    Here is an example of the plot create by this method:
        
    .. image:: C:/Users/plamb_v00y0i4/OneDrive/Bureau/Optimal_Pointing_Point_Code/modeling_result/PSR_J0437-4715/Chandra/img/CS_Chandra_spectral_modeling_close_to_PSR_J0437-4715.png
        :align: center
    
    Here is an example of the plot create by this method for MatchClass:
        
    .. image:: C:/Users/plamb_v00y0i4/OneDrive/Bureau/Optimal_Pointing_Point_Code/modeling_result/PSR_J0437-4715/xmmXchandra/img/xmmXchandra_spectral_modeling_close_to_PSR_J0437-4715.png
        :align: center
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

    # Variability bracketing (merged mode only). When the nearby table carries
    # var_ratio/flux_min/flux_max columns (added by MultiCatalogPipeline from
    # the MasterSource records), build per-source linear scaling ratios so the
    # nominal spectrum can be rescaled to the source's observed flux extrema.
    # Spectra scale linearly with norm at fixed alpha/Nh, so multiplying the
    # nominal spectrum by (flux_min / flux) and (flux_max / flux) gives the
    # min- and max-state forward-folded spectra without re-running fakeit.
    # var_ratio > 1 ↔ MasterSource detected variability beyond per-observation
    # errors. Sources with var_ratio == 1 (single observation or no significant
    # variability) collapse to ratio_lower == ratio_upper == 1.
    nearby_table_var = simulation_data.get("nearby_sources_table")
    has_var_info = (
        nearby_table_var is not None
        and {"var_ratio", "flux_min", "flux_max", "flux"}
            <= set(getattr(nearby_table_var, "colnames", []))
    )
    var_flags = [False] * len(total_spectra)
    var_ratios = [1.0] * len(total_spectra)
    ratio_lower = [1.0] * len(total_spectra)
    ratio_upper = [1.0] * len(total_spectra)
    if has_var_info:
        vr_col = np.array(nearby_table_var["var_ratio"], dtype=float)
        fmin_col = np.array(nearby_table_var["flux_min"], dtype=float)
        fmax_col = np.array(nearby_table_var["flux_max"], dtype=float)
        f_col = np.array(nearby_table_var["flux"], dtype=float)
        for i in range(len(total_spectra)):
            var_ratios[i] = float(vr_col[i])
            if vr_col[i] > 1.0 and f_col[i] > 0.0:
                var_flags[i] = True
                ratio_lower[i] = float(fmin_col[i] / f_col[i])
                ratio_upper[i] = float(fmax_col[i] / f_col[i])
    any_variable = any(var_flags)

    spectrum_summed_lower = 0.0
    spectrum_summed_upper = 0.0
    for i in range(len(total_spectra)):
        spectrum_summed_lower = spectrum_summed_lower + total_spectra[i] * ratio_lower[i]
        spectrum_summed_upper = spectrum_summed_upper + total_spectra[i] * ratio_upper[i]

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
        figure_1, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
        figure_1.suptitle(f"Spectral modeling close to {object_data['object_name']}\ncatalog : {catalog_name}", fontsize=20)
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

        ax0.set_title(f"Top {min(N_BRIGHTEST, len(total_spectra))} brightest + remaining sum")
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
        ax1.set_title("Sum of all source spectra")
        ax1.legend(loc='upper right', fontsize=9)

        key = simulation_data["os_dictionary"]["catalog_key"]
        img_path = os.path.join(os_dictionary['img'], f"{key}_spectral_modeling_close_to_{object_data['object_name']}.png".replace(" ", "_")).replace("\\", "/")
        plt.savefig(img_path, bbox_inches='tight')
        plt.show()

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
                    src_low = np.median(spectra * ratio_lower[i], axis=0)
                    src_up = np.median(spectra * ratio_upper[i], axis=0)
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

                ms_id = int(ms_ids[i])
                detected = list(ms_id_to_detected.get(ms_id, []))
                # Also fold in any catalogs the post-merge spatial dedup attached
                # to this row (e.g. an unmatched XMM detection that got folded
                # into this Chandra-origin row).
                if "extra_detections" in nearby_table.colnames:
                    raw_extra = str(nearby_table["extra_detections"][i]).strip()
                    if raw_extra:
                        for c in raw_extra.split(","):
                            if c and c not in detected:
                                detected.append(c)
                if ms_id == -1 and not detected:
                    detected_str = f"{origins[i]} (not in Master_source.fits)"
                elif not detected:
                    detected_str = origins[i]
                else:
                    # Ensure origin is listed first, then the others.
                    ordered = [origins[i]] + [c for c in detected if c != origins[i]]
                    detected_str = ", ".join(ordered)
                if var_flags[i]:
                    header = f"{iaunames[i]}  (var_ratio = {var_ratios[i]:.2f})"
                else:
                    header = iaunames[i]
                title = (
                    f"{header}\n"
                    f"data from: {origins[i]}  |  detected in: {detected_str}"
                )
                ax_s.set_title(title, fontsize=12)

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
            fig_sum, ax_sum = plt.subplots(1, 1, figsize=(8, 6))
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
            ax_sum.set_title(
                f"Sum of nearby-source spectra close to {object_data['object_name']}\n"
                f"({len(total_spectra)} sources from {sum_title_origins})",
                fontsize=12,
            )
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
                fig_sv, ax_sv = plt.subplots(1, 1, figsize=(8, 6))
                ax_sv.set_xlim([graph_data["min_lim_x"], graph_data["max_lim_x"]])
                ax_sv.set_ylim([graph_data["min_lim_y"], graph_data["max_lim_y"]])
                ax_sv.loglog()
                sum_low_s = np.median(spectrum_summed_lower, axis=0)
                sum_up_s = np.median(spectrum_summed_upper, axis=0)
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
                n_var = sum(1 for f in var_flags if f)
                ax_sv.set_title(
                    f"Sum of nearby-source spectra close to {object_data['object_name']}"
                    f"\n({len(total_spectra)} sources from {sum_title_origins}; "
                    f"variability bracket from {n_var} variable source"
                    f"{'s' if n_var != 1 else ''})",
                    fontsize=12,
                )
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
            fig_cat.suptitle(
                f"Per-catalog and total summed spectra close to "
                f"{object_data['object_name']}",
                fontsize=18,
            )
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
            cat_ax_left.set_title("Sum per origin catalog")
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
            cat_ax_right.set_title("Total summed spectrum")
            cat_ax_right.legend(loc='upper right', fontsize=9)

            cat_img_path = os.path.join(
                os_dictionary['img'],
                f"{key}_per_catalog_sum_close_to_{object_data['object_name']}.png".replace(" ", "_"),
            ).replace("\\", "/")
            plt.savefig(cat_img_path, bbox_inches='tight')
            plt.close(fig_cat)

    # .txt envelope (used by write_txt_file): median + 16th / 84th percentile
    # of the 100 total summed spectra at each energy bin.
    data = {
        "Energy": energies,
        "Counts": np.median(spectrum_summed, axis=0),
        "Upper limit": np.percentile(spectrum_summed, 84, axis=0),
        "Lower limit": np.percentile(spectrum_summed, 16, axis=0),
    }

    return data


def write_txt_file(simulation_data: Dict, data: Dict) -> None:
    """
    Writes spectral modeling data into a formatted text file for analysis and documentation.

    This function takes spectral data and simulation parameters, and writes this information to a text file in a 
    specified directory. The output file is formatted to include a header and rows of data, presenting energy values, 
    count rates, and their upper and lower limits in a readable format. This is useful for subsequent analysis or documentation.

    Args:
        simulation_data (Dict): A dictionary containing simulation parameters and paths, including output directory.
        data (Dict): A dictionary with spectral data. Expected keys are 'Energy', 'Counts', 'Upper limit', and 'Lower limit'.

    Important:
        The text file created includes:
            - A header with column names: 'Energy', 'Counts', 'Upper limit', 'Lower limit'.
            - Data rows formatted for clarity, presenting values for each key in `data`.

    The output file is named based on the 'catalog_key' from `simulation_data` and saved in the specified directory.

    Note:
        - It is expected that `data` dictionary contains lists of equal length for each key.
        - The file path and name are derived from the 'catalog_directory' and 'catalog_key' in `simulation_data`.
    """
    
    catalog_directory = simulation_data['os_dictionary']["catalog_directory"]
    key = simulation_data["os_dictionary"]["catalog_key"]
    txt_path = os.path.join(catalog_directory, f'{key}_output_modeling_plot.txt').replace("\\", "/")
    
    data_to_txt = [
        list(data.keys())
    ]
    
    energy, counts, y_upper, y_lower = list(data.values())
    data_to_txt.extend([energy[index], counts[index], y_upper[index], y_lower[index]] for index in range(len(energy)))
    
    with open(txt_path, 'w') as file:
        header = "{:<15} {:<15} {:<15} {:<15}".format(*data_to_txt[0])
        file.write(header + "\n")

        for row in data_to_txt[1:]:
            new_row = "{:<10.5f}     {:<10.5f}       {:<10.5f}       {:<10.5f}".format(*row)
            file.write(new_row + "\n")
            
    print(f"\n{colored(f'{key}_output_modeling_plot.txt', 'yellow')} has been created in {colored(txt_path, 'blue')}")

