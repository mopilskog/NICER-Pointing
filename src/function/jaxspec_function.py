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

def norm_estimation (nearby_sources_table: Table, model, index, key):
    norm = 1e-5
    param = {
            "tbabs_1": {"N_H": nearby_sources_table["Nh"][index]/1e22},
            "powerlaw_1": {
                "alpha": nearby_sources_table["Photon Index"][index] if nearby_sources_table["Photon Index"][index] > 0.0 else 1.7,
                "norm": norm,
            }
        }
    if key == "Chandra":
        Total_flux = nearby_sources_table["flux_aper_b"][index]
        iauname = "Chandra_IAUNAME"
    elif key =="XMM":
        Total_flux = nearby_sources_table["SC_EP_8_FLUX"][index]
        iauname = "IAUNAME"
    elif key == "eRASS1":
        Total_flux = nearby_sources_table["ML_FLUX_1"][index]
        iauname = "eRASS_IAUNAME"
    elif key == "CS_Chandra":
        Total_flux = nearby_sources_table["flux_aper_b"][index]
        iauname = "Chandra_IAUNAME"
    elif key == "Swift":
        Total_flux = nearby_sources_table["Flux"][index]
        iauname = "Swift_IAUNAME"
    elif key == "merged":
        Total_flux = nearby_sources_table["flux"][index]
        iauname = "iau_name"

    target_flux = Total_flux * (u.erg / u.cm**2 / u.s)

    tolerance = 1e-15 * u.erg / (u.cm**2 * u.s)  # in erg/cm²/s
    max_iterations = 10000  # Maximum number of iterations to avoid infinite loop
    learning_rate = 1.0e7 * (u.cm**2 * u.s) / u.erg  # in (cm² s) / erg
    energies = jnp.geomspace(0.3, 10, 100)
    
    for iteration in range(max_iterations):
        energy_flux = model.energy_flux(param, energies[:-1], energies[1:], n_points=100) * (u.keV/ ( u.s * u.cm**2))
        total_computed_flux = np.sum(energy_flux)  # Sum across the energy bands
        total_computed_flux = total_computed_flux.to(u.erg / u.cm**2/ u.s)  # Sum across the energy bands

        # Calculate the difference between computed flux and target flux
        flux_difference = target_flux - total_computed_flux

        # Check if the computed flux is close enough to the target flux
        if np.abs(flux_difference) < tolerance:
            #print(f"Matching flux found with norm: {param['powerlaw_1']['norm']} at iteration {iteration}")
            break
        #learning_rate = 1.0e9 * (flux_difference / target_flux).value * (u.cm**2 * u.s) / u.erg
        norm = norm + flux_difference * learning_rate
        param['powerlaw_1']['norm'] = norm
    source = nearby_sources_table[iauname][index]
    print(f"Found norm for {source} : {norm}")
    return norm, iauname

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
        norm, iauname = norm_estimation(nearby_sources_table, model=model, index=index, key=key)

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
            # Sample from N(alpha, perr), then floor at 0. A negative alpha
            # gives a rising spectrum (E^(+|alpha|)) at high energies, which is
            # unphysical for X-ray power-law sources in 0.3-10 keV.
            alpha_samples = np.clip(
                np.random.normal(pho_idx, pho_idx_err, size),
                a_min=0.0, a_max=None,
            )
        else:
            # No error info available (e.g. failed fit path, user-added source).
            alpha_samples = np.full(size, pho_idx if pho_idx > 0.0 else 1.7)

        parameters = {
            "tbabs_1": {"N_H": np.full(size, nearby_sources_table["Nh"][index]/1e22)},
            "powerlaw_1": {
                "alpha": alpha_samples,
                "norm": np.full(size, norm),
            }
        }

        spectra = fakeit_for_multiple_parameters(instrument=obsconfig, model=model, parameters=parameters) * vignet_factor

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
    graph_data = {"min_lim_x": 0.2, "max_lim_x": 10.0,
                  "min_lim_y": 5e-3, "max_lim_y": 3e5}

    energies = obsconfig.out_energies[0]

    # Index-wise sum across sources: spectrum_summed has shape (size, n_energy);
    # row i is the total at iteration i (sum of every source's i-th realization).
    spectrum_summed = 0.0
    for item in range(len(total_spectra)):
        spectrum_summed += total_spectra[item]

    # If we're in merged mode, color sources by origin catalog. Each catalog
    # gets a hue family; within a family, the lightness varies so sources are
    # individually distinguishable while their catalog identity is obvious.
    from matplotlib.colors import hsv_to_rgb
    from matplotlib.lines import Line2D
    nearby_table = simulation_data.get("nearby_sources_table")
    has_origin = (
        nearby_table is not None
        and "origin_catalog" in getattr(nearby_table, "colnames", [])
    )
    CATALOG_HUES = {
        "XMM":     0.60,   # blue
        "Chandra": 0.93,   # pink / magenta
        "Swift":   0.35,   # green
        "eRASS1":  0.07,   # orange
        "user":    0.75,   # purple
    }

    if has_origin:
        origins = list(nearby_table["origin_catalog"])
        iaunames = [str(n).strip() for n in nearby_table["iau_name"]]
        # Per-source color: hue from catalog, value/saturation varied within
        # the catalog so sibling sources are distinguishable.
        source_colors = []
        cat_running_idx = {}
        cat_total = {c: origins.count(c) for c in set(origins)}
        for cat in origins:
            n = cat_total[cat]
            i_in_cat = cat_running_idx.get(cat, 0)
            cat_running_idx[cat] = i_in_cat + 1
            hue = CATALOG_HUES.get(cat, 0.5)
            # Spread saturation across [0.35, 0.85] within the catalog.
            sat = 0.35 + 0.5 * (i_in_cat / max(n - 1, 1))
            val = 0.95 - 0.15 * (i_in_cat / max(n - 1, 1))
            source_colors.append(hsv_to_rgb([hue, sat, val]))
    else:
        # Single-catalog mode: keep the previous pastel rainbow.
        n_sources = max(len(total_spectra), 1)
        hues = np.linspace(0.0, 0.85, n_sources) if n_sources > 1 else np.array([0.0])
        source_colors = [hsv_to_rgb([h, 0.55, 0.92]) for h in hues]
        origins = None
        iaunames = None

    # Use scienceplots' "science" style for LaTeX-formatted publication-quality
    # plots. Scoped to this function so other matplotlib output in the project
    # is unaffected.
    with plt.style.context(['science']):
        figure_1, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
        figure_1.suptitle(f"Spectral modeling close to {object_data['object_name']}\ncatalog : {catalog_name}", fontsize=20)
        figure_1.text(0.5, 0.04, 'Energy [keV]', ha='center', va='center', fontsize=16)
        figure_1.text(0.085, 0.5, 'Counts', ha='center', va='center', rotation='vertical', fontsize=16)

        for ax in axes:
            ax.set_xlim([graph_data["min_lim_x"], graph_data["max_lim_x"]])
            ax.set_ylim([graph_data["min_lim_y"], graph_data["max_lim_y"]])
            ax.loglog()

        # Left panel: per-source curve-spaghetti. Every alpha-sampled realization
        # of each source is drawn with low alpha so visual density conveys the
        # photon-index uncertainty. In merged mode the color encodes the source's
        # origin catalog; in single-catalog mode each source gets its own pastel
        # color from a generated rainbow.
        ax0 = axes[0]
        for i, spectra in enumerate(total_spectra):
            color = source_colors[i]
            for j in range(spectra.shape[0]):
                ax0.step(energies, spectra[j, :], where="post", color=color, alpha=0.05)
        ax0.set_title("Spectra from Nearby Sources")

        # Per-source legend with full opacity proxies so the colors are readable
        # even though the actual curves are drawn at alpha=0.05. In merged mode
        # the legend is grouped by catalog. With many sources the legend can be
        # tall; it goes outside the axes on the right so it doesn't obscure data.
        if has_origin:
            legend_entries = []
            # Group sources by catalog in priority order so the legend reads
            # XMM block, then Swift, eRASS1, Chandra, user.
            CATALOG_ORDER = ["XMM", "Swift", "eRASS1", "Chandra", "user"]
            ordered_indices = sorted(
                range(len(origins)),
                key=lambda i: (
                    CATALOG_ORDER.index(origins[i]) if origins[i] in CATALOG_ORDER else 99,
                    iaunames[i],
                ),
            )
            for i in ordered_indices:
                label = f"{origins[i]}: {iaunames[i]}"
                legend_entries.append(Line2D([0], [0], color=source_colors[i], lw=2, label=label))
            ax0.legend(
                handles=legend_entries, loc='center left',
                bbox_to_anchor=(1.02, 0.5), fontsize=6, ncol=1,
                frameon=False,
            )

        # Right panel: every total summed spectrum drawn as a low-alpha step
        # curve. Width of the cloud reflects the propagated photon-index
        # uncertainty.
        ax1 = axes[1]
        for j in range(spectrum_summed.shape[0]):
            ax1.step(energies, spectrum_summed[j, :], where="post", color='lightsteelblue', alpha=0.05)
        ax1.set_title("Sum of spectra")

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

            # One pink plot per source.
            for i, spectra in enumerate(total_spectra):
                fig_s, ax_s = plt.subplots(1, 1, figsize=(8, 6))
                ax_s.set_xlim([graph_data["min_lim_x"], graph_data["max_lim_x"]])
                ax_s.set_ylim([graph_data["min_lim_y"], graph_data["max_lim_y"]])
                ax_s.loglog()
                for j in range(spectra.shape[0]):
                    ax_s.step(energies, spectra[j, :], where="post",
                              color='palevioletred', alpha=0.05)
                ax_s.set_xlabel("Energy [keV]")
                ax_s.set_ylabel("Counts")

                ms_id = int(ms_ids[i])
                detected = ms_id_to_detected.get(ms_id, [])
                if ms_id == -1 or not detected:
                    detected_str = f"{origins[i]} (not in Master_source.fits)"
                else:
                    detected_str = ", ".join(detected)
                title = (
                    f"{iaunames[i]}\n"
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

            # One blue plot for the total sum.
            fig_sum, ax_sum = plt.subplots(1, 1, figsize=(8, 6))
            ax_sum.set_xlim([graph_data["min_lim_x"], graph_data["max_lim_x"]])
            ax_sum.set_ylim([graph_data["min_lim_y"], graph_data["max_lim_y"]])
            ax_sum.loglog()
            for j in range(spectrum_summed.shape[0]):
                ax_sum.step(energies, spectrum_summed[j, :], where="post",
                            color='lightsteelblue', alpha=0.05)
            ax_sum.set_xlabel("Energy [keV]")
            ax_sum.set_ylabel("Counts")
            ax_sum.set_title(
                f"Sum of nearby-source spectra close to {object_data['object_name']}\n"
                f"({len(total_spectra)} sources from "
                f"{', '.join(sorted(set(origins)))})",
                fontsize=12,
            )
            sum_img_path = os.path.join(
                os_dictionary['img'],
                f"{key}_sum_close_to_{object_data['object_name']}.png".replace(" ", "_"),
            ).replace("\\", "/")
            plt.savefig(sum_img_path, bbox_inches='tight')
            plt.close(fig_sum)

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

