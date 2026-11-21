# ------------------------------ #
        # Python's packages

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from typing import Dict, List

import os
import sys
import numpy as np

import catalog_information as dict_cat
from function.log_function import log, RESULT

# ------------------------------ #

"""
Writes summary.txt, a short readable overview of one run: settings, nearby
sources, pointing results, variable sources, sources with the default photon
index and the output files. The full details of the run are in run.log.
"""


def _name_column(table: Table, key: str) -> str:
    return "iau_name" if key == "merged" else dict_cat.dictionary_coord[key]["iauname"]


def _flux_column(key: str) -> str:
    return "flux" if key == "merged" else dict_cat.dictionary_catalog[key]["flux_obs"]


def write_summary(simulation_data: Dict, key: str, settings: Dict, merge_summary: List[str],
                  nominal: Dict, vector_dictionary: Dict, optimal_index: int) -> str:
    """
    Writes summary.txt in the output folder and prints the variability and
    photon-index counts to the console.

    Args:
        simulation_data (Dict): Simulation parameters, paths and the nearby-sources table.
        key (str): Catalog key ('merged', 'XMM', ...).
        settings (Dict): Run settings: 'radius' (arcmin) and 'exp_time' (s).
        merge_summary (List[str]): Merge description from MultiCatalogPipeline (merged mode only).
        nominal (Dict): Nominal-pointing results from c_f.nominal_pointing_info ('snr', 'src_rate').
        vector_dictionary (Dict): Pointing grid results from c_f.calculate_opti_point.
        optimal_index (int): Index of the optimal pointing in the grid.

    Returns:
        str: Path of summary.txt.
    """
    object_data = simulation_data["object_data"]
    telescop_data = simulation_data["telescop_data"]
    output_dir = simulation_data["os_dictionary"]["catalog_directory"]
    table = simulation_data["nearby_sources_table"]
    names = [str(n).strip() for n in table[_name_column(table, key)]]
    catalogs = ([str(c) for c in table["origin_catalog"]] if "origin_catalog" in table.colnames
                else [key] * len(table))
    target = object_data["object_position"]

    lines = ["NICER-Pointing run summary", "=" * 60,
             f"Date (UTC):        {Time.now().iso[:19]}",
             f"Command:           python {' '.join(sys.argv)}",
             f"Target:            {object_data['object_name']} "
             f"(RA {target.ra.deg:.5f}, Dec {target.dec.deg:.5f} deg)",
             f"Target rate:       {object_data['count_rate']} cts/s",
             f"Catalog mode:      {key}",
             f"Search radius:     {settings['radius']} arcmin",
             f"Exposure:          {settings['exp_time']} s",
             ""]

    # ---------- nearby sources ---------- #
    lines += ["Nearby sources", "-" * 60, f"{len(table)} sources within {settings['radius']} arcmin."]
    if merge_summary:
        lines += merge_summary
    lines.append("")

    # ---------- pointing ---------- #
    opt_ra = vector_dictionary["SampleRA"][optimal_index]
    opt_dec = vector_dictionary["SampleDEC"][optimal_index]
    offset = target.separation(SkyCoord(ra=opt_ra * u.deg, dec=opt_dec * u.deg)).arcmin
    # Same interpolation as c_f.vignetting_factor.
    vignetting = np.interp(offset, telescop_data["EffArea"], telescop_data["OffAxisAngle"])
    lines += ["Pointing", "-" * 60,
              f"Nominal (on target): S/N {float(nominal['snr']):.1f}, "
              f"nearby sources {float(nominal['src_rate']):.4f} cts/s",
              f"Optimal:             RA {opt_ra:.5f}, Dec {opt_dec:.5f} deg",
              f"                     S/N {float(vector_dictionary['SNR'][optimal_index]):.1f}, "
              f"target {float(vector_dictionary['PSRrates'][optimal_index]):.4f} cts/s, "
              f"nearby sources {float(vector_dictionary['SRCrates'][optimal_index]):.4f} cts/s",
              f"                     {offset:.3f} arcmin from the target (vignetting {float(vignetting):.3f})",
              ""]

    # ---------- variability ---------- #
    lines += ["Variable sources", "-" * 60]
    n_between = n_within = n_var = 0
    if {"var_ratio", "flux_min", "flux_max"} <= set(table.colnames):
        var_ratio = np.array(table["var_ratio"], dtype=float)
        within = (np.array(table["short_term_var"], dtype=bool) if "short_term_var" in table.colnames
                  else np.zeros(len(table), dtype=bool))
        between = var_ratio > 1.0
        variable = np.where(between | within)[0]
        n_between, n_within, n_var = int(between.sum()), int(within.sum()), len(variable)
        lines += ["Between observations: var_ratio > 1 (flux changed beyond the error bars).",
                  "Within an observation: XMM VAR_FLAG (FVAR used for the flux range).",
                  "Flux range = band used for the variability envelope (erg/cm2/s).", ""]
        if n_var:
            lines.append(f"  {'Source':<24} {'Catalog':<8} {'Type':<16} {'var_ratio':>9}  "
                         f"{'flux':>9}  {'flux range':>21}")
            for i in variable[np.argsort(-var_ratio[variable])]:
                kind = ("between + within" if between[i] and within[i]
                        else "between" if between[i] else "within")
                lines.append(f"  {names[i]:<24} {catalogs[i]:<8} {kind:<16} {var_ratio[i]:>9.2f}  "
                             f"{float(table[_flux_column(key)][i]):>9.2e}  "
                             f"{float(table['flux_min'][i]):.2e} - {float(table['flux_max'][i]):.2e}")
        else:
            lines.append("  None.")
    else:
        lines.append("  Not available in single-catalog mode (use the merged mode).")
    lines.append("")

    # ---------- default photon index ---------- #
    pho_index = np.array(table["Photon Index"], dtype=float)
    pho_index_err = np.array(table["Photon Index Error"], dtype=float)
    default = np.where((pho_index == 1.7) & (pho_index_err == 0.0))[0]
    lines += ["Default photon index", "-" * 60,
              "Photon index set to 1.7 (error 0) after a bad (error > 2) or failed fit.",
              "Fitted values of the bad fits are in run.log ('Bad fit for ...');",
              "a source without such a line had a failed fit.", ""]
    lines += [f"  {names[i]:<24} {catalogs[i]}" for i in default] or ["  None."]
    lines.append("")

    # ---------- output files ---------- #
    lines += ["Output files", "-" * 60]
    for root, dirs, files in os.walk(output_dir):
        dirs.sort()
        rel = os.path.relpath(root, output_dir)
        files = [f for f in files if not f.startswith(".")]
        pngs = [f for f in files if f.endswith(".png")]
        others = sorted(f for f in files if not f.endswith(".png"))
        lines += [f"  {f if rel == '.' else os.path.join(rel, f)}" for f in others]
        if pngs:
            lines.append(f"  {rel}/ ({len(pngs)} plots)" if rel != "." else f"  ({len(pngs)} plots)")
    lines.append("  summary.txt")

    summary_path = os.path.join(output_dir, "summary.txt").replace("\\", "/")
    with open(summary_path, "w") as file:
        file.write("\n".join(lines) + "\n")

    if n_var or "var_ratio" in table.colnames:
        log.info(f"Variability: {n_var} of {len(table)} sources "
                 f"({n_between} between observations, {n_within} within an observation)")
    if len(default):
        log.info(f"Photon index set to 1.7 for {len(default)} sources (bad or failed fit)")
    log.log(RESULT, f"Run summary: {summary_path}")
    return summary_path
