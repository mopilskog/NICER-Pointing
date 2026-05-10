# --------------- Packages --------------- #

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.units import Quantity
from dataclasses import dataclass
from termcolor import colored
from typing import Dict, List, Optional

import numpy as np
import os

# ---------- import class ---------- #

from catalog_class.XmmClass import XmmCatalog
from catalog_class.SwiftClass import SwiftCatalog
from catalog_class.eRASS1Class import eRASS1Catalog
from catalog_class.ChandraClass import ChandraCatalog

# ---------------------------------- #

import catalog_information as dict_cat


# Per-catalog configuration. Only entries that genuinely don't belong in
# catalog_information.py live here: the catalog class to instantiate (a Python
# object reference, not catalog physics) and the output directory name (a
# workflow concern). Everything else (RA/DEC/IAU column names, band fluxes,
# energy bands) is read from dict_cat.dictionary_catalog and dict_cat.dictionary_coord.
CATALOG_REGISTRY: Dict[str, Dict] = {
    "XMM":     {"class": XmmCatalog,     "directory_name": "4XMM_DR13"},
    "Swift":   {"class": SwiftCatalog,   "directory_name": "Swift"},
    "eRASS1":  {"class": eRASS1Catalog,  "directory_name": "eRASS1"},
    "Chandra": {"class": ChandraCatalog, "directory_name": "Chandra"},
}


@dataclass
class PipelineResult:
    """Output of MultiCatalogPipeline.run().

    The fields mirror the variables main.py expects after the catalog-specific
    elif branch, so the post-pipeline call sites (count_rates, calculate_opti_point,
    vignetting_factor, modeling, jaxspec) consume them unchanged.
    """
    nearby_sources_table: Table
    nearby_sources_position: SkyCoord
    model_dictionary: Dict
    column_dictionary: Dict
    key: str


class MultiCatalogPipeline:
    """Orchestrates nearby-source retrieval across one or more X-ray catalogs.

    Single-catalog mode (one entry in `catalogs`): thin wrapper around the
    per-catalog class. Outputs are byte-equivalent to the original inline
    branch in main.py.

    Merged mode (multiple entries): not yet implemented.
    """

    def __init__(self,
                 catalogs: List[str],
                 simulation_data: Dict,
                 radius: Quantity,
                 user_table: Table,
                 catalog_path: Optional[str] = None) -> None:
        for cat in catalogs:
            if cat not in CATALOG_REGISTRY:
                raise ValueError(f"Unknown catalog key: {cat!r}")
        self.catalogs = catalogs
        self.simulation_data = simulation_data
        self.radius = radius
        self.user_table = user_table
        self.catalog_path = catalog_path

    def run(self) -> PipelineResult:
        if len(self.catalogs) == 1:
            return self._run_single_catalog(self.catalogs[0])
        raise NotImplementedError(
            "Merged mode not yet implemented. Call with a single-catalog list."
        )

    def _run_single_catalog(self, cat_key: str) -> PipelineResult:
        cfg = CATALOG_REGISTRY[cat_key]
        self._setup_catalog_directory(cfg)

        catalog_instance = cfg["class"](
            catalog_path=self.catalog_path,
            radius=self.radius,
            simulation_data=self.simulation_data,
            user_table=self.user_table,
        )

        if cat_key == "Chandra":
            return self._build_chandra_result(catalog_instance)
        return self._build_standard_result(cat_key, catalog_instance)

    def _build_standard_result(self, cat_key: str, catalog_instance) -> PipelineResult:
        # Identical pattern for XMM / Swift / eRASS1: the per-catalog class
        # exposes nearby_sources_table, nearby_sources_position, model_dictionary
        # and we build column_dictionary from dict_cat lookups.
        column_dictionary = self._build_column_dictionary(cat_key)
        self.simulation_data["os_dictionary"]["catalog_key"] = cat_key
        return PipelineResult(
            nearby_sources_table=catalog_instance.nearby_sources_table,
            nearby_sources_position=catalog_instance.nearby_sources_position,
            model_dictionary=catalog_instance.model_dictionary,
            column_dictionary=column_dictionary,
            key=cat_key,
        )

    def _build_chandra_result(self, catalog_instance) -> PipelineResult:
        # Chandra single-catalog mode prompts the user to choose between the
        # on-disk Chandra catalog and the Harvard cone-search (CS_Chandra).
        # Same prompt text and retry-on-bad-input loop as the original main.py
        # branch.
        prompt = (
            f"Which Table do you chose to follow the modeling ? "
            f"{colored('Chandra / CS_Chandra', 'magenta')}\n"
        )
        answer = str(input(prompt))
        while answer not in ("Chandra", "CS_Chandra"):
            print(f"{colored('Key error ! ', 'red')}. Please retry !")
            answer = str(input(prompt))

        if answer == "Chandra":
            nst = catalog_instance.nearby_sources_table
            nsp = catalog_instance.nearby_sources_position
            md = catalog_instance.model_dictionary
        else:  # CS_Chandra
            nst = catalog_instance.cone_search_catalog
            nsp = catalog_instance.cs_nearby_sources_position
            md = catalog_instance.cs_model_dictionary

        column_dictionary = self._build_column_dictionary(answer)
        self.simulation_data["os_dictionary"]["catalog_key"] = answer
        return PipelineResult(
            nearby_sources_table=nst,
            nearby_sources_position=nsp,
            model_dictionary=md,
            column_dictionary=column_dictionary,
            key=answer,
        )

    def _setup_catalog_directory(self, cfg: Dict) -> None:
        # Mirrors the original main.py branch: same paths, same legacy quirks
        # (the 'cloesest_dataset_path' typo, the duplicated '_var_sources_path'
        # / 'plot_var_sources_path' entries) so downstream readers keep working.
        seed = self.simulation_data.get("os_dictionary", {})
        modeling_file_path = seed["modeling_file_path"]

        cat_dir = os.path.join(modeling_file_path, cfg["directory_name"]).replace("\\", "/")
        cat_img = os.path.join(cat_dir, "img").replace("\\", "/")
        cat_closest = os.path.join(cat_dir, "closest_catalog").replace("\\", "/")
        if not os.path.exists(cat_dir):
            os.mkdir(cat_dir)
            os.mkdir(cat_img)
            os.mkdir(cat_closest)

        self.simulation_data["os_dictionary"] = {
            "active_workflow": seed["active_workflow"],
            "catalog_datapath": seed["catalog_datapath"],
            "modeling_file_path": modeling_file_path,
            "_var_sources_path": seed["plot_var_sources_path"],
            "plot_var_sources_path": seed["plot_var_sources_path"],
            "catalog_directory": cat_dir,
            "cloesest_dataset_path": cat_closest,
            "img": cat_img,
            "stilts_software_path": seed["stilts_software_path"],
            "topcat_software_path": seed["topcat_software_path"],
        }

    def _build_column_dictionary(self, cat_key: str) -> Dict:
        cat_info = dict_cat.dictionary_catalog[cat_key]
        coord = dict_cat.dictionary_coord[cat_key]
        return {
            "band_flux_obs": cat_info["band_flux_obs"],
            "band_flux_obs_err": cat_info["band_flux_obs_err"],
            "energy_band": cat_info["energy_band_center"],
            "sigma": np.array(list(np.linspace(1e-20, 1e-24, len(cat_info["energy_band_center"])))),
            "data_to_vignetting": [coord["right_ascension"], coord["declination"], coord["iauname"]],
        }
