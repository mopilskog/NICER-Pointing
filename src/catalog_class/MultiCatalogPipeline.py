# --------------- Packages --------------- #

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.units import Quantity
from dataclasses import dataclass
from termcolor import colored
from typing import Dict, List, Optional

import numpy as np
import os
import shlex
import subprocess
import sys

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
    "XMM":     {"class": XmmCatalog,     "directory_name": "4XMM_DR13",
                "catalog_filename": "4XMM_slim_DR13cat_v1.0.fits"},
    "Swift":   {"class": SwiftCatalog,   "directory_name": "Swift",
                "catalog_filename": "Swift.fits"},
    "eRASS1":  {"class": eRASS1Catalog,  "directory_name": "eRASS1",
                "catalog_filename": "eRASS1.fits"},
    "Chandra": {"class": ChandraCatalog, "directory_name": "Chandra",
                "catalog_filename": "Chandra.fits"},
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
        return self._run_merged()

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

    # ---------------- merged-mode helpers ---------------- #

    def _pointings_path(self) -> str:
        """Where Master_source_cone.fits and per-catalog cross-match files live."""
        modeling_file_path = self.simulation_data["os_dictionary"]["modeling_file_path"]
        path = os.path.join(modeling_file_path, "Pointings").replace("\\", "/")
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def _setup_merged_directory(self) -> Dict:
        """Create the merged/ output subtree and return the os_dictionary that
        downstream calls (write_fits_file, total_plot_spectra, ...) should use."""
        seed = self.simulation_data["os_dictionary"]
        modeling_file_path = seed["modeling_file_path"]

        merged_dir = os.path.join(modeling_file_path, "merged").replace("\\", "/")
        merged_img = os.path.join(merged_dir, "img").replace("\\", "/")
        merged_closest = os.path.join(merged_dir, "closest_catalog").replace("\\", "/")
        if not os.path.exists(merged_dir):
            os.mkdir(merged_dir)
            os.mkdir(merged_img)
            os.mkdir(merged_closest)

        return {
            "active_workflow": seed["active_workflow"],
            "catalog_datapath": seed["catalog_datapath"],
            "modeling_file_path": modeling_file_path,
            "_var_sources_path": seed["plot_var_sources_path"],
            "plot_var_sources_path": seed["plot_var_sources_path"],
            "catalog_directory": merged_dir,
            "cloesest_dataset_path": merged_closest,
            "img": merged_img,
            "stilts_software_path": seed["stilts_software_path"],
            "topcat_software_path": seed["topcat_software_path"],
        }

    def _catalog_path_for(self, cat_key: str) -> str:
        catalog_datapath = self.simulation_data["os_dictionary"]["catalog_datapath"]
        filename = CATALOG_REGISTRY[cat_key]["catalog_filename"]
        return os.path.join(catalog_datapath, filename).replace("\\", "/")

    def _build_master_source_cone(self) -> None:
        """Spatially crop Master_source.fits to the FoV and tmatch2 each per-catalog
        FITS against it. Equivalent to the inline code in main.py:1080-1106 — lifted
        here so it runs *before* the merge loop instead of after the catalog
        dispatch. Outputs land in <modeling_file_path>/Pointings/ as before."""
        seed = self.simulation_data["os_dictionary"]
        catalog_datapath = seed["catalog_datapath"]
        stilts_software_path = seed["stilts_software_path"]
        output_name = self._pointings_path()

        obj = self.simulation_data["object_data"]
        ra = obj["object_position"].ra.value
        dec = obj["object_position"].dec.value
        radius_arcmin = self.radius.to(u.arcmin).value

        master_source_path = os.path.join(catalog_datapath, "Master_source.fits").replace("\\", "/")
        master_cone_path = os.path.join(output_name, "Master_source_cone.fits").replace("\\", "/")

        print(colored(
            f"\nBuilding Master_source_cone.fits around RA={ra:.4f}, "
            f"Dec={dec:.4f}, radius={radius_arcmin}'", 'yellow'
        ))
        cmd = (
            f"java -jar {stilts_software_path} tpipe {master_source_path} cmd='"
            f"select skyDistanceDegrees({ra},{dec},MS_RA,MS_DEC)*60<{radius_arcmin} "
            f"' out={master_cone_path}"
        )
        subprocess.run(shlex.split(cmd))

        print(colored("Cross-matching each catalog with the master-source cone...", 'yellow'))
        for cat in dict_cat.catalogs:
            path_in = os.path.join(catalog_datapath, cat).replace("\\", "/")
            path_out = os.path.join(output_name, cat).replace("\\", "/")
            values2 = "eRASS_IAUNAME" if cat == "eRASS1" else f"{cat}_IAUNAME"
            cmd = (
                f"java -jar {stilts_software_path} tmatch2 matcher=exact "
                f"in1='{master_cone_path}' in2='{path_in}.fits' out='{path_out}.fits' "
                f"values1='{cat}' values2='{values2}' find=all progress=none"
            )
            subprocess.run(shlex.split(cmd))

    def _load_master_source_cone(self) -> Table:
        path = os.path.join(self._pointings_path(), "Master_source_cone.fits").replace("\\", "/")
        with fits.open(path, memmap=True) as hdul:
            return Table(hdul[1].data)

    def _lookup_ms_ids(self, catalog_instance, cat_key: str, master_cone: Table) -> Dict[int, Optional[int]]:
        """For each row in the catalog's nearby_sources_table, look up its IAU
        name in master_cone[<cat_key>]. On a name miss, fall back to spatial
        matching against (MS_RA, MS_DEC). Returns dict mapping nearby-table row
        index -> MS_ID (or None if neither lookup succeeds)."""
        iauname_col = dict_cat.dictionary_coord[cat_key]["iauname"]
        ra_col = dict_cat.dictionary_coord[cat_key]["right_ascension"]
        dec_col = dict_cat.dictionary_coord[cat_key]["declination"]

        # Name-based index. Master cone uses '--' as a "not detected" placeholder
        # alongside empty strings; both should be filtered out.
        name_to_ms_id = {}
        for row in master_cone:
            name = str(row[cat_key]).strip()
            if name and name != "--":
                name_to_ms_id[name] = int(row["MS_ID"])

        # Pre-build SkyCoords for the master cone for spatial fallback.
        master_position = SkyCoord(
            ra=np.array(master_cone["MS_RA"]) * u.deg,
            dec=np.array(master_cone["MS_DEC"]) * u.deg,
        )

        # Spatial fallback is restricted in two ways:
        # 1. Threshold tightened to 3" (down from 5") to reduce accidental
        #    cross-source folding in crowded fields.
        # 2. Only allow the fallback if the master cone row's column for THIS
        #    catalog already has a real name. That makes spatial fallback an
        #    intra-catalog naming-rescue mechanism (e.g. trailing whitespace,
        #    minor version differences), not a cross-catalog assignment tool.
        #    Cross-catalog identification stays the master cone's job.
        SPATIAL_FALLBACK_ARCSEC = 3.0

        nearby_table = catalog_instance.nearby_sources_table
        result = {}
        for idx in range(len(nearby_table)):
            name = str(nearby_table[iauname_col][idx]).strip()
            ms_id = name_to_ms_id.get(name)

            if ms_id is None:
                # Name didn't match. Try spatial fallback: find the closest master
                # cone position to this source.
                src_pos = SkyCoord(
                    ra=float(nearby_table[ra_col][idx]) * u.deg,
                    dec=float(nearby_table[dec_col][idx]) * u.deg,
                )
                seps = src_pos.separation(master_position).arcsec
                closest = int(np.argmin(seps))
                if seps[closest] < SPATIAL_FALLBACK_ARCSEC:
                    # Check guard 2: master cone must already record a name
                    # for THIS catalog at this row. Otherwise we'd be making
                    # a cross-catalog assignment, which we leave to the master
                    # cone's own bookkeeping.
                    master_name_here = str(master_cone[cat_key][closest]).strip()
                    if master_name_here and master_name_here != "--":
                        ms_id = int(master_cone["MS_ID"][closest])
                        print(colored(
                            f"  {name} (from {cat_key}) linked to MS_ID={ms_id} via "
                            f"{seps[closest]:.2f}\" position match (master cone "
                            f"already had '{master_name_here}' for {cat_key} at this MS_ID)",
                            'yellow'
                        ))

            result[idx] = ms_id
        return result

    def _add_new_row(self,
                     merged_rows: List[Dict],
                     merged_model_dict: Dict,
                     catalog_instance,
                     cat_key: str,
                     row_idx: int,
                     ms_id: int) -> None:
        """Normalize one source from a per-catalog table into the merged schema and
        append. Also copies the corresponding entry from the catalog class's
        model_dictionary into merged_model_dict, re-keyed to the merged-table
        index so c_f.count_rates can still iterate src_0, src_1, ..."""
        nearby_table = catalog_instance.nearby_sources_table
        coord = dict_cat.dictionary_coord[cat_key]
        cat_info = dict_cat.dictionary_catalog[cat_key]

        iauname = str(nearby_table[coord["iauname"]][row_idx]).strip()
        ra = float(nearby_table[coord["right_ascension"]][row_idx])
        dec = float(nearby_table[coord["declination"]][row_idx])
        pho_idx = float(nearby_table["Photon Index"][row_idx])
        pho_idx_err = (float(nearby_table["Photon Index Error"][row_idx])
                       if "Photon Index Error" in nearby_table.colnames else 0.0)
        nh = float(nearby_table["Nh"][row_idx])
        flux = float(nearby_table[cat_info["flux_obs"]][row_idx])

        merged_idx = len(merged_rows)
        merged_rows.append({
            "MS_ID": int(ms_id),
            "origin_catalog": cat_key,
            "RA": ra,
            "DEC": dec,
            "iau_name": iauname,
            "Photon Index": pho_idx,
            "Photon Index Error": pho_idx_err,
            "Nh": nh,
            "flux": flux,
        })

        # Re-key the per-catalog model_dictionary entry to the merged-table index.
        src_key = f"src_{row_idx}"
        if src_key in catalog_instance.model_dictionary:
            merged_model_dict[f"src_{merged_idx}"] = catalog_instance.model_dictionary[src_key]

    def _append_user_table(self,
                           merged_rows: List[Dict],
                           merged_model_dict: Dict) -> None:
        """Append user-defined sources to the merged table after the merge loop.

        Required columns in `self.user_table`: ``Name``, ``RA``, ``DEC``, ``Flux``.
        Optional columns: ``Photon Index``, ``Nh``, ``Var Value``.

        When ``Photon Index`` is absent, defaults to 1.7. When ``Nh`` is absent,
        the value is looked up from ``NHI_HPX.fits`` at the source position,
        matching what the catalog classes do for sources missing from x2a.

        User sources are tagged ``origin_catalog="user"`` and ``MS_ID=-1``.
        """
        if self.user_table is None or len(self.user_table) == 0:
            return

        REQUIRED = ["Name", "RA", "DEC", "Flux"]
        missing = [c for c in REQUIRED if c not in self.user_table.colnames]
        if missing:
            raise ValueError(
                f"User table is missing required columns: {missing}. "
                f"In merged mode, user-added sources need columns "
                f"{REQUIRED} (case-sensitive). Optional columns: "
                f"'Photon Index', 'Nh', 'Var Value'."
            )

        has_pho = "Photon Index" in self.user_table.colnames
        has_nh = "Nh" in self.user_table.colnames

        # Load NHI sky map lazily — only opened if at least one user source
        # needs the default Nh lookup.
        nh_table = None

        print(colored(
            f"\nAppending {len(self.user_table)} user-added source(s) to the "
            f"merged table...", 'yellow', attrs=['underline']
        ))
        for row in self.user_table:
            name = str(row["Name"]).strip()
            ra = float(row["RA"])
            dec = float(row["DEC"])
            flux = float(row["Flux"])

            if has_pho:
                pho_idx = float(row["Photon Index"])
                if pho_idx <= 0.0:
                    pho_idx = 1.7
            else:
                pho_idx = 1.7

            if has_nh:
                nh = float(row["Nh"])
            else:
                if nh_table is None:
                    catalog_datapath = self.simulation_data["os_dictionary"]["catalog_datapath"]
                    nh_path = os.path.join(catalog_datapath, "NHI_HPX.fits").replace("\\", "/")
                    with fits.open(nh_path, memmap=True) as hdul:
                        nh_table = Table(hdul[1].data)
                distances = np.sqrt(
                    (np.array(nh_table["RA2000"]) - ra) ** 2
                    + (np.array(nh_table["DEC2000"]) - dec) ** 2
                )
                best_match = int(np.argmin(distances))
                nh = float(nh_table["NHI"][best_match])

            merged_idx = len(merged_rows)
            merged_rows.append({
                "MS_ID": -1,
                "origin_catalog": "user",
                "RA": ra,
                "DEC": dec,
                "iau_name": name,
                "Photon Index": pho_idx,
                "Photon Index Error": 0.0,
                "Nh": nh,
                "flux": flux,
            })
            merged_model_dict[f"src_{merged_idx}"] = {
                "model": "power",
                "model_value": pho_idx,
                "flux": flux,
                "column_dentsity": nh,
            }

            print(colored(
                f"  user source '{name}' at RA={ra:.4f}, DEC={dec:.4f}  "
                f"PhI={pho_idx:.3f}, Nh={nh:.3e}, Flux={flux:.3e}",
                'green'
            ))

    def _add_unmatched_row(self,
                           merged_rows: List[Dict],
                           merged_model_dict: Dict,
                           catalog_instance,
                           cat_key: str,
                           row_idx: int) -> None:
        """Source has no MS_ID match in Master_source_cone.fits. Per the design
        decision, emit a warning and add it as a new unique entry rather than
        spatially folding into existing rows."""
        iauname = str(catalog_instance.nearby_sources_table[
            dict_cat.dictionary_coord[cat_key]["iauname"]
        ][row_idx]).strip()
        print(colored(
            f"Warning: source {iauname} (from {cat_key}) has no MS_ID match in "
            f"Master_source.fits. Adding as new unique entry. Master_source.fits "
            f"may be older than the catalog snapshot.",
            'yellow'
        ))
        # Use -1 as the sentinel MS_ID for unmatched sources. They never participate
        # in MS_ID-based deduplication, so multiple unmatched rows don't collide.
        self._add_new_row(merged_rows, merged_model_dict, catalog_instance, cat_key, row_idx, ms_id=-1)

    def _run_merged(self) -> PipelineResult:
        """Iterate the priority list, instantiate each catalog class, deduplicate
        rows by MS_ID (higher-priority catalog wins), and produce a unified table
        with a uniform schema downstream code can consume."""
        merged_os_dictionary = self._setup_merged_directory()
        self._build_master_source_cone()
        master_cone = self._load_master_source_cone()

        merged_rows: List[Dict] = []
        merged_model_dict: Dict = {}
        seen_ms_ids: Dict[int, str] = {}  # MS_ID -> first cat_key that contributed it

        for cat_key in self.catalogs:
            cfg = CATALOG_REGISTRY[cat_key]
            # Each per-catalog class still wants its own subdirectory for its
            # neighborhood plot (useful diagnostic). Set os_dictionary to point
            # there for the duration of that class's __init__.
            self._setup_catalog_directory(cfg)
            catalog_path = self._catalog_path_for(cat_key)

            try:
                catalog_instance = cfg["class"](
                    catalog_path=catalog_path,
                    radius=self.radius,
                    simulation_data=self.simulation_data,
                    user_table=Table(),
                )
            except SystemExit:
                # XmmCatalog / SwiftCatalog / eRASS1Catalog / ChandraCatalog all
                # call sys.exit() if they find zero nearby sources. Catch that and
                # move on to the next catalog instead of aborting the merged run.
                print(colored(f"No nearby sources found in {cat_key}; skipping.", 'yellow'))
                continue

            row_to_ms_id = self._lookup_ms_ids(catalog_instance, cat_key, master_cone)

            for row_idx, ms_id in row_to_ms_id.items():
                if ms_id is None:
                    self._add_unmatched_row(
                        merged_rows, merged_model_dict, catalog_instance, cat_key, row_idx
                    )
                elif ms_id in seen_ms_ids:
                    iauname = str(catalog_instance.nearby_sources_table[
                        dict_cat.dictionary_coord[cat_key]["iauname"]
                    ][row_idx]).strip()
                    print(colored(
                        f"  {iauname} (MS_ID={ms_id}, {cat_key}) superseded by "
                        f"{seen_ms_ids[ms_id]}",
                        'cyan'
                    ))
                else:
                    seen_ms_ids[ms_id] = cat_key
                    self._add_new_row(
                        merged_rows, merged_model_dict, catalog_instance,
                        cat_key, row_idx, ms_id,
                    )

        # Append user-added sources (if any) post-merge. Per the design they're
        # not run through the per-catalog classes — they're added directly with
        # origin_catalog="user" and don't participate in MS_ID-based dedup.
        self._append_user_table(merged_rows, merged_model_dict)

        if not merged_rows:
            print(colored("No nearby sources retained from any catalog.", 'red'))
            sys.exit()

        # Restore the merged os_dictionary for downstream code (write_fits_file,
        # total_plot_spectra, ...). The per-catalog os_dictionaries left in place
        # by the loop are no longer relevant — outputs go under merged/.
        merged_os_dictionary["catalog_key"] = "merged"
        self.simulation_data["os_dictionary"] = merged_os_dictionary

        column_names = [
            "MS_ID", "origin_catalog", "RA", "DEC", "iau_name",
            "Photon Index", "Photon Index Error", "Nh", "flux",
        ]
        column_data = {name: [r[name] for r in merged_rows] for name in column_names}
        merged_table = Table(column_data, names=column_names)
        merged_position = SkyCoord(
            ra=merged_table["RA"], dec=merged_table["DEC"], unit=u.deg,
        )

        # In merged mode the only catalog-specific column we still need downstream
        # is `data_to_vignetting`, which now points at the uniform merged columns.
        # Band-flux entries are unused (c_f.modeling is gated off in main.py for
        # merged mode), so we omit them.
        column_dictionary = {
            "data_to_vignetting": ["RA", "DEC", "iau_name"],
        }

        print(colored(
            f"\nMerged table built: {len(merged_table)} sources from "
            f"{len(seen_ms_ids)} unique MS_IDs across {self.catalogs}.",
            'green'
        ))

        return PipelineResult(
            nearby_sources_table=merged_table,
            nearby_sources_position=merged_position,
            model_dictionary=merged_model_dict,
            column_dictionary=column_dictionary,
            key="merged",
        )
