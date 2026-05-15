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

    def _spatial_dedup_pass(self,
                            merged_rows: List[Dict],
                            merged_model_dict: Dict,
                            stats: Dict,
                            threshold_arcsec: float = 3.0) -> tuple:
        """Post-merge spatial dedup. After the catalog loop and user-table append,
        scan pairs of merged rows for separations below ``threshold_arcsec`` and
        drop the lower-priority row if at least one of the pair is unmatched
        (MS_ID == -1).

        Two rows with different real MS_IDs are left alone: the master file
        considers them distinct physical sources by construction, so we trust
        that even when they sit close on the sky.

        Returns rebuilt (merged_rows, merged_model_dict) with kept rows only;
        ``stats["spatial_merged"]`` is incremented for each dropped row's
        origin catalog.
        """
        if not merged_rows:
            return merged_rows, merged_model_dict

        cat_priority = {c: i for i, c in enumerate(self.catalogs)}
        cat_priority["user"] = 100  # user-added sources are lowest priority

        keep = [True] * len(merged_rows)
        positions = SkyCoord(
            ra=np.array([r["RA"] for r in merged_rows]) * u.deg,
            dec=np.array([r["DEC"] for r in merged_rows]) * u.deg,
        )

        for i in range(len(merged_rows)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(merged_rows)):
                if not keep[j]:
                    continue
                sep = positions[i].separation(positions[j]).arcsec
                if sep >= threshold_arcsec:
                    continue
                is_un_i = int(merged_rows[i]["MS_ID"]) == -1
                is_un_j = int(merged_rows[j]["MS_ID"]) == -1
                if not is_un_i and not is_un_j:
                    # Both have real MS_IDs (necessarily distinct ones, since
                    # the loop already dedup'd matching MS_IDs). The master
                    # file considers them separate physical sources.
                    continue
                # At least one is unmatched. Drop the lower-priority row.
                pi = cat_priority.get(merged_rows[i]["origin_catalog"], 999)
                pj = cat_priority.get(merged_rows[j]["origin_catalog"], 999)
                drop_idx = j if pi <= pj else i
                keep_idx = i if pi <= pj else j
                dropped = merged_rows[drop_idx]
                kept = merged_rows[keep_idx]
                print(colored(
                    f"  post-merge: {dropped['iau_name']} "
                    f"({dropped['origin_catalog']}, MS_ID={dropped['MS_ID']}) "
                    f"folded into {kept['iau_name']} "
                    f"({kept['origin_catalog']}, MS_ID={kept['MS_ID']}) "
                    f"at {sep:.2f}\"",
                    'yellow',
                ))
                origin_dropped = dropped["origin_catalog"]
                if origin_dropped in stats["spatial_merged"]:
                    stats["spatial_merged"][origin_dropped] += 1
                # Preserve the fact that this source was also detected by the
                # dropped row's catalog, even though we're only keeping the
                # higher-priority row's data. The neighborhood-plot rings and
                # per-source spectrum-plot annotation use this to show every
                # catalog that detected the source.
                kept_extras = merged_rows[keep_idx].setdefault("extra_detections", [])
                if origin_dropped not in kept_extras:
                    kept_extras.append(origin_dropped)
                # Also propagate the dropped row's own extra_detections, in
                # case it already collected some before being itself folded
                # (rare with single-pass dedup, but safe).
                for c in dropped.get("extra_detections", []):
                    if c not in kept_extras:
                        kept_extras.append(c)
                keep[drop_idx] = False
                if drop_idx == i:
                    break  # i was dropped; move to next i

        # Rebuild merged_rows and merged_model_dict in order, re-keying src_*.
        new_rows: List[Dict] = []
        new_dict: Dict = {}
        for old_idx, ok in enumerate(keep):
            if not ok:
                continue
            new_idx = len(new_rows)
            new_rows.append(merged_rows[old_idx])
            old_key = f"src_{old_idx}"
            if old_key in merged_model_dict:
                new_dict[f"src_{new_idx}"] = merged_model_dict[old_key]
        return new_rows, new_dict

    def _save_merged_neighborhood_plot(self, merged_table: Table) -> None:
        """Plot every merged source on a single RA/DEC figure, colored by
        ``origin_catalog``. Same per-catalog hue scheme as the spectrum plot so
        the two figures can be cross-referenced visually."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import hsv_to_rgb
        from matplotlib.patches import Ellipse

        CATALOG_HUES = {
            "XMM":     0.60,   # blue
            "Chandra": 0.93,   # pink / magenta
            "Swift":   0.35,   # green
            "eRASS1":  0.07,   # orange
            "user":    0.75,   # purple
        }

        object_data = self.simulation_data["object_data"]
        psr_ra = object_data["object_position"].ra.deg
        psr_dec = object_data["object_position"].dec.deg
        radius_deg = self.radius.to(u.deg).value

        # Load master_source_cone so each source can be ringed by every catalog
        # that detected it (per master), not just its origin catalog.
        ms_id_to_detected: Dict[int, List[str]] = {}
        modeling_file_path = self.simulation_data["os_dictionary"]["modeling_file_path"]
        master_cone_path = os.path.join(
            modeling_file_path, "Pointings", "Master_source_cone.fits",
        ).replace("\\", "/")
        if os.path.exists(master_cone_path):
            with fits.open(master_cone_path, memmap=True) as hdul:
                mc = Table(hdul[1].data)
            cat_columns = ["XMM", "Swift", "eRASS1", "Chandra"]
            for row in mc:
                detected = []
                for c in cat_columns:
                    if c in mc.colnames:
                        v = str(row[c]).strip()
                        if v and v != "--":
                            detected.append(c)
                ms_id_to_detected[int(row["MS_ID"])] = detected

        fig, ax = plt.subplots(figsize=(9, 9))

        # Pre-compute per-catalog facecolor once so origin marker and rings stay
        # consistent.
        cat_color = {
            c: hsv_to_rgb([CATALOG_HUES.get(c, 0.5), 0.65, 0.85])
            for c in ["XMM", "Swift", "eRASS1", "Chandra", "user"]
        }

        # Origin markers (filled, with the origin catalog's color). One scatter
        # group per origin catalog so the legend has one entry each.
        catalog_order = ["XMM", "Swift", "eRASS1", "Chandra", "user"]
        present = [c for c in catalog_order if c in set(merged_table["origin_catalog"])]
        for cat in present:
            ras = [
                float(merged_table["RA"][i])
                for i in range(len(merged_table))
                if str(merged_table["origin_catalog"][i]) == cat
            ]
            decs = [
                float(merged_table["DEC"][i])
                for i in range(len(merged_table))
                if str(merged_table["origin_catalog"][i]) == cat
            ]
            ax.scatter(
                ras, decs,
                s=60, color=cat_color[cat], edgecolor='black', linewidth=0.5,
                label=f"{cat} (origin): {len(ras)}", alpha=0.85, zorder=4,
            )

        # Concentric rings: for each merged row, draw an empty circle for every
        # OTHER catalog that detected this source. Two sources of that info:
        #   - master_source_cone columns at this MS_ID (for MS_ID != -1)
        #   - the row's extra_detections column, populated when the post-merge
        #     spatial dedup folded another catalog's nearby row into this one
        # Inner marker already shows the origin. Ring sizes grow outward.
        BASE_SIZE = 60
        RING_STEP = 80
        has_extra_col = "extra_detections" in merged_table.colnames
        for i in range(len(merged_table)):
            ms_id = int(merged_table["MS_ID"][i])
            origin = str(merged_table["origin_catalog"][i])
            detected_by_master = ms_id_to_detected.get(ms_id, []) if ms_id != -1 else []
            spatial_extras = []
            if has_extra_col:
                raw = str(merged_table["extra_detections"][i]).strip()
                if raw:
                    spatial_extras = [c for c in raw.split(",") if c]
            others = []
            for c in detected_by_master + spatial_extras:
                if c != origin and c not in others:
                    others.append(c)
            if not others:
                continue
            ra_i = float(merged_table["RA"][i])
            dec_i = float(merged_table["DEC"][i])
            for ring_idx, other_cat in enumerate(others):
                ring_size = BASE_SIZE + RING_STEP * (ring_idx + 1)
                ax.scatter(
                    [ra_i], [dec_i],
                    s=ring_size,
                    facecolors='none',
                    edgecolor=cat_color.get(other_cat, (0.5, 0.5, 0.5)),
                    linewidth=1.4,
                    alpha=0.9,
                    zorder=3 - 0.01 * ring_idx,
                )

        ax.scatter(
            [psr_ra], [psr_dec],
            marker="*", s=350, color="red",
            edgecolor="black", linewidth=1,
            label=object_data["object_name"], zorder=5,
        )
        # On the sky, "radius in arcmin" is a spherical cap. In RA/DEC degree
        # space at declination ≠ 0 that's an ellipse, not a circle:
        # 1 arcmin of RA spans 1/cos(dec) arcmin of true sky distance. So the
        # cap's RA half-width is radius/cos(dec) while the DEC half-width is
        # just radius. Drawing a circle would visually under-cover the cap and
        # make in-FoV sources at high |dec| look as if they're outside.
        cos_dec = np.cos(np.deg2rad(psr_dec))
        ax.add_patch(Ellipse(
            (psr_ra, psr_dec),
            width=2 * radius_deg / cos_dec,
            height=2 * radius_deg,
            fill=False, color="gray", linestyle="--", linewidth=1,
        ))

        ax.set_xlabel("RA [deg]")
        ax.set_ylabel("DEC [deg]")
        ax.set_title(
            f"Neighbourhood of {object_data['object_name']} "
            f"(merged catalog, radius = {self.radius.to(u.arcmin).value:.1f}')\n"
            f"Filled marker = data origin.  Outer ring(s) = also detected in "
            f"that catalog per Master_source_cone."
        )
        ax.legend(loc='upper right', fontsize=9)
        ax.invert_xaxis()       # RA increases to the left, astronomy convention
        ax.set_aspect('equal', adjustable='datalim')

        img_dir = self.simulation_data["os_dictionary"]["img"]
        target_name = object_data["object_name"].replace(" ", "_")
        out_path = os.path.join(
            img_dir, f"merged_neighbourhood_of_{target_name}.png"
        ).replace("\\", "/")
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(colored(f"Merged neighborhood plot saved: {out_path}", 'green'))

    def _write_manifest(self,
                        stats: Dict,
                        merged_rows: List[Dict],
                        seen_ms_ids: Dict[int, str]) -> None:
        """Write a readable summary of the merge to ``merged/manifest.txt``.

        Per-catalog: how many sources added (with MS_ID vs unmatched), how many
        superseded by which catalog. Plus skipped catalogs and the user-source
        count.
        """
        output_dir = self.simulation_data["os_dictionary"]["catalog_directory"]
        manifest_path = os.path.join(output_dir, "manifest.txt").replace("\\", "/")

        target = self.simulation_data["object_data"]["object_name"]
        radius_arcmin = self.radius.to(u.arcmin).value
        n_user = sum(1 for r in merged_rows if r["origin_catalog"] == "user")
        total_unmatched = sum(stats["added_unmatched"].values())

        lines = []
        lines.append("Merged-mode manifest")
        lines.append("=" * 60)
        lines.append(f"Target:           {target}")
        lines.append(f"FoV radius:       {radius_arcmin:.1f} arcmin")
        lines.append(f"Priority order:   {' > '.join(self.catalogs)}")
        lines.append("")
        lines.append(f"Final merged table: {len(merged_rows)} sources "
                     f"({len(seen_ms_ids)} unique MS_IDs + "
                     f"{total_unmatched} unmatched + {n_user} user)")
        lines.append("")
        lines.append("Per-catalog contribution:")
        for cat in self.catalogs:
            if cat in stats["skipped_empty"]:
                lines.append(f"  {cat:<8}: 0 nearby sources found (skipped)")
                continue
            added_msid = stats["added_by_msid"][cat]
            added_un = stats["added_unmatched"][cat]
            total_added = added_msid + added_un
            sup_parts = [
                f"{n} by {by_cat}"
                for by_cat, n in stats["superseded_by"][cat].items()
                if n > 0
            ]
            sup_str = ", ".join(sup_parts) if sup_parts else "none"
            spatial_merged = stats["spatial_merged"][cat]
            spatial_str = (
                f", spatial-merged: {spatial_merged}"
                if spatial_merged > 0 else ""
            )
            lines.append(
                f"  {cat:<8}: {total_added:>3} added "
                f"({added_msid} with MS_ID, {added_un} unmatched), "
                f"superseded: {sup_str}{spatial_str}"
            )
        if n_user > 0:
            lines.append(f"  {'user':<8}: {n_user:>3} added "
                         f"(post-merge from add_sources.fits)")

        lines.append("")
        lines.append(f"Total unmatched (added with MS_ID=-1): {total_unmatched}")

        with open(manifest_path, 'w') as f:
            f.write("\n".join(lines) + "\n")
        print(colored(f"Manifest written: {manifest_path}", 'green'))

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

        # Per-catalog stats accumulated during the loop, used by the manifest.
        stats: Dict[str, Dict] = {
            "added_by_msid":   {c: 0 for c in self.catalogs},
            "added_unmatched": {c: 0 for c in self.catalogs},
            "superseded_by":   {c: {other: 0 for other in self.catalogs} for c in self.catalogs},
            "spatial_merged":  {c: 0 for c in self.catalogs},
            "skipped_empty":   [],
        }

        # Suppress per-catalog plot popups during the merged-mode catalog loop.
        # Each per-catalog class still SAVES its neighborhood plot to its own
        # subdirectory via plt.savefig(...), which we want; only the plt.show()
        # calls that open separate windows are silenced. The original is
        # restored in the finally clause so other plotting paths are unaffected.
        # To opt back in, set self.show_individual_plots = True before run().
        import matplotlib.pyplot as plt
        _original_show = plt.show
        if not getattr(self, "show_individual_plots", False):
            plt.show = lambda *a, **kw: None

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
                stats["skipped_empty"].append(cat_key)
                continue

            row_to_ms_id = self._lookup_ms_ids(catalog_instance, cat_key, master_cone)

            for row_idx, ms_id in row_to_ms_id.items():
                if ms_id is None:
                    self._add_unmatched_row(
                        merged_rows, merged_model_dict, catalog_instance, cat_key, row_idx
                    )
                    stats["added_unmatched"][cat_key] += 1
                elif ms_id in seen_ms_ids:
                    iauname = str(catalog_instance.nearby_sources_table[
                        dict_cat.dictionary_coord[cat_key]["iauname"]
                    ][row_idx]).strip()
                    print(colored(
                        f"  {iauname} (MS_ID={ms_id}, {cat_key}) superseded by "
                        f"{seen_ms_ids[ms_id]}",
                        'cyan'
                    ))
                    stats["superseded_by"][cat_key][seen_ms_ids[ms_id]] += 1
                else:
                    seen_ms_ids[ms_id] = cat_key
                    self._add_new_row(
                        merged_rows, merged_model_dict, catalog_instance,
                        cat_key, row_idx, ms_id,
                    )
                    stats["added_by_msid"][cat_key] += 1

        # Restore plt.show now that the per-catalog popups are no longer a risk.
        # The downstream merged-mode plots (spectrum, neighborhood, etc.) still
        # need to display normally.
        plt.show = _original_show
        plt.close('all')  # close any figures the catalog classes left open

        # Append user-added sources (if any) post-merge. Per the design they're
        # not run through the per-catalog classes — they're added directly with
        # origin_catalog="user" and don't participate in MS_ID-based dedup.
        self._append_user_table(merged_rows, merged_model_dict)

        # Post-merge spatial dedup: drop unmatched rows that sit within 3" of
        # another row. Handles cases that the in-loop dedup can't (e.g. a
        # matched row added after an unmatched row at the same position).
        merged_rows, merged_model_dict = self._spatial_dedup_pass(
            merged_rows, merged_model_dict, stats
        )

        if not merged_rows:
            print(colored("No nearby sources retained from any catalog.", 'red'))
            sys.exit()

        # Restore the merged os_dictionary for downstream code (write_fits_file,
        # total_plot_spectra, ...). The per-catalog os_dictionaries left in place
        # by the loop are no longer relevant — outputs go under merged/.
        merged_os_dictionary["catalog_key"] = "merged"
        self.simulation_data["os_dictionary"] = merged_os_dictionary

        # Write the manifest now that os_dictionary points back at merged/.
        self._write_manifest(stats, merged_rows, seen_ms_ids)

        # Save the merged-mode neighborhood plot (all sources, colored by
        # origin_catalog, with rings showing every catalog that detected each
        # source via the master cone OR via post-merge spatial folding).
        plot_cols = ["MS_ID", "origin_catalog", "RA", "DEC", "iau_name"]
        self._save_merged_neighborhood_plot(Table(
            {
                **{name: [r[name] for r in merged_rows] for name in plot_cols},
                "extra_detections": [",".join(r.get("extra_detections", [])) for r in merged_rows],
            },
            names=plot_cols + ["extra_detections"],
        ))

        column_names = [
            "MS_ID", "origin_catalog", "RA", "DEC", "iau_name",
            "Photon Index", "Photon Index Error", "Nh", "flux",
            "extra_detections",
        ]
        column_data = {
            name: (
                [",".join(r.get("extra_detections", [])) for r in merged_rows]
                if name == "extra_detections"
                else [r[name] for r in merged_rows]
            )
            for name in column_names
        }
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
