"""Diagnostic script for the merged-mode output.

Runs a set of consistency checks against a merged-mode result on disk:

  python verify_merged.py <target_dir_name>

e.g.  python verify_merged.py PSR_J0437-4715

Checks (a) through (g):
  a. No duplicate MS_IDs (except -1) in the merged table.
  b. Origin catalog matches IAU name prefix.
  c. For non-unmatched rows, master_source_cone[<origin>] contains the IAU
     name. (Rows whose origin_catalog column doesn't match a name in the
     master cone but ARE within 3" of a master position are spatial-fallback
     links and counted separately.)
  d. Photon Index / Photon Index Error / Nh match per-catalog single-mode runs.
  e. Priority order respected: when an MS_ID is detected by multiple catalogs
     in the master cone, the merged row's origin_catalog is the highest one
     present in the priority list.
  f. flux column matches the origin catalog's broad-band flux (catalog file).
  g. All "spatial-fallback" links are within 3" of the MS_RA/MS_DEC.

Each check prints PASS / FAIL with details. The script exits with code 0 if
all checks pass, 1 otherwise.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table


PRIORITY = ["XMM", "Swift", "eRASS1", "Chandra"]

# Mapping from origin_catalog tag → (master-cone column, name-prefix(es) the
# IAU column is expected to use, single-catalog FITS path stub, IAU column
# name in the single-catalog file, flux column name in the single-catalog
# file).
CATALOG_META = {
    "XMM": {
        "master_col": "XMM",
        "name_prefixes": ("4XMM ",),
        "single_dir": "4XMM_DR13",
        "single_file": "XMM_nearby_sources_table.fits",
        "iauname_col": "IAUNAME",
        "flux_col": "SC_EP_8_FLUX",
    },
    "Swift": {
        "master_col": "Swift",
        "name_prefixes": ("LSXPS ", "SXPS "),
        "single_dir": "Swift",
        "single_file": "Swift_nearby_sources_table.fits",
        "iauname_col": "Swift_IAUNAME",
        "flux_col": "Flux",
    },
    "eRASS1": {
        "master_col": "eRASS1",
        "name_prefixes": ("1eRASS ", "eRASS "),
        "single_dir": "eRASS1",
        "single_file": "eRASS1_nearby_sources_table.fits",
        "iauname_col": "eRASS_IAUNAME",
        "flux_col": "ML_FLUX_1",
    },
    "Chandra": {
        "master_col": "Chandra",
        "name_prefixes": ("2CXO ",),
        "single_dir": "Chandra",
        "single_file": "Chandra_nearby_sources_table.fits",
        "iauname_col": "Chandra_IAUNAME",
        "flux_col": "flux_aper_b",
    },
}

SPATIAL_FALLBACK_ARCSEC = 3.0


def color(s: str, c: str) -> str:
    codes = {"red": 31, "green": 32, "yellow": 33, "blue": 34, "cyan": 36, "bold": 1}
    return f"\033[{codes.get(c, 0)}m{s}\033[0m"


def print_header(check: str, title: str) -> None:
    print()
    print(color("=" * 78, "bold"))
    print(color(f"Check {check} — {title}", "bold"))
    print(color("=" * 78, "bold"))


def result(passed: bool, msg: str) -> bool:
    tag = color(" PASS ", "green") if passed else color(" FAIL ", "red")
    print(f"{tag} {msg}")
    return passed


def check_a_no_duplicate_ms_ids(merged: Table) -> bool:
    print_header("a", "No duplicate MS_IDs (except -1)")
    real = [int(x) for x in merged["MS_ID"] if int(x) != -1]
    dupes = [v for v in set(real) if real.count(v) > 1]
    if dupes:
        return result(False, f"Duplicate MS_IDs found: {dupes}")
    return result(True, f"{len(real)} unique MS_IDs, no duplicates")


def check_b_name_prefix_matches_origin(merged: Table) -> bool:
    print_header("b", "Origin catalog matches IAU name prefix")
    fails = []
    for i, row in enumerate(merged):
        origin = str(row["origin_catalog"])
        name = str(row["iau_name"]).strip()
        if origin not in CATALOG_META:
            if origin != "user":
                fails.append((i, name, origin, "unknown origin"))
            continue
        expected = CATALOG_META[origin]["name_prefixes"]
        if not any(name.startswith(p) for p in expected):
            fails.append((i, name, origin, f"expected prefix in {expected}"))
    if fails:
        print(f"  {len(fails)} mismatch(es):")
        for i, n, o, msg in fails:
            print(f"    row {i:>3}  origin={o:<8}  name='{n}'  ← {msg}")
        return result(False, f"{len(fails)} prefix mismatches")
    return result(True, f"All {len(merged)} rows have origin-consistent name prefixes")


def check_c_name_in_master_or_spatial_fallback(merged: Table, master_cone: Table) -> bool:
    print_header("c", "MS_ID-matched rows: name appears in master_cone[origin] (or spatial-fallback)")
    master_position = SkyCoord(
        ra=np.array(master_cone["MS_RA"]) * u.deg,
        dec=np.array(master_cone["MS_DEC"]) * u.deg,
    )
    master_names_by_cat = {
        c: {str(v).strip() for v in master_cone[CATALOG_META[c]["master_col"]]
            if str(v).strip() and str(v).strip() != "--"}
        for c in CATALOG_META
    }
    n_name_match = 0
    n_spatial = 0
    n_unmatched = 0
    name_failures = []
    for i, row in enumerate(merged):
        ms_id = int(row["MS_ID"])
        origin = str(row["origin_catalog"])
        name = str(row["iau_name"]).strip()
        if ms_id == -1:
            n_unmatched += 1
            continue
        if origin not in master_names_by_cat:
            continue
        if name in master_names_by_cat[origin]:
            n_name_match += 1
            continue
        # Not by name — check spatial fallback
        src_pos = SkyCoord(ra=float(row["RA"]) * u.deg, dec=float(row["DEC"]) * u.deg)
        sep = src_pos.separation(master_position).arcsec
        closest_idx = int(np.argmin(sep))
        if sep[closest_idx] < SPATIAL_FALLBACK_ARCSEC and int(master_cone["MS_ID"][closest_idx]) == ms_id:
            n_spatial += 1
        else:
            name_failures.append((i, origin, name, ms_id, float(sep[closest_idx])))
    print(f"  name match: {n_name_match}, spatial fallback: {n_spatial}, unmatched (MS_ID=-1): {n_unmatched}")
    if name_failures:
        print(f"  {len(name_failures)} rows have MS_ID but neither name nor close spatial match:")
        for i, o, n, m, s in name_failures[:10]:
            print(f"    row {i:>3}  origin={o:<8}  name='{n}'  MS_ID={m}  closest_sep={s:.2f}\"")
        return result(False, f"{len(name_failures)} unexplained MS_ID assignments")
    return result(True, "Every MS_ID-assigned row is either name-matched or spatial-fallback-matched")


def _try_load_single(target: str, cat: str) -> Optional[Table]:
    meta = CATALOG_META[cat]
    path = os.path.join("modeling_result", target, meta["single_dir"], "closest_catalog", meta["single_file"])
    if not os.path.exists(path):
        return None
    return Table.read(path)


def check_d_photon_index_parity(target: str, merged: Table) -> bool:
    print_header("d", "Photon Index / Photon Index Error / Nh parity vs per-catalog single-mode")
    overall = True
    for cat in PRIORITY:
        single = _try_load_single(target, cat)
        if single is None:
            print(color(f"  skip {cat}: no single-mode FITS at {CATALOG_META[cat]['single_dir']}/closest_catalog/", "yellow"))
            continue
        iauname_col = CATALOG_META[cat]["iauname_col"]
        single_dict = {str(r[iauname_col]).strip(): r for r in single}
        has_perr_col = "Photon Index Error" in single.colnames

        merged_rows = [r for r in merged if str(r["origin_catalog"]) == cat]
        diffs = []
        skipped_excl = 0
        for r in merged_rows:
            name = str(r["iau_name"]).strip()
            if name not in single_dict:
                # Single-mode file doesn't have this source (e.g. it was the
                # target and got excluded in this run but not the older one,
                # or the single file is stale)
                skipped_excl += 1
                continue
            s = single_dict[name]
            d = {}
            if abs(float(r["Photon Index"]) - float(s["Photon Index"])) > 1e-6:
                d["PhI"] = (float(r["Photon Index"]), float(s["Photon Index"]))
            if has_perr_col:
                if abs(float(r["Photon Index Error"]) - float(s["Photon Index Error"])) > 1e-6:
                    d["perr"] = (float(r["Photon Index Error"]), float(s["Photon Index Error"]))
            if abs(float(r["Nh"]) - float(s["Nh"])) > 1e-3:
                d["Nh"] = (float(r["Nh"]), float(s["Nh"]))
            if d:
                diffs.append((name, d))
        if diffs:
            print(f"  {cat}: {len(diffs)} mismatches (of {len(merged_rows)} merged rows). First 5:")
            for name, d in diffs[:5]:
                print(f"    {name}: {d}")
            overall = result(False, f"{cat}: parity failed") and overall
        else:
            note = f" ({skipped_excl} rows not in single-mode file — likely stale single file)" if skipped_excl else ""
            extra = "" if has_perr_col else " (single-mode file pre-dates 'Photon Index Error' column — perr not compared)"
            result(True, f"{cat}: {len(merged_rows) - skipped_excl} rows match{note}{extra}")
    return overall


def check_e_priority_order_respected(merged: Table, master_cone: Table) -> bool:
    print_header("e", "Priority order respected (highest-priority detector wins)")
    ms_id_to_origin = {int(r["MS_ID"]): str(r["origin_catalog"]) for r in merged if int(r["MS_ID"]) != -1}
    fails = []
    for row in master_cone:
        ms_id = int(row["MS_ID"])
        if ms_id not in ms_id_to_origin:
            continue  # this MS_ID isn't in our merged table at all
        detected_in = [
            cat for cat in PRIORITY
            if str(row[CATALOG_META[cat]["master_col"]]).strip()
            and str(row[CATALOG_META[cat]["master_col"]]).strip() != "--"
        ]
        if not detected_in:
            continue  # no detection in our 4 priority catalogs
        expected = detected_in[0]  # PRIORITY is already ordered
        actual = ms_id_to_origin[ms_id]
        if actual != expected:
            fails.append((ms_id, actual, expected, detected_in))
    if fails:
        print(f"  {len(fails)} priority violations:")
        for m, a, e, d in fails:
            print(f"    MS_ID={m}: merged.origin={a}, but {e} is higher priority (detected_in={d})")
        return result(False, f"{len(fails)} rows with priority order violated")
    return result(True, f"All {len(ms_id_to_origin)} MS_IDs follow priority order")


def check_f_flux_matches_origin_catalog(target: str, merged: Table) -> bool:
    print_header("f", "flux column matches origin catalog's broad-band flux")
    overall = True
    for cat in PRIORITY:
        single = _try_load_single(target, cat)
        if single is None:
            print(color(f"  skip {cat}: no single-mode FITS", "yellow"))
            continue
        iauname_col = CATALOG_META[cat]["iauname_col"]
        flux_col = CATALOG_META[cat]["flux_col"]
        single_dict = {str(r[iauname_col]).strip(): float(r[flux_col]) for r in single}

        merged_rows = [r for r in merged if str(r["origin_catalog"]) == cat]
        diffs = []
        for r in merged_rows:
            name = str(r["iau_name"]).strip()
            if name not in single_dict:
                continue
            merged_flux = float(r["flux"])
            single_flux = single_dict[name]
            if single_flux != 0:
                relerr = abs(merged_flux - single_flux) / abs(single_flux)
                if relerr > 1e-4:
                    diffs.append((name, merged_flux, single_flux, relerr))
            elif abs(merged_flux - single_flux) > 1e-30:
                diffs.append((name, merged_flux, single_flux, float("inf")))
        if diffs:
            print(f"  {cat}: {len(diffs)} flux mismatches. First 5:")
            for name, mf, sf, re_ in diffs[:5]:
                print(f"    {name}: merged={mf:.4e}  single={sf:.4e}  relerr={re_:.2e}")
            overall = result(False, f"{cat}: flux parity failed") and overall
        else:
            result(True, f"{cat}: flux values match single-mode catalog")
    return overall


def check_g_spatial_fallback_threshold(merged: Table, master_cone: Table) -> bool:
    print_header("g", "All spatial-fallback links are within 3\"")
    master_position = SkyCoord(
        ra=np.array(master_cone["MS_RA"]) * u.deg,
        dec=np.array(master_cone["MS_DEC"]) * u.deg,
    )
    master_names_by_cat = {
        c: {str(v).strip() for v in master_cone[CATALOG_META[c]["master_col"]]
            if str(v).strip() and str(v).strip() != "--"}
        for c in CATALOG_META
    }
    spatial_links = []  # (i, origin, name, sep)
    violations = []
    for i, row in enumerate(merged):
        ms_id = int(row["MS_ID"])
        origin = str(row["origin_catalog"])
        name = str(row["iau_name"]).strip()
        if ms_id == -1 or origin not in master_names_by_cat:
            continue
        if name in master_names_by_cat[origin]:
            continue  # name-matched, not spatial fallback
        src_pos = SkyCoord(ra=float(row["RA"]) * u.deg, dec=float(row["DEC"]) * u.deg)
        sep = src_pos.separation(master_position).arcsec
        closest_idx = int(np.argmin(sep))
        if int(master_cone["MS_ID"][closest_idx]) == ms_id:
            spatial_links.append((i, origin, name, float(sep[closest_idx])))
            if sep[closest_idx] >= SPATIAL_FALLBACK_ARCSEC:
                violations.append((i, origin, name, float(sep[closest_idx])))
    print(f"  {len(spatial_links)} spatial-fallback links detected")
    for i, o, n, s in spatial_links:
        print(f"    row {i:>3}  origin={o:<8}  name='{n}'  sep={s:.2f}\"")
    if violations:
        return result(False, f"{len(violations)} spatial-fallback links exceed 3\"")
    return result(True, f"All {len(spatial_links)} spatial-fallback links within 3\"")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    target = sys.argv[1]
    base = os.path.join("modeling_result", target)
    merged_path = os.path.join(base, "merged", "closest_catalog", "merged_nearby_sources_table.fits")
    master_path = os.path.join(base, "Pointings", "Master_source_cone.fits")
    for p in (merged_path, master_path):
        if not os.path.exists(p):
            print(color(f"ERROR: {p} not found", "red"))
            return 2

    merged = Table.read(merged_path)
    master = Table.read(master_path)

    print(color(f"\nVerifying merged run for {target}", "bold"))
    print(f"  merged table:        {len(merged)} rows")
    print(f"  master cone:         {len(master)} rows")

    checks = [
        check_a_no_duplicate_ms_ids(merged),
        check_b_name_prefix_matches_origin(merged),
        check_c_name_in_master_or_spatial_fallback(merged, master),
        check_d_photon_index_parity(target, merged),
        check_e_priority_order_respected(merged, master),
        check_f_flux_matches_origin_catalog(target, merged),
        check_g_spatial_fallback_threshold(merged, master),
    ]

    print()
    print(color("=" * 78, "bold"))
    if all(checks):
        print(color(f" ALL CHECKS PASSED for {target}", "green"))
        return 0
    failed = [c for c, ok in zip("abcdefg", checks) if not ok]
    print(color(f" SOME CHECKS FAILED: {failed}", "red"))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
