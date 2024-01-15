# --------------- Packages --------------- #

from astropy import units as u
from astropy.table import Table
from termcolor import colored
from astroquery.simbad import Simbad
from jaxspec.model.multiplicative import Tbabs
from jaxspec.model.additive import Powerlaw
from jax.config import config
from jaxspec.data.instrument import Instrument

# ---------- import class ---------- #

from catalog_class.XmmClass import XmmCatalog
from catalog_class.ChandraClass import ChandraCatalog
from catalog_class.SwiftClass import SwiftCatalog
from catalog_class.eRositaClass import eRositaCatalog
from catalog_class.CompareCatalogClass import CompareCatalog
from catalog_class.MatchClass import MatchCatalog

# ---------------------------------- #

# ---------- import function ---------- #

import function.init_function as i_f
import function.calculation_function as c_f
import function.software_function as s_f
import function.jaxspec_function as j_f

# ------------------------------------- #

import argparse
import numpy as np
import os
import subprocess
import sys
import shlex
import catalog_information as dict_cat
import numpyro
import platform

# ---------------------------------------- #

# --------------- Initialization --------------- #

catalogs = ["XMM", "Chandra", "Swift", "eRosita", "Slew", "RASS", "WGACAT", "Stacked"]

parser = argparse.ArgumentParser(description="Code optimal pointing point for NICER",
                                 epilog="Focus an object with his name or his coordinate")

main_group = parser.add_mutually_exclusive_group()
main_group.add_argument("--info", '-i', action='store_true',
                        help="Display a pulsar table")
main_group.add_argument('--name', '-n', type=str, 
                        help="Enter an object name")
main_group.add_argument('--coord', '-co', type=float, 
                        nargs=2, help="Enter your object coordinates : ra dec")

parser.add_argument('--radius', '-r', type=float, 
                    help="Enter the radius of the field of view (unit = arcmin)")

parser.add_argument('--exp_time', '-e_t', type=int,
                    help="Enter the exposure time to modeling data")

parser.add_argument('--catalog', '-ca', type=str, 
                    help="Enter catalog keyword : Xmm_DR13/CSC_2.0/Swift/eRosita/compare_catalog/match")

args = parser.parse_args()

psr_name = np.array(["PSR J0437-4715", "PSR J2124-3358", "PSR J0751+1807", "PSR J1231-1411"], dtype=str)
psr_coord = np.array([f"{i_f.get_coord_psr(name).ra} {i_f.get_coord_psr(name).dec}" for name in psr_name])
psr_count_rate = np.array([1.319, 0.1, 0.025, 0.27])
psr_table = Table(names=["full psr name", "psr coord", "psr count rate"],
                    data=[psr_name, psr_coord, psr_count_rate])

if args.info :
    print(psr_table)
    sys.exit()
    
if args.name:
    while True:
        if '_' in args.name:
            object_name = args.name.replace('_', " ")
            print(f"\nCollecting data for {colored(object_name, 'magenta')}")
        try:
            object_position = i_f.get_coord_psr(object_name)
            print(f"\n{colored(object_name, 'green')} is in Simbad Database, here is his coordinate :\n{object_position}")
            break
        except Exception as error:
            print(f"Error : {colored(object_name, 'red')}, isn't in Simbad Database")
            object_name = str(input("Enter another name : \n"))
            args.name = object_name
            print(f"\nCollecting data for {colored(object_name, 'magenta')}")
    catalog_path, catalog_name = i_f.choose_catalog(args.catalog)
elif args.coord:
    ra, dec = args.coord
    while True:
        print(f"\nCollecting data for coord : {colored([ra, dec], 'magenta')}")
        try:
            object_name = Simbad.query_region(f"{ra}d {dec}d", radius="1s")['MAIN_ID'][0]
            print(f"{colored([ra, dec], 'green')} is in Simbad Database, here is his name :\n{object_name}")
            break
        except Exception as error:
            print(f"{colored([ra, dec], 'red')} isn't Simbad Database")
            new_coord = str(input("Enter new coordinates : ra dec\n"))
            ra, dec = new_coord.split()
    object_position = i_f.get_coord_psr(object_name)
    catalog_path, catalog_name = i_f.choose_catalog(args.catalog)
    
while True:
    if object_name in psr_name:
        count_rate = psr_table["psr count rate"][psr_table['full psr name'] == object_name][0]
        break
    else:
        try:
            count_rate = float(input("Enter the count rate of your object : \n"))
            break
        except ValueError as error:
            print(f"Error: {error}")
            print("Please enter a valid float value for Count Rate.")
            continue

# ------------------------------------------------- #

# --------------- object_data --------------- #

object_data = {"object_name": object_name,
               "object_position": object_position,
               "count_rate": count_rate}

# ------------------------------------------- #

# --------------- modeling file --------------- #

# get the active workflow path
active_workflow = os.getcwd()
active_workflow = active_workflow.replace("\\","/")
data_path = os.path.join(active_workflow, "data").replace("\\", "/")

# catalog_data_path
catalog_datapath = os.path.join(data_path, "catalog_data").replace("\\", "/")

# path of stilts and topcat software 
stilts_software_path = os.path.join(data_path, 'softwares/stilts.jar').replace("\\", "/")
topcat_software_path = os.path.join(data_path, 'softwares/topcat-extra.jar').replace("\\", "/")

result_path = os.path.join(active_workflow, "modeling_result")
if not os.path.exists(result_path):
    os.mkdir(result_path)

# creation of modeling file 
name = object_data['object_name'].replace(" ", "_")
modeling_file_path = os.path.join(active_workflow, 'modeling_result', name).replace("\\", "/")

if not os.path.exists(modeling_file_path):
    os.mkdir(modeling_file_path)

# creation of plot_var_sources
plot_var_sources_path = os.path.join(modeling_file_path, "plot_var_sources").replace("\\", "/")
if not os.path.exists(plot_var_sources_path):
    os.mkdir(plot_var_sources_path)

output_name = os.path.join(modeling_file_path, 'Pointings').replace("\\", "/")
if not os.path.exists(output_name):
    os.mkdir(output_name)

# --------------------------------------------- #

# --------------- User table --------------- #

add_source_table = i_f.add_source_list(active_workflow=active_workflow)

if len(add_source_table) != 0:
    colnames = ['Name', 'Right Ascension', 'Declination', 'Var Value']
    print("\nHere is the list given by the User : \n", add_source_table, "\n")
else:
    print("\nUser don't defined any additionnal sources. \n")

# ------------------------------------------ #

# --------------- Load Nicer parameters --------------- #

print('-'*50)
print(f"{colored('Load NICER parameters : ', 'magenta')}")
nicer_data_path = os.path.join(data_path, "NICER_data")
PSF_data = os.path.join(nicer_data_path, "NICER_PSF.dat")
ARF_data = os.path.join(nicer_data_path, "nixtiaveonaxis20170601v005.arf")
RMF_data = os.path.join(nicer_data_path, "nixtiref20170601v003.rmf")
nicer_parameters_path = i_f.get_valid_file_path(PSF_data)
nicer_data_arf = i_f.get_valid_file_path(ARF_data)
nicer_data_rmf = i_f.get_valid_file_path(RMF_data)
EffArea, OffAxisAngle = np.loadtxt(nicer_parameters_path, unpack=True, usecols=(0, 1))
print('-'*50, '\n')

telescop_data = {"telescop_name": "nicer",
                 "EffArea": EffArea,
                 "OffAxisAngle": OffAxisAngle,
                 "nicer_data_arf": nicer_data_arf,
                 "nicer_data_rmf": nicer_data_rmf,
                 "min_value": 0.3,
                 "max_value": 10.0,
                 "energy_band": "0.2-12.0"}

# ----------------------------------------------------- #

# --------------- simulation_data --------------- #

simulation_data = {"object_data": object_data,
                   "telescop_data": telescop_data,
                   "INSTbkgd": 0.2,
                   "EXPtime": args.exp_time
                   }

# ----------------------------------------------- #

radius = args.radius*u.arcmin

if catalog_name == "Xmm_DR13":
    # Find the optimal pointing point with the Xmm_DR13 catalog
    
    # creation of 4XMM_DR13 directory
    xmm_directory = os.path.join(modeling_file_path, '4XMM_DR13'.replace("\\", "/"))
    xmm_img = os.path.join(xmm_directory, 'img'.replace("\\", "/"))
    xmm_closest_catalog = os.path.join(xmm_directory, "closest_catalog")
    if not os.path.exists(xmm_directory):
        os.mkdir(xmm_directory)
        os.mkdir(xmm_img)
        os.mkdir(xmm_closest_catalog)
    
    os_dictionary = {"active_workflow": active_workflow,
                     "catalog_datapath": catalog_datapath,
                     "modeling_file_path": modeling_file_path,
                     "plot_var_sources_path": plot_var_sources_path,
                     "catalog_directory" : xmm_directory,
                     "cloesest_dataset_path": xmm_closest_catalog,
                     "img": xmm_img,
                     "stilts_software_path": stilts_software_path,
                     "topcat_software_path": topcat_software_path}
    
    simulation_data["os_dictionary"] = os_dictionary
    
    # call XmmCatalog Class to make modeling
    xmm = XmmCatalog(catalog_path=catalog_path, radius=radius, simulation_data=simulation_data, user_table=add_source_table)
    nearby_sources_table, nearby_sources_position = xmm.nearby_sources_table,  xmm.nearby_sources_position
    model_dictionary = xmm.model_dictionary
    
    key = "XMM"
    column_dictionary = {"band_flux_obs" : dict_cat.dictionary_catalog['XMM']["band_flux_obs"],
                         "band_flux_obs_err": dict_cat.dictionary_catalog["XMM"]["band_flux_obs_err"],
                         "energy_band": [0.35, 0.75, 1.5, 3.25, 8.25],
                         "sigma": np.array(list(np.linspace(1e-20, 1e-24, len(dict_cat.dictionary_catalog[key]["energy_band_center"])))),
                         "data_to_vignetting": ["SC_RA", "SC_DEC", "IAUNAME"]}
    
    simulation_data["os_dictionary"]["catalog_key"] = key
    
elif catalog_name == "CSC_2.0":
    # Find the optimal pointing point with the Chandra catalog
    
    # creation of Chandra directory
    chandra_directory = os.path.join(modeling_file_path, 'Chandra'.replace("\\", "/"))
    chandra_img = os.path.join(chandra_directory, 'img'.replace("\\", "/"))
    chandra_closest_catalog = os.path.join(chandra_directory, "closest_catalog")
    if not os.path.exists(chandra_directory):
        os.mkdir(chandra_directory)
        os.mkdir(chandra_img)
        os.mkdir(chandra_closest_catalog)
    
    os_dictionary = {"active_workflow": active_workflow,
                     "modeling_file_path": modeling_file_path,
                     "plot_var_sources_path": plot_var_sources_path,
                     "catalog_directory": chandra_directory,
                     "cloesest_dataset_path": chandra_closest_catalog,
                     "img": chandra_img,
                     "stilts_software_path": stilts_software_path,
                     "topcat_software_path": topcat_software_path}
    
    simulation_data["os_dictionary"] = os_dictionary
    
                    # cs = cone search (Harvard features)
    # call Chandra Class to make modeling
    csc = ChandraCatalog(catalog_path=catalog_path, radius=radius, simulation_data=simulation_data, user_table=add_source_table)
    table_1, sources_1 = csc.nearby_sources_table, csc.nearby_sources_position
    table_2, sources_2 = csc.cone_search_catalog, csc.cs_nearby_sources_position
    
    answer = str(input(f"Which Table do you chose to follow the modeling ? {colored('Chandra / CS_Chandra', 'magenta')}\n"))
    while True:
        if answer == "Chandra":
            key = "Chandra"
            nearby_sources_table, nearby_sources_position = table_1, sources_1
            column_dictionary = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                                 "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                                 "energy_band": dict_cat.dictionary_catalog[key]["energy_band_center"],
                                 "sigma": np.array(list(np.linspace(1e-20, 1e-24, len(dict_cat.dictionary_catalog[key]["energy_band_center"])))),
                                 "data_to_vignetting": ["RA", "DEC", "Chandra_IAUNAME"]}
            model_dictionary = csc.model_dictionary
            simulation_data["os_dictionary"]["catalog_key"] = key
            break
        elif answer == "CS_Chandra":
            key = "CS_Chandra"
            nearby_sources_table, nearby_sources_position = table_2, sources_2
            column_dictionary = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                                 "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                                 "energy_band": dict_cat.dictionary_catalog[key]["energy_band_center"],
                                 "sigma": np.array(list(np.linspace(1e-20, 1e-24, len(dict_cat.dictionary_catalog[key]["energy_band_center"])))),
                                 "data_to_vignetting": ["ra", "dec", "name"]}
            model_dictionary = csc.cs_model_dictionary
            simulation_data["os_dictionary"]["catalog_key"] = key
            break
        else:
            print(f"{colored('Key error ! ', 'red')}. Please retry !")
            answer = str(input(f"Which Table do you chose to follow the modeling ? {colored('Chandra / CS_Chandra', 'magenta')}\n"))
    
elif catalog_name == "Swift":
    # Find the optimal pointing point with the Swift catalog
    
    # creation of Swift directory
    swi_directory = os.path.join(modeling_file_path, 'Swift'.replace("\\", "/"))
    swi_img = os.path.join(swi_directory, 'img'.replace("\\", "/"))
    swi_closest_catalog = os.path.join(swi_directory, "closest_catalog")
    if not os.path.exists(swi_directory):
        os.mkdir(swi_directory)
        os.mkdir(swi_img)
        os.mkdir(swi_closest_catalog)
    
    os_dictionary = {"active_workflow": active_workflow,
                     "modeling_file_path": modeling_file_path,
                     "plot_var_sources_path": plot_var_sources_path,
                     "catalog_directory" : swi_directory,
                     "cloesest_dataset_path": swi_closest_catalog,
                     "img": swi_img,
                     "stilts_software_path": stilts_software_path,
                     "topcat_software_path": topcat_software_path}
    
    simulation_data["os_dictionary"] = os_dictionary
    
    # call Swift Class to make modeling
    swi = SwiftCatalog(catalog_path=catalog_path, radius=radius, simulation_data=simulation_data, user_table=add_source_table)
    nearby_sources_table, nearby_sources_position = swi.nearby_sources_table, swi.nearby_sources_position
    model_dictionary = swi.model_dictionary
    
    key = "Swift"
    column_dictionary = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                         "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                         "energy_band": dict_cat.dictionary_catalog[key]["energy_band_center"],
                         "sigma": np.array(list(np.linspace(1e-20, 1e-24, len(dict_cat.dictionary_catalog[key]["energy_band_center"])))),
                         "data_to_vignetting": ["RA", "DEC", "Swift_IAUNAME"]}
    
    simulation_data["os_dictionary"]["catalog_key"] = key
    
elif catalog_name == "eRosita":
    # Find the optimal pointing with the eRosita catalog
    
    # creation of eRosita directory
    eRo_directory = os.path.join(modeling_file_path, 'eRosita'.replace("\\", "/"))
    eRo_img = os.path.join(eRo_directory, 'img'.replace("\\", "/"))
    eRo_closest_catalog = os.path.join(eRo_directory, "closest_catalog")
    if not os.path.exists(eRo_directory):
        os.mkdir(eRo_directory)
        os.mkdir(eRo_img)
        os.mkdir(eRo_closest_catalog)
        
    os_dictionary = {"active_workflow": active_workflow,
                     "modeling_file_path": modeling_file_path,
                     "plot_var_sources_path": plot_var_sources_path,
                     "catalog_directory" : eRo_directory,
                     "cloesest_dataset_path": eRo_closest_catalog,
                     "img": eRo_img,
                     "stilts_software_path": stilts_software_path,
                     "topcat_software_path": topcat_software_path}
    
    # call eRosita Class to make modeling
    eRo = eRositaCatalog(catalog_path=catalog_path, radius=radius, simulation_data=simulation_data, user_table=add_source_table)
    nearby_sources_table, nearby_sources_position = eRo.nearby_sources_table, eRo.nearby_sources_position
    model_dictionary = eRo.model_dictionary
    
    key = "eRosita"
    column_dictionary = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                         "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                         "energy_band": dict_cat.dictionary_catalog[key]["energy_band_center"],
                         "sigma": np.array(list(np.linspace(1e-20, 1e-24, len(dict_cat.dictionary_catalog[key]["energy_band_center"])))),
                         "data_to_vignetting": ["RA", "DEC", "Swift_IAUNAME"]}
    
    simulation_data["os_dictionary"]["catalog_key"] = key
    
elif catalog_name == "match":
    # Find optimal pointing point with using two catalog Xmm and Chandra
    
    # creation of match directory
    mixed_directory = os.path.join(modeling_file_path, 'xmmXchandra'.replace("\\", "/"))
    mixed_img = os.path.join(mixed_directory, 'img'.replace("\\", "/"))
    mixed_closest_catalog = os.path.join(mixed_directory, "closest_catalog")
    if not os.path.exists(mixed_directory):
        os.mkdir(mixed_directory)
        os.mkdir(mixed_img)
        os.mkdir(mixed_closest_catalog)
    
    os_dictionary = {"active_workflow": active_workflow,
                     "data_path": data_path,
                     "plot_var_sources_path": plot_var_sources_path,
                     "catalog_datapath": catalog_datapath,
                     "stilts_software_path": stilts_software_path,
                     "topcat_software_path": topcat_software_path,
                     "output_name": output_name,
                     "modeling_file_path": modeling_file_path,
                     "catalog_directory": mixed_directory,
                     "cloesest_dataset_path": mixed_closest_catalog,
                     "img": mixed_img}
    
    simulation_data["os_dictionary"] = os_dictionary
    os_dictionary["catalog_key"] = "xmmXchandra"
    
    # call CatalogMatch Class to make modeling
    mixed_catalog = MatchCatalog(catalog_name=("Xmm_DR13", "Chandra"), radius=radius, simulation_data=simulation_data)
    nearby_sources_table = mixed_catalog.nearby_sources_table
    var_index = mixed_catalog.var_index
    
    # --------------- modeling spectra with jaxspec --------------- #

    # setup jaxspec
    config.update("jax_enable_x64", True)
    numpyro.set_platform("cpu")

    # define caracteristic model here --> exp(-nh*$\sigma$) * x ** (-$\Gamma$)
    model = Tbabs() * Powerlaw()

    # load instrument parameters
    instrument = Instrument.from_ogip_file(nicer_data_arf, nicer_data_rmf, exposure=args.exp_time)

    # load all of the sources spetcra
    total_spectra, total_var_spectra = j_f.modeling_source_spectra(nearby_sources_table=nearby_sources_table, instrument=instrument, model=model, var_index=var_index)

    # plot of all spectra data
    data = j_f.total_plot_spectra(total_spectra=total_spectra, total_var_spectra=total_var_spectra, instrument=instrument, simulation_data=simulation_data, catalog_name="xmmXchandra")

    # output spectre plot
    j_f.write_txt_file(simulation_data=simulation_data, data=data)
    
    # ------------------------------------------------------------- # 
    
    sys.exit()

elif catalog_name == "compare_catalog":
    # Find the optimal pointing point with two catalogs to compare data
    
    # creation of compare_catalog directory
    compare_catalog_directory = os.path.join(modeling_file_path, 'Compare_catalog'.replace("\\", "/"))
    compare_catalog_img = os.path.join(compare_catalog_directory, 'img'.replace("\\", "/"))
    compare_catalog_closest_catalog = os.path.join(compare_catalog_directory, "closest_catalog")
    if not os.path.exists(compare_catalog_directory):
        os.mkdir(compare_catalog_directory)
        os.mkdir(compare_catalog_img)
        os.mkdir(compare_catalog_closest_catalog)
    
    os_dictionary = {"active_workflow": active_workflow,
                     "data_path": data_path,
                     "modeling_file_path": modeling_file_path,
                     "catalog_datapath": catalog_datapath,
                     "output_name": output_name,
                     "plot_var_sources_path": plot_var_sources_path,
                     "stilts_software_path": stilts_software_path,
                     "catalog_directory": compare_catalog_directory,
                     "cloesest_dataset_path": compare_catalog_closest_catalog,
                     "img": compare_catalog_img}
    
    simulation_data["os_dictionary"] = os_dictionary
    
    # call CompareCatalog Class to make calculation
    compare_class = CompareCatalog(catalog_path=catalog_path, radius=radius, simulation_data=simulation_data, exp_time=args.exp_time)
    sys.exit()
    
else:
    print(f"{colored('Invalid key workd !', 'red')}")
    sys.exit()
    
# --------------- count_rates --------------- #

excel_data_path = os.path.join(data_path, 'excel_data').replace("\\", "/")
if not os.path.exists(excel_data_path):
    os.mkdir(excel_data_path)
    
if platform.system() != "Windows":
    count_rates, nearby_sources_table = c_f.count_rates(nearby_sources_table, model_dictionary, telescop_data)
    # i_f.py_to_xlsx(excel_data_path=excel_data_path, count_rates=count_rates, object_data=object_data, args=(args.catalog, key), radius=args.radius)
elif platform.system() == "Windows":
    count_rates, nearby_sources_table = i_f.xlsx_to_py(excel_data_path=excel_data_path, nearby_sources_table=nearby_sources_table, object_data=object_data, args=(args.catalog, key), radius=args.radius)
else:
    sys.exit()
    
simulation_data['nearby_sources_table'] = nearby_sources_table

# -------------------------------------------------- #

# --------------- Nominal pointing infos --------------- #
            
c_f.nominal_pointing_info(simulation_data, nearby_sources_position)

# ------------------------------------------------------ #

# --------------- Value of optimal pointing point and infos --------------- #

            
OptimalPointingIdx, SRCoptimalSEPAR, SRCoptimalRATES, vector_dictionary = c_f.calculate_opti_point(simulation_data, nearby_sources_position)

c_f.optimal_point_infos(vector_dictionary, OptimalPointingIdx, SRCoptimalRATES)

# ------------------------------------------------------------------------- #

# --------------- Visualized data Matplotlib with S/N --------------- #

c_f.data_map(simulation_data, vector_dictionary, OptimalPointingIdx, nearby_sources_position)

# ------------------------------------------------------------------- #

# --------------- Calculate vignetting factor --------------- #

vignetting_factor, nearby_sources_table = c_f.vignetting_factor(OptimalPointingIdx=OptimalPointingIdx, vector_dictionary=vector_dictionary, simulation_data=simulation_data, data=column_dictionary["data_to_vignetting"], nearby_sources_table=nearby_sources_table)

# ----------------------------------------------------------- #

# --------------- Modeling nearby sources --------------- #

c_f.modeling(vignetting_factor=vignetting_factor, simulation_data=simulation_data, column_dictionary=column_dictionary, catalog_name=args.catalog)

# ------------------------------------------------------- #

# --------------- write fits file --------------- #

c_f.write_fits_file(nearby_sources_table=nearby_sources_table, simulation_data=simulation_data)

# ----------------------------------------------- #

# --------------- software --------------- # 

master_source_path = os.path.join(catalog_datapath, 'Master_source.fits').replace("\\", "/")


def select_master_sources_around_region(ra, dec, radius, output_name):
    """Radius is in arcminutes"""
    print(f"Extracting sources around region: RA {ra} and Dec {dec}")
    master_cone_path = os.path.join(output_name, 'Master_source_cone.fits').replace("\\", "/")
    command = (f"java -jar {stilts_software_path} tpipe {master_source_path} cmd='"+
            f'select skyDistanceDegrees({ra},{dec},MS_RA,MS_DEC)*60<{radius} '+
            f"' out={master_cone_path}")
    command = shlex.split(command)
    subprocess.run(command)


def select_catalogsources_around_region(output_name):
    print('Selecting catalog sources')
    master_cone_path = os.path.join(output_name, 'Master_source_cone.fits').replace("\\", "/")
    for cat in catalogs:
        path_to_cat_init = os.path.join(catalog_datapath, cat).replace("\\", "/")
        path_to_cat_final = os.path.join(output_name, cat).replace("\\", "/")
        command = (f"java -jar {stilts_software_path} tmatch2 matcher=exact \
                in1='{master_cone_path}' in2='{path_to_cat_init}.fits' out='{path_to_cat_final}.fits'\
                    values1='{cat}' values2='{cat}_IAUNAME' find=all progress=none")
        command = shlex.split(command)
        subprocess.run(command)

right_ascension = object_data["object_position"].ra.value
declination = object_data["object_position"].dec.value
try:
    print(f"\n{colored('Load Erwan s code for :', 'yellow')} {object_data['object_name']}")
    select_master_sources_around_region(ra=right_ascension, dec=declination, radius=radius.value, output_name=output_name)
    select_catalogsources_around_region(output_name=output_name)
    master_sources = s_f.load_master_sources(output_name)
    s_f.master_source_plot(master_sources=master_sources, simulation_data=simulation_data, number_graph=len(master_sources))
except Exception as error :
    print(f"{colored('An error occured : ', 'red')} {error}")

# ---------------------------------------- #

# --------------- modeling spectra with jaxspec --------------- #

var_index =  j_f.cross_catalog_index(output_name=output_name, key=key, iauname=column_dictionary["data_to_vignetting"][2], nearby_sources_table=nearby_sources_table)

# setup jaxspec
config.update("jax_enable_x64", True)
numpyro.set_platform("cpu")

# define caracteristic model here --> exp(-nh*$\sigma$) * x ** (-$\Gamma$)
model = Tbabs() * Powerlaw()

# load instrument parameters
instrument = Instrument.from_ogip_file(nicer_data_arf, nicer_data_rmf, exposure=args.exp_time)

# load all of the sources spetcra
total_spectra, total_var_spectra = j_f.modeling_source_spectra(nearby_sources_table=nearby_sources_table, instrument=instrument, model=model, var_index=var_index)

# plot of all spectra data
data = j_f.total_plot_spectra(total_spectra=total_spectra, total_var_spectra=total_var_spectra, instrument=instrument, simulation_data=simulation_data, catalog_name=args.catalog)

# output spectre plot
j_f.write_txt_file(simulation_data=simulation_data, data=data)

# ------------------------------------------------------------- # 
