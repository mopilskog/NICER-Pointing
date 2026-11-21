# --------------- Packages --------------- #

from astropy import units as u
from astropy.table import Table
from termcolor import colored
from astroquery.simbad import Simbad
from jaxspec.model.multiplicative import Tbabs
from jaxspec.model.additive import Powerlaw
#from jax.config import config
from jax import config
from jaxspec.data.instrument import Instrument
from jaxspec.data.observation import Observation
from jaxspec.data import ObsConfiguration
from jaxspec.data.util import data_path_finder

# ---------- import class ---------- #

from catalog_class.XmmClass import XmmCatalog
from catalog_class.ChandraClass import ChandraCatalog
from catalog_class.SwiftClass import SwiftCatalog
from catalog_class.eRASS1Class import eRASS1Catalog
from catalog_class.MultiCatalogPipeline import MultiCatalogPipeline

# ---------------------------------- #

# ---------- import function ---------- #

import function.init_function as i_f
import function.calculation_function as c_f
import function.software_function as s_f
import function.jaxspec_function as j_f
from function.log_function import log, setup_logging, start_log_file
import function.summary_function as sum_f

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

# ---------- for documentation --------- #

# from src.catalog_class.XmmClass import XmmCatalog
# from src.catalog_class.ChandraClass import ChandraCatalog
# from src.catalog_class.SwiftClass import SwiftCatalog
# from src.catalog_class.eRASS1Class import eRASS1Catalog
# import src.function.init_function as i_f
# import src.function.calculation_function as c_f
# import src.function.software_function as s_f
# import src.function.jaxspec_function as j_f
# import src.catalog_information as dict_cat

# -------------------------------------- #

def main():
    """
    This script (main.py) serves as the entry point for the Optimal Pointing Point Code for the NICER telescope. It leverages various classes and functions to load data, perform calculations, and generate visualizations and outputs for astronomical studies.

    Note:
        The script is structured into several main sections:
        
        - Initialization: Setting up basic parameters and loading necessary catalogs.
        - Argument Processing: Using `argparse` to handle command-line options.
        - Data Loading and Analysis: Loading data from specified catalogs and performing analysis.
        - Modeling and Visualization: Performing modeling calculations and generating visualizations.

    Each section is documented to aid in understanding and maintaining the code.
    
    Example:
        This is how we can execute the program:
        
        >>> Optimal_Pointing_Point_Code > python ./src/main.py [*args]


    Important:
        Here are the different arguments to execute the project:
        
        - --info, -i : Display a table of pulsars.
        - --name, -n : Specify the name of a celestial object for analysis.
        - --coord, -co : Specify the coordinates of a celestial object.
        - --radius, -r : Define the radius of the area to be analyzed.
        - --exp_time, -e_t : Set the exposure time for modeling.
        - --catalog, -ca : Choose the catalog to use for analysis:
            - Key : Xmm_DR13 / CSC_2.0 / Swift / eRASS1 / merged
        
    Example:
        Here is an example demonstrating the use of the argument:
        
        With the name of the object -->
        
        >>> Optimal_Pointing_Point_Code > python ./src/main.py --name OBJECT_NAME --radius RADIUS_VALUE --exp_time MODELING_TIME --CATALOG CATALOG_KEY
        
        With the coordinates of the object -->
        
        >>> Optimal_Pointing_Point_Code > python ./src/main.py --coord RA DEC --radius RADIUS_VALUE --exp_time MODELING_TIME --CATALOG CATALOG_KEY
        
        
    Initialization:
    ===============
        This script serves as the primary interface for the Optimal Pointing Point Code designed for the NICER telescope. It initializes a set of predefined 
        astronomical catalogs and sets up a command-line argument parser to process user inputs. The available catalogs include 'XMM', 'Chandra', 'Swift', 'eRASS1'. 

        Users can interact with the script by specifying options such as displaying a pulsar table, providing the name or coordinates of a celestial object, defining the 
        field of view radius, setting the exposure time for data modeling, or selecting a specific catalog for analysis. The script then performs data collection and 
        processing based on these inputs. For instance, if a celestial object name is provided, the script fetches its coordinates from the Simbad Database and proceeds 
        with the chosen catalog for further analysis. Similarly, when coordinates are provided, the script identifies the corresponding celestial object. 
        The script is designed to handle exceptions and guide users through valid inputs, ensuring a smooth data retrieval and processing experience.
        
    Dictionary to stock data:
    =========================
    This script section initializes several dictionaries and data structures, crucial for the NICER telescope pointing optimization process.

    object data Dictionary:
    -----------------------
    The 'object_data' dictionary holds crucial information about the celestial object of interest. It includes:
    
    - 'object_name': Name of the celestial object.
    - 'object_position': Astropy SkyCoord object representing the object's celestial coordinates.
    - 'count_rate': The observed count rate of the object.
    
    This dictionary is pivotal for various calculations and data retrieval related to the target object.
    
    Example:
    
    >>> object_data = {"object_name": "name",
                       "object_position": object_position,
                       "count_rate": count_rate
                       }

    telescop data Dictionary:
    -------------------------
    The 'telescop_data' dictionary contains parameters specific to the NICER telescope and its observational capabilities. It includes:
    
    - 'telescop_name': Name of the telescope, in this case, 'nicer'.
    - 'EffArea': The effective area of the telescope.
    - 'OffAxisAngle': Off-axis angle data.
    - 'nicer_data_arf', 'nicer_data_rmf': Paths to the Auxiliary Response File (ARF) and the Response Matrix File (RMF).
    - 'min_value', 'max_value': The minimum and maximum values for certain telescope parameters.
    - 'energy_band': The energy band of interest for observations.
    
    Example:
    
    >>> telescop_data = {"telescop_name": "nicer",
                        "EffArea": EffArea,
                        "OffAxisAngle": OffAxisAngle,
                        "nicer_data_arf": nicer_data_arf,
                        "nicer_data_rmf": nicer_data_rmf,
                        "min_value": 0.3,
                        "max_value": 10.0,
                        "energy_band": "0.2-12.0"
                        }

    simulation data Dictionary:
    ---------------------------
    The 'simulation_data' dictionary aggregates data required for simulation and modeling. It includes:
    
    - 'object_data': A reference to the 'object_data' dictionary.
    - 'telescop_data': A reference to the 'telescop_data' dictionary.
    - 'INSTbkgd': The instrumental background value.
    - 'EXPtime': The exposure time for the observations.
    
    Example:
    
    >>> simulation_data = {"object_data": object_data,
                           "telescop_data": telescop_data,
                           "INSTbkgd": 0.2,
                           "EXPtime": args.exp_time
                           }
    
    This dictionary is used extensively throughout the simulation process.

    Note:
        - The accuracy and reliability of the simulation are heavily dependent on the correctness and precision of the data provided in these dictionaries.
        - Users should verify the integrity and relevance of the data, especially when using custom input or dealing with unfamiliar celestial objects.

    Important:
        - Changes to these dictionaries should be made cautiously, as they can significantly impact the simulation results.
        - Ensure that the paths to data files and software tools are correctly set according to your system's configuration.

    
    This section of the script is dedicated to handling various astronomical catalogs for the NICER telescope pointing optimization. Depending on the user-selected catalog, the script sets up the necessary directories, data paths, and initializes specific classes for modeling and analysis.


    Accessing Class Functionality
    =============================
    
    XMM-DR13 Catalog Handling [Xmm_DR13]
    ------------------------------------
    
    This section of the script focuses on using the XMM-DR13 catalog for optimizing NICER telescope pointing. The XMM-Newton observatory, from which this catalog is derived, is known for its detailed X-ray observations of the universe.

    Functionality:
        - The script establishes a dedicated directory specifically for the 4XMM_DR13 catalog, organizing data and images for analysis.
        - The `XmmCatalog` class is initialized to handle the processing and analysis of data from the XMM-DR13 catalog.
        - Essential directories for image storage and catalog data are created to support the processing.

    XMM-DR13 Catalog Processing:
        - Configuration settings for the XMM-DR13 catalog, including paths for modeling files and software tools, are stored in `os_dictionary`.
        - The `XmmCatalog` class specializes in analyzing the high-energy X-ray data from the XMM-Newton observatory.
        - The script sets up columns and parameters specific to the XMM-DR13 catalog, facilitating the analysis process.

    Note:
        - XMM-DR13 is an invaluable resource for studying cosmic X-ray sources, providing extensive data for NICER's pointing optimization.
        - The catalog's detailed spectral and timing data enhance the understanding of various astrophysical phenomena.

    Important:
        - Users must ensure that the XMM-DR13 catalog data is current and correctly structured for effective integration into the analysis.
        - Precise configuration in the `XmmCatalog` class is crucial for accurate data processing and extraction of meaningful results.

    Upon completing the setup and processing of the XMM-DR13 catalog data, the script updates `simulation_data` for subsequent analysis stages, leveraging the detailed X-ray observations from XMM-Newton.


    Chandra Catalog Handling [CSC_2.0]
    ----------------------------------
    This segment of the script is tailored for the analysis and optimization of NICER telescope pointing using the Chandra catalog. The Chandra X-ray Observatory's data is renowned for its high spatial resolution and depth in X-ray astronomy.

    Functionality:
        - Creation of a Chandra-specific directory to manage and organize the catalog data efficiently.
        - The `ChandraCatalog` class is initialized for detailed processing and analysis of the Chandra catalog data.
        - Directories for image storage and the closest catalog data are established to aid in data handling.

    Chandra Catalog Processing:
        - Configuration settings specific to the Chandra catalog are stored in `os_dictionary`, including paths for modeling files, images, and relevant software tools.
        - The `ChandraCatalog` class manages the intricacies of Chandra data, focusing on high-resolution X-ray observations.

    Note:
        - The Chandra catalog is particularly useful for studying detailed structures in X-ray emitting regions, benefiting NICER's pointing accuracy.

    Important:
        - Users must verify the availability and correctness of Chandra catalog data in the specified directories for successful processing.

    After setting up the necessary configuration and making the choice of catalog, the script proceeds with the processing of the Chandra data, updating `simulation_data` accordingly for the next stages of analysis.


    Swift Catalog Handling [Swift]
    ------------------------------
    This part of the script is focused on utilizing the Swift catalog for NICER telescope pointing optimization. The Swift Gamma-Ray Burst Mission provides valuable multi-wavelength astronomical data, which is crucial for various astrophysical studies.

    Functionality:
        - The script creates a dedicated directory for the Swift catalog to organize the processing and storage of data.
        - The `SwiftCatalog` class is initialized for handling the data analysis and modeling from the Swift catalog.
        - Necessary directories for image storage and catalog data are established.

    Swift Catalog Processing:
        - Configuration for the Swift catalog, including paths for modeling files, images, and relevant software tools, is stored in `os_dictionary`.
        - The `SwiftCatalog` class efficiently processes data from the Swift mission, particularly focusing on gamma-ray burst data and other astronomical phenomena.

    Note:
        - Swift's rapid-response capability and broad wavelength coverage provide a unique dataset for astrophysical research, enhancing NICER's observational strategies.
        - The catalog's data is instrumental in studying transient and high-energy phenomena in the universe.

    Important:
        - Users must ensure that the Swift catalog data is up-to-date and accurately formatted for seamless integration into the analysis process.
        - Adjustments in the `SwiftCatalog` class or the parameters used for analysis should be approached with careful consideration to preserve the accuracy of the results.

    After processing the Swift catalog data, the script updates `simulation_data` and prepares for subsequent stages of analysis, leveraging the unique capabilities of the Swift mission for NICER's optimization.


    eRASS1 Catalog Handling [eRASS1]
    ----------------------------------
    This section of the script focuses on the utilization of the eRASS1 catalog for the NICER telescope pointing optimization. The eRASS1 space telescope provides comprehensive all-sky survey data in X-ray bands, making it invaluable for astronomical research and analysis.

    Functionality:
        - The script sets up a dedicated directory for the eRASS1 catalog, ensuring organized storage and retrieval of data.
        - The `eRASS1Catalog` class is initialized to process and analyze the data from this catalog.
        - Directories for storing images and the closest catalog data are created to facilitate effective data handling and visualization.

    eRASS1 Catalog Processing:
        - Configuration settings specific to the eRASS1 catalog are stored in `os_dictionary`, including paths for modeling files, images, and software tools.
        - The `eRASS1Catalog` class manages the data from the eRASS1 space telescope, focusing on optimal point determination for NICER.

    Note:
        - The eRASS1 catalog's comprehensive sky coverage offers a wide range of data for analysis, enhancing the scope of the NICER pointing optimization.
        - The data from eRASS1 is crucial for understanding large-scale structures and energetic phenomena in the universe.

    Important:
        - Users should ensure that the eRASS1 catalog data is current and correctly formatted for effective processing.
        - Adjustments in the `eRASS1Catalog` class or the analysis parameters should be made with caution to maintain the integrity of the results.

    Upon completion of processing, the script updates the `simulation_data` with key information from the eRASS1 catalog and prepares the data for subsequent analysis stages.


    Count Rate - Optimal Pointing Point - Vignetting Factor
    =======================================================
    
    If the user selects one of the following keys: {"Xmm_DR13", "CSC_2.0", "Swift", "eRASS1"}, then the classes respectively generate the nearby sources table (nearby_sources_table), 
    a list of the positions of these sources (nearby_sources_position), photon index data, hydrogen density column, and a dictionary (model_dictionary) containing the model for determining count rates, 
    the value of this model, the median flux, and Nh.
    
    First, we will call the 'c_f.count_rates' function from the 'calculation_function.py' module, which, with the help of 'model_dictionary' and the WebPIMMS software, will accurately 
    determine the value of the count rates
    
    Example:
        Use c_f.count_rate function -->
        
        >>> count_rates, nearby_sources_table = c_f.count_rates(nearby_sources_table, model_dictionary, telescop_data)
    
    Subsequently, now that we have all the necessary data at our disposal, we can determine the optimal line of sight for the NICER telescope. To do this, we will utilize the 'c_f.calculate_opti_point' 
    function from the 'calculation_function.py' module.
    
    Example:
        Use c_f.calculate_opti_point function -->
        
        >>> OptimalPointingIdx, SRCoptimalSEPAR, SRCoptimalRATES, vector_dictionary = c_f.calculate_opti_point(simulation_data, nearby_sources_position)
    
    Here is an example of this function
    
    .. image:: C:/Users/plamb_v00y0i4/OneDrive/Bureau/Optimal_Pointing_Point_Code/modeling_result/PSR_J0437-4715/Chandra/img/Chandra_SNR_PSR_J0437-4715.png
        :align: center
        
    Once all these values are determined, we can calculate the vignetting factor based on this line of sight. This factor will enable us to generate the spectra of the nuisance sources in the most
    effective manner, allowing us to determine the sum of contributions around the targeted object. We can get these values using "c_f.vignetting_factor"from the"calculation_function.py" module.
    
    Example:
        Use c_f.vignetting_factor function -->
        
        >>> vignetting_factor, nearby_sources_table = c_f.vignetting_factor(OptimalPointingIdx=OptimalPointingIdx, vector_dictionary=vector_dictionary, simulation_data=simulation_data, data=column_dictionary["data_to_vignetting"], nearby_sources_table=nearby_sources_table)

    Variable Sources detected with Erwan's Quintin Code
    ===================================================
    
    With a portion of code from the S.T.O.N.K.S project, it is possible for us to determine the sources around the targeted object that have been classified as variable. To achieve this, we will use several 
    other catalogs such as RASS, Stacked, Slew, WGACAT in addition to our 4 catalogs. The idea is to compare these data to identify whether the sources exhibit variations or not. A table named 'Master_Source.fits'
    stores all the source names, including those that are common across all catalogs.

    All the variable sources are stored in the table 'Master_source_cone.fits'. With this information, we can, for example, generate spectra for these sources.
    
    Note:
        - Ensure that you have the necessary data files, including 'Master_Source.fits' and 'Master_source_cone.fits', to perform variable source detection.

    Important:
        - The variable source detection process may require significant computational resources, depending on the size of the datasets and the complexity of the analysis.
    
    .. image:: C:/Users/plamb_v00y0i4/OneDrive/Bureau/Optimal_Pointing_Point_Code/modeling_result/PSR_J0437-4715/plot_var_sources/sources_plot_6.png
        :align: center
        
    Link to the S.T.O.N.K.S project : <https://github.com/ErwanQuintin/STONKS>
    
    Spectra Modeling with Jaxspec
    =============================

    Jaxspec, developed at Irap by Dr. Simon Dupourqué and Dr. Didier Barret, is a Python package designed to facilitate spectral analysis.
    Link to the project : <https://github.com/renecotyfanboy/jaxspec>

    Note:
        - Before using Jaxspec, make sure to install it by following the instructions provided in the project's documentation.

    Important:
        - Ensure that you have the necessary data files, including 'Master_source_cone.fits', 'arf', and 'rmf', to perform spectral modeling.

    First, we define a model to generate these spectra and load the NICER instrument parameters (the .arf and .rmf files).

    Example:
        >>> model = Tbabs() * Powerlaw()
        >>> instrument = Instrument.from_ogip_file(nicer_data_arf, nicer_data_rmf, exposure=args.exp_time)

    Finally, we only need to generate the spectra of the sources, which we do using the 'j_f.modeling_source_spectra' function from the 'jaxspec_function.py' module.

    Example:
        Use j_f.modeling_source_spectra function -->

        >>> total_spectra = j_f.modeling_source_spectra(nearby_sources_table=nearby_sources_table, obsconfig=obsconfig, model=model, key=key)

    Once these spectra are obtained, we can plot the different spectra of these sources as well as the sum of contributions using the 'j_f.total_plot_spectra' function from the 'jaxspec_function.py' module.

    Example:
        Use j_f.total_plot_spectra function ->

        >>> j_f.total_plot_spectra(total_spectra=total_spectra, obsconfig=obsconfig, simulation_data=simulation_data, catalog_name=args.catalog)

    Here is an example of this plot :

    .. image:: C:/Users/plamb_v00y0i4/OneDrive/Bureau/Optimal_Pointing_Point_Code/modeling_result/PSR_J0437-4715/4XMM_DR13/img/XMM_spectral_modeling_close_to_PSR_J0437-4715.png
        :align: center

    In this graph, we have an envelope of variability from the variable sources in the modeling. The last function we use, 'j_f.write_pha_file' from the 'jaxspec_function.py' module,
    writes the summed spectrum (with and without variability), the per-source spectra and the source information to one PHA FITS file.

    Example:
        Use j_f.write_pha_file function ->

        >>> j_f.write_pha_file(total_spectra=total_spectra, obsconfig=obsconfig, simulation_data=simulation_data, key=key, pointing_ra=ra_opti, pointing_dec=dec_opti, radius_arcmin=args.radius, exposure=args.exp_time, rmf_path=nicer_data_rmf, arf_path=nicer_data_arf)

    """
    # --------------- Initialization --------------- #
 
    catalogs = ["XMM", "Chandra", "Swift", "eRASS1", "Slew", "RASS", "WGACAT", "Stacked"]

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
                        help="Catalog keyword: Xmm_DR13/CSC_2.0/Swift/eRASS1/merged (default: merged)")

    parser.add_argument('--interactive','-in', action='store_true', help="Interactive mode toggles plotting and asks for user input regarding file specifications. By default, the system will not show plots, to allow the user to run the program without having to interact with it at runtime.")

    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument('--verbose', '-v', action='store_true',
                                 help="Show all details in the console")
    verbosity_group.add_argument('--quiet', '-q', action='store_true',
                                 help="Show only warnings, errors and the final results in the console")

    args = parser.parse_args()

    # Console detail chosen by the user; everything is always written to run.log.
    setup_logging("verbose" if args.verbose else "quiet" if args.quiet else "normal")

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
                log.info(f"\nCollecting data for {colored(object_name, 'magenta')}")
            else:
                object_name = args.name
            try:
                object_position = i_f.get_coord_psr(object_name)
                log.debug(f"{object_name} found in Simbad: RA {object_position.ra.deg:.5f}, Dec {object_position.dec.deg:.5f} deg")
                break
            except Exception as error:
                log.warning(f"{colored(object_name, 'red')} was not found in Simbad")
                object_name = str(input("Enter another name : \n"))
                args.name = object_name
                log.info(f"\nCollecting data for {colored(object_name, 'magenta')}")
        catalog_path, catalog_name = i_f.choose_catalog(args.catalog, interactive=args.interactive)
    elif args.coord:
        ra, dec = args.coord
        while True:
            log.info(f"\nCollecting data for coord : {colored([ra, dec], 'magenta')}")
            try:
                object_name = Simbad.query_region(f"{ra}d {dec}d", radius="1s")['main_id'][0]
                log.debug(f"Simbad object at these coordinates: {object_name}")
                break
            except Exception as error:
                log.warning(f"No Simbad object found at {colored([ra, dec], 'red')}")
                new_coord = str(input("Enter new coordinates : ra dec\n"))
                ra, dec = new_coord.split()
        object_position = i_f.get_coord_psr(object_name)
        catalog_path, catalog_name = i_f.choose_catalog(args.catalog, interactive=args.interactive)
        
    while True:
        if object_name in psr_name:
            count_rate = psr_table["psr count rate"][psr_table['full psr name'] == object_name][0]
            break
        else:
            try:
                count_rate = float(input("Enter the count rate of your object : \n"))
                break
            except ValueError as error:
                log.warning(f"Invalid count rate: {error}")
                log.warning("Please enter a valid float value for Count Rate.")
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
    
    add_source_table = i_f.add_source_list(active_workflow=active_workflow, interactive=args.interactive)

    if len(add_source_table) != 0:
        colnames = ['Name', 'Right Ascension', 'Declination', 'Var Value']
        log.info(f"User-added sources:\n{add_source_table}")
    else:
        log.debug("No user-added sources.")

    # ------------------------------------------ #

    # --------------- Load Nicer parameters --------------- #

    log.debug("Loading NICER parameters")
    nicer_data_path = os.path.join(data_path, "NICER_data")
    PSF_data = os.path.join(nicer_data_path, "NICER_PSF.dat")
    ARF_data = os.path.join(nicer_data_path, "nixtiaveonaxis20170601v005.arf")
    RMF_data = os.path.join(nicer_data_path, "nixtiref20170601v003.rmf")
    nicer_parameters_path = i_f.get_valid_file_path(PSF_data)
    nicer_data_arf = i_f.get_valid_file_path(ARF_data)
    nicer_data_rmf = i_f.get_valid_file_path(RMF_data)
    EffArea, OffAxisAngle = np.loadtxt(nicer_parameters_path, unpack=True, usecols=(0, 1))

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
    # Catalog modes that go through MultiCatalogPipeline. Single-catalog modes
    # use a one-element list; merged uses the full priority list. To add a new
    # catalog or mode that fits the orchestrator pattern, add one entry here.
    CATALOG_DISPATCH = {
        "Xmm_DR13": ["XMM"],
        "CSC_2.0":  ["Chandra"],
        "Swift":    ["Swift"],
        "eRASS1":   ["eRASS1"],
        "merged":   ["XMM", "Swift", "eRASS1", "Chandra"],
    }

    if catalog_name in CATALOG_DISPATCH:
        simulation_data["os_dictionary"] = {
            "active_workflow": active_workflow,
            "catalog_datapath": catalog_datapath,
            "modeling_file_path": modeling_file_path,
            "plot_var_sources_path": plot_var_sources_path,
            "stilts_software_path": stilts_software_path,
            "topcat_software_path": topcat_software_path,
        }
        pipeline = MultiCatalogPipeline(
            catalogs=CATALOG_DISPATCH[catalog_name],
            simulation_data=simulation_data,
            radius=radius,
            user_table=add_source_table,
            catalog_path=catalog_path,
        )
        result = pipeline.run()
        nearby_sources_table = result.nearby_sources_table
        nearby_sources_position = result.nearby_sources_position
        model_dictionary = result.model_dictionary
        column_dictionary = result.column_dictionary
        key = result.key
        log_path = start_log_file(simulation_data["os_dictionary"]["catalog_directory"])
        log.debug(f"Full log of this run: {log_path}")

    else:
        log.error(f"{colored('Invalid catalog keyword!', 'red')}")
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
                
    nominal_pointing = c_f.nominal_pointing_info(simulation_data, nearby_sources_position)

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

    # c_f.modeling re-fits each source's band fluxes for a diagnostic plot. In
    # merged mode the table mixes sources from catalogs with different band
    # schemas (XMM has 5 bands, Swift/Chandra 3, eRASS1 4), so this step is
    # skipped. Single-catalog modes still run it as before.
    if key != "merged":
        c_f.modeling(vignetting_factor=vignetting_factor, simulation_data=simulation_data, column_dictionary=column_dictionary, catalog_name=args.catalog)

    # ------------------------------------------------------- #

    # --------------- write fits file --------------- #

    c_f.write_fits_file(nearby_sources_table=nearby_sources_table, simulation_data=simulation_data,interactive=args.interactive)

    # ----------------------------------------------- #

    # --------------- software --------------- # 

    master_source_path = os.path.join(catalog_datapath, 'Master_source.fits').replace("\\", "/")


    def select_master_sources_around_region(ra, dec, radius, output_name):
        """Radius is in arcminutes"""
        log.debug(f"Extracting sources around region: RA {ra} and Dec {dec}")
        master_cone_path = os.path.join(output_name, 'Master_source_cone.fits').replace("\\", "/")
        command = (f"java -jar {stilts_software_path} tpipe {master_source_path} cmd='"+
                f'select skyDistanceDegrees({ra},{dec},MS_RA,MS_DEC)*60<{radius} '+
                f"' out={master_cone_path}")
        command = shlex.split(command)
        subprocess.run(command)


    def select_catalogsources_around_region(output_name):
        log.debug('Selecting catalog sources')
        master_cone_path = os.path.join(output_name, 'Master_source_cone.fits').replace("\\", "/")
        for cat in catalogs:
            path_to_cat_init = os.path.join(catalog_datapath, cat).replace("\\", "/")
            path_to_cat_final = os.path.join(output_name, cat).replace("\\", "/")
            if cat=="eRASS1":
                command = (f"java -jar {stilts_software_path} tmatch2 matcher=exact \
                        in1='{master_cone_path}' in2='{path_to_cat_init}.fits' out='{path_to_cat_final}.fits'\
                            values1='{cat}' values2='eRASS_IAUNAME' find=all progress=none")
            else: 
                command = (f"java -jar {stilts_software_path} tmatch2 matcher=exact \
                        in1='{master_cone_path}' in2='{path_to_cat_init}.fits' out='{path_to_cat_final}.fits'\
                            values1='{cat}' values2='{cat}_IAUNAME' find=all progress=none")
            command = shlex.split(command)
            subprocess.run(command)

    right_ascension = object_data["object_position"].ra.value
    declination = object_data["object_position"].dec.value
    try:
        log.debug(f"Master-source variability plots for {object_data['object_name']}")
        if result.master_sources is not None:
            # Merged mode: the pipeline already built Master_source_cone.fits
            # and loaded the master sources for the variability information.
            master_sources = result.master_sources
        else:
            select_master_sources_around_region(ra=right_ascension, dec=declination, radius=radius.value, output_name=output_name)
            select_catalogsources_around_region(output_name=output_name)
            master_sources = s_f.load_master_sources(output_name)
        log.debug(f"Master-source files in: {output_name}")
        s_f.master_source_plot(master_sources=master_sources, simulation_data=simulation_data, number_graph=len(master_sources))
    except Exception as error :
        log.warning(f"Master-source variability plots failed: {error}")

    # ---------------------------------------- #

    # --------------- modeling spectra with jaxspec --------------- #

    # setup jaxspec
    config.update("jax_enable_x64", True)
    numpyro.set_platform("cpu")

    # define caracteristic model here --> exp(-nh*$\sigma$) * x ** (-$\Gamma$)
    model = Tbabs() * Powerlaw()

    # load instrument parameters
    instrument = Instrument.from_ogip_file(nicer_data_rmf, nicer_data_arf)
    #instrument = Instrument.from_ogip_file(nicer_data_arf, nicer_data_rmf, exposure=args.exp_time)
    
    obsconfig= ObsConfiguration.mock_from_instrument(instrument, exposure=args.exp_time)

    #observation = Observation.from_ogip_container(ogip_container)
    #obsconfig = ObsConfiguration.from_instrument(instrument, observation)
    
    # load all of the sources spetcra
    total_spectra = j_f.modeling_source_spectra(nearby_sources_table=nearby_sources_table, obsconfig=obsconfig, model=model, key=key)
    
    # plot of all spectra data
    j_f.total_plot_spectra(total_spectra=total_spectra, obsconfig=obsconfig, simulation_data=simulation_data, catalog_name=args.catalog)

    # summed and per-source spectra + source information in one PHA FITS file
    j_f.write_pha_file(total_spectra=total_spectra, obsconfig=obsconfig, simulation_data=simulation_data, key=key,
                       pointing_ra=vector_dictionary['SampleRA'][OptimalPointingIdx],
                       pointing_dec=vector_dictionary['SampleDEC'][OptimalPointingIdx],
                       radius_arcmin=args.radius, exposure=args.exp_time,
                       rmf_path=nicer_data_rmf, arf_path=nicer_data_arf)

    # ------------------------------------------------------------- #

    # --------------- run summary --------------- #

    sum_f.write_summary(simulation_data=simulation_data, key=key,
                        settings={"radius": args.radius, "exp_time": args.exp_time},
                        merge_summary=result.merge_summary, nominal=nominal_pointing,
                        vector_dictionary=vector_dictionary, optimal_index=OptimalPointingIdx)

    # ------------------------------------------- # 

if __name__ == "__main__":
    main()