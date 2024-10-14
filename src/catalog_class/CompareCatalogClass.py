# --------------- Packages --------------- #

from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from typing import Dict, Tuple, List
from termcolor import colored
from tqdm import tqdm
from jaxspec.model.multiplicative import Tbabs
from jaxspec.model.additive import Powerlaw
from jaxspec.data.instrument import Instrument
from jaxspec.data.observation import Observation
from jaxspec.data import ObsConfiguration
from jaxspec.data.util import fakeit_for_multiple_parameters

# ---------- import function ---------- #

import function.calculation_function as c_f
import function.software_function as s_f
import function.unique_function as u_f

# ------------------------------------- #

import openpyxl
import catalog_information as dict_cat
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyvo as vo
import subprocess
import os
import platform
import shlex

# ---------------------------------------- #

# ---------- for documentation ---------- #

# import src.function.calculation_function as c_f
# import src.function.software_function as s_f
# import src.function.unique_function as u_f
# import src.catalog_information as dict_cat

# --------------------------------------- #

class CompareCatalog:
    """
    The CompareCatalog class is designed to analyze and compare astronomical catalogs, specifically focusing on X-ray sources. It processes two astronomical catalogs, calculating photon indices and hydrogen column densities (NH values) for sources from catalogs such as Xmm_DR13, CS_Chandra, Swift, and eRASS1. This class is essential for studies involving the comparison of X-ray sources across different catalogs and determining optimal pointing positions for telescopic observations.

    Note:
        - The class handles catalog comparison and analysis, including calculations of photon index and NH values for various X-ray catalogs.
        - It also includes functionality for generating FITS tables, text files, and plots for the analyzed data.
        - The class supports different operating systems, catering to specific functionalities based on the system.
        - Key operations include identifying variable sources, modeling source spectra, and calculating vignetting factors for sources.

    Examples
    --------
    Creating an instance of CompareCatalog with necessary parameters:
    
    >>> catalog_path = ("path/to/catalog_1.fits","path/to/catalog_2.fits", "key_1", "key_2")
    >>> compare_catalog = CompareCatalog(catalog_path, 5*u.arcmin, simulation_data, user_table)

    Important:
        - The initialization of this class requires the paths to two catalogs, a radius for analysis, and a dictionary containing simulation data.
        - The methods within the class perform a range of analytical tasks, from opening and processing catalogs to modeling spectra and plotting data.
        - The class supports a broad spectrum of X-ray astronomy catalog formats and provides tools for detailed comparative analysis.
    """
   
    def __init__(self, catalog_path: tuple, radius, simulation_data: dict, exp_time: int) -> None:
        """
        Initializes the CompareCatalog class for comparing astronomical catalogs.

        This constructor sets up the class by opening and processing two specified catalogs, identifying nearby sources, and performing various calculations and analyses such as photon index and NH value computations. It also handles different functionalities based on the operating system and prepares for the visualization of source spectra and the calculation of vignetting factors.

        Args:
            catalog_path (tuple): A tuple containing the paths and keys for two catalogs (path_1, path_2, key_1, key_2).
            radius: The radius around the object position to consider for analysis.
            simulation_data (dict): A dictionary containing simulation data such as object data and telescope data.
            exp_time (int): The exposure time used in the analysis.

        Attributes:
        
        .. attribute:: nearby_sources_table_1
            :type: Table
            :value: A table of nearby sources from the first catalog.

        .. attribute:: nearby_sources_table_2
            :type: Table
            :value: A table of nearby sources from the second catalog.

        .. attribute:: nearby_sources_position_1
            :type: SkyCoord
            :value: Sky coordinates of nearby sources from the first catalog.

        .. attribute:: nearby_sources_position_2
            :type: SkyCoord
            :value: Sky coordinates of nearby sources from the second catalog.

        .. attribute:: var_index_1
            :type: List
            :value: Indices of variable sources in the first catalog.

        .. attribute:: var_index_2
            :type: List
            :value: Indices of variable sources in the second catalog.

        .. attribute:: total_spectra_1
            :type: List
            :value: Modeled total spectra for the first catalog.

        .. attribute:: total_spectra_2
            :type: List
            :value: Modeled total spectra for the second catalog.

        .. attribute:: total_var_spectra_1
            :type: List
            :value: Modeled spectra for variable sources in the first catalog.

        .. attribute:: total_var_spectra_2
            :type: List
            :value: Modeled spectra for variable sources in the second catalog.

        .. attribute:: count_rates_1
            :type: List
            :value: Calculated count rates for sources in the first catalog.

        .. attribute:: count_rates_2
            :type: List
            :value: Calculated count rates for sources in the second catalog.

        .. attribute:: data_1
            :type: Dict
            :value: Data for plotting spectra of the first catalog.

        .. attribute:: data_2
            :type: Dict
            :value: Data for plotting spectra of the second catalog.

        .. attribute:: instrument
            :type: Instrument
            :value: Instrument used for spectral modeling.

        Note:
        - The constructor performs preliminary data preparation, which is critical for the subsequent analytical methods in the class.
        - It assumes the catalogs are in a format compatible with the class methods and the simulation data provided is accurate.

        Important:
        - Proper initialization with valid parameters is crucial for the correct functioning of the class.
        - The class is capable of handling large datasets, but performance might vary based on the system's capabilities and the size of the data.

        The class is designed to handle a variety of tasks, including the processing of FITS tables, modeling of source spectra, and plotting of data. It also includes methods for writing analysis results to text files.
        """
        
        path_1, path_2 = catalog_path[0], catalog_path[1]
        key_1, key_2 = catalog_path[2], catalog_path[3]

        table_1, table_2 = self.open_catalog(key=(key_1, key_2), path=(path_1, path_2), radius=radius, object_data=simulation_data["object_data"])
        self.nearby_sources_table_1, self.nearby_sources_table_2, self.nearby_sources_position_1, self.nearby_sources_position_2 = self.find_nearby_sources(table=(table_1, table_2), radius=radius, simulation_data=simulation_data, key=(key_1, key_2))

        # ---------- Xmm_DR13 photon index and nh value --------- #
        if "Xmm_DR13" in [key_1, key_2]:
            print(f"\nCalculation of photon index and nh value for {colored('Xmm DR13 catalog', 'yellow')}.")
            #TODO add method to find variable sources in xmm 2 athena catalog with index_table
            xmm_index = [key_1, key_2].index("Xmm_DR13")
            if xmm_index == 0:
                self.nearby_sources_table_1, self.index_table = self.photon_index_nh_for_xmm(os_dictionary=simulation_data["os_dictionary"], xmm_index=xmm_index)
                vignet_data_1 = ["SC_RA", "SC_DEC", "IAUNAME"]
            elif xmm_index == 1:
                self.nearby_sources_table_2, self.index_table = self.photon_index_nh_for_xmm(os_dictionary=simulation_data["os_dictionary"], xmm_index=xmm_index)
                vignet_data_2 = ["SC_RA", "SC_DEC", "IAUNAME"]

        # ---------- CS_Chandra photon index and nh value ---------- #
        if "CSC_2.0" in [key_1, key_2]:
            print(f"\nCalculation of photon index and nh value for {colored('Chandra cone search catalog', 'yellow')}.")
            csc_index = [key_1, key_2].index("CSC_2.0")
            #TODO add method to find variable sources in chandra cone search catalog
            if csc_index == 0:
                self.nearby_sources_table_1 = self.threshold(cone_search_catalog=self.nearby_sources_table_1)
                self.nearby_sources_table_1 = self.photon_index_nh_for_csc(csc_index=csc_index)
                vignet_data_1 = ["ra", "dec", "name"]
            elif csc_index == 1:
                self.nearby_sources_table_2 = self.threshold(cone_search_catalog=self.nearby_sources_table_2)
                self.nearby_sources_table_2 = self.photon_index_nh_for_csc(csc_index=csc_index)
                vignet_data_2 = ["ra", "dec", "name"]

        # ---------- Swift photon index and nh value ---------- #
        if "Swift" in [key_1, key_2]:
            print(f"\nCalculation of photon index and nh value for {colored('Swift catalog', 'yellow')}.")
            swi_index = [key_1, key_2].index("Swift")
            if swi_index == 0:
                self.nearby_sources_table_1 = self.photon_index_nh_for_other_catalog(key="Swift", table=self.nearby_sources_table_1)
                vignet_data_1 = ["RA", "DEC", "Swift_IAUNAME"]
            if swi_index == 1:
                self.nearby_sources_table_2 = self.photon_index_nh_for_other_catalog(key="Swift", table=self.nearby_sources_table_2)
                vignet_data_2 = ["RA", "DEC", "Swift_IAUNAME"]

        # ---------- eRASS1 photon index and nh value ---------- #
        if "eRASS1" in [key_1, key_2]:
            print(f"\nCalculation of photon index and nh value for {colored('eRASS1 catalog', 'yellow')}.")
            ero_index = [key_1, key_2].index("eRASS1")
            if ero_index == 0:
                self.nearby_sources_table_1 = self.photon_index_nh_for_other_catalog(key="eRASS1", table=self.nearby_sources_table_1)
                vignet_data_1 = ["RA", "DEC", "eRASS_IAUNAME"]
            if ero_index == 1:
                self.nearby_sources_table_2 = self.photon_index_nh_for_other_catalog(key="eRASS1", table=self.nearby_sources_table_2)
                vignet_data_2 = ["RA", "DEC", "eRASS_IAUNAME"]

        self.neighbourhood_of_object(key=(key_1, key_2), simulation_data=simulation_data, radius=radius)
        self.model_dictionary_1, self.model_dictionary_2 = self.dictionary_model(key=(key_1, key_2))

        if platform.system() != "Windows":
            self.count_rates_1, self.count_rates_2 = self.count_rate()
        elif platform.system() == "Windows":
            self.count_rates_1, self.nearby_sources_table_1 = self.xslx_to_py(args=key_1, table=self.nearby_sources_table_1, simulation_data=simulation_data, radius=radius.value)
            self.count_rates_2, self.nearby_sources_table_2 = self.xslx_to_py(args=key_2, table=self.nearby_sources_table_2, simulation_data=simulation_data, radius=radius.value)


        self.vector_dictionary_1, self.vector_dictionary_2, self.OptimalPointingIdx_1, self.OptimalPointingIdx_2 = self.calculate_opti_point(simulation_data=simulation_data, key=(key_1, key_2))


        self.vignetting_factor_1, self.nearby_sources_table_1 = c_f.vignetting_factor(OptimalPointingIdx=self.OptimalPointingIdx_1, vector_dictionary=self.vector_dictionary_1, simulation_data=simulation_data, data=vignet_data_1, nearby_sources_table=self.nearby_sources_table_1)
        self.vignetting_factor_2, self.nearby_sources_table_2 = c_f.vignetting_factor(OptimalPointingIdx=self.OptimalPointingIdx_2, vector_dictionary=self.vector_dictionary_2, simulation_data=simulation_data, data=vignet_data_2, nearby_sources_table=self.nearby_sources_table_2)

        self.master_source_path = self.variability_table(simulation_data=simulation_data, radius=radius.value)
        
        print(f"{colored(f'Find variable index for {key_1} catalog', 'blue')}")
        self.var_index_1 = self.variability_index(key=key_1, iauname=vignet_data_1[2], nearby_sources_table=self.nearby_sources_table_1)
        print(f"{colored(f'Find variable index for {key_2} catalog', 'blue')}")
        self.var_index_2 = self.variability_index(key=key_2, iauname=vignet_data_2[2], nearby_sources_table=self.nearby_sources_table_2)
        
        # ---------- fits table ---------- #
        self.write_fits_table(table=self.nearby_sources_table_1, key=key_1, os_dictionary=simulation_data["os_dictionary"])
        self.write_fits_table(table=self.nearby_sources_table_2, key=key_2, os_dictionary=simulation_data["os_dictionary"])
        # -------------------------------- #
        
        self.total_spectra_1, self.total_spectra_2, self.total_var_spectra_1, self.total_var_spectra_2, self.obsconfig = self.modeling_source_spectra(simulation_data=simulation_data, exp_time=exp_time, key=(key_1, key_2))
        self.data_1, self.data_2 = self.total_spectra_plot(simulation_data=simulation_data, radius=radius.value,obsconfig=self.obsconfig, key=(key_1, key_2))
        self.write_txt_file(simulation_data=simulation_data, data_1=self.data_1, data_2=self.data_2, key=(key_1, key_2))
        
    
    def open_catalog(self, key: Tuple, path:Tuple, radius, object_data: Dict) -> Tuple[Table, Table]:
        """
        Opens and processes astronomical catalogs for further analysis.

        This method loads data from specified catalogs, identified by keys and paths, and processes it for analysis. It handles different types of catalogs, employing specific procedures to load and convert the data into a format suitable for analytical operations.

        Parameters:
            key (Tuple[str, str]): A tuple containing the keys of the catalogs to be opened. These keys identify the specific catalogs and dictate the corresponding data-loading procedures.
            path (Tuple[str, str]): A tuple containing the file paths of the catalogs to be opened.
            radius (float): The radius around the object's position, used for catalog searching, typically measured in arcminutes.
            object_data (Dict): A dictionary containing information about the object of interest, such as its name.

        Returns:
            Tuple[Table, Table]: A tuple consisting of two `Table` objects that contain the data from the opened catalogs.

        Note:
            The method executes the following steps:
            - Utilizes the Chandra Source Catalog cone search service to find sources within the specified radius from the object's position if one of the catalogs is "CSC_2.0".
            - Opens FITS files from the provided paths to load data for other catalogs.
            - Converts the loaded data into `Table` objects, facilitating further analysis.
            - Outputs a confirmation message once the catalogs are successfully loaded.

        This method streamlines the access to astronomical data from different sources, thereby aiding in conducting comparative analyses between various catalogs.
        """
        
        if key[0] == "CSC_2.0":
            cone = vo.dal.SCSService('http://cda.cfa.harvard.edu/csc2scs/coneSearch') 
            name = SkyCoord.from_name(object_data['object_name'])
            cone_search_catalog = cone.search(pos=name, radius=radius, verbosity=3)
            table_1 = cone_search_catalog.to_table()
            self.cone_search_table = table_1
            
            with fits.open(path[1]) as data:
                table_2 = Table(data[1].data)
                
            print(f"{key[0]} and {key[1]} catalog are loaded !")
            
        elif key[1] == "CSC_2.0":
            cone = vo.dal.SCSService('http://cda.cfa.harvard.edu/csc2scs/coneSearch') 
            name = SkyCoord.from_name(object_data['object_name'])
            cone_search_catalog = cone.search(pos=name, radius=radius, verbosity=3)
            table_2 = cone_search_catalog.to_table()
            self.cone_search_table = table_2
            
            with fits.open(path[0]) as data:
                table_1 = Table(data[1].data)
                
            print(f"{key[0]} and {key[1]} catalog are loaded !")
        
        else:
            with fits.open(path[0]) as data_1, fits.open(path[1]) as data_2:
                table_1 = Table(data_1[1].data)
                table_2 = Table(data_2[1].data)
                
            print(f"{key[0]} and {key[1]} catalog are loaded !")
                
        return table_1, table_2
    
    
    def find_nearby_sources(self, table, radius, simulation_data, key: Tuple) -> Tuple[Table, Table, SkyCoord, SkyCoord]:
        """
        Identifies and processes nearby sources from astronomical catalogs based on the specified object's position.

        Args:
            table (Tuple[Table, Table]): A tuple containing two `Table` objects with data from the catalogs to be processed.
            radius (float): The radius around the object's position to consider when finding nearby sources, typically in arcminutes.
            simulation_data (dict): A dictionary containing simulation data, including the object's position and other relevant information.
            key (Tuple[str, str]): A tuple containing the keys identifying the specific catalogs to be processed.

        Returns:
            Tuple[Table, Table, SkyCoord, SkyCoord]: A tuple containing two `Table` objects with nearby sources from each catalog and two `SkyCoord` objects representing the positions of these sources.

        Note:
            This method is tailored to handle specific astronomical catalogs. Modifications may be required to adapt it to other catalog formats or to handle additional catalogs.

        Important:
            The accuracy of the results depends significantly on the quality and completeness of the input catalog data. Users should ensure that the catalog data is up-to-date and comprehensive to achieve the best results.

        This method processes the following steps:
        - Determines the field of view based on the specified radius and the object's position.
        - Filters the sources in each catalog within the field of view and closer than the specified radius to the object's position.
        - Applies specific procedures for handling different combinations of catalogs (e.g., CSC_2.0, Xmm_DR13, Swift, eRASS1).
        - Calculates sky coordinates for the nearby sources.
        - Returns the processed tables and coordinates for further analysis.

        This method is essential for focusing on sources in the vicinity of a specified object, enabling detailed analysis and comparison of these sources across different catalogs.
        """
    
        
        object_data = simulation_data["object_data"]
        object_position = object_data['object_position']
        object_name = object_data["object_name"]
        
        table_1, table_2 = table
        
        field_of_view = radius + 5*u.arcmin
        min_ra, max_ra = object_position.ra - field_of_view, object_position.ra + field_of_view
        min_dec, max_dec = object_position.dec - field_of_view, object_position.dec + field_of_view
        
        if key[0] == "CSC_2.0":
            if key [1] == "Xmm_DR13":
                # ---------- Xmm catalog ---------- #
                small_table_2 = Table(names=table_2.colnames,
                                      dtype=table_2.dtype)
                nearby_sources_table_2 = Table(names=table_2.colnames,
                                               dtype=table_2.dtype)
                
                print(fr"{colored('Reducing Xmm catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
                for number in tqdm(range(len(table_2))):
                    if min_ra/u.deg < table_2["SC_RA"][number] < max_ra/u.deg and min_dec/u.deg < table_2["SC_DEC"][number] < max_dec/u.deg:
                        small_table_2.add_row(table_2[number])
                        
                src_position = SkyCoord(ra=small_table_2['SC_RA'], dec=small_table_2['SC_DEC'], unit=u.deg)

                print(f"{colored(f'Find sources close to {object_name} with Chandra catalog', 'blue')}")
                for number in tqdm(range(len(small_table_2))):
                    if c_f.ang_separation(object_position, src_position[number]) < radius:
                        nearby_sources_table_2.add_row(small_table_2[number])
                nearby_sources_position_2 = SkyCoord(ra=nearby_sources_table_2["SC_RA"], dec=nearby_sources_table_2["SC_DEC"], unit=u.deg)
                
                # ---------- Chandra cone search catalog ---------- #
                nearby_sources_table_1 = self.cone_search_table
                nearby_sources_position_1 = SkyCoord(ra=nearby_sources_table_1["ra"], dec=nearby_sources_table_1["dec"], unit=u.deg)
                
                print(f"{len(nearby_sources_table_1)} sources was detected with Chandra catalog and {len(nearby_sources_table_2)} sources with Xmm catalog close to {object_name}")
                return nearby_sources_table_1, nearby_sources_table_2, nearby_sources_position_1, nearby_sources_position_2
            
            else:
                # ---------- catalog Swift or eRASS1 ---------- #
                small_table_2 = Table(names=table_2.colnames,
                                      dtype=table_2.dtype)
                not_unique_table = Table(names=table_2.colnames,
                                               dtype=table_2.dtype)
                
                print(f"{colored(f'Reducing {key[1]} catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
                for number in tqdm(range(len(table_2))):
                    if min_ra/u.deg < table_2["RA"][number] < max_ra/u.deg and min_dec/u.deg < table_2["DEC"][number] < max_dec/u.deg:
                        small_table_2.add_row(table_2[number])
                        
                src_position = SkyCoord(ra=small_table_2['RA'], dec=small_table_2['DEC'], unit=u.deg)

                print(f"{colored(f'Find sources close to {object_name} with {key[1]} catalog', 'blue')}")
                for number in tqdm(range(len(small_table_2))):
                    if c_f.ang_separation(object_position, src_position[number]) < radius:
                        not_unique_table.add_row(small_table_2[number])
                
                if len(not_unique_table) != 0:
                    if key[1] == "Swift":
                        column_name = {"source_name": "Swift_IAUNAME",
                                       "right_ascension": "RA",
                                       "declination": "DEC",
                                       "catalog_name": "Swift"}
                    elif key[1] == "eRASS1":
                        column_name = {"source_name": "eRASS_IAUNAME",
                                       "right_ascension": "RA",
                                       "declination": "DEC",
                                       "catalog_name": "eRASS1"}

                    nearby_sources_table_2 = u_f.create_unique_sources_catalog(nearby_sources_table=not_unique_table, column_name=column_name)
                    nearby_sources_position_2 = SkyCoord(ra=nearby_sources_table_2["RA"], dec=nearby_sources_table_2["DEC"], unit=u.deg)
                    
                    # ---------- Chandra cone search catalog ---------- #
                    nearby_sources_table_1 = self.cone_search_table
                    nearby_sources_position_1 = SkyCoord(ra=nearby_sources_table_1["ra"], dec=nearby_sources_table_1["dec"], unit=u.deg)
                    
                    print(f"{len(nearby_sources_table_1)} sources was detected with Chandra catalog and {len(nearby_sources_table_2)} sources with {key[1]} catalog close to {object_name}")
                    return nearby_sources_table_1, nearby_sources_table_2, nearby_sources_position_1, nearby_sources_position_2
                else:
                    print(f"No sources detected in {colored(key[1], 'red')} catalog !")
                    sys.exit()
                
        elif key[1] == "CSC_2.0":
            if key[0] == "Xmm_DR13":
                # ---------- Xmm catalog ---------- #
                small_table_1 = Table(names=table_1.colnames,
                                      dtype=table_1.dtype)
                nearby_sources_table_1 = Table(names=table_1.colnames,
                                               dtype=table_1.dtype)
                
                print(fr"{colored(f'Reducing {key[0]} catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
                for number in tqdm(range(len(table_1))):
                    if min_ra/u.deg < table_1["SC_RA"][number] < max_ra/u.deg and min_dec/u.deg < table_1["SC_DEC"][number] < max_dec/u.deg:
                        small_table_1.add_row(table_1[number])
                        
                src_position = SkyCoord(ra=small_table_1['SC_RA'], dec=small_table_1['SC_DEC'], unit=u.deg)

                print(f"{colored(f'Find sources close to {object_name} with {key[0]} catalog', 'blue')}")
                for number in tqdm(range(len(small_table_1))):
                    if c_f.ang_separation(object_position, src_position[number]) < radius:
                        nearby_sources_table_1.add_row(small_table_1[number])
                nearby_sources_position_1 = SkyCoord(ra=nearby_sources_table_1["SC_RA"], dec=nearby_sources_table_1["SC_DEC"], unit=u.deg)
                
                # ---------- Chandra cone search catalog ---------- #
                nearby_sources_table_2 = self.cone_search_table
                nearby_sources_position_2 = SkyCoord(ra=nearby_sources_table_2["ra"], dec=nearby_sources_table_2["dec"], unit=u.deg)
                
                print(f"{len(nearby_sources_table_1)} sources was detected with Xmm catalog and {len(nearby_sources_table_2)} sources with Chandra catalog close to {object_name}")
                return nearby_sources_table_1, nearby_sources_table_2, nearby_sources_position_1, nearby_sources_position_2
                
            else:
                # ---------- catalog Swift or eRASS1 ---------- #
                small_table_1 = Table(names=table_1.colnames,
                                      dtype=table_1.dtype)
                not_unique_table = Table(names=table_1.colnames,
                                         dtype=table_1.dtype)
                
                print(f"{colored(f'Reducing {key[0]} catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
                for number in tqdm(range(len(table_1))):
                    if min_ra/u.deg < table_1["RA"][number] < max_ra/u.deg and min_dec/u.deg < table_1["DEC"][number] < max_dec/u.deg:
                        small_table_1.add_row(table_1[number])
                        
                src_position = SkyCoord(ra=small_table_1["RA"], dec=small_table_1["DEC"], unit=u.deg)

                print(f"{colored(f'Find sources close to {object_name} with {key[0]} catalog', 'blue')}")
                for number in tqdm(range(len(small_table_1))):
                    if c_f.ang_separation(object_position, src_position[number]) < radius:
                        not_unique_table.add_row(small_table_1[number])
                if len(not_unique_table) != 0:
                    if key[0] == "Swift":
                        column_name = {"source_name": "Swift_IAUNAME",
                                       "right_ascension": "RA",
                                       "declination": "DEC",
                                       "catalog_name": "Swift"}
                    elif key[0] == "eRASS1":
                        column_name = {"source_name": "eRASS_IAUNAME",
                                       "right_ascension": "RA",
                                       "declination": "DEC",
                                       "catalog_name": "eRASS1"}
                    
                    nearby_sources_table_1 = u_f.create_unique_sources_catalog(nearby_sources_table=not_unique_table, column_name=column_name)
                    nearby_sources_position_1 = SkyCoord(ra=nearby_sources_table_1["RA"], dec=nearby_sources_table_1["DEC"], unit=u.deg)
                    
                    # ---------- Chandra cone search catalog ---------- #
                    nearby_sources_table_2 = self.cone_search_table
                    nearby_sources_position_2 = SkyCoord(ra=nearby_sources_table_2["ra"], dec=nearby_sources_table_2["dec"], unit=u.deg)
                    
                    print(f"{len(nearby_sources_table_1)} sources was detected with {key[0]} catalog and {len(nearby_sources_table_2)} sources with Chandra catalog close to {object_name}")
                    return nearby_sources_table_1, nearby_sources_table_2, nearby_sources_position_1, nearby_sources_position_2
                else:
                    print(f"No sources detected in {colored(key[0], 'red')} catalog !")
                    sys.exit()
        else:
            if key[0] == "Xmm_DR13" :
                # ---------- Xmm catalog ---------- #
                small_table_1 = Table(names=table_1.colnames,
                                      dtype=table_1.dtype)
                nearby_sources_table_1 = Table(names=table_1.colnames,
                                               dtype=table_1.dtype)
                
                # On gère ici la première table dans le cas ou elle est Xmm_DR13
                print(f"{colored(f'Reducing {key[0]} catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
                for number in tqdm(range(len(table_1))):
                    if min_ra/u.deg < table_1["SC_RA"][number] < max_ra/u.deg and min_dec/u.deg < table_1["SC_DEC"][number] < max_dec/u.deg:
                        small_table_1.add_row(table_1[number])
                        
                src_position = SkyCoord(ra=small_table_1["SC_RA"], dec=small_table_1["SC_DEC"], unit=u.deg)

                print(f"{colored(f'Find sources close to {object_name} with {key[0]} catalog', 'blue')}")
                for number in tqdm(range(len(small_table_1))):
                    if c_f.ang_separation(object_position, src_position[number]) < radius:
                        nearby_sources_table_1.add_row(small_table_1[number])
                        
                nearby_sources_position_1 = SkyCoord(ra=nearby_sources_table_1["SC_RA"], dec=nearby_sources_table_1["SC_DEC"], unit=u.deg)
                
                # ---------- catalog Swift or eRASS1 ---------- #
                
                small_table_2 = Table(names=table_2.colnames,
                                      dtype=table_2.dtype)
                not_unique_table = Table(names=table_2.colnames,
                                         dtype=table_2.dtype)
                
                print(f"{colored(f'Reducing {key[1]} catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
                for number in tqdm(range(len(table_2))):
                    if min_ra/u.deg < table_2["RA"][number] < max_ra/u.deg and min_dec/u.deg < table_2["DEC"][number] < max_dec/u.deg:
                        small_table_2.add_row(table_2[number])
                        
                src_position = SkyCoord(ra=small_table_2["RA"], dec=small_table_2["DEC"], unit=u.deg)

                print(f"{colored(f'Find sources close to {object_name} with {key[1]} catalog', 'blue')}")
                for number in tqdm(range(len(small_table_2))):
                    if c_f.ang_separation(object_position, src_position[number]) < radius:
                        not_unique_table.add_row(small_table_2[number])
                if len(not_unique_table) != 0:
                    if key[1] == "Swift":
                        column_name = {"source_name": "Swift_IAUNAME",
                                       "right_ascension": "RA",
                                       "declination": "DEC",
                                       "catalog_name": "Swift"}
                    elif key[1] == "eRASS1":
                        column_name = {"source_name": "eRASS_IAUNAME",
                                       "right_ascension": "RA",
                                       "declination": "DEC",
                                       "catalog_name": "eRASS1"}
                                    
                    nearby_sources_table_2 = u_f.create_unique_sources_catalog(nearby_sources_table=not_unique_table, column_name=column_name)
                    nearby_sources_position_2 = SkyCoord(ra=nearby_sources_table_2["RA"], dec=nearby_sources_table_2["DEC"], unit=u.deg)
                    
                    print(f"{len(nearby_sources_table_1)} sources was detected with {key[0]} catalog and {len(nearby_sources_table_2)} sources with {key[1]} catalog close to {object_name}")
                    return nearby_sources_table_1, nearby_sources_table_2, nearby_sources_position_1, nearby_sources_position_2
                else:
                    print(f"No sources detected in {colored(key[1], 'red')} catalog !")
                    sys.exit()
                
            elif key[1] == "Xmm_DR13":
                # ---------- catalog Swift or eRASS1 ---------- #
                small_table_1 = Table(names=table_1.colnames,
                                      dtype=table_1.dtype)
                not_unique_table = Table(names=table_1.colnames,
                                         dtype=table_1.dtype)
                
                print(f"{colored(f'Reducing {key[0]} catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
                for number in tqdm(range(len(table_1))):
                    if min_ra/u.deg < table_1["RA"][number] < max_ra/u.deg and min_dec/u.deg < table_1["DEC"][number] < max_dec/u.deg:
                        small_table_1.add_row(table_1[number])
                        
                src_position = SkyCoord(ra=small_table_1["RA"], dec=small_table_1["DEC"], unit=u.deg)

                print(f"{colored(f'Find sources close to {object_name} with {key[0]} catalog', 'blue')}")
                for number in tqdm(range(len(small_table_1))):
                    if c_f.ang_separation(object_position, src_position[number]) < radius:
                        not_unique_table.add_row(small_table_1[number])
                
                if len(not_unique_table) != 0:
                    if key[0] == "Swift":
                        column_name = {"source_name": "Swift_IAUNAME",
                                       "right_ascension": "RA",
                                       "declination": "DEC",
                                       "catalog_name": "Swift"}
                    elif key[0] == "eRASS1":
                        column_name = {"source_name": "eRASS_IAUNAME",
                                       "right_ascension": "RA",
                                       "declination": "DEC",
                                       "catalog_name": "eRASS1"}
                    
                    nearby_sources_table_1 = u_f.create_unique_sources_catalog(nearby_sources_table=not_unique_table, column_name=column_name)
                    nearby_sources_position_1 = SkyCoord(ra=nearby_sources_table_1["RA"], dec=nearby_sources_table_1["DEC"], unit=u.deg)
                    
                    # ---------- Xmm catalog ---------- #
                    
                    small_table_2 = Table(names=table_2.colnames,
                                        dtype=table_2.dtype)
                    nearby_sources_table_2 = Table(names=table_2.colnames,
                                                dtype=table_2.dtype)
                    
                    print(f"{colored(f'Reducing {key[1]} catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
                    for number in tqdm(range(len(table_2))):
                        if min_ra/u.deg < table_2["SC_RA"][number] < max_ra/u.deg and min_dec/u.deg < table_2["SC_DEC"][number] < max_dec/u.deg:
                            small_table_2.add_row(table_2[number])
                            
                    src_position = SkyCoord(ra=small_table_2['SC_RA'], dec=small_table_2['SC_DEC'], unit=u.deg)

                    print(f"{colored(f'Find sources close to {object_name} with {key[1]} catalog', 'blue')}")
                    for number in tqdm(range(len(small_table_2))):
                        if c_f.ang_separation(object_position, src_position[number]) < radius:
                            nearby_sources_table_2.add_row(small_table_2[number])
                    nearby_sources_position_2 = SkyCoord(ra=nearby_sources_table_2["SC_RA"], dec=nearby_sources_table_2["SC_DEC"], unit=u.deg)
                    
                    print(f"{len(nearby_sources_table_1)} sources was detected with {key[0]} catalog and {len(nearby_sources_table_2)} sources with {key[1]} catalog close to {object_name}")
                    return nearby_sources_table_1, nearby_sources_table_2, nearby_sources_position_1, nearby_sources_position_2
                else:
                    print(f"No sources detected in {colored(key[0], 'red')} catalog !")
                    sys.exit()
                
            else:
                # ---------- eRASS1 or Swift catalog ---------- #
                small_table_1 = Table(names=table_1.colnames,
                                      dtype=table_1.dtype)
                not_unique_table_1 = Table(names=table_1.colnames,
                                           dtype=table_1.dtype)
                
                print(f"{colored(f'Reducing {key[0]} catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
                for number in tqdm(range(len(table_1))):
                    if min_ra/u.deg < table_1["RA"][number] < max_ra/u.deg and min_dec/u.deg < table_1["DEC"][number] < max_dec/u.deg:
                        small_table_1.add_row(table_1[number])
                        
                src_position = SkyCoord(ra=small_table_1["RA"], dec=small_table_1["DEC"], unit=u.deg)

                print(f"{colored(f'Find sources close to {object_name} with {key[0]} catalog', 'blue')}")
                for number in tqdm(range(len(small_table_1))):
                    if c_f.ang_separation(object_position, src_position[number]) < radius:
                        not_unique_table_1.add_row(small_table_1[number])
                        
                if len(not_unique_table_1) != 0:
                    if key[0] == "Swift":
                        column_name = {"source_name": "Swift_IAUNAME",
                                       "right_ascension": "RA",
                                       "declination": "DEC",
                                       "catalog_name": "Swift"}
                    elif key[0] == "eRASS1":
                        column_name = {"source_name": "eRASS_IAUNAME",
                                       "right_ascension": "RA",
                                       "declination": "DEC",
                                       "catalog_name": "eRASS1"}
                        
                    nearby_sources_table_1 = u_f.create_unique_sources_catalog(nearby_sources_table=not_unique_table_1, column_name=column_name)
                    nearby_sources_position_1 = SkyCoord(ra=nearby_sources_table_1["RA"], dec=nearby_sources_table_1["DEC"], unit=u.deg)

                    # ---------- eRASS1 or Swift catalog ---------- #
                    small_table_2 = Table(names=table_2.colnames,
                                        dtype=table_2.dtype)
                    not_unique_table_2 = Table(names=table_2.colnames,
                                            dtype=table_2.dtype)
                    
                    print(f"{colored(f'Reducing {key[1]} catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
                    for number in tqdm(range(len(table_2))):
                        if min_ra/u.deg < table_2["RA"][number] < max_ra/u.deg and min_dec/u.deg < table_2["DEC"][number] < max_dec/u.deg:
                            small_table_2.add_row(table_2[number])
                            
                    src_position = SkyCoord(ra=small_table_2["RA"], dec=small_table_2["DEC"], unit=u.deg)

                    print(f"{colored(f'Find sources close to {object_name} with {key[1]} catalog', 'blue')}")
                    for number in tqdm(range(len(small_table_2))):
                        if c_f.ang_separation(object_position, src_position[number]) < radius:
                            not_unique_table_2.add_row(small_table_2[number])
                    
                    if len(not_unique_table_2) != 0:
                        if key[1] == "Swift":
                            column_name = {"source_name": "Swift_IAUNAME",
                                           "right_ascension": "RA",
                                           "declination": "DEC",
                                           "catalog_name": "Swift"}
                        elif key[1] == "eRASS1":
                            column_name = {"source_name": "eRASS_IAUNAME",
                                           "right_ascension": "RA",
                                           "declination": "DEC",
                                           "catalog_name": "eRASS1"}
                            
                        nearby_sources_table_2 = u_f.create_unique_sources_catalog(nearby_sources_table=not_unique_table_2, column_name=column_name)
                        nearby_sources_position_2 = SkyCoord(ra=nearby_sources_table_2["RA"], dec=nearby_sources_table_2["DEC"], unit=u.deg)
                        
                        print(f"{len(nearby_sources_table_1)} sources was detected with {key[0]} catalog and {len(nearby_sources_table_2)} sources with {key[1]} catalog close to {object_name}")
                        return nearby_sources_table_1, nearby_sources_table_2, nearby_sources_position_1, nearby_sources_position_2
                    else:
                        print(f"No sources detected in {colored(key[1], 'red')} catalog !")
                        sys.exit()
                else:
                    print(f"No sources detected in {colored(key[0], 'red')} catalog !")
                    sys.exit()
                    
    
    def optimization(self, index: int, key: str, table: Table) -> Tuple[List, List]:
        """
        Optimizes the parameters of an absorbed power-law model based on observed fluxes in different energy bands.

        Args:
            index (int): The index of the source in the catalog for optimization.
            key (str): The key identifying the catalog (e.g., 'XMM', 'CS_Chandra', 'Swift', 'eRASS1').
            table (Table): The table with observed flux data and other relevant information.

        Returns:
            Tuple[List, List]: A tuple where the first element is a list of optimized parameters, and the second is the absorbed photon index.

        Note:
            This method is specifically tailored for astronomical data and assumes familiarity with astrophysical concepts and data structures.

        Important:
            The reliability of the optimization depends on the quality and precision of the input data. Users must ensure data accuracy and appropriateness of the model for the specific astrophysical context.

        The method performs steps including retrieving observed fluxes and their errors, defining an absorbed power-law function, curve fitting to find best-fit parameters, and returning optimized parameters and photon index.
        """
        
        if key == "XMM":
            interp_data = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                           "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                           "energy_band_center": dict_cat.dictionary_catalog[key]["energy_band_center"],
                           "energy_band_half_width": dict_cat.dictionary_catalog[key]["energy_band_half_width"]}
        
        if key == "CS_Chandra":
            interp_data = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                           "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                           "energy_band_center": dict_cat.dictionary_catalog[key]["energy_band_center"],
                           "energy_band_half_width": dict_cat.dictionary_catalog[key]["energy_band_half_width"]}
            
        if key == "Swift":
            interp_data = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                           "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                           "energy_band_center": dict_cat.dictionary_catalog[key]["energy_band_center"],
                           "energy_band_half_width": dict_cat.dictionary_catalog[key]["energy_band_half_width"]}
        
        if key == "eRASS1":
            interp_data = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                           "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                           "energy_band_center": dict_cat.dictionary_catalog[key]["energy_band_center"],
                           "energy_band_half_width": dict_cat.dictionary_catalog[key]["energy_band_half_width"]}

        def absorbed_power_law(energy_band, constant, gamma):
            sigma = np.array(np.linspace(1e-20, 1e-24, len(energy_band)), dtype=float)
            return (constant * energy_band ** (-gamma)) * (np.exp(-sigma * 3e20))
        
        def absorbed_power_law2(energy_band, constant, gamma):
            energy_band = np.array(energy_band, dtype=float)
            sigma = np.array([1e-20, 1e-22, 1e-24], dtype=float)
            return (constant * energy_band **(-gamma)) * (np.exp(-sigma*3e20))
        
        tab_width = 2 * interp_data["energy_band_half_width"]
        
        flux_obs = [table[name][index] for name in interp_data["band_flux_obs"]]
        flux_err = [[table[err_0][index] for err_0 in interp_data["band_flux_obs_err"][0]],
                    [table[err_1][index] for err_1 in interp_data["band_flux_obs_err"][1]]]

        flux_err_obs = [np.mean([flux_err_0, flux_err_1]) for flux_err_0, flux_err_1 in zip(flux_err[0], flux_err[1])]
        
        y_array = [num/det for num, det in zip(flux_obs, tab_width)]
        yerr_array = [num/det for num, det in zip(flux_err_obs, tab_width)]

        try:
            if key == "CS_Chandra":
                popt, pcov = curve_fit(absorbed_power_law2, interp_data["energy_band_center"], y_array, sigma=yerr_array)
                constant, absorb_pho_index = popt
            else:
                popt, pcov = curve_fit(absorbed_power_law, interp_data["energy_band_center"], y_array, sigma=yerr_array)
                constant, absorb_pho_index = popt
        except Exception as error:
            #popt = (1e-14, 1.7)
            if key == "CS_Chandra":
                constant = 1.0  # Reasonable default for constant
                absorb_pho_index = 1.7  # Default for photon index
                popt = [constant, absorb_pho_index] 
            else:
                popt = (1e-14, 1.7) # Use these defaults for popt
    
        #popt, pcov = curve_fit(absorbed_power_law, interp_data["energy_band_center"], y_array, sigma=yerr_array)
        #constant, absorb_pho_index = popt
        if key == "CS_Chandra":
            optimization_parameters = (interp_data["energy_band_center"], y_array, yerr_array, absorbed_power_law2(interp_data["energy_band_center"], *popt))
        else:
            optimization_parameters = (interp_data["energy_band_center"], y_array, yerr_array, absorbed_power_law(interp_data["energy_band_center"], *popt))
        
        return optimization_parameters, absorb_pho_index
    

    def visualization_interp(self, optimization_parameters, photon_index, key) -> None:
        """
        Visualizes the interpolation of photon indices using an absorbed power-law model.

        Args:
            optimization_parameters (List): Optimization parameters from the absorbed power-law model fitting.
            photon_index (List): List of photon indices for each source or observation.
            key (str): The catalog key (e.g., 'XMM', 'CS_Chandra', 'Swift', 'eRASS1').

        Note:
            This visualization method is designed for astrophysical data and requires an understanding of spectral analysis in astronomy.

        Important:
            The clarity and interpretability of the plots depend on the resolution and range of the input data. Users should carefully choose the energy bands and flux ranges for meaningful visualizations.

        The method involves plotting observed fluxes against energy bands, overlaying the best-fit model, and annotating photon indices, offering insights into the spectral properties of astronomical sources.
        """
        
        energy_band = dict_cat.dictionary_catalog[key]["energy_band_center"]
        
        fig, axes = plt.subplots(1, 1, figsize=(15, 8))
        fig.suptitle("Interpolation Photon Index plot", fontsize=20)
        fig.text(0.5, 0.04, 'Energy [keV]', ha='center', va='center')
        fig.text(0.04, 0.5, 'Flux $[erg.cm^{-2}.s^{-1}.keV^{-1}]$', ha='center', va='center', rotation='vertical')
        
        for item in range(len(optimization_parameters)):
            flux_obs = optimization_parameters[item][1]
            flux_obs_err = optimization_parameters[item][2]
            absorbed_power_law = optimization_parameters[item][3]
            absorb_pho_index = photon_index[item]
            
            axes.errorbar(energy_band, flux_obs, flux_obs_err, fmt='*', color='red', ecolor='black')
            axes.plot(energy_band, absorbed_power_law, label=f"$\Gamma$ = {absorb_pho_index:.8f}")
    
        axes.legend(loc="upper left", ncol=4, fontsize=6)
        axes.loglog()
        
        plt.show()
    
    
    def photon_index_nh_for_xmm(self, os_dictionary: Dict, xmm_index: int) -> Tuple[Table, List]:
        """
        Calculates the photon index and hydrogen column density (NH) for sources in the XMM-Newton catalog.

        Args:
            os_dictionary (Dict): A dictionary containing paths and OS-related information for catalog access.
            xmm_index (int): An index indicating which nearby sources table corresponds to the XMM-Newton catalog.

        Returns:
            Tuple[Table, List]: An updated table with 'Photon Index' and 'Nh' columns, and a table of indexes mapping sources to the XMM catalogs.

        Note:
            Assumes familiarity with X-ray astronomy catalogs, particularly XMM-Newton. Suitable for users with astrophysical data processing experience.

        Important:
            Accuracy of NH and photon index calculations is contingent upon the quality and completeness of the XMM catalog data and the robustness of the optimization process.

        The method accesses XMM-Newton catalogs, matches sources, calculates photon indices and NH values, visualizes absorbed power-law fits, and updates the table with these parameters.
        """
        
        catalog_datapath = os_dictionary["catalog_datapath"]
        xmm_dr11_path = os.path.join(catalog_datapath, "4XMM_DR11cat_v1.0.fits").replace("\\", "/")
        xmm_2_athena_path = os.path.join(catalog_datapath, "xmm2athena_D6.1_V3.fits").replace("\\", "/")

        while True:
            try:
                with fits.open(xmm_dr11_path) as data:
                    xmm_dr11_table = Table(data[1].data)
                    print(f"{colored('Xmm DR11', 'green')} catalog is loaded.")
                    break
            except Exception as error:
                print(f"{colored('An error occured : ', 'red')} {error}")
                xmm_dr11_path = str(input("Enter a new path for xmm_dr11 catalog : \n"))
        while True:
            try:
                with fits.open(xmm_2_athena_path) as data:
                    xmm_2_athena_table = Table(data[1].data)
                    print(f"{colored('Xmm 2 Athena', 'green')} catalog is loaded.")
                    break
            except Exception as error:
                print(f"{colored('An error occured : ', 'red')} {error}")
                xmm_2_athena_path = str(input("Enter a new path for xmm_2_athena catalog : \n"))

        if xmm_index == 0:
            nearby_sources_table = self.nearby_sources_table_1
        else: 
            nearby_sources_table = self.nearby_sources_table_2

        sources_number = len(nearby_sources_table)
        name_list = nearby_sources_table["IAUNAME"]
        nearby_sources_table_dr11 = Table(names=xmm_dr11_table.colnames,
                                          dtype=xmm_dr11_table.dtype)

        index_dr11 = np.array([], dtype=int)
        for name in name_list:
            if name in xmm_dr11_table["IAUNAME"]:
                index_name = list(xmm_dr11_table["IAUNAME"]).index(name)
                index_dr11 = np.append(index_dr11, index_name)
                nearby_sources_table_dr11.add_row(xmm_dr11_table[index_name])

        index_x2a = np.array([], dtype=int)
        for det_id in nearby_sources_table_dr11["DETID"]:
            if det_id in xmm_2_athena_table["DETID"]:
                index_det_id = list(xmm_2_athena_table["DETID"]).index(det_id)
                index_x2a = np.append(index_x2a, index_det_id)
            else:
                index_x2a = np.append(index_x2a, "No data found")

        column_names = ["Index in nearby_sources_table", "Index in xmm_dr11", "Index in x2a"]
        column_data = [[item for item in range(sources_number)], index_dr11, index_x2a]
        index_table = Table(names=column_names,
                            data=column_data)
        
        nh_list, photon_index_list, optimization_parameters = [], [], []
        for index in range(sources_number):
            if index_table["Index in x2a"][index] != "No data found":
                nh = xmm_2_athena_table["logNH_med"][index]
                nh_list.append(np.exp(nh * np.log(10)))
                photon_index_list.append(xmm_2_athena_table['PhoIndex_med'][index])
            else:
                nh_list.append(3e20)
                params, photon_index = self.optimization(index=index, key="XMM", table=nearby_sources_table)
                photon_index_list.append(photon_index)
                optimization_parameters.append(params)
                
        self.visualization_interp(optimization_parameters=optimization_parameters, photon_index=photon_index_list, key="XMM")
        
        col_names = ["Photon Index", "Nh"]
        col_data = [photon_index_list, nh_list]
        
        for name, data in zip(col_names, col_data):
            nearby_sources_table[name] = data
            
        return nearby_sources_table, index_table
        

    def threshold(self, cone_search_catalog: Table) -> Table:
        """
        Corrects and standardizes flux values in an astronomical catalog, addressing missing or undefined data points.

        Parameters:
            cone_search_catalog (Table): The table with flux data from the CSC or a similar catalog.

        Returns:
            Table: The corrected and standardized table with processed flux values.

        Note:
            This method is tailored for astronomical data processing, particularly for handling flux data in catalogs like CSC.

        Important:
            The effectiveness of this method depends on the consistency of data formats in the input catalog. It's crucial for users to verify that the catalog adheres to expected data structures.

        The method performs the following steps:
        - Iterates over each item in the catalog to check for undefined or missing flux values, represented as `np.ma.core.MaskedConstant`.
        - Replaces missing values in observed flux (`flux_obs`) and flux error (`flux_obs_err`) fields with the minimum numerical value found in the respective field.
        - Processes other flux-related fields (like `flux_aper_b`, `flux_aper__s/m/h`, and their error limits) in a similar manner.
        - Ensures that all flux-related fields in the catalog have consistent and usable numerical values, facilitating further analysis.

        This method is crucial for preparing astronomical data for analysis, ensuring that all flux-related fields are consistent and numerically valid, especially important in catalogs where missing or undefined values are common.
        """
        
        source_number = len(cone_search_catalog)
        key = "CS_Chandra"

        # name : flux_aper_b
        # algo to replace -- by the min numerical value of the list
        flux_obs = dict_cat.dictionary_catalog[key]["flux_obs"]
        flux_data = []
        for item in range(source_number):
            # iterate for each item in the list
            if not isinstance(cone_search_catalog[flux_obs][item], np.ma.core.MaskedConstant):
                # put in flux_data list each item of type different to np.ma.core.MaskedConstant
                flux_data.append(cone_search_catalog[flux_obs][item])
        flux = list(cone_search_catalog[flux_obs])
        corrected_flux_obs = np.nan_to_num(flux, nan=np.min(flux_data))
        cone_search_catalog[flux_obs] = corrected_flux_obs

        for name in dict_cat.dictionary_catalog[key]["flux_obs_err"]:
            flux_err = []
            for item in range(source_number):
                # iterate for each item in the list
                if not isinstance(cone_search_catalog[name][item], np.ma.core.MaskedConstant):
                    # put in flux_err list each item of type different to np.ma.core.MaskedConstant
                    flux_err.append(cone_search_catalog[name][item])
            flux_err_obs = list(cone_search_catalog[name])
            corrected_flux_err_obs = np.nan_to_num(flux_err_obs, nan=np.min(flux_err))
            cone_search_catalog[name] = corrected_flux_err_obs

        # name : flux_aper__s/m/h
        # algo to replace -- by the min numerical value of the list
        for name in dict_cat.dictionary_catalog[key]["band_flux_obs"]:
            # itera name in band_flux_obs 
            data = []
            for item in range(source_number):
                # iterate for each item in the list
                if not isinstance(cone_search_catalog[name][item], np.ma.core.MaskedConstant):
                    # put in data list each item of type different to np.ma.core.MaskedConstant
                    data.append(cone_search_catalog[name][item])
            flux = list(cone_search_catalog[name])
            corrected_flux = np.nan_to_num(flux, nan=np.min(data))
            cone_search_catalog[name] = corrected_flux


        # name : flux_aper_lo/hi_lim_s/m/h
        err_flux_neg, err_flux_pos = dict_cat.dictionary_catalog[key]["band_flux_obs_err"][0], dict_cat.dictionary_catalog[key]["band_flux_obs_err"][1]
        # algo to replace -- by the min numerical value of the list
        for err_name_0, err_name_1 in zip(err_flux_neg, err_flux_pos):
            neg_data, pos_data = [], []
            for item in range(source_number):
                # iterate for each item in the list
                if not isinstance(cone_search_catalog[err_name_0][item], np.ma.core.MaskedConstant):
                    # put in neg_data list each item of type different to np.ma.core.MaskedConstant
                    neg_data.append(cone_search_catalog[err_name_0][item])
                if not isinstance(cone_search_catalog[err_name_1][item], np.ma.core.MaskedConstant):
                    # put in pos_data list each item of type different to np.ma.core.MaskedConstant
                    pos_data.append(cone_search_catalog[err_name_1][item])
                    
            neg_flux, pos_flux = list(cone_search_catalog[err_name_0]), list(cone_search_catalog[err_name_1])
            corrected_neg_flux = np.nan_to_num(neg_flux, nan=np.min(neg_data))
            corrected_pos_flux = np.nan_to_num(pos_flux, nan=np.min(pos_data))
            
            cone_search_catalog[err_name_0] = corrected_neg_flux
            cone_search_catalog[err_name_1] = corrected_pos_flux
            
        return cone_search_catalog


    def photon_index_nh_for_csc(self, csc_index: int) -> Table:
        """
        Calculates and updates the photon index and hydrogen column density (NH) for sources in the Chandra Source Catalog (CSC).

        Args:
            csc_index (int): An index indicating which of the nearby sources tables (1 or 2) corresponds to the CSC catalog.

        Returns:
            Table: The updated nearby sources table with new columns for 'Photon Index' and 'Nh' added.

        Note:
            Assumes the user has familiarity with CSC data structures and astrophysical parameters like photon index and NH.

        Important:
            The method relies on the availability and accuracy of initial values in the CSC catalog. Inaccuracies in source data may affect the calculated photon index and NH values.

        The method performs the following steps:
            - Determines the appropriate nearby sources table based on the csc_index.
            - Iterates through each source in the table. For each source:
                - If a valid photon index is already present, it is used; otherwise, it's calculated using the `optimization` method.
                - If a valid NH value is present, it is used; otherwise, a default value is assigned.
            - Each source's photon index and NH value are added to the photon_index_list and nh_list, respectively.
            - Calls the `visualization_interp` method to visualize the interpolated photon indices.
            - The 'Photon Index' and 'Nh' columns are added or updated in the nearby sources table with the calculated values.
            - Returns the updated nearby sources table.

        This method is crucial for enhancing the CSC data by calculating key astrophysical parameters (photon index and NH) that are essential for analyzing the X-ray spectra of astronomical sources.
        """
        
        if csc_index == 0:
            nearby_sources_table = self.nearby_sources_table_1
        else:
            nearby_sources_table = self.nearby_sources_table_2
            
        photon_index_list, nh_list, optimization_parameters = [], [], []
        
        for (index, photon), nh_value in zip(enumerate(nearby_sources_table["powlaw_gamma"]), nearby_sources_table["nh_gal"]):
            if photon != 0:
                photon_index_list.append(photon)
            else:
                params, photon_index = self.optimization(index=index, key="CS_Chandra", table=nearby_sources_table)
                photon_index_list.append(photon_index if photon_index > 0.0 else 1.7)
                optimization_parameters.append(params)
                
            if nh_value != 0:
                nh_list.append(nh_value)
            else:
                nh_list.append(3e20)

        self.visualization_interp(optimization_parameters=optimization_parameters, photon_index=photon_index_list, key="CS_Chandra")

        col_names = ["Photon Index", "Nh"]
        col_data = [photon_index_list, nh_list]
        
        for name, data in zip(col_names, col_data):
            nearby_sources_table[name] = data
            
        return nearby_sources_table
        
        
    def photon_index_nh_for_other_catalog(self, key: str, table: Table) -> Table:
        """
        Computes and updates photon index and hydrogen column density (NH) for sources in a given astronomical catalog other than XMM and CSC.

        Args:
            key (str): The key identifying the catalog (e.g., 'Swift', 'eRASS1').
            table (Table): The table containing the catalog data.

        Returns:
            Table: The updated table with 'Photon Index' and 'Nh' columns added.

        Note:
            Designed for users with proficiency in handling astronomical catalog data and spectral analysis.

        Important:
            Results are dependent on the quality of the catalog data and the effectiveness of the optimization method. Inconsistent data may lead to unreliable photon index and NH values.

        Methodology:
        - Iterates over each source in the table.
        - For each source, calculates the photon index using the `optimization` method.
        - Assigns a default NH value of 3e20.
        - The calculated photon index and NH values are added to the photon_index_list and nh_list, respectively.
        - Calls `visualization_interp` for visualizing the photon index interpolation.
        - Updates the table with the new 'Photon Index' and 'Nh' columns.
        - Returns the updated table.

        This method is essential for spectral analysis in astrophysics, allowing for the enhancement of catalog data with key spectral parameters.
        """
        
        photon_index_list, nh_list, optimization_parameters = [], [], []
        
        number_source = len(table)
        
        for index in range(number_source):
            params, photon_index = self.optimization(index=index, key=key, table=table)
            photon_index_list.append(photon_index if photon_index > 0.0 else 1.7)
            optimization_parameters.append(params)
            nh_list.append(3e20)
        
        self.visualization_interp(optimization_parameters=optimization_parameters, photon_index=photon_index_list, key=key)

        col_names = ["Photon Index", "Nh"]
        col_data = [photon_index_list, nh_list]
        
        for name, data in zip(col_names, col_data):
            table[name] = data
            
        return table
        
    
    def neighbourhood_of_object(self, key: Tuple[str, str], simulation_data: Dict, radius) -> None:
        """
        Visualizes the astronomical sources in the vicinity of a given object, using data from two different catalogs.

        Parameters:
            key (Tuple[str, str]): A tuple containing the keys of the two catalogs being compared.
            simulation_data (Dict): A dictionary containing simulation data, including object information.
            radius (float): The radius around the object within which sources are considered.

        Returns:
            None: This method does not return anything but produces a visualization plot.

        Note:
            The method requires correct alignment and calibration of catalog data for accurate representation of source positions.

        Important:
            This method provides a visual representation and does not quantify the statistical significance of the distribution of sources around the object. Interpretation should be done with caution.

        Description:
            - Retrieves object data, including name and celestial position.
            - Based on the catalog keys, determines the right ascension (RA) and declination (DEC) column names for each catalog.
            - Creates a plot with two subplots, each representing one of the catalogs.
            - In each subplot:
                - Plots the positions of the sources from the respective catalog.
                - Highlights the position of the main object.
            - Saves the plot image to a specified path.

        This method is useful for astronomers to visually assess the distribution of sources around a particular object across different catalogs.

        Here is an example of the plot create by this method:

        .. image:: C:/Users/plamb_v00y0i4/OneDrive/Bureau/Optimal_Pointing_Point_Code/modeling_result/PSR_J0437-4715/Compare_catalog/img/Xmm_DR13_CSC_2.0_neighbourhood_of_PSR_J0437-4715.png
            :align: center
        """
        
        object_data = simulation_data["object_data"]
        object_name = object_data["object_name"]
        obj_ra, obj_dec = object_data["object_position"].ra, object_data["object_position"].dec
        
        if key[0] == "Xmm_DR13" and key[1] == "CSC_2.0":
            ra_1, ra_2 = "SC_RA", "ra"
            dec_1, dec_2 = "SC_DEC", "dec"
        elif key[0] == "CSC_2.0" and key[1] == "Xmm_DR13":
            ra_1, ra_2 = "ra", "SC_RA"
            dec_1, dec_2 = "dec", "SC_DEC"
        elif key[0] == "Xmm_DR13" and (key[1] == "Swift" or key[1] == "eRASS1"):
            ra_1, ra_2 = "SC_RA", "RA"
            dec_1, dec_2 = "SC_DEC", "DEC"
        elif key[0] == "CSC_2.0" and (key[1] == "Swift" or key[1] == "eRASS1"):
            ra_1, ra_2 = "ra", "RA"
            dec_1, dec_2 = "dec", "DEC"
        elif (key[0] == "Swift" or key[0] == "eRASS1") and key[1] == "Xmm_DR13":
            ra_1, ra_2 = "RA", "SC_RA"
            dec_1, dec_2 = "DEC", "SC_DEC"
        elif (key[0] == "Swift" or key[0] == "eRASS1") and key[1] == "CSC_2.0":
            ra_1, ra_2 = "RA", "ra"
            dec_1, dec_2 = "DEC", "dec"
        else:
            print("Invalid key !")
            sys.exit()
            
        figure_1, axes = plt.subplots(1, 2, figsize=(17, 9), sharey=True)
        figure_1.suptitle(f"Neighbourhood of {object_name}, radius = {radius}", fontsize=20)
        figure_1.text(0.5, 0.04, 'Right Ascension [deg]', ha='center', va='center', fontsize=16)
        figure_1.text(0.04, 0.5, 'Declination [deg]', ha='center', va='center', rotation='vertical', fontsize=16)
        
        ax0, ax1 = axes[0], axes[1]
        
        cat_1_ra, cat_1_dec = self.nearby_sources_table_1[ra_1], self.nearby_sources_table_1[dec_1]
        ax0.scatter(cat_1_ra, cat_1_dec, s=30, edgecolors='black', facecolors='none', label=f"{len(cat_1_ra)} sources")
        ax0.scatter(obj_ra, obj_dec, s=120, marker="*", facecolors='none', edgecolors='red', label=f"{object_name}")
        ax0.legend(loc="upper right")
        ax0.set_title(f"With {key[0]} catalog")
        
        cat_2_ra, cat_2_dec = self.nearby_sources_table_2[ra_2], self.nearby_sources_table_2[dec_2]
        ax1.scatter(cat_2_ra, cat_2_dec, s=30, edgecolors='black', facecolors='none', label=f"{len(cat_2_ra)} sources")
        ax1.scatter(obj_ra, obj_dec, s=120, marker="*", facecolors='none', edgecolors='red', label=f"{object_name}")
        ax1.legend(loc="upper right")
        ax1.set_title(f"With {key[1]} catalog")
        
        os_dictionary = simulation_data["os_dictionary"]
        img = os_dictionary["img"]
        img_path = os.path.join(img, f"{key[0]}_{key[1]}_neighbourhood_of_{object_name}.png".replace(" ", "_")).replace("\\", "/")
        plt.savefig(img_path)
        plt.show()


    def dictionary_model(self, key: Tuple[str, str]) -> Tuple[Dict, Dict]:
        """
        Creates dictionaries for each source in the nearby sources tables, specifying the astrophysical model parameters.

        Args:
            key (Tuple[str, str]): Tuple containing the keys identifying the two catalogs being compared.

        Returns:
            Tuple[Dict, Dict]: Two dictionaries, each containing the model parameters for the sources in the corresponding nearby sources table.

        Note:
            The method assumes the availability of key spectral parameters in the catalogs. Incomplete data may impact the accuracy of the models.

        Important:
            The choice of 'power' as a model type is a simplification. In real-world scenarios, models should be chosen based on the source characteristics and scientific goals.

        Description:
        - Converts catalog keys to standard format if necessary.
        - Iterates over each source in both nearby sources tables.
        - For each source, creates a dictionary with model type ('power'), photon index, observed flux, and column density.
        - The model parameters for each source are stored in separate dictionaries for each catalog.
        - Returns a tuple containing these two dictionaries.

        This method is crucial for preparing data for further spectral analysis, encapsulating key parameters in an accessible format.
        """
        
        model_dictionary_1, model_dictionary_2 = {}, {}
        
        key_0 = key[0]
        key_1 = key[1]
        
        if key_0 == "Xmm_DR13":
            key_0 = "XMM"
        elif key_1 == "Xmm_DR13":
            key_1 = "XMM"
            
        if key_0 == "CSC_2.0":
            key_0 = "CS_Chandra"
        elif key_1 == "CSC_2.0":
            key_1 = "CS_Chandra"
        
        flux_obs_1 = dict_cat.dictionary_catalog[key_0]["flux_obs"]
        flux_obs_2 = dict_cat.dictionary_catalog[key_1]["flux_obs"]
        
        for index in range(len(self.nearby_sources_table_1)):

            dictionary = {
                "model": 'power',
                "model_value": self.nearby_sources_table_1["Photon Index"][index],
                "flux": self.nearby_sources_table_1[flux_obs_1][index],
                "column_dentsity": self.nearby_sources_table_1["Nh"][index]
            }

            model_dictionary_1[f"src_{index}"] = dictionary
            
        for index in range(len(self.nearby_sources_table_2)):

            dictionary = {
                "model": 'power',
                "model_value": self.nearby_sources_table_2["Photon Index"][index],
                "flux": self.nearby_sources_table_2[flux_obs_2][index],
                "column_dentsity": self.nearby_sources_table_2["Nh"][index]
            }

            model_dictionary_2[f"src_{index}"] = dictionary
            
        return model_dictionary_1, model_dictionary_2
        

    def count_rate(self) -> Tuple[List, List]:
        """
        Calculates the count rate for each source in both nearby sources tables using the PIMMS tool.

        Returns:
            Tuple[List, List]: Two lists containing the calculated count rates for each source in the corresponding nearby sources table.

        Note:
            The accuracy of count rate calculations depends on the correctness of the input astrophysical model parameters and the accuracy of the PIMMS tool itself.

        Important:
            PIMMS (Portable, Interactive Multi-Mission Simulator) is a tool used for predicting count rates from astrophysical models. The user must ensure that PIMMS is correctly installed and configured for accurate calculations.

        Description:
        - Iterates over each source in both nearby sources tables.
        - For each source, extracts the astrophysical model parameters from the model dictionaries.
        - Generates and runs PIMMS commands to calculate the count rate for each source based on its model parameters.
        - The resulting count rates are stored in two separate lists.
        - Updates the nearby sources tables with the calculated count rates.
        - Returns a tuple containing these two lists of count rates.

        This method enables the quantification of expected detector count rates based on the observed astrophysical parameters, which is essential for planning and analyzing astronomical observations.
        """
        
        number_source_1 = len(self.nearby_sources_table_1)
        number_source_2 = len(self.nearby_sources_table_2)
        
        ct_rates_1, ct_rates_2 = np.array([], dtype=float), np.array([], dtype=float)

        for item in range(number_source_1):
            model_1 = self.model_dictionary_1[f"src_{item}"]["model"]
            model_value_1 = self.model_dictionary_1[f"src_{item}"]["model_value"]
            flux_1 =  self.model_dictionary_1[f"src_{item}"]["flux"]
            nh_1 = self.model_dictionary_1[f"src_{item}"]["column_dentsity"]
                    
            pimms_cmds = f"instrument nicer 0.3-10.0\nfrom flux ERGS 0.2-12.0\nmodel galactic nh {nh_1}\nmodel {model_1} {model_value_1} 0.0\ngo {flux_1}\nexit\n"
            
            with open('pimms_script.xco', 'w') as file:
                file.write(pimms_cmds)
                file.close()
                
            result = subprocess.run(['pimms', '@pimms_script.xco'], stdout=subprocess.PIPE).stdout.decode(sys.stdout.encoding)
            ct_rate_value = float(result.split("predicts")[1].split('cps')[0])
            ct_rates_1 = np.append(ct_rates_1, ct_rate_value)
            
        self.nearby_sources_table_1["count_rate"] = ct_rates_1
            
        for item in range(number_source_2):
            model_2 = self.model_dictionary_2[f"src_{item}"]["model"]
            model_value_2 = self.model_dictionary_2[f"src_{item}"]["model_value"]
            flux_2 =  self.model_dictionary_2[f"src_{item}"]["flux"]
            nh_2 = self.model_dictionary_2[f"src_{item}"]["column_dentsity"]
                    
            pimms_cmds = f"instrument nicer 0.3-10.0\nfrom flux ERGS 0.2-12.0\nmodel galactic nh {nh_2}\nmodel {model_2} {model_value_2} 0.0\ngo {flux_2}\nexit\n"
            
            with open('pimms_script.xco', 'w') as file:
                file.write(pimms_cmds)
                file.close()
                
            result = subprocess.run(['pimms', '@pimms_script.xco'], stdout=subprocess.PIPE).stdout.decode(sys.stdout.encoding)
            ct_rate_value = float(result.split("predicts")[1].split('cps')[0])
            ct_rates_2 = np.append(ct_rates_2, ct_rate_value)
            
        self.nearby_sources_table_2["count_rate"] = ct_rates_2
        
        return ct_rates_1, ct_rates_2
        
    
    
    def xslx_to_py(self, args, table, simulation_data, radius) -> Tuple[List, Table]:
        """
        Reads count rate data from an Excel file based on specified parameters and updates the provided table with these values.

        Args:
            args (str): Identifier for the catalog to be used (e.g., "Xmm_DR13", "CSC_2.0", "Swift", "eRASS1", "match").
            table (Table): The table to be updated with count rate data.
            simulation_data (Dict): Dictionary containing simulation parameters and paths.
            radius (float): The radius value used to define the file name for the Excel data.

        Returns:
            Tuple[List, Table]: A tuple where the first element is a list of count rates and the second element is the updated table with these rates.

        This method constructs the path to the relevant Excel file based on the catalog and object name, reads the count rates, and updates the table's 'count_rate' column.
        """
        
        data_path = simulation_data["os_dictionary"]["data_path"]
        object_data = simulation_data["object_data"]
        excel_data_path = os.path.join(data_path, "excel_data").replace("\\", "/")

        if args == "Xmm_DR13":
            cat = "xmm"
        elif args == "CSC_2.0":
            cat = "csc_CS_Chandra"
        elif args == "Swift":
            cat = "swi"
        elif args == "eRASS1":
            cat = "ero"
        elif args == "match":
            cat = "xmmXchandra"
        
        ct_rates_path = os.path.join(excel_data_path, f"{cat}_{radius}_{object_data['object_name']}.xlsx".replace(" ", "_"))
        wb = openpyxl.load_workbook(ct_rates_path)
        sheet = wb.active

        count_rates = []
        for item in range(len(table)): 
            count_rates.append(sheet.cell(row = item + 1, column = 1).value)
            
        table["count_rate"] = count_rates

        return count_rates, table
    
    
    def calculate_opti_point(self, simulation_data, key) -> Tuple[Dict, Dict, int, int]:
        """
        Calculates the optimal pointing coordinates for an astronomical object to maximize the signal-to-noise ratio (S/N) based on nearby sources.

        Args:
            simulation_data (Dict): Dictionary containing simulation parameters, including telescope and object data.
            key (Tuple[str, str]): A tuple of two catalog identifiers used for the analysis.

        Returns:
            Tuple[Dict, Dict, int, int]: A tuple containing two dictionaries for each catalog, and two integers representing the indices of optimal pointing coordinates.

        Note:
            Accurate input parameters are critical for achieving reliable S/N optimization. Ensure that telescope data, object properties, and catalogs are up-to-date and precise. Inaccuracies can lead to suboptimal or misleading results.

        Important:
            The accuracy of the generated S/N maps and optimal pointing coordinates heavily relies on the provided simulation data and catalog information. Verify that the input data accurately represents the astronomical scenario you intend to analyze.

        Description:
            - This method first creates a grid of potential pointing coordinates around the astronomical object, considering a range of right ascension and declination offsets. It then calculates the signal-to-noise ratio (S/N) for each pointing position. The S/N is computed by comparing the expected count rates from the astronomical object of interest with those from nearby sources, accounting for telescope characteristics and background noise.
            - The function identifies the optimal pointing location by selecting the coordinates with the highest S/N ratio. Two separate dictionaries are generated, each containing information for one of the specified catalogs. Additionally, the indices of the optimal pointing coordinates are returned.
            - The generated S/N maps provide valuable insights into the quality of potential observations, aiding astronomers in selecting the best pointing coordinates for maximizing scientific outcomes during telescope observations.

        This method is crucial for planning and conducting astronomical observations, enabling astronomers to optimize telescope pointing for enhanced data quality.
        
        Here is an example of the plot create by this method:
        
        .. image:: C:/Users/plamb_v00y0i4/OneDrive/Bureau/Optimal_Pointing_Point_Code/modeling_result/PSR_J0437-4715/Compare_catalog/img/Xmm_DR13_CSC_2.0_SNR_PSR_J0437-4715.png
            :align: center
        """
        
        min_value, max_value, step = -7.0, 7.1, 0.05
        DeltaRA = Angle(np.arange(min_value, max_value, step), unit=u.deg)/60
        DeltaDEC = Angle(np.arange(min_value, max_value, step), unit=u.deg)/60
        
        telescop_data = simulation_data["telescop_data"]
        object_data = simulation_data["object_data"]
        object_name = object_data["object_name"]
        
        RA_grid, DEC_grid = np.meshgrid(DeltaRA, DeltaDEC)

        SampleRA = object_data["object_position"].ra.deg + RA_grid.flatten().deg
        SampleDEC = object_data["object_position"].dec.deg + DEC_grid.flatten().deg
        
        NICERpointing = SkyCoord(ra=SampleRA*u.deg, dec=SampleDEC*u.deg)
        NICERpointing = NICERpointing.reshape(-1, 1)
        
        PSRseparation = c_f.ang_separation(object_data["object_position"], NICERpointing).arcmin
        PSRcountrateScaled = c_f.scaled_ct_rate(PSRseparation, object_data['count_rate'], telescop_data["EffArea"], telescop_data["OffAxisAngle"])
        
        sources_1 = self.nearby_sources_position_1.reshape(1, -1)
        SRCseparation_1 = c_f.ang_separation(sources_1, NICERpointing).arcmin
        count_rate_1 = self.nearby_sources_table_1['count_rate']
        SRCcountrateScaled_1 = c_f.scaled_ct_rate(SRCseparation_1, count_rate_1, telescop_data["EffArea"], telescop_data["OffAxisAngle"])
        SNR_1, PSRrates, SRCrates_1  = np.zeros((3, len(DeltaRA) * len(DeltaDEC)))
        for item in range(len(PSRcountrateScaled)):
            PSRrates[item] = PSRcountrateScaled[item]
            SRCrates_1[item] = np.sum(SRCcountrateScaled_1[item])
            SNR_1[item] = c_f.signal_to_noise(PSRrates[item], SRCrates_1[item], simulation_data["INSTbkgd"], simulation_data["EXPtime"])
        
        sources_2 = self.nearby_sources_position_2.reshape(1, -1)
        SRCseparation_2 = c_f.ang_separation(sources_2, NICERpointing).arcmin
        count_rate_2 = self.nearby_sources_table_2['count_rate']
        SRCcountrateScaled_2 = c_f.scaled_ct_rate(SRCseparation_2, count_rate_2, telescop_data["EffArea"], telescop_data["OffAxisAngle"])
        SNR_2, PSRrates, SRCrates_2  = np.zeros((3, len(DeltaRA) * len(DeltaDEC)))
        for item in range(len(PSRcountrateScaled)):
            PSRrates[item] = PSRcountrateScaled[item]
            SRCrates_2[item] = np.sum(SRCcountrateScaled_2[item])
            SNR_2[item] = c_f.signal_to_noise(PSRrates[item], SRCrates_2[item], simulation_data["INSTbkgd"], simulation_data["EXPtime"])
        
        OptimalPointingIdx_1 = np.where(SNR_1==max(SNR_1))[0][0]
        SRCoptimalSEPAR_1 = c_f.ang_separation(self.nearby_sources_position_1, SkyCoord(ra=SampleRA[OptimalPointingIdx_1]*u.degree, dec=SampleDEC[OptimalPointingIdx_1]*u.degree)).arcmin
        SRCoptimalRATES_1 = c_f.scaled_ct_rate(SRCoptimalSEPAR_1, self.count_rates_1, telescop_data["EffArea"], telescop_data["OffAxisAngle"])
        
        OptimalPointingIdx_2 = np.where(SNR_2==max(SNR_2))[0][0]
        SRCoptimalSEPAR_2 = c_f.ang_separation(self.nearby_sources_position_2, SkyCoord(ra=SampleRA[OptimalPointingIdx_1]*u.degree, dec=SampleDEC[OptimalPointingIdx_2]*u.degree)).arcmin
        SRCoptimalRATES_2 = c_f.scaled_ct_rate(SRCoptimalSEPAR_2, self.count_rates_2, telescop_data["EffArea"], telescop_data["OffAxisAngle"])     
        
        vector_dictionary_1 = {'SampleRA': SampleRA,
                               'SampleDEC': SampleDEC,
                               'PSRrates': PSRrates,
                               'SRCrates_2': SRCrates_2,
                               'SRCoptimalRATES_1':SRCoptimalRATES_1,
                               'SNR_1': SNR_1}
        
        vector_dictionary_2 = {'SampleRA': SampleRA,
                               'SampleDEC': SampleDEC,
                               'PSRrates': PSRrates,
                               'SRCrates_2': SRCrates_2,
                               'SRCoptimalRATES_2':SRCoptimalRATES_2,
                               'SNR_2': SNR_2}

        opti_ra_1, opti_dec_1 = vector_dictionary_1['SampleRA'][OptimalPointingIdx_1], vector_dictionary_1['SampleDEC'][OptimalPointingIdx_1]
        opti_ra_2, opti_dec_2 = vector_dictionary_2['SampleRA'][OptimalPointingIdx_2], vector_dictionary_2['SampleDEC'][OptimalPointingIdx_2]
        
        figure_map, axes = plt.subplots(1, 2, figsize=(17, 9), sharey=True)
        figure_map.suptitle(f"S/N map for {object_data['object_name']}", fontsize=20)
        figure_map.text(0.5, 0.04, 'Right Ascension [deg]', ha='center', va='center', fontsize=16)
        figure_map.text(0.04, 0.5, 'Declination [deg]', ha='center', va='center', rotation='vertical', fontsize=16)

        ax0 = axes[0]
        ax0.plot(self.nearby_sources_position_1.ra, self.nearby_sources_position_1.dec, marker='.', color='black', linestyle='', label=f"{len(self.nearby_sources_position_1.ra)} sources")
        ax0.plot(object_data["object_position"].ra, object_data["object_position"].dec, marker='*', color="green", linestyle='', label=f"{object_data['object_name']}")
        ax0.plot(opti_ra_1, opti_dec_1, marker="+", color='red', linestyle='', label="Optimal pointing point")
        ax0.scatter(vector_dictionary_1['SampleRA'], vector_dictionary_1['SampleDEC'], c=vector_dictionary_1["SNR_1"], s=10, edgecolor='face')
        ax0.set_title(f"With {key[0]} catalog\n Optimal pointing point : {opti_ra_1} deg, {opti_dec_1} deg")
        ax0.legend(loc="upper right", ncol=2)

        ax1 = axes[1]
        ax1.plot(self.nearby_sources_position_2.ra, self.nearby_sources_position_2.dec, marker='.', color='black', linestyle='', label=f"{len(self.nearby_sources_position_2.ra)} sources")
        ax1.plot(object_data["object_position"].ra, object_data["object_position"].dec, marker='*', color="green", linestyle='', label=f"{object_data['object_name']}")
        ax1.plot(opti_ra_2, opti_dec_2, marker="+", color='red', linestyle='', label="Optimal pointing point")
        ax1.scatter(vector_dictionary_2['SampleRA'], vector_dictionary_2['SampleDEC'], c=vector_dictionary_2["SNR_2"], s=10, edgecolor='face')
        ax1.set_title(f"With {key[1]} catalog\n Optimal pointing point : {opti_ra_2} deg, {opti_dec_2} deg")
        ax1.legend(loc="upper right", ncol=2)
 
        norm = plt.Normalize(vmin=min(vector_dictionary_1["SNR_1"] + vector_dictionary_2["SNR_2"]), vmax=max(vector_dictionary_1["SNR_1"] + vector_dictionary_2["SNR_2"]))
        sm = ScalarMappable(cmap='viridis', norm=norm)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        color_bar = figure_map.colorbar(sm, cax=cax)
        color_bar.set_label("S/N")
        
        os_dictionary = simulation_data["os_dictionary"]
        img = os_dictionary["img"]
        img_path = os.path.join(img, f"{key[0]}_{key[1]}_SNR_{object_name}.png".replace(" ", "_")).replace("\\", "/")
        plt.savefig(img_path)
        plt.show()

        return vector_dictionary_1, vector_dictionary_2, OptimalPointingIdx_1, OptimalPointingIdx_2
    
    
    def variability_table(self, simulation_data: Dict, radius: int) -> str:
        """
        Creates a table of master sources around a specific astronomical object based on a given radius and simulation data.

        Args:
            simulation_data (Dict): Dictionary containing various parameters and paths used in the simulation.
            radius (int): Radius in arcminutes defining the area around the object of interest.

        Returns:
            str: Path to the created master source FITS file.

        Description:
            This method first extracts sources around the specified region (RA, Dec) within the given radius from the master source catalog. Then, it selects relevant catalog sources around this region. It uses STILTS (Software for the Treatment of Image Data from Large Telescopes) for data processing. Finally, it visualizes and saves the resulting master sources.

        Exceptions:
            Any exceptions during processing are caught and printed to the console.

        Note:
            - This method relies on external software (STILTS) and assumes the existence of a master source catalog and additional catalogs defined in `simulation_data`.
            - Accurate input parameters are critical for achieving reliable results. Ensure that the telescope data, object properties, and catalogs are up-to-date and precise. Inaccuracies can lead to suboptimal or misleading results.
            - The generated master source table is saved in the directory specified in the simulation data.

        Important:
            The accuracy of the generated master source table heavily relies on the provided simulation data, including the path to the master source catalog, the STILTS software path, and other parameters. Verify that the input data accurately represents the astronomical scenario you intend to analyze.
        """
        
        os_dictionary = simulation_data["os_dictionary"]
        catalog_datapath = os_dictionary["catalog_datapath"]
        stilts_software_path = os_dictionary["stilts_software_path"]
        output_name = os_dictionary["output_name"]
        
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
            for cat in dict_cat.catalogs:
                if cat == "eRASS1":
                    path_to_cat_init = os.path.join(catalog_datapath, cat).replace("\\", "/")
                    path_to_cat_final = os.path.join(output_name, cat).replace("\\", "/")
                    command = (f"java -jar {stilts_software_path} tmatch2 matcher=exact \
                            in1='{master_cone_path}' in2='{path_to_cat_init}.fits' out='{path_to_cat_final}.fits'\
                                values1='{cat}' values2='eRASS_IAUNAME' find=all progress=none")
                else:
                    path_to_cat_init = os.path.join(catalog_datapath, cat).replace("\\", "/")
                    path_to_cat_final = os.path.join(output_name, cat).replace("\\", "/")
                    command = (f"java -jar {stilts_software_path} tmatch2 matcher=exact \
                            in1='{master_cone_path}' in2='{path_to_cat_init}.fits' out='{path_to_cat_final}.fits'\
                                values1='{cat}' values2='{cat}_IAUNAME' find=all progress=none")
                command = shlex.split(command)
                subprocess.run(command)

        object_data = simulation_data["object_data"]
        right_ascension = object_data["object_position"].ra.value
        declination = object_data["object_position"].dec.value
        try:
            print(f"\n{colored('Load Erwan s code for :', 'yellow')} {object_data['object_name']}")
            select_master_sources_around_region(ra=right_ascension, dec=declination, radius=radius, output_name=output_name)
            select_catalogsources_around_region(output_name=output_name)
            master_sources = s_f.load_master_sources(output_name)
            s_f.master_source_plot(master_sources=master_sources, simulation_data=simulation_data, number_graph=len(master_sources))
        except Exception as error :
            print(f"{colored('An error occured : ', 'red')} {error}")
            
        return master_source_path
    
    
    def variability_index(self, key: str, iauname: str, nearby_sources_table: Table) -> List:
        """
        Identifies and returns the indices of variable sources from a nearby sources table based on a master source catalog.

        Args:
            key (str): Key representing the catalog name (e.g., 'CSC_2.0', 'Xmm_DR13').
            iauname (str): The column name in `nearby_sources_table` representing source names.
            nearby_sources_table (Table): A table containing data of nearby sources.

        Returns:
            List: A list of indices in `nearby_sources_table` corresponding to variable sources found in the master source catalog.

        Description:
            This method opens the master source FITS file and filters out variable sources based on the catalog specified by `key`. It then matches these sources with those in the `nearby_sources_table` and compiles a list of indices representing these variable sources within the table.

        Note:
            - This method assumes that the master source path is already set in the instance variable `self.master_source_path`.
            - The returned list contains indices that can be used to access variable sources in the `nearby_sources_table`.
        """
        
        with fits.open(self.master_source_path) as data:
            master_source_cone = Table(data[1].data)
        
        if key == "CSC_2.0":
            key = "Chandra"
        elif key == "Xmm_DR13":
            key = "XMM"
        
        msc_name = [name for name in master_source_cone[key] if name != ""]
        var_index_in_nearby_sources_table = []
        for name in msc_name:
            if name in nearby_sources_table[iauname]:
                index_in_table = list(nearby_sources_table[iauname]).index(name)
                var_index_in_nearby_sources_table.append(index_in_table)
                
        return var_index_in_nearby_sources_table
    
    
    def write_fits_table(self, table: Table, key: str, os_dictionary: Dict) -> None:
        """
        Writes a given table to a FITS file, using a specific catalog key and directory paths from a dictionary.

        Args:
            table (Table): The table to be written to the FITS file.
            key (str): Key representing the catalog name used for naming the FITS file.
            os_dictionary (Dict): Dictionary containing various file paths used in the operation.

        Description:
            This method attempts to write the `table` to a FITS file in the directory specified in `os_dictionary["cloesest_dataset_path"]`. The file is named using the `key` parameter. If an error occurs during this process, it is caught and printed to the console.

        Note:
            - The method overwrites any existing file with the same name.
            - The path to the generated FITS file is printed to the console upon successful creation.
        """
        
        try:            
            cloesest_dataset_path = os_dictionary["cloesest_dataset_path"]
            nearby_sources_table_path = os.path.join(cloesest_dataset_path, f"{key}_nearby_sources_table.fits").replace("\\", "/")

            cat_name = key + "_IAUNAME"
        #print(cat_name)
            if key =="CS_Chandra":
                cat_name = "name"
                max_len = max(len(str(col_name)) for col_name in table["likelihood_class"])
                table["likelihood_class"] = np.array([str(col_name).ljust(max_len) for col_name in table["likelihood_class"]])

            if key =="eRASS1":
                cat_name = "eRASS_IAUNAME"
                max_len = max(len(str(name)) for name in table[cat_name])
                table[cat_name] = np.array([str(name).ljust(max_len) for name in table[cat_name]])

                table.write(nearby_sources_table_path, format='fits', overwrite=True)
                print(f"Nearby sources table was created in : {colored(nearby_sources_table_path, 'magenta')}")
        except Exception as error:
            print(f"{colored('An error occured : ', 'red')} {error}")
    
    
    def modeling_source_spectra(self, simulation_data: Dict, exp_time: int, key: Tuple[str, str]) -> Tuple[List, List]:
        """
        Generates modeled spectra for sources in nearby sources tables for given catalogs.

        Args:
            simulation_data (Dict): A dictionary containing simulation data including telescope data.
            exp_time (int): Exposure time used in the simulation.
            key (Tuple[str, str]): A tuple containing the keys of the catalogs to be used.

        Returns:
            Tuple[List, List, List, List, Instrument]: A tuple containing lists of total spectra, total variable spectra for both catalogs, and the instrument used.

        Description:
            This method creates and models spectral data for each source in the nearby sources tables of the specified catalogs (key[0] and key[1]). It utilizes an X-ray spectrum model (Tbabs * Powerlaw) and simulates the spectra using the NICER instrument's ARF and RMF files. The method also accounts for the vignetting factor for each source. The result is a collection of spectra for all sources, as well as a separate collection for variable sources as identified by their indices.

        Note:
            - This method assumes that the necessary data paths and instrumental information are provided in `simulation_data`.
            - Accurate input parameters are crucial for obtaining meaningful results. Ensure that the telescope data, exposure time, and other simulation data are correctly specified.
            - The generated spectra are stored as lists and can be used for further analysis or visualization.

        Important:
            The accuracy of the generated spectra depends on the quality of the input data and the accuracy of the spectral model used. Verify that the instrument files (ARF and RMF), exposure time, and catalog data are appropriate for the analysis. Inaccurate input data can lead to incorrect results and interpretations.
        """
        
        model = Tbabs() * Powerlaw()
        telescop_data = simulation_data["telescop_data"]
        nicer_data_arf = telescop_data["nicer_data_arf"]
        nicer_data_rmf = telescop_data["nicer_data_rmf"]
        
        instrument = Instrument.from_ogip_file(nicer_data_rmf, nicer_data_arf)
        
        obsconfig = ObsConfiguration.mock_from_instrument(instrument, exp_time)

        print(f"\n{colored(f'Modeling spectra for {key[0]} catalog... ', 'yellow')}")
        size = 10_000
        
        total_spectra_1, total_var_spectra_1 = [], []
        for index, vignet_factor in tqdm(enumerate(self.nearby_sources_table_1["vignetting_factor"])):
            parameters = {}
            parameters = {
                "tbabs_1": {"N_H": np.full(size, self.nearby_sources_table_1["Nh"][index]/1e22)},
                "powerlaw_1": {
                    "alpha": np.full(size, self.nearby_sources_table_1["Photon Index"][index] if self.nearby_sources_table_1["Photon Index"][index] > 0.0 else 1.7),
                    "norm": np.full(size, 1e-5),
                }
            }
            
            spectra = fakeit_for_multiple_parameters(instrument=obsconfig, model=model, parameters=parameters) * vignet_factor
            
            if index in self.var_index_1:
                total_var_spectra_1.append(spectra)
            
            total_spectra_1.append(spectra)
        
        print(f"\n{colored(f'Modeling spectra for {key[1]} catalog... ', 'yellow')}")
        total_spectra_2, total_var_spectra_2 = [], []
        for index, vignet_factor in tqdm(enumerate(self.nearby_sources_table_2["vignetting_factor"])):
            parameters = {}
            parameters = {
                "tbabs_1": {"N_H": np.full(size, self.nearby_sources_table_2["Nh"][index]/1e22)},
                "powerlaw_1": {
                    "alpha": np.full(size, self.nearby_sources_table_2["Photon Index"][index] if self.nearby_sources_table_2["Photon Index"][index] > 0.0 else 1.7),
                    "norm": np.full(size, 1e-5),
                }
            }
            
            spectra = fakeit_for_multiple_parameters(instrument=obsconfig, model=model, parameters=parameters) * vignet_factor
            
            if index in self.var_index_1:
                total_var_spectra_2.append(spectra)
            
            total_spectra_2.append(spectra)
            
            
        return total_spectra_1, total_spectra_2, total_var_spectra_1, total_var_spectra_2, obsconfig
     
    
    def total_spectra_plot(self, simulation_data: Dict, radius: float, obsconfig, key: Tuple[str, str]):
        """
        Plots the modeled spectra for sources around a specific object from two different catalogs.

        Args:
        - simulation_data (Dict): A dictionary containing the following keys:
            - "object_data" (Dict): Information about the object of interest, including its name.
            - "os_dictionary" (Dict): A dictionary with file paths, including the path to save the generated image.
            - Other necessary data for modeling.

        - radius (float): The radius (in some unit) within which sources are considered for modeling.

        - key (Tuple[str, str]): A tuple containing two strings representing the keys of the catalogs to be used for modeling.

        Returns:
            - Tuple[Dict[str, Union[List[float], List[float]]]]: A tuple containing two dictionaries, one for each catalog, with the following keys:
                - "Energy" (List[float]): The energy values (in keV) for the spectral data.
                - "Counts" (List[float]): The counts (cts/s) corresponding to the modeled spectra.
                - "Upper limit" (List[float]): The upper limits of the counts, representing variability.
                - "Lower limit" (List[float]): The lower limits of the counts, representing variability.

        This method generates plots of the modeled spectra for sources in the vicinity of a specified object from two different catalogs. It uses the provided `simulation_data`, which includes object and instrumental data. The generated plots include individual spectra for each catalog, as well as a combined plot showcasing the summed spectra and variability errors. The method calculates upper and lower limits for the spectra to provide an envelope for the variability. Each subplot is appropriately labeled, and the overall figure title indicates the object around which the spectra are modeled.

        Here is an example of the plot create by this method:

        .. image:: C:/Users/plamb_v00y0i4/OneDrive/Bureau/Optimal_Pointing_Point_Code/modeling_result/PSR_J0437-4715/Compare_catalog/img/Xmm_DR13_CSC_2.0_spectral_modeling_close_to_PSR_J0437-4715.png
            :align: center
        """
        
        object_name = simulation_data["object_data"]["object_name"]
        
        figure_spectra, axes = plt.subplots(2, 3, figsize=(17, 8), sharey=True, sharex=True)
        figure_spectra.suptitle(f"Spectral modeling close to {object_name}", fontsize=20)
        figure_spectra.text(0.5, 0.04, 'Energy [keV]', ha='center', va='center', fontsize=16)
        figure_spectra.text(0.04, 0.5, 'Counts [cts/s]', ha='center', va='center', rotation='vertical', fontsize=16)
        
        figure_spectra.text(0.07, 0.75, f'{key[0]}', ha='center', va='center', rotation='vertical', fontsize=14)
        figure_spectra.text(0.07, 0.25, f'{key[1]}', ha='center', va='center', rotation='vertical', fontsize=14)
        
        graph_data = {"min_lim_x": 0.2,
                      "max_lim_x": 10.0}
        
        for ax_row in axes:
            for ax in ax_row:
                ax.loglog()
                ax.set_xlim([graph_data["min_lim_x"], graph_data["max_lim_x"]])
        
        ax00, ax01, ax02 = axes[0][0], axes[0][1], axes[0][2]
        ax10, ax11, ax12 = axes[1][0], axes[1][1], axes[1][2]
        figure_spectra.delaxes(ax02)
        figure_spectra.delaxes(ax12)
        
        # ---------- first row ---------- #
        
        for spectra in self.total_spectra_1:
            ax00.step(obsconfig.out_energies[0],
                      np.median(spectra, axis=0),
                      where="post")
        ax00.set_title("All spectra of nearby sources table")
        
        spectrum_summed_1 = 0.0
        for index in range(len(self.total_spectra_1)):
            spectrum_summed_1 += self.total_spectra_1[index]

        spectrum_var_summed_1 = 0.0
        for index in range(len(self.total_var_spectra_1)):
            spectrum_var_summed_1 += self.total_var_spectra_1[index]
        
        ax01.errorbar(obsconfig.out_energies[0], y=np.median(spectrum_summed_1, axis=0), yerr=np.median(spectrum_var_summed_1, axis=0), 
                    fmt="none", ecolor='red', capsize=2, capthick=3,
                    label='error')
        ax01.step(obsconfig.out_energies[0], np.median(spectrum_summed_1, axis=0), color='black', label="sum powerlaw")
        ax01.set_title("Spectrum Summed with var sources error")
        ax01.legend(loc='upper right')
        
        # ---------- second row ---------- #
        
        for spectra in self.total_spectra_2:
            ax10.step(obsconfig.out_energies[0],
                      np.median(spectra, axis=0),
                      where="post")
        
        spectrum_summed_2 = 0.0
        for index in range(len(self.total_spectra_2)):
            spectrum_summed_2 += self.total_spectra_2[index]

        spectrum_var_summed_2 = 0.0
        for index in range(len(self.total_var_spectra_2)):
            spectrum_var_summed_2 += self.total_var_spectra_2[index]
            
        ax11.errorbar(obsconfig.out_energies[0], y=np.median(spectrum_summed_2, axis=0), yerr=np.median(spectrum_var_summed_2, axis=0), 
                    fmt="none", ecolor='red', capsize=2, capthick=3,
                    label='error')
        ax11.step(obsconfig.out_energies[0], np.median(spectrum_summed_2, axis=0), color='black', label="sum powerlaw")
        ax11.legend(loc='upper right')
        
        # ---------- big axes ---------- #
        
        new_ax = figure_spectra.add_subplot(1, 3, 3)
        new_ax.errorbar(obsconfig.out_energies[0], y=np.median(spectrum_summed_1, axis=0), yerr=np.median(spectrum_var_summed_1, axis=0),
                        fmt="none", ecolor='red', capsize=2, capthick=3, alpha=0.1, label='error_1')
        new_ax.step(obsconfig.out_energies[0], np.median(spectrum_summed_1, axis=0), color='black', label=f"sum powerlaw {key[0]}")
        new_ax.set_title("Spectrum Summed with var sources error")
        new_ax.legend(loc='upper right')
        
        new_ax.errorbar(obsconfig.out_energies[0], y=np.median(spectrum_summed_2, axis=0), yerr=np.median(spectrum_var_summed_2, axis=0),
                        fmt="none", ecolor='darkorange', capsize=2, capthick=3, alpha=0.1, label='error_2')
        new_ax.step(obsconfig.out_energies[0], np.median(spectrum_summed_2, axis=0), color='navy', label=f"sum powerlaw {key[1]}")
        new_ax.set_title("Spectrum Summed with var sources error")
        new_ax.legend(loc='upper right')
        
        new_ax.legend(loc="upper right", ncol=2)
        new_ax.set_xlim([graph_data["min_lim_x"], graph_data["max_lim_x"]])
        new_ax.loglog()
        
        # ---------- get envelop ---------- #
        
        y_upper_1 = np.median(spectrum_summed_1, axis=0) + np.median(spectrum_var_summed_1, axis=0)
        y_lower_1 = np.median(spectrum_summed_1, axis=0) - np.median(spectrum_var_summed_1, axis=0)
        
        y_upper_2 = np.median(spectrum_summed_2, axis=0) + np.median(spectrum_var_summed_2, axis=0)
        y_lower_2 = np.median(spectrum_summed_2, axis=0) - np.median(spectrum_var_summed_2, axis=0)
        
        data_1 = {
            "Energy": obsconfig.out_energies[0],
            "Counts": np.median(spectrum_summed_1, axis=0),
            "Upper limit": y_upper_1,
            "Lower limit": y_lower_1
        }
        
        data_2 = {
            "Energy": obsconfig.out_energies[0],
            "Counts": np.median(spectrum_summed_2, axis=0),
            "Upper limit": y_upper_2,
            "Lower limit": y_lower_2
        }

        os_dictionary = simulation_data["os_dictionary"]
        img = os_dictionary["img"]
        img_path = os.path.join(img, f"{key[0]}_{key[1]}_spectral_modeling_close_to_{object_name}.png".replace(" ", "_")).replace("\\", "/")
        plt.savefig(img_path)
        plt.show()
        
        return data_1, data_2
    
    
    def write_txt_file(self, simulation_data: Dict, data_1: Dict, data_2: Dict, key: Tuple[str, str]) -> None:
        """
        Writes the spectral modeling data into text files for each of the specified catalogs.

        Args:
            simulation_data (Dict): A dictionary containing simulation data including directory paths.
            data_1 (Dict): A dictionary containing the spectral data for the first catalog (key[0]).
            data_2 (Dict): A dictionary containing the spectral data for the second catalog (key[1]).
            key (Tuple[str, str]): A tuple containing the keys of the catalogs.

        This method exports the spectral data for two different catalogs into separate text files. Each file contains data such as energy, counts, and upper and lower limits of the spectra. The data is formatted into columns for readability. The files are named according to the catalogs' keys and saved in the specified directory.

        The method iterates through the provided spectral data, formats each row according to the given specifications, and writes the rows to the respective text files. The headers of the files include the names of the data columns.

        Note:
            The method assumes that the directory for saving the text files is provided in `simulation_data['os_dictionary']["catalog_datapath"]`.

        Example:
            Energy        Counts        Upper Limit   Lower Limit
            [value]       [value]       [value]       [value]
            [value]       [value]       [value]       [value]
        """
    
        catalog_directory = simulation_data['os_dictionary']["catalog_directory"]
        txt_path_1 = os.path.join(catalog_directory, f'{key[0]}_output_modeling_plot.txt').replace("\\", "/")
        
        data_to_txt_1 = [
            list(data_1.keys())
        ]
        
        energy, counts, y_upper, y_lower = list(data_1.values())
        data_to_txt_1.extend([energy[index], counts[index], y_upper[index], y_lower[index]] for index in range(len(energy)))
        
        with open(txt_path_1, 'w') as file:
            header = "{:<15} {:<15} {:<15} {:<15}".format(*data_to_txt_1[0])
            file.write(header + "\n")

            for row in data_to_txt_1[1:]:
                new_row = "{:<10.5f}     {:<10.5f}       {:<10.5f}       {:<10.5f}".format(*row)
                file.write(new_row + "\n")
                
        print(f"\n{colored(f'{key[0]}_output_modeling_plot.txt', 'yellow')} has been created in {colored(txt_path_1, 'blue')}")
        
        
        txt_path_2 = os.path.join(catalog_directory, f'{key[1]}_output_modeling_plot.txt').replace("\\", "/")
        
        data_to_txt_2 = [
            list(data_2.keys())
        ]
        
        energy, counts, y_upper, y_lower = list(data_2.values())
        data_to_txt_2.extend([energy[index], counts[index], y_upper[index], y_lower[index]] for index in range(len(energy)))
        
        with open(txt_path_2, 'w') as file:
            header = "{:<15} {:<15} {:<15} {:<15}".format(*data_to_txt_2[0])
            file.write(header + "\n")

            for row in data_to_txt_2[1:]:
                new_row = "{:<10.5f}     {:<10.5f}       {:<10.5f}       {:<10.5f}".format(*row)
                file.write(new_row + "\n")
                
        print(f"\n{colored(f'{key[1]}_output_modeling_plot.txt', 'yellow')} has been created in {colored(txt_path_2, 'blue')}")
    
    