# ------------------------------ #
        # Python's packages
        
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from typing import Tuple, Dict, Union
from termcolor import colored
from scipy.optimize import curve_fit
from astropy.units import Quantity
from tqdm import tqdm
from astroquery.esasky import ESASky
from astropy.visualization import PercentileInterval, ImageNormalize, LinearStretch
from astropy.wcs import WCS
import csv

# ---------- import function ---------- #

import function.init_function as i_f
import function.calculation_function as c_f

# ------------------------------------- #

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import catalog_information as dict_cat

# ------------------------------ #

# ---------- for documentation ---------- #

# import src.function.init_function as i_f
# import src.function.calculation_function as c_f
# import src.catalog_information as dict_cat

# --------------------------------------- #

class XmmCatalog:
    """
    A class for managing and analyzing XMM (X-ray Multi-Mirror Mission) source catalogs.

    This class offers functionalities for loading, processing, and analyzing XMM source catalogs,
    enabling users to discover nearby sources relative to a given position, analyze their variability,
    and visualize the corresponding data.

    XmmCatalog leverages the rich data from the XMM mission to provide detailed insights into astronomical
    objects, enhancing the understanding of X-ray sources in the universe. Its capabilities include identifying
    nearby sources, estimating photon indices, hydrogen column density, and constructing models for source analysis.

    Important:
        - XmmCatalog is an essential tool for astronomers and researchers in the field of X-ray astrophysics.
        - The effective use of this class can significantly enhance the quality and depth of astronomical data analysis.
        - Understanding the basic principles of X-ray astronomy and data catalogs is necessary to utilize this class fully.

    Examples
    --------
    Creating an instance of XmmCatalog with necessary parameters:
    
    >>> xmm_catalog = XmmCatalog("path/to/catalog.fits", 5*u.arcmin, simulation_data, user_table)

    Note:
        The class serves as a comprehensive interface for X-ray data analysis, facilitating advanced studies
        in astrophysics and contributing to the broader understanding of celestial phenomena.
    """

    def __init__(self, catalog_path: str, radius: Quantity, simulation_data:Dict, user_table: Table) -> None:
        """
        Initializes the XmmCatalog class by setting up the X-ray Multi-Mirror (XMM) mission source catalogs.

        This constructor is responsible for loading and processing data from specified paths. It prepares the
        XMM catalog along with additional data tables and models for further astronomical analysis. This includes
        identifying nearby sources, calculating their properties, and visualizing relevant data.

        Parameters
        ----------
            catalog_path : str
                Path to the primary XMM catalog file. This file is used to initialize the 'xmm_catalog' attribute.
            radius : Quantity
                The search radius for finding nearby sources. Used in conjunction with 'object_data' from
                'simulation_data' to determine the area of interest.
            simulation_data : Dict
                A dictionary containing simulation data. Key components include 'object_data' for nearby source
                identification and 'os_dictionary' for paths to additional catalogs.
            user_table : Table, optional
                A user-provided table to include in the analysis. Enhances source finding capabilities when
                used with 'find_nearby_sources'.
        
        Attributes:

        .. attribute:: key
            :type: str
            :value: XMM
            
            A key identifying the catalog type.
            
        .. attribute:: ra
            :type: str
            :value: SC_RA
            
            Right ascension field name in the XMM catalog.
            
        .. attribute:: dec
            :type: str
            :value: SC_DEC
            
            Declination field name in the XMM catalog.
            
        .. attribute:: xmm_catalog
            :type: Table
            
            The main catalog data loaded from 'catalog_path'.
            
        .. attribute:: nearby_sources_table
            :type: Table
            
            A table containing sources found near the specified object of interest.
            
        .. attribute:: nearby_sources_position
            :type: List[SkyCoord]
            
            Coordinates of nearby sources.
            
        .. attribute:: xmm_dr11_catalog
            :type: Table
            
            Data from the XMM DR11 catalog.
            
        .. attribute:: x2a_catalog
            :type: Table
            
            Data from the Xmm2Athena catalog.
            
        .. attribute:: Table
            :type: Table
            
            An indexing table mapping sources across different catalogs.
            
        .. attribute:: variability_table
            :type: Table
            
            A table detailing the variability of sources.
            
        .. attribute:: model_dictionary
            :type: Dict[str, Dict[str, Union[str, float]]]
        
            Dictionary of model parameters for each source.

        Important:
            - Initializes 'xmm_catalog' by opening the catalog from the given path.
            - Finds nearby sources using 'find_nearby_sources', storing results in 'nearby_sources_table' and 'nearby_sources_position'.
            - Loads additional catalogs (XMM DR11 and Xmm2Athena) specified in 'simulation_data', initializing 'xmm_dr11_catalog' and 'x2a_catalog'.
            - Calculates photon indices and column densities for nearby sources using 'get_phoindex_nh'.
            - Constructs a variability table for nearby sources and visualizes the neighborhood of the object of interest.
        """
        self.key = "XMM"
        # ---------- coord ---------- #
        self.ra = dict_cat.dictionary_coord[self.key]["right_ascension"]
        self.dec = dict_cat.dictionary_coord[self.key]["declination"]
        # -------------------------- #
        
        catalog = self.open_catalog(catalog_path=catalog_path)
        
        if len(user_table) != 0:
            self.xmm_catalog = self.add_row(user_table=user_table, catalog=catalog)
        else:
            self.xmm_catalog = catalog
        
        self.nearby_sources_table, self.nearby_sources_position = self.find_nearby_sources(radius=radius, object_data=simulation_data["object_data"], user_table=user_table)
        
        test_dr11_path = os.path.join(simulation_data["os_dictionary"]["catalog_datapath"], "4XMM_DR11cat_v1.0.fits").replace("\\", "/")
        test_x2a_path = os.path.join(simulation_data["os_dictionary"]["catalog_datapath"], "xmm2athena_D6.1_V3.fits").replace("\\", "/")
        test_nh_path = os.path.join(simulation_data["os_dictionary"]["catalog_datapath"], "NHI_HPX.fits").replace("\\", "/")
        xmm_dr11_path = i_f.get_valid_file_path(test_dr11_path)
        x2a_path = i_f.get_valid_file_path(test_x2a_path)
        nh_path = i_f.get_valid_file_path(test_nh_path)

        self.xmm_dr11_catalog = self.open_catalog(catalog_path=xmm_dr11_path)
        self.x2a_catalog = self.open_catalog(catalog_path=x2a_path)
        self.nh_dictionnary = self.open_catalog(catalog_path=nh_path)
        self.nearby_sources_table, self.index_table = self.get_phoindex_nh()
        self.variability_table = self.variability_table(object_data=simulation_data["object_data"])
        self.neighbourhood_of_object(radius=radius, simulation_data=simulation_data, user_table=user_table)
        self.model_dictionary = self.dictionary_model()
        
        
    
    def open_catalog(self, catalog_path: str) -> Table:
        """
        Opens an XMM catalog from a specified file path and returns it as an Astropy Table.

        This method is designed to open FITS files which contain astronomical data, typically used in X-ray astronomy. 
        It utilizes the 'fits' module to read the file and convert the data into a format that is easily manageable 
        and usable for further analysis.

        Args:
            catalog_path (str): The file path to the XMM catalog FITS file. This file is expected to be in a 
                                format compatible with the FITS standard, commonly used in astrophysics and astronomy.

        Returns:
            Table: An Astropy Table object containing the data from the FITS file. The table structure allows for 
                easy manipulation and analysis of the catalog data, leveraging the capabilities of the Astropy package.

        """
        with fits.open(catalog_path, memmap=True) as data:
            return Table(data[1].data)
        
    
    def add_row(self, user_table: Table, catalog:Table) -> Table:
        """
        Adds rows from a user-provided table to the catalog.

        This method iterates through each row in the `user_table` and appends it to the `catalog`. 
        It's useful for incorporating additional data, provided by the user, into the existing catalog data. 
        The method ensures that all rows from the user's table are seamlessly integrated into the catalog.

        Args:
            user_table (Table): An Astropy Table containing user-defined data. Each row of this table will be added to the `catalog`.
            catalog (Table): The main catalog (an Astropy Table) to which the rows from `user_table` will be added.

        Returns:
            Table

        Note:
            - The `user_table` and `catalog` must have compatible column structures for the addition to work.
            - If `user_table` contains columns not present in `catalog`, these new columns will be added to `catalog`.
            - The method modifies the `catalog` in-place and also returns it for convenience.
        """
        for row in range(len(user_table)):
            catalog.add_row(user_table[row])
        return catalog

    
    def find_nearby_sources(self, radius: Quantity, object_data: dict, user_table: Table) -> Tuple[Table, SkyCoord]:
        """
        Searches for sources within a specified radius around a given astronomical object and returns a table of these sources along with their coordinates.

        This method filters the XMM catalog to find sources that are within a certain radius of the specified object's position. 
        It can incorporate additional user-provided data for a more tailored search.

        Args:
            radius (Quantity): The search radius around the object of interest. This radius is used to define a circular region in the sky within which the sources will be searched for.
            object_data (dict): A dictionary containing data about the object of interest. It should include keys 'object_position' (SkyCoord) and 'object_name' (str), representing the astronomical coordinates and the name of the object, respectively.
            user_table (Table): An optional Astropy Table containing user-defined data. If provided, this table is used in addition to the XMM catalog for finding nearby sources.

        Returns:
            Tuple[Table, SkyCoord]: A tuple containing two elements:
                - An Astropy Table of sources found within the specified radius. The table includes various details of these sources as defined in the XMM catalog.
                - A SkyCoord object representing the coordinates of the found sources.

        Important:
            The method involves the following steps:
            
            - Calculating the minimum and maximum right ascension (RA) and declination (Dec) based on the object's position and the specified radius.
            - Filtering the XMM catalog to create a smaller table of sources within this RA and Dec range.
            - Further filtering these sources based on the angular separation from the object's position to ensure they are within the specified radius.
            - Handling user-provided data if available, to include additional sources in the search.
            - Returning the final list of nearby sources and their positions.
        
        """
        object_position = object_data['object_position']
        object_name = object_data['object_name']
        
        pointing_area = radius + 5*u.arcmin
        min_ra, max_ra = object_position.ra - pointing_area, object_position.ra + pointing_area
        min_dec, max_dec = object_position.dec - pointing_area, object_position.dec + pointing_area
        
        small_table = Table(names=self.xmm_catalog.colnames,
                            dtype=self.xmm_catalog.dtype)
        nearby_src_table = Table(names=self.xmm_catalog.colnames,
                                 dtype=self.xmm_catalog.dtype)
        
        print(fr"{colored('Reducing XMM catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
        for number in tqdm(range(len(self.xmm_catalog))):
            if min_ra/u.deg < self.xmm_catalog[self.ra][number] < max_ra/u.deg and min_dec/u.deg < self.xmm_catalog[self.dec][number] < max_dec/u.deg:
                small_table.add_row(self.xmm_catalog[number])

        src_position = SkyCoord(ra=small_table[self.ra], dec=small_table[self.dec], unit=u.deg)
        print(f"{colored(f'Find sources close to {object_name} with XMM catalog', 'blue')}")
        for number in tqdm(range(len(small_table))):
            if c_f.ang_separation(object_position, src_position[number]) < radius:
                nearby_src_table.add_row(small_table[number])
        nearby_src_position = SkyCoord(ra=nearby_src_table[self.ra], dec=nearby_src_table[self.dec], unit=u.deg)
            
        try:
            if len(nearby_src_table) != 0:
                print((f"We have detected {len(nearby_src_table)} sources close to {object_name}.\n"))
                return nearby_src_table, nearby_src_position
            else:
                print(f"No sources detected close to {object_name}.")
                sys.exit()
        except Exception as error:
            print(f"An error occured : {error}")
            
       
    def optim_index(self, number: int) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[float, float],Tuple[float]]:
        """
        Optimizes the photon index for a specific source in the catalog using an absorbed power law model.

        This method applies curve fitting to observed flux data to determine the best-fit parameters of an absorbed power law model. 
        It is primarily used to analyze the energy distribution of a source and estimate its photon index, a key parameter in astrophysical analysis.

        Args:
            number (int): The index of the source in the nearby_sources_table for which the photon index is to be optimized.

        Returns:
            Tuple[Tuple[np.ndarray, ...], Tuple[float, float]]: A tuple containing two elements:
                - A tuple of numpy arrays representing the energy band, observed flux (y_array), flux error (yerr_array), 
                and the fitted power law values for the given energy band.
                - A tuple containing the optimized constant and photon index values from the power law fit.

        Important:
            The method involves:
            
            - Retrieving the observed flux and its error for the given source based on predefined band names.
            - Normalizing the flux values based on the energy band width.
            - Using the `curve_fit` function from `scipy.optimize` to fit the absorbed power law model to the observed data.
            - Returning the optimized parameters along with the energy band and normalized flux values.

        Note:
            The method assumes specific keys in 'dict_cat.dictionary_catalog' for retrieving energy band and flux information.
            The absorbed power law model is defined internally within the method.
        """    
        
        def absorbed_power_law(x, constant, gamma):
            sigma = np.array([1e-20, 5e-21, 1e-22, 1e-23, 1e-24], dtype=float)
            return (constant * x ** (-gamma)) * (np.exp(-sigma * 3e20))
        

        energy_band = dict_cat.dictionary_catalog[self.key]["energy_band_center"]
        energy_band_half_width = dict_cat.dictionary_catalog[self.key]["energy_band_half_width"]
        tab_width = 2 * energy_band_half_width
        
        band_flux_obs_name = dict_cat.dictionary_catalog[self.key]["band_flux_obs"]
        band_flux_obs_err_name = dict_cat.dictionary_catalog[self.key]["band_flux_obs_err"]
        
        flux_obs = [self.nearby_sources_table[name][number] for name in band_flux_obs_name]
        flux_err = [[self.nearby_sources_table[err_0][number] for err_0 in band_flux_obs_err_name[0]],
                    [self.nearby_sources_table[err_1][number] for err_1 in band_flux_obs_err_name[1]]]

        flux_err_obs = [np.mean([flux_err_0, flux_err_1]) for flux_err_0, flux_err_1 in zip(flux_err[0], flux_err[1])]
        
        y_array = [num/det for num, det in zip(flux_obs, tab_width)]
        yerr_array = [num/det for num, det in zip(flux_err_obs, tab_width)]
        
        try:
            popt, pcov = curve_fit(absorbed_power_law, energy_band, y_array, sigma=yerr_array)
            constant, absorb_pho_index = popt
            print(f" (Source: Curve fit) - "f"pho_index: {absorb_pho_index}")
            perr = np.sqrt(np.diag(pcov))
            absorb_random = np.random.normal(absorb_pho_index, perr[1])
            #perr will have 2 values one associated with the const and another with the absorb_pho_index

        except Exception as error:
            constant = 1e-14
            absorb_pho_index = 1.7
            popt = constant, absorb_pho_index
            perr = 0.0
            print(f" (Source: constant) - "f"pho_index: {absorb_pho_index}")

        optimization_parameters = (energy_band, y_array, yerr_array, absorbed_power_law(energy_band, *popt))
        #return optimization_parameters, absorb_pho_index, perr[1]
        return optimization_parameters, absorb_pho_index, perr[1]


    def visualization_interp(self, optimization_parameters, photon_index) -> None:
        """
        Visualizes the results of the photon index interpolation for each source.

        This method creates a plot for each source showing the observed flux versus the energy band, along with the fitted power law model. 
        It's useful for visually assessing the quality of the fit and understanding the energy distribution of the sources.

        Args:
            optimization_parameters (tuple): A tuple containing the energy band, observed flux, flux error, and the fitted power law values for each source. These parameters are obtained from the optimization_phoindex method.
            photon_index (array): An array of photon index values for each source. These are the optimized photon index values obtained from the absorbed power law fit.

        Important:
            The method performs the following steps:
            
            - For each set of optimization parameters, it plots the observed flux (with errors) against the energy band.
            - It also plots the corresponding absorbed power law model, using the optimized photon index for each source.
            - The plots are annotated with the photon index value for each source.
            - The plot uses a logarithmic scale for both axes for better visualization of the power law behavior.

        Note:
            - The method assumes that the energy band data is provided in keV and the flux in erg.cm^-2.s^-1.keV^-1.
            - This method does not return any value but displays the plots directly using matplotlib.
        """
        
        energy_band = dict_cat.dictionary_catalog["XMM"]["energy_band_center"]
        
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
        
        plt.close()

    def get_phoindex_nh(self) -> Tuple[Table, Table]:
        """
        Retrieves and calculates the photon index and hydrogen column density (Nh) for each source in the nearby_sources_table.

        This method performs a series of operations to determine the photon index and Nh values for sources 
        based on data from the XMM DR11 and Xmm2Athena catalogs. It also involves optimizing the photon index 
        for sources not found in the Xmm2Athena catalog using an absorbed power law model.

        Returns:
            Tuple[Table, Table]: A tuple of two Astropy Tables:
                - The first table is the nearby_sources_table, updated with new columns for Photon Index and Nh.
                - The second table is an index table that maps each source to its corresponding index in the xmm_dr11 and x2a catalogs.

        Important:
            The method includes the following steps:
            
            - Compiling an index table that correlates the sources in the nearby_sources_table with their respective indices in the xmm_dr11 and x2a catalogs.
            - Calculating the Nh value for each source. If the source is found in the x2a catalog, the Nh value is taken from there; otherwise, a default value is used.
            - Optimizing the photon index for each source using the optimization_phoindex method. For sources in the x2a catalog, the photon index is taken from there.
            - Updating the nearby_sources_table with the calculated Photon Index and Nh values.
            - Visualizing the photon index interpolation results for sources not found in the x2a catalog.

        Note:
            - The method assumes specific keys and structures in the xmm_dr11 and x2a catalogs for data extraction.
            - It prints messages to the console regarding missing data in the xmm_dr11 catalog and visualizes the interpolation results.
        """
        
        name_list = [self.nearby_sources_table["IAUNAME"][number] for number in range(len(self.nearby_sources_table)) if self.nearby_sources_table["SRCID"][number] != 0]
        
        xmm_dr_11_table = Table(names=self.xmm_dr11_catalog.colnames,
                                dtype=self.xmm_dr11_catalog.dtype)
        index_dr11 = np.array([], dtype=int)
        
        for name in name_list:
            if name in self.xmm_dr11_catalog['IAUNAME']:
                index = list(self.xmm_dr11_catalog['IAUNAME']).index(name)
                index_dr11 = np.append(index_dr11, index)
                xmm_dr_11_table.add_row(self.xmm_dr11_catalog[index])
            else:
                print(f"{colored('Missing data in Xmm_DR11 : ', 'red')} {name}")
                index_dr11 = np.append(index_dr11, np.nan)

        index_x2a = np.array([], dtype=int)
        for det_id in xmm_dr_11_table["DETID"]:
            if det_id in self.x2a_catalog["DETID"]: 
                index = list(self.x2a_catalog["DETID"]).index(det_id)
                index_x2a = np.append(index_x2a, index)
            else:
                index_x2a = np.append(index_x2a, "No data found")

        column_names = ["Index in nearby_sources_table", "Index in xmm_dr11", "Index in x2a"]
        column_data = [[item for item in range(len(name_list))], index_dr11, index_x2a]        
        index_table = Table(names=column_names,
                            data=column_data)
        
        #column_nh, column_phoindex = np.array([], dtype=float), np.array([], dtype=float)
        #pho_index_min_med, pho_index_max_med = np.array([], dtype=float), np.array([], dtype=float)
        column_nh = np.array([], dtype=float)
        column_nh_min = np.array([], dtype=float)
        column_nh_max = np.array([], dtype=float)

        column_phoindex, pho_index_min_med, pho_index_max_med = [], [], []
        optimization_parameters, photon_index = [], []

    

        for number in range(len(name_list)):
            print("---------------------------------------------------------")
            print(f"NAME : {name_list[number]}")
            i = 0
            if i == 1:
            #if index_table["Index in x2a"][number] != "No data found":

                nh_value = self.x2a_catalog["logNH_med"][number]
                nh_value_min = self.x2a_catalog["logNH_med_min"][number]
                nh_value_max = self.x2a_catalog["logNH_med_max"][number]

                column_nh = np.append(column_nh, np.exp(nh_value * np.log(10)))
                column_nh_min = np.append(column_nh_min, np.exp(nh_value_min * np.log(10)))
                column_nh_max = np.append(column_nh_max, np.exp(nh_value_max * np.log(10)))

                transformed_nh_value = np.exp(nh_value * np.log(10))
                transformed_nh_min = np.exp(nh_value_min * np.log(10))
                transformed_nh_max = np.exp(nh_value_max * np.log(10))
                
                column_phoindex = np.append(column_phoindex, self.x2a_catalog['PhoIndex_med'][number])
                pho_index_min_med = np.append(pho_index_min_med, self.x2a_catalog['PhoIndex_med_min'][number])
                pho_index_max_med = np.append(pho_index_max_med, self.x2a_catalog['PhoIndex_med_max'][number])
                
                # Retrieve the photon index values for printing
                pho_index = self.x2a_catalog['PhoIndex_med'][number]
                pho_index_min = self.x2a_catalog['PhoIndex_med_min'][number]
                pho_index_max = self.x2a_catalog['PhoIndex_med_max'][number]

                

               
            else:
                
                ra_val = self.nearby_sources_table[self.ra][number]
                dec_val = self.nearby_sources_table[self.dec][number]
                
                distances = np.sqrt((self.nh_dictionnary['RA2000'] - ra_val)**2 + (self.nh_dictionnary["DEC2000"] - dec_val)**2)
                best_match_index = np.argmin(distances)

                nhi_value = self.nh_dictionnary['NHI'][best_match_index]
                column_nh = np.append(column_nh, nhi_value)
                column_nh_min = np.append(column_nh_min, nhi_value)
                column_nh_max = np.append(column_nh_max, nhi_value)
                
                print(f"Nh values(Source: Nh sky map) - Nh: {nhi_value}, "
                    f"Nh min: {nhi_value}, Nh max: {nhi_value}")
                
                parameters, pho_value, perr = self.optim_index(number)
                optimization_parameters.append(parameters)

                pho_value_min = pho_value - perr
                pho_value_max = pho_value + perr

                photon_index.append(pho_value)
                column_phoindex = np.append(column_phoindex, pho_value)
                pho_index_max_med = np.append(pho_index_max_med, pho_value_max) 
                pho_index_min_med = np.append(pho_index_min_med, pho_value_min) 

                print(f"Photon Index values (Source: Optimized value) - "
                    f"pho_index: {pho_value}, pho_index min: {pho_value_min}, pho_index max: {pho_value_max}")
                  
        index_add_source = []
        if len(name_list) != len(self.nearby_sources_table):
            for name in self.nearby_sources_table["IAUNAME"]:
                if name not in name_list:
                    index_name = list(self.nearby_sources_table["IAUNAME"]).index(name)
                    index_add_source.append(index_name)
                    
        for index in index_add_source:

            ra_val = self.nearby_sources_table[self.ra][number]
            dec_val = self.nearby_sources_table[self.dec][number]
            
            distances = np.sqrt((self.nh_dictionnary['RA2000'] - ra_val)**2 + (self.nh_dictionnary["DEC2000"] - dec_val)**2)
            best_match_index = np.argmin(distances)

            nhi_value = self.nh_dictionnary['NHI'][best_match_index]
            column_nh = np.append(column_nh, nhi_value)
            column_nh_min = np.append(column_nh_min, nhi_value)
            column_nh_max = np.append(column_nh_max, nhi_value)
            print(f"Nh values(Source: Nh sky map) - Nh: {nhi_value}, "
                    f"Nh min: {nhi_value}, Nh max: {nhi_value}")
                
            parameters, pho_value, perr = self.optim_index(index)
            optimization_parameters.append(parameters)

            pho_value_min = pho_value - perr
            pho_value_max = pho_value + perr

            pho_index_min_med = np.append(pho_index_min_med, pho_value_min) 
            pho_index_max_med = np.append(pho_index_max_med, pho_value_max) 
            column_phoindex = np.append(column_phoindex, pho_value)
            print(f"Photon Index values (Source: Optimized value) - "
                    f"pho_index: {pho_value}, pho_index min: {pho_value_min}, pho_index max: {pho_value_max}")
                  
        self.visualization_interp(optimization_parameters=optimization_parameters, photon_index=photon_index)
        
        #col_names = ["Photon Index", "Photon Index Error", "Nh", "Nh min value", "Nh max value"]
        #col_data = [column_phoindex, perr,  column_nh, column_nh_min, column_nh_max]
        col_names = ["Photon Index", "Nh", "Nh min value", "Nh max value"]
        col_data = [column_phoindex, column_nh, column_nh_min, column_nh_max]
    
        for name, data in zip(col_names, col_data):
            self.nearby_sources_table[name] = data

        return self.nearby_sources_table, index_table
    
    
    def variability_table(self, object_data: dict) -> Table:
        """
        Creates a table summarizing the variability of sources near a specified astronomical object.

        This method assesses the variability of each source in the nearby_sources_table using data from the xmm_dr11_catalog. 
        It compiles a new table that includes details about the variability of these sources, along with their inclusion in the Xmm2Athena catalog.

        Args:
            object_data (dict): A dictionary containing data about the object of interest, including its name.

        Returns:
            Table: An Astropy Table containing information on the variability of each source. The table includes the source's index, 
                IAU name, coordinates (RA and Dec), variability factor (SC_FVAR), and a boolean indicating if the source is included 
                in the Xmm2Athena catalog.

        Important:
            The method performs the following operations:
            
            - Iterates through the nearby_sources_table to check for each source's variability factor from the xmm_dr11_catalog.
            - Constructs a new table with relevant information for sources that have a defined variability factor.
            - The table is populated with indexes, names, coordinates, variability factors, and Xmm2Athena catalog inclusion status.
            - Prints a summary message to the console about the number of variable sources and their distribution in the Xmm2Athena catalog.

        Note:
            - The method assumes the presence of specific keys in 'dict_cat.dictionary_coord' for coordinate comparison and in 'object_data' for the object's name.
            - It prints summary messages to the console for informative purposes.
        """
        basic_index = [index for index in range(len(self.nearby_sources_table)) if self.nearby_sources_table["SRCID"][index] != 0]
        object_name = object_data["object_name"]

        index_array, iauname_array, sc_ra_array = np.array([], dtype=int), np.array([], dtype=str), np.array([], dtype=float)
        sc_dec_array, sc_fvar_array, in_x2a_array = np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=bool)

        for number in basic_index:
            if not np.isnan(self.xmm_dr11_catalog["SC_FVAR"][self.index_table["Index in xmm_dr11"][number]]):

                index_array = np.append(index_array, self.index_table["Index in nearby_sources_table"][number])
                iauname_array = np.append(iauname_array, self.nearby_sources_table["IAUNAME"][number])
                sc_ra_array = np.append(sc_ra_array, self.nearby_sources_table[self.ra][number])
                sc_dec_array = np.append(sc_dec_array, self.nearby_sources_table[self.dec][number])
                sc_fvar_array = np.append(sc_fvar_array, self.nearby_sources_table["SC_FVAR"][number])

                if self.index_table["Index in x2a"][number] != "No data found":
                    in_x2a_array = np.append(in_x2a_array, True)
                else:
                    in_x2a_array = np.append(in_x2a_array, False)

        column_names = ["INDEX", "IAUNAME", self.ra, self.dec, "SC_FVAR", "IN_X2A"]
        data_array = [index_array, iauname_array, sc_ra_array, sc_dec_array, sc_fvar_array, in_x2a_array]
        variability_table = Table()

        for data, name in zip(data_array, column_names):
            variability_table[name] = data

        message_xmm = f"Among {len(self.nearby_sources_table)} sources detected close to {object_name}, {len(index_array)} of them are variable. Using DR13 Catalog."
        print(message_xmm)
        message_xmm2ath = f"Among {len(index_array)} variable sources, {list(variability_table['IN_X2A']).count(True)} are in Xmm2Athena and {list(variability_table['IN_X2A']).count(False)} are not in Xmm2Athena. "    
        print(message_xmm2ath)

        return variability_table
    
    
    def neighbourhood_of_object(self, radius: Quantity, simulation_data: dict, user_table: Table) -> None:
        """
        Visualizes the neighborhood of a specified astronomical object, highlighting the variable and invariable sources around it.

        This method creates a visual representation of sources around the given object. It differentiates between sources that are variable 
        and those that are not, based on their presence in the Xmm2Athena catalog, and plots their positions relative to the object.

        Args:
            radius (Quantity): The radius around the object within which to search for neighboring sources.
            simulation_data (dict): A dictionary containing simulation data, including 'object_data' with the target object's information and 'os_dictionary' for saving the plot.

        Important:
            The method performs the following operations:
            
            - Queries the ESASky catalog for XMM-EPIC observations of the object and retrieves corresponding images if available.
            - Identifies variable and invariable sources from the nearby_sources_table and the variability_table.
            - Creates two plots: 
                - The first plot shows all sources around the object.
                - The second plot differentiates between variable sources found in Xmm2Athena and those not found.
            - Both plots include the position of the target object and are saved as images.

        Note:
            - This method attempts to query ESASky for relevant data and handles any exceptions that may occur during this process.
            - It assumes the availability of right ascension (RA) and declination (Dec) keys in 'dict_cat.dictionary_coord'.
            - The plots are saved to the directory specified in 'os_dictionary' within 'simulation_data'.
        
        Here is an example of the plot create by this method:
        
        .. image:: C:/Users/plamb_v00y0i4/OneDrive/Bureau/Optimal_Pointing_Point_Code/modeling_result/PSR_J0437-4715/4XMM_DR13/img/neighbourhood_of_PSR_J0437-4715.png
            :align: center
        """

        print("\n")
        object_data = simulation_data["object_data"]
        object_name = object_data['object_name']
        obj_ra, obj_dec = object_data['object_position'].ra, object_data['object_position'].dec
        try:
            result = ESASky.query_object_catalogs(position=object_name, catalogs="XMM-EPIC")
            xmm_epic = Table(result[0])
            xmm_obs_id = list(xmm_epic["observation_id"])
            result_fits_images = ESASky.get_images(observation_ids=xmm_obs_id[0], radius=radius, missions="XMM")
        except Exception as error:
            print(f"{colored('An error occured : ', 'red')} {error}")
            result_fits_images = {}
            
        
        ra_in_x2a = [ra_value for index, ra_value in enumerate(self.variability_table[self.ra]) if self.variability_table['IN_X2A'][index] == True]
        dec_in_x2a = [dec_value for index, dec_value in enumerate(self.variability_table[self.dec]) if self.variability_table['IN_X2A'][index] == True]
        ra_in_dr11 = [ra_value for index, ra_value in enumerate(self.variability_table[self.ra]) if self.variability_table['IN_X2A'][index] == False]
        dec_in_dr11 = [dec_value for index, dec_value in enumerate(self.variability_table[self.dec]) if self.variability_table['IN_X2A'][index] == False]
        invar_ra = [ra_value for ra_value in self.nearby_sources_table[self.ra] if ra_value not in self.variability_table[self.ra]]
        invar_dec = [dec_value for dec_value in self.nearby_sources_table[self.dec] if dec_value not in self.variability_table[self.dec]]
        
        if len(user_table) != 0:
            add_ra = [ra_value for ra_value in user_table[self.ra]]
            add_dec = [dec_value for dec_value in user_table[self.dec]]
        
        figure = plt.figure(figsize=(17, 8))
        figure.suptitle(f"Neighbourhood of {object_name}", fontsize=20)
        
        if result_fits_images == {}:
            figure.text(0.5, 0.04, 'Right Ascension [deg]', ha='center', va='center')
            figure.text(0.04, 0.5, 'Declination [deg]', ha='center', va='center', rotation='vertical')
            
            axes_0 = figure.add_subplot(121)
            axes_0.invert_xaxis()
            axes_0.scatter(list(self.nearby_sources_table[self.ra]), list(self.nearby_sources_table[self.dec]), s=20, color='darkorange', label=f"Sources : {len(self.nearby_sources_table)}")
            if len(user_table) != 0:
                axes_0.scatter(add_ra, add_dec, s=20, color='limegreen', label=f"Add sources : {len(user_table)}")
            axes_0.scatter(obj_ra, obj_dec, s=100, color='red', marker="*", label=f"{object_name}")
            axes_0.legend(loc='upper right', ncol=2, fontsize=7)
            axes_0.set_title(f"Sources close to {object_name}")
            
            axes_1 = figure.add_subplot(122)
            axes_1.invert_xaxis()
            axes_1.scatter(invar_ra, invar_dec, color='black', s=20, label=f"Invariant sources, {len(invar_ra)}")
            if len(user_table) != 0:
                axes_1.scatter(add_ra, add_dec, s=20, color='limegreen', label=f"Add sources : {len(user_table)}")
            axes_1.scatter(obj_ra, obj_dec, color='red', marker='x', s=100, label=f"Position of {object_name}")
            axes_1.scatter(ra_in_x2a, dec_in_x2a, color='darkorange', marker="*", label=f"Variable sources in X2A, {len(ra_in_x2a)}")
            axes_1.scatter(ra_in_dr11, dec_in_dr11, color='royalblue', marker="*", label=f"Variable sources not in X2A, {len(ra_in_dr11)}")
            axes_1.legend(loc="lower right", ncol=2)
            axes_1.set_title(f"Variable and invariable sources close to {object_name} ")
            
        else:
            image = result_fits_images["XMM"][0][0].data[0, :, :]
            wcs = WCS(result_fits_images["XMM"][0][0].header)
            _wcs_ = wcs.dropaxis(2)
            
            figure.text(0.5, 0.04, 'Right Ascension [hms]', ha='center', va='center', fontsize=16)
            figure.text(0.04, 0.5, 'Declination [dms]', ha='center', va='center', rotation='vertical', fontsize=16)
            
            norm = ImageNormalize(image,interval=PercentileInterval(98.0), stretch=LinearStretch())
            
            axes_0 = figure.add_subplot(121, projection=_wcs_)
            axes_0.coords[0].set_format_unit(u.hourangle)
            axes_0.coords[1].set_format_unit(u.deg)  
            axes_0.imshow(image, cmap='gray', origin='lower', norm=norm, interpolation='nearest', aspect='equal')
            axes_0.scatter(list(self.nearby_sources_table[self.ra]), list(self.nearby_sources_table[self.dec]), s=30, transform=axes_0.get_transform('fk5'), facecolors='none', edgecolors='orange', label=f"Sources : {len(self.nearby_sources_table)}")
            if len(user_table) != 0:
                axes_0.scatter(add_ra, add_dec, s=30, transform=axes_0.get_transform('fk5'), facecolors='none', edgecolors='limegreen', label=f"Add sources : {len(user_table)}")
            axes_0.scatter(obj_ra, obj_dec, s=100, color='red', marker="*", transform=axes_0.get_transform('fk5'), facecolors='none', edgecolors='red', label=f"{object_name}")
            axes_0.legend(loc='upper right', ncol=2, fontsize=7)
            xlim, ylim = plt.xlim(), plt.ylim()
            value_x, value_y = 180, 180
            axes_0.set_xlim(xmin=xlim[0]+value_x, xmax=xlim[1]-value_x)
            axes_0.set_ylim(ymin=ylim[0]+value_y, ymax=ylim[1]-value_y)
            axes_0.set_xlabel(" ")
            axes_0.set_ylabel(" ")
            axes_0.set_title(f"Sources close to {object_name}")
            
            axes_1 = figure.add_subplot(122, projection=_wcs_, sharex=axes_0, sharey=axes_0)
            axes_1.imshow(image, cmap='gray', origin='lower', norm=norm, interpolation='nearest', aspect='equal')
            axes_1.scatter(invar_ra, invar_dec, s=30, transform=axes_1.get_transform('fk5'), facecolors='none', edgecolors='orange', label=f"Invar src : {len(invar_ra)}")
            axes_1.scatter(ra_in_dr11, dec_in_dr11, s=30, transform=axes_1.get_transform('fk5'), facecolors='none', edgecolors='blue', label=f"Var src not in x2a : {len(ra_in_dr11)} sources")
            axes_1.scatter(ra_in_x2a, dec_in_x2a, s=30, transform=axes_1.get_transform('fk5'), facecolors='none', edgecolors='hotpink', label=f"Var src in x2a : {len(ra_in_x2a)} sources")
            if len(user_table) != 0:
                axes_1.scatter(add_ra, add_dec, s=30, transform=axes_1.get_transform('fk5'), facecolors='none', edgecolors='limegreen', label=f"Add sources : {len(user_table)}")
            axes_1.scatter(obj_ra, obj_dec, s=100, marker="*", transform=axes_1.get_transform('fk5'), facecolors='none', edgecolors='red', label=f"{object_name}")
            axes_1.legend(loc='upper right', ncol=2, fontsize=7)
            axes_1.set_xlabel(" ")
            axes_1.set_ylabel(" ")
            axes_1.set_title(f"Variable and invariable sources close to {object_name} ")
            
        os_dictionary = simulation_data["os_dictionary"]
        plt.savefig(os.path.join(os_dictionary["img"], f"neighbourhood_of_{object_name}.png".replace(" ", "_")))
        plt.close()
        print("\n")


    def dictionary_model(self) -> Dict[str, Dict[str, Union[str, float]]]:
        """
        Creates a dictionary of models for each source in the nearby_sources_table, detailing the model type, its parameters, and source flux.

        This method compiles a dictionary where each entry corresponds to a source in the nearby_sources_table. 
        It includes details like the model used for analysis, the model's parameters, the observed flux, and the hydrogen column density (Nh).

        Returns:
            Dict[str, Dict[str, Union[str, float]]]: A dictionary with keys as source identifiers (e.g., 'src_0', 'src_1', etc.). 
            Each value is another dictionary containing the following keys:
                - 'model': The name of the model used for the source (e.g., 'power').
                - 'model_value': The value of a key parameter in the model (e.g., photon index for a power-law model).
                - 'flux': The observed flux of the source.
                - 'column_density': The hydrogen column density (Nh) associated with the source.

        Important:
            The method performs the following operations:
            
            - Iterates through each source in the nearby_sources_table.
            - Determines the appropriate model and its parameters based on predefined criteria.
            - Compiles this information into a dictionary, alongside the source's observed flux and Nh value.

        Note:
            - Currently, the method only implements the 'power' model, with the photon index as the model value.
            - Future extensions may include additional models like 'black_body' or 'temp'.
            - The method assumes that flux and Nh values are available in the nearby_sources_table.
        """
        
        model_dictionary = {}
        number_source = len(self.nearby_sources_table)

        flux_obs = dict_cat.dictionary_catalog[self.key]["flux_obs"]
        
        model = np.array([], dtype=str)
        model_value = np.array([], dtype=float)
        xmm_flux = np.array([self.nearby_sources_table[flux_obs][item] for item in range(number_source)], dtype=float)
        nh_value = np.array([self.nearby_sources_table["Nh"][item] for item in range(number_source)], dtype=float)
        
        # Pour le moment seulement 'power' indiquant le modèle a utiliser pour la commande pimms
        for item in range(number_source):
            model = np.append(model, 'power')    
            
        for item in range(number_source):
            if model[item] == 'power':
                model_value = np.append(model_value, self.nearby_sources_table["Photon Index"][item])
            elif model[item] == 'black_body':
                pass # Pas de valeur pour le moment...
            elif model[item] == 'temp':
                pass # Pas de valeur pour le moment... (dernier model pimms)

        for item in range(number_source):

            dictionary = {
                "model": model[item],
                "model_value": model_value[item],
                "flux": xmm_flux[item],
                "column_dentsity": nh_value[item]
            }

            model_dictionary[f"src_{item}"] = dictionary
            
        return model_dictionary 

