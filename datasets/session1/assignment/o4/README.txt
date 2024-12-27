GENERAL INFORMATION

1. Title of Dataset: Dataset for greenhouse gas modelling in diesel dependent communities transitioning to bioenergy 

2. Author Information
	A. Principal Investigator Contact Information
		Name: Dr. Nicolas Mansuy
		Institution: Natural Resources Canada, Canadian Forest Service, Northern Forestry Centre
		Address: Northern Forestry Centre, 5320 122 St., Edmonton, AB, T6H 3S5, Canada
		Email: nicolas.mansuy@NRCan-RNCan.gc.ca

	B. Co-investigator Contact Information
		Name: Dr. Jérôme Laganière
		Institution: Natural Resources Canada, Canadian Forest Service, Laurentian Forestry Centre

	C. Alternate Contact Information
		Name: Jennifer Buss
		Institution: Natural Resources Canada/ Canadian Forest Service
		Email: jennifer.buss@NRCan-RNCan.gc.ca

3. Date of data collection: 2020

4. Geographic location of data collection: Data are from northern/western Canada, with as many parameters as possible specific to Fort McPherson, NWT

5. Information about funding sources that supported the collection of the data: 
		This work was supported by the Office of Energy Research and Development, Project 
		“Supporting Indigenous clean energy shift with asset-based community development” granted to Dr. Nicolas Mansuy.

SHARING/ACCESS INFORMATION

1. Licenses/restrictions placed on the data: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication

2. Links to publications that cite or use the data: 
	Buss, Jennifer; Mansuy, Nicolas; Laganière, Jérôme; Persson, Daniel. (2022), Greenhouse gas mitigation potential of replacing diesel fuel with wood-based bioenergy in an arctic 
	Indigenous community: A pilot study in Fort McPherson, Canada, Biomass and Bioenergy, https://doi.org/10.1016/j.biombioe.2022.106367 

3. Links to other publicly accessible locations of the data: 
	https://github.com/jbuss11/GHG_LCA_code_2022

4. Links/relationships to ancillary data sets: NA

5. Was data derived from another source? yes
	A. If yes, list source(s): List of sources is included in the metadata page of the Bioenergy_Modelling_Dataset.csv file and can be found in Buss, Jennifer; Mansuy, Nicolas; 
	Laganière, Jérôme; Persson, Daniel. (2022), Greenhouse gas mitigation potential of replacing diesel fuel with wood-based bioenergy in an arctic 
	Indigenous community: A pilot study in Fort McPherson, Canada, Biomass and Bioenergy, https://doi.org/10.1016/j.biombioe.2022.106367 

6. Recommended citation for this dataset: 
	Buss, Jennifer; Mansuy, Nicolas; Laganière, Jérôme. (2022), Dataset for greenhouse gas modelling in diesel dependent communities transitioning to bioenergy, Dryad, Dataset, 
	https://doi.org/10.5061/dryad.79cnp5hxw

DATA & FILE OVERVIEW

1. File List: 
	File 1/3: Bioenergy_Modelling_Dataset.csv. This file contains GHG emissions factors, forest growth parameters, and biomass decomposition parameters used in the greenhouse gas (GHG)
	modelling performed in Buss et al. 2022. A metadata tab in the file describes all parameters included in the data file, their sources, as well as the naming convention for 
	scenarios. The data file includes a total of 72 scenarios.

	File 2/3: Life_Cycle_Analysis_(LCA)_Script.txt. This file contains the R code for the GHG model. The code guides users through a seven-step process designed to aid in the 
	development of biomass and reference fossil fuel scenarios, and calculation of the quantity and timing of GHG benefits. It also includes a description of model parameters and 
	outputs.

	File 3/3: Model_Output_Summary_Table.docx. This file contains a small subset of model outputs for all 72 scenarios. Model outputs included in this file are carbon parity times and
	GHG emissions at 25, 50, and 100 years.

2. Relationship between files, if important: 
	File 1/3: Bioenergy_Modelling_Dataset.csv contains the data parameters required to run the model in File 2/3: LCA_Script.csv.
	File 3/3: Model_Output_Summary_Table.csv includes a small subset of outputs from the model (File 2/3: LCA_Script.csv).

3. Additional related data collected that was not included in the current data package: 
	The raw model output files for all 72 scenarios were not included in the current data package because they can be recreated using the provided dataset and script. However, a subset
        of the main results have been included in the Model_Output_Summary_Table.csv. 

4. Are there multiple versions of the dataset? NA


METHODOLOGICAL INFORMATION

1. Description of methods used for collection/generation of data: 
	Data were collected and curated from a community-based bioenergy project, published literature, and other GHG models and inventories. 
	Further details on the methods used for data collection can be found in Buss, Jennifer; Mansuy, Nicolas; Laganière, Jérôme; Persson, Daniel. (2022), 
	Greenhouse gas mitigation potential of replacing diesel fuel with wood-based bioenergy in an arctic Indigenous community: A pilot study in Fort McPherson, Canada, 
	Biomass and Bioenergy, https://doi.org/10.1016/j.biombioe.2022.106367 

2. Methods for processing the data: 
	Buss, Jennifer; Mansuy, Nicolas; Laganière, Jérôme; Persson, Daniel. (2022), Greenhouse gas mitigation potential of replacing diesel fuel with wood-based bioenergy in an arctic 
	Indigenous community: A pilot study in Fort McPherson, Canada, Biomass and Bioenergy, https://doi.org/10.1016/j.biombioe.2022.106367 

3. Instrument- or software-specific information needed to interpret the data: 
	Any spreadsheet programs and most data analysis softwares such as the R software. 
	R packages needed in the model code include dplyr and xlsx

4. Standards and calibration information, if appropriate: NA

5. Environmental/experimental conditions: NA

6. Describe any quality-assurance procedures performed on the data: NA

7. People involved with sample collection, processing, analysis and/or submission: 
	Jennifer Buss, Nicolas Mansuy, Jérôme Laganière


DATA-SPECIFIC INFORMATION FOR: Bioenergy_Modelling_Dataset.csv

1. Number of variables: 28

2. Number of cases/rows: 72 scenarios

3. Variable List: 

C_Bio: collection emissions factor for biomass (kg CO2eq/GJ)
P_Bio: processing emissions factor for biomass (kg CO2eq/GJ)
Bio_km: transport distance (one-way) for biomass (km)
Bio_vehicle_EF: transport emissions factor for the biomass transport vehicle (kg CO2 eq/t.km)
Bio_km_multiplier: multiplier for load type (empty vs full truck) and/or road type (paved vs unpaved) for biomass transport
Bio_EC: energy content of biomass feedstock (GJ/t)
Co_Bio: biomass conversion/combustion emissions factor (kg CO2 eq/GJ)
CE_Bio: conversion efficiency of biomass boiler
a: maximum volume of the stand (odt/ha)
b: slope of biomass growth curve (odt/ha.year)
c: age to half volume of the stand (years)
d: volume to reach harvest age (100 % regeneration) (odt/ha)
e: stand age at the time of the first harvest (years)
e2: stand age at the time of the second harvest (years) 
e3: stand age at the time of the third harvest (years) 
e4: stand age at the time of the fourth harvest (years)
C0: inital quantity of carbon stored in biomass
CP_FF: collection and processing emissions factor for fossil fuel (kg CO2eq/GJ)
FF_km: transport distance (one-way) for fossil fuel (km)
FF_vehicle_EF: transport emissions factor for fossil fuel transport vehicle (kg CO2 eq/t.km)
FF_km_multiplier: multiplier for load type (empty vs full truck) and/or road type (paved vs unpaved) for fossil fuel transport
EC_FF: energy content of fossil fuel (GJ/t)
Co_FF: conversion/combustion emissions factor for fossil fuel (kg CO2eq/GJ)
CE_Fossil: conversion efficiency of fossil fuel generator (enter as a proportion)
MATf: mean annual temperature of location where feedstock will decompose (C)
RefMAT: reference mean annual temperature (C)
Q10: temperature sensitivity of decomposition
BDRk: base decomposition rate of feedstock (year^-1)

4. Missing data codes: NA


5. Specialized formats or other abbreviations used: NA
