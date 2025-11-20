# Weather/Atmospheric Science

- <mark>Arevalo-2023</mark>: Sensitivity of Simulated Conditions to Different Parameterization Choices Over Complex Terrain in Central Chile

	- This study evaluates the performance of 14 high-resolution WRF runs with different combinations of parameterizations in simulating the atmospheric conditions over the complex terrain of central Chile during austral winter and spring. 
	- We focus on the **validation of results for coastal, interior valleys, and mountainous areas independently**, and also present an in-depth analysis of two synoptic-scale events that occurred during the study period: a frontal system and a cut-off low. 
	- The performance of the simulations decreases from the coast to higher altitudes, even though the differences are not very clear between the coast and interior valleys for ws10 and precipitation. The simulated vertical profiles show a warmer and drier boundary layer and a cooler and moister free atmosphere than observed. The choice of the land-surface model has the largest positive impact on near-surface variables with the five-layer thermal diffusion scheme showing the smallest errors. 
	- Precipitation is more sensitive to the choice of cumulus parameterizations, with the simplified **Arakawa–Schubert scheme** generally providing the best performance for absolute errors. When examining the performance of the model simulating rain/no-rain events for different thresholds, also the cumulus parameterizations better represented the false alarm ratio (FAR) and the bias score (BS). However, the **Morrison microphysics scheme** resulted in the best critical success index (CSI), while the probability of detection (POD) was better in the simulation without analysis nudging. 
	- **Overall**, these results provide guidance to other researchers and help to identify the best WRF configuration for a specific research or operational goal.

---

- <mark>Kratzert 2019</mark>: Toward Improved Predictions in Ungauged Basins: Exploiting the Power of Machine Learning

	- LSTM networks offer unprecedented accuracy for prediction in ungauged basins. 
	- We trained and tested several LSTMs on 531 basins from the **CAMELS** data set using **k-fold validation**, so that predictions were made in basins that supplied no training data. The training and test data set included  $\sim30$ years of daily rainfall-runoff data from catchments in the US ranging in size from 4 to 2,000 km 2 with aridity index from 0.22 to 5.20, and including 12 of the 13 IGPB vegetated land cover classifications. 
	- This effectively “ungauged” model was benchmarked over a **15-year validation period** against the Sacramento Soil Moisture Accounting (SAC-SMA) model and also against the NOAA National Water Model reanalysis. SAC-SMA was calibrated separately for each basin using 15 years of daily data. The out-of-sample LSTM had higher median Nash-Sutcliffe Efficiencies across the 531 basins (0.69) than either the calibrated SAC-SMA (0.64) or the National Water Model (0.58). 
	- This indicates that there is (typically) sufficient information in available catchment attributes data about similarities and differences between catchment-level rainfall-runoff behaviors to provide out-of-sample simulations that are generally more accurate than current models under ideal (i.e., calibrated) conditions. 
	- We found evidence that adding **physical constraints to the LSTM models might improve simulations**, which we suggest motivates future research related to physics-informed ML.

---

- <mark>Kratzert-2019</mark>: Towards Learning Universal, Regional, and Local Hydrological Behaviors via Machine-Learning Applied to Large-Sample Datasets

	- Regional rainfall-runoff modeling is an old but still mostly out-standing problem in Hydrological Sciences. The problem currently is that traditional hydrological models **degrade significantly in performance when calibrated for multiple basins together instead of for a single basin alone**. 
	- In this paper, we propose a novel, data-driven approach using LSTMs, and demonstrate that under a ’big data’ paradigm, this is not necessarily the case. By training a single LSTM model on 531 basins from the CAMELS data set using **meteorological time series data and static catchment attributes**, we were able to significantly improve performance compared to a set of several different hydrological benchmark models. 
	- Our proposed approach not only significantly outperforms hydrological models that were calibrated regionally but also achieves **better performance than hydrological models that were calibrated for each basin individually**. Furthermore, we propose an adaption to the standard LSTM architecture, which we call an **Entity-Aware-LSTM** (EA-LSTM), that allows for learning, and embedding as a feature layer in a deep learning model, **catchment similarities**. We show that this learned catchment similarity corresponds well with what we would expect from prior hydrological understanding.

---

- <mark>Fernandez 2017</mark>: Sensitivity Analysis of the WRF Model: Wind-Resource Assessment for Complex Terrain

	- Wind energy requires accurate forecasts for adequate integration into the electric grid system. In addition, global atmospheric models are not able to simulate local winds in complex terrain, where wind farms are sometimes placed. For this reason, the use of mesoscale models is vital for estimating wind speed at wind turbine hub height. In this regard, the Weather Research and Forecasting (WRF) Model allows a user to apply different initial and boundary conditions as well as physical parameterizations. In this research, a sensitivity analysis of several physical schemes and initial and boundary conditions was performed for the Alaiz mountain range in the northern Iberian Peninsula, where several wind farms are located. Model performance was evaluated under various atmospheric stabilities and wind speeds. For validation purposes, a mast with anemometers installed at 40, 78, 90, and 118 m above ground level was used. 
	- The results indicate that performance of the Global Forecast System analysis and European Centre for Medium-Range Weather Forecasts interim reanalysis (ERA-Interim) as initial and boundary conditions was similar, although each performed better under certain meteorological conditions. 
	- With regard to physical schemes, there is no single combination of parameterizations that performs best during all weather conditions. Nevertheless, some combinations have been identified as inefficient, and therefore their use is discouraged. As a result, the validation of an ensemble prediction system composed of the best 12 deterministic simulations shows the most accurate results, obtaining relative errors in wind speed forecasts that are ,15%.

---

- <mark>Addor 2018</mark>: Selection of hydrological signatures for large-sample hydrology 

	- Hydrological signatures are now used for a wide range of purposes, including **catchment classification**, process exploration and **hydrological model calibration**. The recent boost in the popularity and number of signatures has however not been accompanied by the development of clear guidance on signature selection, meaning that signature selection is often arbitrary. Here we use three complementary approaches to compare and rank 15 commonly-used signatures, which we evaluate in 671 US catchments from the CAMELS data set (Catchment Attributes and MEteorology for Large-sample Studies). 
	- Firstly, we employ random forests to explore how attributes characterizing the climatic conditions, topography, land cover, soil and geology influence (or not) the signatures. 
	- Secondly, we use a conceptual hydrological model (Sacramento) to critically assess which signatures are well captured by the simulations. 
	- Thirdly, we take advantage of the large sample of CAMELS catchments to characterize the spatial smoothness (using Moran’s I) of the signature field. 
	- These three approaches lead to **remarkably similar rankings of the signatures**. We show that **signatures with the noisiest spatial pattern tend to be poorly captured by hydrological simulations**, that their relationship to catchments attributes are elusive (in particular they are not correlated to climatic indices like aridity) and that they are particularly sensitive to discharge uncertainties. We question the utility and reliability of those signatures in experimental and modeling hydrological studies, and we underscore the general importance of accounting for uncertainties in hydrological signatures.

---

- <mark>Poblete 2020</mark>: Optimization of Hydrologic Response Units (HRUs) Using Gridded Meteorological Data and Spatially Varying Parameters

	- Although complex hydrological models with detailed physics are becoming more common, **lumped and semi-distributed models** are still used for many applications and offer some advantages, such as reduced computational cost. Most of these semi-distributed models use the concept of the hydrological response unit or HRU. In the original conception, HRUs are defined as homogeneous structured elements with similar climate, land use, soil and/or pedotransfer properties, and hence a homogeneous hydrological response under equivalent meteorological forcing. 
	- This work presents a **quantitative methodology**, called hereafter the principal component analysis and hierarchical cluster analysis or **PCA/HCPC** method, to construct HRUs using gridded meteorological data and hydrological parameters. 
	- The PCA/HCPC method is tested using the water evaluation and planning system (WEAP) model for the **Alicahue River Basin**, a small and semi-arid catchment of the **Andes, in Central Chile**. 
	- The results show that **with four HRUs, it is possible to reduce the relative within variance of the catchment up to about 10%, an indicator of the homogeneity of the HRUs**. The evaluation of the simulations shows a good agreement with streamflow observations in the outlet of the catchment with an Nash–Sutcliffe efficiency (NSE) value of **0.79** and also shows the presence of small hydrological extreme areas that generally are neglected due to their relative size.

---

- <mark>Di Giuseppe 2016</mark>: The Potential Predictability of Fire Danger Provided by Numerical Weather Prediction

	- A **global fire danger rating system driven by atmospheric model forcing** has been developed with the aim of providing early warning information to civil protection authorities. The daily predictions of fire danger conditions are based on the U.S. Forest Service National Fire-Danger Rating System (NFDRS), the Canadian Forest Service Fire Weather Index Rating System (FWI), and the Australian McArthur (Mark 5) rating systems. 
	- Weather forcings are provided in real time by the European Centre for Medium-Range Weather Forecasts forecasting system at **25-km resolution**. The global system’s potential predictability is assessed using reanalysis fields as weather forcings. 
	- The Global Fire Emissions Database (**GFED4**) provides **11 yr of observed** **burned areas** from satellite measurements and is used as a validation dataset. 
	- The **fire indices** implemented are **good predictors to highlight dangerous conditions**. High values are correlated with observed fire, and low values correspond to nonobserved events. A more quantitative skill evaluation was performed using the **extremal dependency index**, which is a skill score specifically designed for rare events. It revealed that the three indices were more skillful than the random forecast to detect large fires on a global scale. The performance peaks in the boreal forests, the Mediterranean region, the Amazon rain forests, and Southeast Asia. 
	- The skill scores were then aggregated at the country level to reveal which nations could potentially benefit from the system information to aid decision-making and fire control support. Overall it was found that fire danger modeling based on weather forecasts can provide **reasonable predictability over large parts of the global landmass**

---

- <mark>Carrasco-Escaff 2024</mark>:The key role of extreme weather and climate change in the occurrence of exceptional fire seasons in south-central Chile

	- Unprecedentedly large areas were burned during the 2016/17 and 2022/23 fire seasons in south-central Chile (34-39 ◦ S). These seasonal-aggregated values were **mostly accounted for human-caused wildfires** within a limited period in late January 2017 and early February 2023. 
	- We provide a comprehensive analysis of the meteorological conditions during these events, from **local to hemispheric scales**, and formally **assess the contribution of climate change** to their occurrence. To achieve this, we gathered monthly fire data from the Chilean Forestry Corporation and daily burned area estimates from satellite sources. In-situ and gridded data provided near-surface atmospheric insights, ERA5 reanalysis helped analyze broader wildfire features, high-resolution simulations were used to obtain details of the wind field, and large-ensemble simulations allowed the assessment of climate change’s impact on extreme temperatures during the fires. 
	- This study found extraordinary daily burned area values (>65,000 ha) occurring under extreme surface weather conditions (temperature, humidity, and winds), fostered by strong mid-level subsidence ahead of a ridge and downslope winds converging towards a coastal low. Daytime temperatures and the **water vapor deficit** reached the maximum values observed across the region, well above the previous historical records. We hypothesize that these conditions were crucial in **exacerbating the spread of fire**, along with longer-term atmospheric processes and other non-climatic factors such as fuel availability and increasing human-driven ignitions. 
	- Our findings further reveal that climate change has increased the probability and intensity of extremely warm temperatures in south-central Chile, underscoring anthropogenic forcing as a significant driver of the extreme fire activity in the region.

---





[1]: Not a Paper 
[2]: Prepared for Submission to JCAP (_Journal of Cosmology and Astroparticle Physics_)