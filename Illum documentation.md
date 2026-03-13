

### 1. Core Data Structures

#### `MultiScaleData.py`

This is the heart of Illumina’s data handling. It manages the multi-resolution nested grid structure.

- **Purpose**: Handles HDF5 files containing multiple "layers" of spatial data, where each layer represents a different spatial scale/resolution.
    
- **Key Class**: `MultiScaleData`
    
    - `_get_col_row()`: Converts geographic coordinates to pixel indices for a specific layer.      
    - `extract_observer()`: Isolates data surrounding a specific observer location.        
    - `set_circle()`: Utility to apply values (like light zones) within a radius.
        
- **Visualization**: Includes `plot()` (2D map) and `scatter()` (intensity vs. distance) methods.
    

---

### 2. Preprocessing & Input Generation

#### `domain.py`

Defines the spatial "playing field" for the simulation.

- **Input**: `domain_params.in` (YAML).
    
- **Functionality**:
    
    - Calculates the bounding box for the simulation.
        
    - Determines the appropriate **UTM projection** (SRS) based on the observer's location.
        
    - Outputs `domain.ini`, which acts as the spatial metadata for all subsequent steps.
        

#### `inputs.py`

The main orchestrator for preparing a simulation run.

- **Tasks**:
    
    - Validates light inventories against the domain.
        
    - Computes road orientations (if enabled).
        
    - Handles spectral binning and reflectance interpolation.
        
    - Interpolates obstacle properties (height, distance, etc.) using `scipy.interpolate.griddata`.
        

#### `inventory.py`

Processes light source data into the multi-scale format.

- **Two Modes**:
    
    1. **Lamps**: Discrete point sources.
        
    2. **Zones**: Area-based sources (often derived from VIIRS satellite data).
        
- **Physics**: It calculates "generalized lamps" by combining Light Output Patterns (LOP) and Spectral Power Distributions (SPD).
    

---

### 3. Light & Physics Properties

#### `AngularPowerDistribution.py` (LOP)

Manages the spatial distribution of light from a fixture.

- **Formats**: Supports `.ies` (standard lighting format) and `.lop` (internal text format).
    
- **Capabilities**: Normalization of lumens, 1D/2D/3D plotting, and interpolation of intensity across a sphere.
    

#### `SpectralPowerDistribution.py` (SPD)

Manages the "color" or wavelength-dependent energy of light.

- **Capabilities**:
    
    - Interpolation to match simulation wavelength bins.
        
    - Normalization relative to photopic vision or total power.
        
    - Supports `.spct`, `.spdx` (XML), and ASTER reflectance files.
        

---

### 4. Utilities & Analysis

#### `extract.py`

The post-simulation tool.

- Walks through execution directories to find `.out` and `pcl.bin` files.
    
- Aggregates total skyglow and can reconstruct **contribution maps** (showing which geographic areas contribute most to skyglow at a point).
    

#### `integrate.py`

- Integrates the light intensity from a binary output file over a specific geographic area defined in a **KML file**. Useful for calculating total light flux from a specific neighborhood or park.
    

#### `convert.py`

- **Export Tool**: Converts the internal HDF5 multi-scale format into standard GIS formats:
    
    - **Raster**: GeoTiff (using GDAL).
        
    - **Vector**: GeoJSON (using GeoPandas).

### 5. Initialization & Project Structure

#### `init.py`

The starting point for any new experiment.

- **Purpose**: Prepares a fresh execution folder by copying example configuration files and a standard `Lights` library from the Illumina installation directory.
    
- **Workflow**: Creates the directory structure required for `domain`, `warp`, and `inputs` to function.
    

#### `main.py`

The command-line interface (CLI) entry point.

- **Purpose**: Uses the `click` library to organize all modules into a single `illum` command-line tool.
    
- **Command Sequence**: Defines the standard logic flow: `init` $\rightarrow$ `domain` $\rightarrow$ `warp` $\rightarrow$ `inputs` $\rightarrow$ `batches` $\rightarrow$ `extract`.
    

---

### 6. Geospatial Processing (Satellite Data)

#### `warp.py`

Handles the ingestion and projection of satellite imagery to match the simulation domain.

- **Core Dependencies**: Relies heavily on **GDAL** (`gdalwarp`, `gdal_rasterize`) to transform spatial data.
    
- **Processed Datasets**:
    
    - **SRTM**: Elevation data (Digital Elevation Model).
        
    - **GHSL**: Global Human Settlement Layer (used for obstacle/building masks).
        
    - **VIIRS-DNB**: Satellite night-light data, used to define lighting "zones".
        
- **Water Masking**: Uses `hydropolys.zip` to identify and mask water bodies, ensuring light sources aren't incorrectly placed in the ocean or lakes.
    

#### `street_orientation.py`

Determines the direction of streets relative to observers.

- **Purpose**: Uses **OpenStreetMap (OSMNX)** to fetch road networks for the simulation area.
    
- **Functionality**: Calculates the bearing of the nearest road to an observer, allowing the model to simulate how street-side buildings or trees shield light.
    

---

### 7. Simulation Management & Execution

#### `batches.py`

Prepares the simulation to run on high-performance computing (HPC) clusters.

- **Grid Splitting**: It takes the large domain and splits it into individual "observer" directories, converting data into binary `.bin` files for the Fortran-based `illumina` engine.
    
- **HPC Integration**: Generates **Slurm** batch scripts (`sbatch`) for parallel execution.
    
- **Parametric Sweeps**: It can automatically generate batches for multiple wavelengths, viewing angles, or atmospheric conditions (Aerosol Optical Depth, etc.).
    

#### `failed.py`

A diagnostic tool for cluster management.

- **Purpose**: Scans the `exec/` subdirectories to identify simulations that crashed or were interrupted.
    
- **Functionality**: Checks for the existence of `.out` files and validates that the last line of the output contains the expected results. It can generate a script to automatically rerun only the failed jobs.
    

#### `alternate.py`

Allows for "what-if" scenario testing.

- **Purpose**: Generates an alternate scenario (e.g., "What if we changed all high-pressure sodium lamps to LED?") while keeping the total lumen output constant.
    
- **Workflow**: It re-calculates the spectral and angular distributions but scales them based on the original scenario's intensities.
    

---

### 8. Atmospheric Modeling

#### `OPAC.py`

Handles aerosol physics via the Optical Properties of Aerosols and Clouds (OPAC) database.

- **Aerosol Types**: Includes presets for soot, sea salt, minerals, and sulfate droplets.
    
- **Functionality**: Calculates the **Single Scattering Albedo** and the **Phase Function** (the probability of light scattering at specific angles) based on relative humidity and aerosol mixture.