### Multi-Gaussian Model for fitting the clump detected by LDC
#### Usage
```
from main_fit_gauss_3d import MGM_main


outcat_name_loc = r'F:\Parameter_reduction\LDC\0170+010_L/MGM_problem_cell/0155+030_L/LDC_auto_loc_outcat.csv'
origin_name = r'F:\Parameter_reduction\LDC\0170+010_L/MGM_problem_cell/0155+030_L\0155+030_L.fits'
mask_name = r'F:\Parameter_reduction\LDC\0170+010_L/MGM_problem_cell/0155+030_L\LDC_auto_mask.fits'
save_path = 'fitting_result7'

MGM_main(outcat_name_loc, origin_name, mask_name, save_path)
```
#### Result
You can find the structure of folder if the code runs runs successfully.
where, LDC_MGM_log.txt records the log of the MGM, which contains the fitted clump outcat, running time and some other information.

MWISP_outcat.csv is the fitted clump outcat, which contains some column names as follows:

ID	Galactic_Longitude	Galactic_Latitude	Velocity	Size_major	Size_minor	Size_velocity	Theta	Peak	Flux	Success	Cost
```
└── 1805_000_L
    ├── LDC_MGM_log.txt
    ├── LDC_MGM_outcat
    │   ├── csv
    │   │   └── fit_item000.csv
    │   ├── png
    │       └── touch_clumps_000.png
    ├── MWISP_outcat.csv
    ├── Fitting_outcat.csv
    └── points
        └── clump_id_xyz_intensity_0004.csv

```
