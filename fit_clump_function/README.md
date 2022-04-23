### Multi-Gaussian Model for fitting the clump detected by LDC
#### Usage
```
from fit_clump_function import main_fit_gauss_3d as mgm
from tools.ultil_lxy import create_folder
import os


outcat_name_loc = r'/home/data/clumps_share/data_luoxy/detect_R16_LDC/R16_LDC_auto/R16_200_detect/1805+000_L/LDC_auto_loc_outcat.csv'
origin_name = r'/home/data/clumps_share/MWISP/R16_200/1805+000/1805+000_L.fits'
mask_name = r'/home/data/clumps_share/data_luoxy/detect_R16_LDC/R16_LDC_auto/R16_200_detect/1805+000_L/LDC_auto_mask.fits'

R2_200_fitting_path = create_folder(r'/home/data/clumps_share/data_luoxy/detect_R16_LDC/R16_LDC_auto/R16_200_MGM')
item = '1805_000_L'
save_path = create_folder(os.path.join(R2_200_fitting_path, item))

mgm.LDC_para_fit_Main(outcat_name_loc, origin_name, mask_name,save_path)
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
    │   ├── fitting_outcat.csv
    │   └── png
    │       └── touch_clumps_000.png
    ├── MWISP_outcat.csv
    └── points
        └── clump_id_xyz_intensity_0004.csv

```
