# LDC-MGM
Molecular Clump extraction algorithm based on Local Density Clustering

*Note* The core idea of the algorithm comes from [this paper](https://ui.adsabs.harvard.edu/abs/2014Sci...344.1492R/abstract)
```
Rodriguez A, Laio A. Clustering by fast search and find of density peaks[J]. Science, 2014, 344(6191):1492.
```

## Dependencies
The code is completed with Python  3. The following dependencies are needed to run the code:

* numpy~=1.19.2
* pandas~=1.3.2
* tabulate~=0.8.9
* matplotlib~=3.3.4
* scikit-image~=0.18.1
* scipy~=1.6.2
* astropy~=4.2



# Install
I suggest you install the code using pip from an Anaconda Python 3 environment. From that environment:
```
git clone https://github.com/Luoxiaoyu828/LDC-MGM.git
cd LDC-MGM/dist
pip install DensityClust-1.4.4.tar.gz
```
or you can install LDC package directly in pypi.com. using:
```
pip install DensityClust
```

# Usage

## import package
```
import astropy.io.fits as fits
from tools.make_plot import make_plot
import LDC_MGM.LDC_MGM_main as ldc_mgm
import LDC_MGM.LDC_main as ldc

```
## setting params & filename
```
data_name = r'*******.fits'
para = Param(delta_min=4, gradmin=0.01, v_min=[25, 5], noise_times=5, rms_times=2, rms_key='RMS')
para.rm_touch_edge = False
save_folder = r'########'
```
## LDC
```
ldc.LDC_main(data_name, para, save_folder)
```
## LDC MGM
```
save_mgm_png = False
ldc_mgm.LDC_MGM_main(data_name, para, save_folder, split=False, save_mgm_png=save_mgm_png)
```

        
## make picture
```
data = fits.getdata(r'data\3d_Clumps\gaussian_out_000.fits')
outcat = r'***.csv'
make_plot.make_plot(outcat, data, lable_num=False)
```



<img src="https://github.com/Luoxiaoyu828/LDC-MGM/blob/main/data/2d_Clumps/gaussian2D_out_000/result.png" width="400px">


# Citation
If you use this code in a scientific publication, I would appreciate citation/reference to this repository. 

```
@ARTICLE{2022RAA....22a5003L,
       author = {{Luo}, Xiaoyu and {Zheng}, Sheng and {Huang}, Yao and {Zeng}, Shuguang and {Zeng}, Xiangyun and {Jiang}, Zhibo and {Chen}, Zhiwei},
        title = "{Molecular Clump Extraction Algorithm Based on Local Density Clustering}",
      journal = {Research in Astronomy and Astrophysics},
     keywords = {molecular data, molecular processes, methods: laboratory: molecular, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2022,
        month = jan,
       volume = {22},
       number = {1},
          eid = {015003},
        pages = {015003},
          doi = {10.1088/1674-4527/ac321d},
archivePrefix = {arXiv},
       eprint = {2110.11620},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022RAA....22a5003L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
