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
pip install DensityClust-1.0.8.tar.gz
```
or you can install LDC package directly in pypi.com. using:
```
pip install DensityClust
```

# Usage
```
import DensityClust.LocalDensityClustering_main as LDC
import astropy.io.fits as fits
from DensityClust import make_plot

para = {"gradmin": 0.01, "rhomin": 0.7, "deltamin": 4, "v_min": 27, "rms": 0.46, "dc": 0.6, "is_plot": 0}
LDC.localDenCluster(r'data\3d_Clumps\gaussian_out_000.fits', para=para)

# make picture
data = fits.getdata(r'data\3d_Clumps\gaussian_out_000.fits')
outcat = r'data\3d_Clumps\gaussian_out_000\LDC_outcat.txt'
make_plot.make_plot(outcat, data, lable_num=False)

```
<img src="https://github.com/Luoxiaoyu828/LDC-MGM/blob/main/data/2d_Clumps/gaussian2D_out_000/result.png" width="400px">


# Citation
If you use this code in a scientific publication, I would appreciate citation/reference to this repository. 

```
@ARTICLE{2021arXiv211011620L,
       author = {{Luo}, Xiaoyu and {Zheng}, Sheng and {Huang}, Yao and {Zeng}, Shuguang and {Zeng}, Xiangyun and {Jiang}, Zhibo and {Chen}, Zhiwei},
        title = "{Molecular Clump Extraction Algorithm Based on Local Density Clustering}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2021,
        month = oct,
          eid = {arXiv:2110.11620},
        pages = {arXiv:2110.11620},
archivePrefix = {arXiv},
       eprint = {2110.11620},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv211011620L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
