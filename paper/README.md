# LDC&MGM 论文仿真数据集说明及算法参数设置

This repo provides a clean implementation of LDC algorithm using all the best practices.

## 仿真数据集
论文中所用到的仿真数据集由Starlink软件包中cupid.makeclumps仿真产生。Starlink 软件详情见：http://starlink.eao.hawaii.edu/starlink.

仿真数据集的参数如下：
- Peak = [2, 10]
- Size1 = [2, 2.5]
- Size2 = [3.5, 4]
- Size3 = [3, 5]  (FWHM = 2.3548*Size)
- Trunc = 0.0001
- rms = 1
- [m,n] 表示区间[m, n]的均匀分布
- 数据尺寸：120*120*120
- 单个数据块中仿真云核个数：10(低)、40(中)、100(高)
- 总的仿真云核个数：10000个

![demo](https://github.com/Luoxiaoyu828/LDC-MGM/blob/main/simulated_clump_3d_show.jpg)

## 合成数据集
合成数据集中的仿真云核生成的参数如下：
- Peak = [2， 5]
- Size1 = [0.5, 2]
- Size2 = [0.5, 2]
- Size3 = [2, 7]
- rms = 0
- [m,n] 表示区间[m, n]的均匀分布
- 数据尺寸：181*361*68(与M16天区13CO谱线数据同尺寸)
- 将制作成单个的仿真云核(不截断)，随机的选取15个仿真分子云核任意的放入，超出13CO数据块边界的部分去掉。
- 总的仿真云核个数：10000个


## 算法参数设置

### 仿真数据集的实验
####LDC&MGM参数设置
- ρ_0=3σ
- δ_0=4
- ∇_0=0.01
- n_0=128
- d_c=0.8

#### FellWalker算法参数设置
- FELLWALKER.MAXJUMP=3.3
- FELLWALKER.MINDIP=1*RMS
- FELLWALKER.MINHEIGHT=1*RMS
- FELLWALKER.MINPIX=128
- FELLWALKER.NOISE=RMS

####GaussClumps算法参数设置
- GAUSSCLUMPS.MINWF=0.8
- GAUSSCLUMPS.MODELLIM=0.5
- GAUSSCLUMPS.NPAD=10
- GAUSSCLUMPS.NPEAK=9
- GAUSSCLUMPS.NSIGMA=3
- GAUSSCLUMPS.NWF=10
- GAUSSCLUMPS.RMS=1
- GAUSSCLUMPS.S0=1
- GAUSSCLUMPS.SA=1
- GAUSSCLUMPS.SB=0.1
- GAUSSCLUMPS.SC=1
- GAUSSCLUMPS.THRESH=2

### 合成数据集的实验
####LDC&MGM参数设置
- ρ_0=1.5
- δ_0=4
- ∇_0=0.01
- n_0=27
- d_c=0.6

#### FellWalker算法参数设置
- FELLWALKER.MAXJUMP=3.3
- FELLWALKER.MINDIP=1*RMS
- FELLWALKER.MINHEIGHT=1*RMS
- FELLWALKER.MINPIX=32
- FELLWALKER.NOISE=RMS

####GaussClumps算法参数设置
- GAUSSCLUMPS.MINWF=0.8
- GAUSSCLUMPS.MODELLIM=0.5
- GAUSSCLUMPS.NPAD=10
- GAUSSCLUMPS.NPEAK=9
- GAUSSCLUMPS.NSIGMA=3
- GAUSSCLUMPS.NWF=10
- GAUSSCLUMPS.RMS=RMS
- GAUSSCLUMPS.S0=1
- GAUSSCLUMPS.SA=1
- GAUSSCLUMPS.SB=0.1
- GAUSSCLUMPS.SC=1
- GAUSSCLUMPS.THRESH=2
