python setup.py build

python setup.py sdist

python setup.py bdist_wheel

twine upload dist/[file_name]

Enter your username: vastlxy_-.828

Enter your password: vastlxy@828


1.1.4--->1.1.5  
对结果的处理的改动:
利用skimage包实现对云核参数的计算
https://scikit-image.org/docs/stable/api/skimage.measure.html?highlight=measure#skimage.measure.regionprops

去掉了out的输出
中间结果可选择性的打印在控制台
对于一些cell头文件没有提供rms,采用设置rms=0.23

在自适应密度估计阶段 提前将密度估计好保存为fits文件
(1825+015_L.fits-->1825+015_L_esti.fits),放到和原始数据同级目录下

show_clumps.py 增加了一个参数loc
用于指定绘制"局部区域"的检测到的云核的图片

function: make_plot_wcs_1()
修改了以前核表中没有云核时，无法画出积分图的bug. 当没有检测到核的时候，只画积分图
在积分图上的标记点改大了一些


1.1.5--->1.1.6
修复bug:
clump_size[clump_ii, :] = 2.3548 * np.array([size_1[0], size_2[0], size_3[0])
  --->  clump_size[clump_ii, :] = 2.3548 * np.array([size_1[ind[0]], size_2[ind[0]], size_3[ind[0]]])
在选择流量最大连通域中，对应的size也应跟着一起选择 而不是默认选择第一个

R16和R2的结果，size: 第一个轴和第三个轴需要调换一下，基于wcs的核表需要转换成像素后再转换

轴的对应关系为：
-->  clump_size[clump_ii, :] = 2.3548 * np.array([size_3[ind[0]], size_2[ind[0]], size_1[ind[0]]])

Bug:同一个核被分到两个cell了：
/home/data/clumps_share/data_luoxy/detect_R16_LDC/R16_LDC_auto/R16_200_fig_loc/1865-010_L_fig_loc/MWISP186.681-01.253+06.392.png
/home/data/clumps_share/data_luoxy/detect_R16_LDC/R16_LDC_auto/R16_200_fig_loc/1865-015_L_fig_loc/MWISP186.681-01.253+06.392.png
 12.0	38.0	88.0	1244.0	39.304	**90.634**	1244.499	6.632	9.548	5.855	3.763	1404.071	1180.0  ——这种情况是bug
 8.0	38.0	28.0	1244.0	39.304	**30.634**	1244.499	6.632	9.548	5.855	3.763	1404.071	1180.0
 
 原因：cell的尺寸为121*121*2411，在银经银纬面上，原来取局部的方式为：
        cen1_min = 30
        cen1_max = size_v - 30
        aa = outcat.loc[outcat['Cen1'] > cen1_min]
        aa = aa.loc[outcat['Cen1'] <= cen1_max]
        即31到91     
 而前一个的91即为下一个的31，故出现一个核在相邻的cell中出现的情况
 现在改为：
        cen1_min = 30
        cen1_max = size_v - 30 - 1
        aa = outcat.loc[outcat['Cen1'] > cen1_min]
        aa = aa.loc[outcat['Cen1'] < = cen1_max]
        即从31到90 
 

1.1.9-->1.2.0:
基本完成多高斯拟合代码，优化了以前的代码

1.2.4
增加分块检测再拼接的模式，将para提到检测前方便用户自己设置

1.2.4-->1.2.5
采用拼接模式时，保存核表重名了，已修改。
![img.png](pic/img.png)
split_cube.py 增加分块检测结果的拼接程序
compare_version(sop, dop, msp)函数 用于匹配两个核表及对应的云核形态参数

1.2.5-->1.2.6
解决了拼接核表不能保存的问题，同时增加了检测核表再原始数据上的可视化结果

1.2.6--1.2.7
将MGM加入到LDC_MGM_main.py中，和原来的LocalDensityClustering_main.py调用方式保存一致

1.2.7-->1.2.8
修复MGM过程中，没有取局部核表。

1.2.8--1.2.9
优化了计算核表的代码，将原来一个一个的计算改为把mask计算好了，利用mask结合原始数据计算核表。

1.2.9-->1.3.0
MGM过程中,会有一些点的强度为nan，导致拟合失败，已解决：
ultil_lxy.py 167:points_all_df = points_all_df.dropna()   # 删除存在nan的行数据
MGM执行完毕后，在积分图上绘制拟合过后的质心
完善2维数据的处理流程.

1.3.0-->1.3.1
Data类中增加了：
    将原始数据块按照指定速度范围、经度范围、纬度范围进行切割保存为fits文件再检测
    针对数据块中存在的Nan值的情况，采用将其替换成-1000的处理方式，进行标记
    将云核的体积参数pix_min=27改为pix-min=[9,3]  分为空间平面的面积和速度轴上的跨度阈值

1.3.1--> 1.4.0
    考虑了望远镜的空间分辨率和速度分辨率参数，对算法中的距离参数做修正，以MWISP的望远镜参数[30, 30, 0.166]为基准, 同时体积参数也会随着分辨率的不同而改变

1.4.0--1.4.1
    mask中出现一些离群点，采用保留最大的连通域作为最终的mask. 
    Data类：
        数据块的经纬度和速度的范围改用spectral_cube中的属性得到，不依赖与数据的头文件.
        Nan值由原来的-1000改为由0替换，因为-1000会影响到最后的积分图成像.

1.4.1--1.4.2
    主要改变是和软件接口的衔接.
    multi_gauss_fitting_new.py 260行：
    cost = np.sqrt(((fit_value - Intensity)**2).sum()) / points_num   # 均方差平方和
    公式不是均方差平方和，改为：
    cost = np.sqrt(((fit_value - Intensity)**2).sum() / points_num)   # 均方差平方和

