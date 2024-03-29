from setuptools import setup


setup(name='DensityClust',
      version='1.4.6',
      author='Luo Xiaoyu',
      description='Molecular Clump extraction algorithm based on Local Density Clustering*',
      author_email='vastlxy@163.com',
      packages=['DensityClust', 't_match', 'tools', 'fit_clump_function', 'Generate', 'LDC_MGM_main'],
      url="https://github.com/Luoxiaoyu828/LDC_MGM",
      project_urls={
        "Bug Tracker": "https://github.com/Luoxiaoyu828/LDC_MGM/issues",
    })