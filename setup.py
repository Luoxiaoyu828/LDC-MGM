from setuptools import setup


def readme_file():
      with open("README.md", encoding="utf-8") as wb:
            return wb.read()

setup(name='DensityClust',
      version='1.2.4',
      author='Luo Xiaoyu',
      description='Molecular Clump extraction algorithm based on Local Density Clustering*',
      author_email='vastlxy@163.com',
      packages=['DensityClust', 't_match', 'tools', 'fit_clump_function', 'Generate'],
      url="https://github.com/Luoxiaoyu828/LDC_MGM",
      project_urls={
        "Bug Tracker": "https://github.com/Luoxiaoyu828/LDC_MGM/issues",
    })