from setuptools import setup


def readme_file():
      with open("README.md", encoding="utf-8") as wb:
            return wb.read()

setup(name='DensityClust',
      version='1.1.9',
      author='Luo Xiaoyu',
      description='the local density clustering algorithm',
      author_email='vastlxy@163.com',
      packages=['DensityClust', 't_match', 'tools', 'fit_clump_function']
      )