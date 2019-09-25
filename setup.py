from setuptools import setup, find_packages

__version__ = "1.0.a"

setup(name='alig',
      description='Implementation of ALI-G',
      author='Anonymized',
      packages=find_packages(),
      license="GNU General Public License",
      url='Anonymized',
      version=str(__version__),
      install_requires=["numpy",
                        "tensorflow",
                        "torch>=1.0"],)
