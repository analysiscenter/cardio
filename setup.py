from setuptools import setup
#from setuptools import find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='dac_ecg',
      version='0.1',
      description='DAC ECG library',
      url='https://github.com/analysiscenter/ecg',
      author='Dmitry Podvyaznikov, Egor Illarionov, Alexander Kuvaev',
      author_email='d.podvyaznikov@analysiscenter.ru, e.illarionov@analysiscenter.ru, a.kuvaev@analysiscenter.ru',
      license='proprietary',
      packages=["dac_ecg"],
      zip_safe=False,
      install_requires=['pandas',
                        'wfdb',
                        'numpy'],
      package_data={'': ['README.md']},
      include_package_data=True)