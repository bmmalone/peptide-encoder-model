from setuptools import find_packages, setup
from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop

import importlib
import logging
import shutil

def _safe_read_lines(f):        
    with open(f) as in_f:
        r = in_f.readlines()
    r = [l.strip() for l in r]
    return r

console_scripts = [
    'embed-peptides=pepenc_model.models.embed_peptides:main',
]

install_requires = [    
    "dm-tree",
    "gym",
    "joblib",
    "lifesci",
    "numpy",
    "opencv-python",
    "pandas",
    "peptide-encoder",
    "pyllars",
    "pyyaml",
    "ray[tune]",
    "torch",
    "tqdm",
]

tests_require = [
    'pytest',
    'coverage',
    'pytest-cov',
    'coveralls',
    'pytest-runner',
]

gpu_requires = []

docs_require = [
    'sphinx',
    'sphinx_rtd_theme'
]

all_requires = (
    tests_require +
    gpu_requires +
    docs_require
)

extras = {
    'test': tests_require,
    'gpu': gpu_requires,
    'docs': docs_require,
    'all': all_requires
}

classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Natural Language :: English',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
]

def _post_install(self):
    import site
    importlib.reload(site)


class my_install(_install):
    def run(self):
        level = logging.getLevelName("INFO")
        logging.basicConfig(level=level,
            format='%(levelname)-8s : %(message)s')

        _install.run(self)
        _post_install(self)

class my_develop(_develop):  
    def run(self):
        level = logging.getLevelName("INFO")
        logging.basicConfig(level=level,
            format='%(levelname)-8s : %(message)s')

        _develop.run(self)
        _post_install(self)

def readme():
    with open('README.md') as f:
        return f.read()

def description():
    description = ("A trained encoder for peptides (short amino acid sequences) based on BLOSUM similarity.")
    return description

setup(
    name='peptide_encoder_model',
    version='0.1.1',
    description=description(),
    long_description=readme(),
    keywords="peptide encoding blossum trained model",
    url="https://github.com/bmmalone/peptide-encoder-model",
    author="Brandon Malone",
    author_email="bmmalone@gmail.com",
    license='MIT',
    packages=find_packages(),
    install_requires=install_requires,
    cmdclass={'install': my_install,  # override install
                'develop': my_develop   # develop is used for pip install -e .
    },
    include_package_data=True,
    tests_require=tests_require,
    extras_require=extras,
    entry_points = {
        'console_scripts': console_scripts
    },
    zip_safe=False,
    classifiers=classifiers,    
)
