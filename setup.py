from setuptools import setup, find_packages

exec(open('pyramid3dunet/__version__.py').read())
setup(
    name="pyramid3dunet",
    packages=find_packages(exclude=["tests"]),
    version=__version__,
    author="Adrian Wolny, Lorenzo Cerrone",
    license="MIT",
    python_requires='>=3.7', 
    entry_points={'console_scripts': [
        'trainpyramidunet=pyramid3dunet.train:main',
        'predictpyramidunet=pyramid3dunet.predict:main',
        'evalpyramidunet=pyramid3dunet.evaluate:main']
        }
)
