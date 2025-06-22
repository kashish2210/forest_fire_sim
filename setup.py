from setuptools import setup, find_packages

setup(
    name="ferse_fire",
    version="0.1.0",
    author="Your Name",
    description="A library for simulating and predicting forest fire spread using AI/ML",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "rasterio",
        "xarray",
        "matplotlib",
        "scikit-learn",
        "pandas"
    ],
    python_requires=">=3.8",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
