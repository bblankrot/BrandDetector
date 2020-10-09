from setuptools import setup

setup(
    name="brand_detector",
    version="0.1.0",
    description="Identifying brand and industry information in audio advertisements",
    url="https://github.com/bblankrot/BrandDetector",
    author="Boaz Blankrot",
    license="MIT",
    install_requires=[
        "cupy-cuda==7.8.0",
        "faker",
        "gibberish",
        "numpy",
        "pandas",
        "spacy==2.3.2",
        "spacy-lookups-data==0.3.2",
    ],
    zip_safe=False,
)
