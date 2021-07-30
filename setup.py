import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="trustscore", # Replace with your own username
    version="0.0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google/TrustScore",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "sklearn",
    ],
    extras_require={
        "evaluation": ["matplotlib", "tensorflow"]
    },
    python_requires='>=3.6',
)
