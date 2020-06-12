from setuptools import setup, find_packages

setup(
    name="train-plot-utils",
    version=1.0,
    author="Cuong V. Nguyen",
    author_email="nguyencuongcl1215@gmail.com",
    description="Some helper scripts to visualize neural networks' training process",
    url="https://github.com/cuongvng/neural-networks-with-PyTorch",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6"
)