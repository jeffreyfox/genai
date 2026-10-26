# Initialize a project using Python 3.12
uv init --python 3.12

# Install PyTorch and Jupyter Notebook
uv add torch torchvision notebook matplotlib

# Launch Jupyter Notebook
uv run jupyter notebook

# When installing new packages, you just need to run `uv add <package>`
