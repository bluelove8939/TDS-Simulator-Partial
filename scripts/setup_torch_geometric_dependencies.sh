TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo installing torch_geometric compatible with $TORCH_VERSION
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/${TORCH_VERSION}.html