echo "----------------------------------------"
echo "* STEP1: Setup Environments"
echo "----------------------------------------"

ENV_NAME=.tds-sim
VENV_PYTHON_VERSION="3.11"
VENV_ROOT=$(realpath ./${ENV_NAME})
VENV_SITE_PACKAGES=${VENV_ROOT}/lib/python${VENV_PYTHON_VERSION}/site-packages/
VENV_BIN=${VENV_ROOT}/bin/

echo "python version: ${VENV_PYTHON_VERSION}"
echo "venv path: ${VENV_ROOT}"


echo "----------------------------------------"
echo "* STEP2: Install Dependencies"
echo "----------------------------------------"

sudo apt update
sudo apt install python${VENV_PYTHON_VERSION} -y
sudo apt install python${VENV_PYTHON_VERSION}-dev -y
sudo apt install python${VENV_PYTHON_VERSION}-venv -y
sudo apt install build-essential make cmake sudo -y
sudo apt install git autoconf flex bison -y


echo "----------------------------------------"
echo "* STEP3: Install Python Dependencies"
echo "----------------------------------------"

if [ ! -d $VENV_ROOT ]; then
    python${VENV_PYTHON_VERSION} -m venv ${ENV_NAME}
else
    echo "python venv already exists"
fi
source ./.tds-sim/bin/activate

pip3 install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install tqdm matplotlib
pip3 install torch_geometric

# TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
# echo installing torch_geometric compatible with $TORCH_VERSION
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/${TORCH_VERSION}+cpu.html

# # install ramulator wrapper python API
# cd ./ramulator2/wrapper
# make install-wrapper
# cd ../..

deactivate


echo "----------------------------------------"
echo "* STEP4: Generate env.sh"
echo "----------------------------------------"

cp ./scripts/env.sh ./env.sh

echo "" >> ./env.sh

echo "env.sh is generated"