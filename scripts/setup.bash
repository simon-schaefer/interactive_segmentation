#!/bin/bash

export YUMI_PUSH_SCRIPTS="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export YUMI_PUSH_HOME="$YUMI_PUSH_SCRIPTS/.."
export YUMI_PUSH_CONFIG="$YUMI_PUSH_HOME/config"
export YUMI_PUSH_MODELS="$YUMI_PUSH_HOME/models"
export YUMI_PUSH_OUTS="$YUMI_PUSH_HOME/outs"
export YUMI_PUSH_POLICIES="$YUMI_PUSH_HOME/policies"

# Source environment (create env. variables).
source "$YUMI_PUSH_SCRIPTS/header.bash"
echo "Setting up yumi push ..."

# Login to virtual environment.
cd $YUMI_PUSH_HOME
if [ ! -d "yumi_venv" ]; then
    echo "Creating virtual environment ..."
    mkdir yumi_venv
    virtualenv -p python3 yumi_venv
fi
source $YUMI_PUSH_HOME/yumi_venv/bin/activate

# Install self-python-package.
echo $'\nInstalling package ...'
cd $YUMI_PUSH_HOME
#brew install mpich
#xcode-select --install
pip3 install -r requirements.txt
pip3 install tensorflow
CFLAGS="-DB3_NO_PYTHON_FRAMEWORK" pip3 install pybullet
pip3 install -e git+git://github.com/simon-schaefer/baselines.git#egg=baselines
pip3 install -e .

# Build outs directory.
echo $'\nBuilding output directory ...'
cd $YUMI_PUSH_HOME
if [ ! -d "outs" ]; then
    mkdir outs
fi

cd $YUMI_PUSH_HOME
echo $'\nSuccessfully set up project !'
