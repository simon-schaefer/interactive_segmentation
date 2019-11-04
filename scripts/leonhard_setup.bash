#!/bin/bash

export YUMI_PUSH_SCRIPTS="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export YUMI_PUSH_HOME="$YUMI_PUSH_SCRIPTS/.."
export YUMI_PUSH_CONFIG="$YUMI_PUSH_HOME/config"
export YUMI_PUSH_MODELS="$YUMI_PUSH_HOME/models"
export YUMI_PUSH_OUTS="$YUMI_PUSH_HOME/outs"
export YUMI_PUSH_POLICIES="$YUMI_PUSH_HOME/policies"

#load python module
module load python_gpu/3.6.4
#load tensorflow module
module load cudnn/7.2
# Source environment (create env. variables).
source "$YUMI_PUSH_SCRIPTS/header.bash"
echo "Setting up yumi push ..."

# Install self-python-package.
echo $'\nInstalling package ...'
cd $YUMI_PUSH_HOME
pip3 install --user -r requirements.txt
CFLAGS="-DB3_NO_PYTHON_FRAMEWORK" pip3 install --user pybullet
pip3 install --user -e git+git://github.com/simon-schaefer/baselines.git#egg=baselines
pip3 install --user -e .
# Build outs directory.
echo $'\nBuilding output directory ...'
cd $YUMI_PUSH_HOME
if [ ! -d "outs" ]; then
    mkdir outs
fi

cd $YUMI_PUSH_HOME
echo $'\nSuccessfully set up project !'
