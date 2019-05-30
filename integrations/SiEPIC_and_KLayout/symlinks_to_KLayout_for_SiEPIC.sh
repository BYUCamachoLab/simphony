#!/bin/bash

# Unix/Linux GitHub repository installation of Simphony files for KLayout and SiEPIC.

# assumes that 
# - Simphony repositories are in ~/Documents/GitHub
# - KLAYOUT_HOME is ~/.klayout

# to run:
# source symlinks_to_KLayout_for_SiEPIC.sh

export SRC=$HOME/Documents/GitHub
export DEST=$HOME/.klayout

mkdir $DEST/pymacros/Simphony
ln -s $SRC/simphony/integrations/SiEPIC_and_KLayout/klayout_dot_config/pymacros/* $DEST/pymacros/Simphony/
ln -s $SRC/simphony/simphony $DEST/python/