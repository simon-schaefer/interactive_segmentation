#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Michel Breyer, Simon Schaefer
# Description : Input/Output utility collection. 
# =============================================================================
import os
import yaml

def load_yaml(file_path):
    """Load a YAML file into a Python dict.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        A dict with the loaded configuration.
    """
    with open(file_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save_yaml(config, file_path):
    """Save a dict to a YAML file.

    Args:
        config (dict): The dict to be saved.
        file_path (str): The path to the YAML file.
    """
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=None)
