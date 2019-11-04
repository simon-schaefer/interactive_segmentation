#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Michel Breyer, Simon Schaefer
# Description : Definition of task constants.    
# =============================================================================
from enum import Enum

class Status(Enum):
    RUNNING = 0
    SUCCESS = 1
    FAIL = 2
    TIME_LIMIT = 3
