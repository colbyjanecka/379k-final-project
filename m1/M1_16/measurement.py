#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 22:37:03 2021

@author: dawei
"""

import psutil
import telnetlib as tel
import sysfs_paths as sysfs
import time


def getTelnetPower(SP2_tel, last_power):
    """
    read power values using telnet.
    """
	# Get the latest data available from the telnet connection without blocking
    tel_dat = str(SP2_tel.read_very_eager()) 
    #print('telnet reading:', tel_dat)
    # find latest power measurement in the data
    findex = tel_dat.rfind('\n')
    findex2 = tel_dat[:findex].rfind('\n')
    findex2 = findex2 if findex2 != -1 else 0
    ln = tel_dat[findex2:findex].strip().split(',')
    if len(ln) < 2:
        total_power = last_power
    else:
        total_power = float(ln[-2])
    return total_power

def getTemps():
    """
    obtain the temp values from sysfs_paths.py
    """
    templ = []
    # get temp from temp zones 0-3 (the big cores)
    for i in range(4):
        temp = float(file(sysfs.fn_thermal_sensor.format(i),'r').readline().strip())/1000
        templ.append(temp)
	# Note: on the 5422, cpu temperatures 5 and 7 (big cores 1 and 3, counting from 0) appear to be swapped. Therefore, swap them back.
    t1 = templ[1]
    templ[1] = templ[3]
    templ[3] = t1
    return templ
