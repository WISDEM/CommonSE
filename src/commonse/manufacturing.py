import numpy as np

from commonse import eps

def steel_cutting_plasma_time(length, thickness):
    # Length input as meters, thickness in mm
    time = length / (-0.180150943 + 41.03815215/(1e3*thickness+eps)) # minutes
    return np.sum( time)

def steel_rolling_time(theta, radius, thickness):
    # Length input as meters, converted to mm
    time = theta * np.exp(6.8582513 - 4.527217/np.sqrt(1e3*thickness+eps) + 0.009541996*np.sqrt(2*1e3*radius)) 
    return np.sum( time )

def steel_welding_time(theta, npieces, mtotal, length, thickness, coeff):
    # Length input as meters, thickness as mm
    time  = np.sum( theta * np.sqrt(npieces * mtotal) )
    time += np.sum( 1.3e-3 * coeff * (length) * (1e3*thickness)**1.9358 )
    return time

def steel_butt_welding_time(theta, npieces, mtotal, length, thickness):
    return steel_welding_time(theta, npieces, mtotal, length, thickness, 0.152)

def steel_filett_welding_time(theta, npieces, mtotal, length, thickness):
    return steel_welding_time(theta, npieces, mtotal, length, thickness, 0.3394)

