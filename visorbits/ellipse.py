'''
    ellipse.py

    This file containts functions to recover the elliptical orbit from
    given velocity and distance to the host.
    -------------------------------------------------------------------------------------------------------
    ecc         :   eccentricity (unitless)
    a_ax        :   semi-major axis (unitless)
    angl_ecc    :   eccentric anomaly (unitless)
    angl_c      :   true anomaly (unitless; positional angle)
    rc          :   radius-vector from eccentric anomaly
    -------------------------------------------------------------------------------------------------------
    NOTE:       For MW halo mass 1.5*10^12 MSun. Units are included
    -------------------------------------------------------------------------------------------------------
'''

from astropy import units as u
import astropy as a0
import numpy as np

# Orbit in its own plane
M = 1.5*10**12*u.kg*2*10**30 # MW halo
G = a0.constants.G

def ecc(r0, v0, vtau0):
    '''
    ---------------------------------------------------------------------
    Arguments:
    --------------------------------------------------------------------
    r0       :   initial radius-vector (u.kpc)
    v0       :   initial scalar velocity (u.km/u.s)
    vtau0    :   perpendicular projection of initial velocity (u.km/u.s)
    ---------------------------------------------------------------------
    Output:
    ---------------------------------------------------------------------
                 (Unitless) eccentricity
    ---------------------------------------------------------------------
    NOTE     :
    ---------------------------------------------------------------------
    '''
    return np.sqrt(1.+4.*(vtau0.to(u.m/u.s))**2*r0.to(u.m)/2./G/M*((v0.to(u.m/u.s))**2*r0.to(u.m)/2./G/M-1)).value

def a_ax(r0, v0):
    '''
    ---------------------------------------------------------------------
    Arguments:
    --------------------------------------------------------------------
    r0       :   initial radius-vector (u.kpc)
    v0       :   initial scalar velocity (u.km/u.s)
    ---------------------------------------------------------------------
    Output:
    ---------------------------------------------------------------------
                 (Unitless) semi-major axis
    ---------------------------------------------------------------------
    NOTE     :
    ---------------------------------------------------------------------
    '''
    return (r0.to(u.m)/(1-(v0.to(u.m/u.s))**2*r0.to(u.m)/2./G/M)/2.).to(u.kpc).value

def angl_ecc(rc, e0, a0):
    '''
    ---------------------------------------------------------------------
    Arguments:
    --------------------------------------------------------------------
    rc       :   initial radius-vector
    e0       :   eccentricity
    a0       :   semi-major axis
    ---------------------------------------------------------------------
    Output:
    ---------------------------------------------------------------------
                 (Unitless by type; radians) eccentric anomaly
    ---------------------------------------------------------------------
    NOTE     :
    ---------------------------------------------------------------------
    '''
    psihalf = np.arctan(np.sqrt(2/(1./e0*(1-rc/a0)+1)-1.))
    if rc>a0:
        psihalf += np.pi
    return (psihalf*2)

def angl_c(psic, e0):
    '''
    ---------------------------------------------------------------------
    Arguments:
    --------------------------------------------------------------------
    psic     :   eccentric anomaly in radians, 0 to 2*pi.
    e0       :   eccentricity
    ---------------------------------------------------------------------
    Output:
    ---------------------------------------------------------------------
                 (Unitless by type; radians) true anomaly angle
    ---------------------------------------------------------------------
    NOTE     :    psic/2 is between 0, pi
    ---------------------------------------------------------------------
    '''
    theta_signed = np.arctan(np.sqrt((1+e0)/(1-e0))*np.tan(psic/2))
    if psic-np.pi > np.pi/2.:
        theta_signed +=np.pi
    theta = 2.*theta_signed
    return theta

def rc(psic, e0, a0):
    '''
    ---------------------------------------------------------------------
    Arguments:
    --------------------------------------------------------------------
    psic     :   eccentric anomaly in radians, 0 to 2*pi.
    e0       :   eccentricity
    a0       :   semi-major axis
    ---------------------------------------------------------------------
    Output:
    ---------------------------------------------------------------------
                 (Unitless by type) radius-vector
    ---------------------------------------------------------------------
    NOTE     :
    ---------------------------------------------------------------------
    '''
    return a0*(1-e0*np.cos(psic))
