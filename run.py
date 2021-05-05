#!/usr/local/bin/python3

import math

from astropy import coordinates as coord, units as u
from astropy.time import Time
from astropy.coordinates import TEME

from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from poliastro.twobody import angles
from poliastro.examples import iss

from sgp4.api import Satrec, SGP4_ERRORS, WGS84

from rv2tle import rv2el

if __name__ == "__main__":

    # Display some initial data
    print(f" Orbit: {iss}")
    print(" State vector [poliastro]")
    print(f"     r = {iss.r}")
    print(f"     v = {iss.v}")
    print()

    # Reference epoch
    epoch = Time(iss.epoch, format='datetime', scale='utc')

    # Store poliastro orbital elements (osculating)
    ecc_anomaly = angles.nu_to_E(iss.nu, iss.ecc)
    mean_anomaly = angles.E_to_M(ecc_anomaly, iss.ecc)

    # Compute orbital elements required by spg4 (mean)
    inc, raan, ecc, argp, m_ano, m_mot = rv2el(iss.r, iss.v, iss.epoch)

    # Display differences
    print("                Poliastro(osc)       rv2tle.rv2el(mean)")
    print(f"Ecc            :    {iss.ecc:10.5f}{'':15}{ecc:10.5f}")
    print(f"Incl  [deg]    :    {math.degrees(iss.inc.value):10.5f}{'':15}{math.degrees(inc):10.5f}")
    print(f"n [deg/min]    :    {math.degrees(iss.n.to(u.rad/u.minute).value):10.5f}{'':15}{math.degrees(m_mot):10.5f}")
    print(f"RAAN  [deg]    :    {math.degrees(iss.raan.value):10.5f}{'':15}{math.degrees(raan):10.5f}")
    print(f"Argp + M [deg] :    {math.degrees(iss.argp.value+mean_anomaly.value):10.5f}{'':15}{math.degrees(argp+m_ano):10.5f}")
    print()

    # Obtain state vector from spg4 and mean elements
    sat = Satrec()
    sat.sgp4init(WGS84, 'i', 0, epoch.jd - 2433281.5, 0.0, 0.0, 0.0, ecc, argp, inc, m_ano, m_mot, raan)
    errorCode, rTEME, vTEME = sat.sgp4(epoch.jd1, epoch.jd2)
    if errorCode != 0:
        raise RuntimeError(SGP4_ERRORS[errorCode])

    # Convert state vector from TEME (True Equator Mean Equinox) to ITRS
    pTEME = coord.CartesianRepresentation(rTEME*u.km)
    vTEME = coord.CartesianDifferential(vTEME*u.km/u.s)
    svTEME = TEME(pTEME.with_differentials(vTEME), obstime=iss.epoch)
    svITRS = svTEME.transform_to(coord.ITRS(obstime=iss.epoch))
    sv = Orbit.from_coords(Earth, svITRS)

    # Display results
    print("State vector [rv2el/spg4]")
    print(f"     r = {sv.r}")
    print(f"     v = {sv.v}")
    print()

    print("State vector differences [poliastro - rv2el/spg4]")
    print(f"    dr = {iss.r - sv.r}")
    print(f"    dv = {iss.v - sv.v}")
    print()
