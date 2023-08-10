"""
Python implementation of RV2TLE program for computation of mean orbital elements
from state vector. Original C++ implementation available in http://sat.belastro.net/satelliteorbitdetermination.com/

Variables names have been similar to the original implementation, for easier comparison.
"""

from astropy import coordinates as coord, units as u
from astropy.coordinates import TEME
from astropy.time import Time
from poliastro.twobody import Orbit
from poliastro.bodies import Earth

import numpy as np
import math

from sgp4.api import Satrec, SGP4_ERRORS, WGS84


def norm(v):
    return np.sqrt(np.dot(v, v))


def unitv(v):
    return v/norm(v)


def acose(x):
    result = 0
    if x > 1:
        result = 0
    elif x < -1:
        result = math.pi
    else:
        result = math.acos(x)
    return result


def fmod2p(x):
    rval = math.fmod(x, 2*math.pi)
    if rval < 0:
        rval += 2*math.pi
    return rval


def rvel(r, v):
    xj3 = -2.53881e-6
    xke = 0.0743669161331734132  # = (G*M)^(1/2)*(er/min)^(3/2) where G
    ck2 = 5.413079e-4
    a3ovk2 = -1 * xj3 / ck2

    rr2 = r.value / 6378.135
    vv2 = v.value * 60 / 6378.135

    vk = 1.0 / xke * vv2  # smult
    h = np.cross(rr2, vk)  # cross
    pl = np.dot(h, h)
    vz = [0, 0, 1.0]
    n = np.cross(vz, h)
    n = unitv(n)  # unitv
    rk = norm(rr2)  # norm
    rdotk = np.dot(rr2, vv2) / rk
    rfdotk = norm(h) * xke / rk
    temp = np.dot(rr2, n) / rk
    uk = acose(temp)
    if rr2[2] < 0.0:
        uk = 2*math.pi - uk
    vz = np.cross(vk, h)
    vy = -1 / rk * rr2
    vec = vz + vy
    ek = norm(vec)
    if (ek > 1.0):
        print(ek)
        return  # open orbit
    xnodek = math.atan2(n[1], n[0])
    if (xnodek < 0.0):
        xnodek += 2*math.pi
    temp = np.sqrt(h[0]*h[0]+h[1]*h[1])
    xinck = math.atan2(temp, h[2])
    temp = np.dot(vec, n) / ek
    wk = acose(temp)
    if (vec[2] < 0):
        wk = fmod2p(2*math.pi - wk)
    aodp = pl / (1.0 - ek*ek)
    xn = xke * aodp ** (-1.5)

    # In the first loop the osculating elements rk, uk, xnodek, xinck, rdotk,
    # and rfdotk are used as anchors to find the corresponding final SGP4
    # mean elements r, u, xnodeo, xincl, rdot, and rfdot.  Several other final
    # mean values based on these are also found: betal, cosio, sinio, theta2,
    # cos2u, sin2u, x3thm1, x7thm1, x1mth2.  In addition, the osculating values
    # initially held by aodp, pl, and xn are replaced by intermediate
    # (not osculating and not mean) values used by SGP4.  The loop converges
    # on the value of pl in about four iterations.

    # seed value for first loop
    xincl = xinck
    u = uk

    for iter in range(0, 99):
        a2 = pl
        betal = math.sqrt(pl / aodp)
        temp1 = ck2 / pl
        temp2 = temp1 / pl
        cosio = math.cos(xincl)
        sinio = math.sin(xincl)
        sin2u = math.sin(2*u)
        cos2u = math.cos(2*u)
        theta2 = cosio * cosio
        x3thm1 = 3 * theta2 - 1
        x1mth2 = 1 - theta2
        x7thm1 = 7 * theta2 - 1
        r = (rk - 0.5 * temp1 * x1mth2 * cos2u) \
            / (1 - 1.5 * temp2 * betal * x3thm1)
        u = uk + .25 * temp2 * x7thm1 * sin2u
        xnodeo = xnodek - 1.5 * temp2 * cosio * sin2u
        xincl = xinck - 1.5 * temp2 * cosio * sinio * cos2u
        rdot = rdotk + xn * temp1 * x1mth2 * sin2u
        rfdot = rfdotk - xn * temp1 * (x1mth2 * cos2u + 1.5 * x3thm1)
        temp = r * rfdot / xke
        pl = temp * temp

        # vis-viva equation
        temp = 2 / r - (rdot*rdot + rfdot*rfdot) / (xke * xke)
        aodp = 1 / temp

        xn = xke * aodp ** (-1.5)
        if math.fabs(a2-pl) < 1e-13:
            break

    # The next values are calculated from constants and a combination of mean
    # and intermediate quantities from the first loop.  These values all remain
    # fixed and are used in the second loop.

    # preliminary values for the second loops
    ecose = 1 - r / aodp
    esine = r * rdot / (xke * np.sqrt(aodp))  # needed for Kepler's eqn
    elsq = 1 - pl / aodp  # intermediate eccentricity squared
    xlcof = .125 * a3ovk2 * sinio * (3 + 5 * cosio) / (1 + cosio)
    aycof = 0.25 * a3ovk2 * sinio
    temp1 = esine / (1 + np.sqrt(1-elsq))
    cosu = math.cos(u)
    sinu = math.sin(u)

    # The second loop normally converges in about six iterations to the final
    # mean value for the eccentricity, eo.  The mean perigee, omegao, is also
    # determined.  Cosepw and sinepw are found to high accuracy and
    # are used to calculate an intermediate value for the eccentric anomaly,
    # temp2.  Temp2 is then used in Kepler's equation to find an intermediate
    # value for the true longitude, capu.

    # seed values for loop
    eo = np.sqrt(elsq)
    omegao = wk
    axn = eo * math.cos(omegao)

    for iter in range(0,99):
        a2 = eo
        beta = 1 - eo*eo
        temp = 1 / (aodp * beta)
        aynl = temp * aycof
        ayn = eo * math.sin(omegao) + aynl
        cosepw = r * cosu / aodp + axn - ayn * temp1
        sinepw = r * sinu / aodp + ayn + axn * temp1
        axn = cosepw * ecose + sinepw * esine
        ayn = sinepw * ecose - cosepw * esine
        omegao = fmod2p(math.atan2(ayn - aynl, axn))
        # use weighted average to tame instability at high eccentricities
        eo = 0.9 * eo + 0.1 * (axn / math.cos(omegao))
        if eo > 0.999:
            eo = 0.999
        if math.fabs(a2-eo) < 1e-13:
            break

    temp2 = math.atan2(sinepw, cosepw)
    capu = temp2 - esine    # Kepler's equation
    xll = temp * xlcof * axn

    # xll adjusts the intermediate true longitude
    # capu, to the mean true longitude, xl
    xl = capu - xll

    xmo = fmod2p(xl-omegao)  # mean anomaly

    # The third loop usually converges after three iterations to the
    # mean semi-major axis, a1, which is then used to find the mean motion, xno.
    a0 = aodp
    a1 = a0
    beta2 = np.sqrt(beta)
    temp = 1.5 * ck2 * x3thm1 / (beta * beta2)
    for iter in range(0,99):
        a2 = a1
        d0 = temp / (a0*a0)
        a0 = aodp * (1.0 - d0)
        d1 = temp / (a1*a1)
        a1 = a0 / (1 - d1 / 3 - d1*d1 - 134 * d1*d1*d1 / 81)
        if math.fabs(a2-a1) < 1e-13:
            break

    xno = xke * a1 ** (-1.5)

    return xincl, xnodeo, eo, omegao, xmo, xno


def el2rv(inc, raan, ecc, argp, mean_anomaly, mean_motion, epoch):

    time_tle = epoch.jd - 2433281.5
    sat = Satrec()
    sat.sgp4init(WGS84, 'i', 0, time_tle, 0.0, 0.0, 0.0, ecc, argp, inc, mean_anomaly, mean_motion, raan)

    errorCode, rTEME, vTEME = sat.sgp4(epoch.jd1, epoch.jd2)
    if errorCode != 0:
        raise RuntimeError(SGP4_ERRORS[errorCode])

    pTEME = coord.CartesianRepresentation(rTEME*u.km)
    vTEME = coord.CartesianDifferential(vTEME*u.km/u.s)
    svTEME = TEME(pTEME.with_differentials(vTEME), obstime=epoch)

    svITRS = svTEME.transform_to(coord.ITRS(obstime=epoch))

    return Orbit.from_coords(Earth, svITRS)


def rv2el(rr, vv, epoch):
    epoch_time = Time(epoch, format='datetime', scale='utc')

    # SPG4 k-elements from state vector
    inck, raank, ecck, argpk, mAnomalyk, mMotionk = rvel(rr, vv)

    # SPG4 propagation of k-elements to rr', vv'
    sv = el2rv(inck, raank, ecck, argpk, mAnomalyk, mMotionk, epoch_time)
    inc2, raan2, ecc2, argp2, mAnomaly2, mMotion2 = rvel(sv.r, sv.v)  # SPG4 x-elements from state vectors

    # First correction
    incz = 2*inck - inc2
    raanz = 2*raank - raan2
    eccz = 2*ecck - ecc2
    argpz = 2*argpk - argp2
    mAnomalyz = 2*mAnomalyk - mAnomaly2
    mMotionz = 2*mMotionk - mMotion2

    # second correction is a small adjustment to z-elements
    sv = el2rv(incz, raanz, eccz, argpz, mAnomalyz, mMotionz, epoch_time)
    inc3, raan3, ecc3, argp3, mAnomaly3, mMotion3 = rvel(sv.r, sv.v)

    inc = incz + inck - inc3
    raan = raanz + raank - raan3
    ecc = eccz + ecck - ecc3
    argp = argpz + argpk - argp3
    mAnomaly = mAnomalyz + mAnomalyk - mAnomaly3
    mMotion = mMotionz + mMotionk - mMotion3

    return inc, raan, ecc, argp, mAnomaly, mMotion
