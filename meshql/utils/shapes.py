from typing import Sequence
from meshql.utils.types import Number
import numpy as np
import numpy.typing as npt

def get_sampling(start: Number, end: Number, num_samples: int, is_cosine_sampling: bool):
    if is_cosine_sampling:
        beta = np.linspace(0.0,np.pi, num_samples, endpoint=True)
        return 0.5*(1.0-np.cos(beta))*(end-start) + start
    else:
        return np.linspace(start, end, num_samples, endpoint=True)


def generate_circle(r, num_points=100):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def generate_naca4_airfoil(naca_string: str, num_points: int = 100, use_cosine_sampling=True) -> npt.NDArray[np.float64]:

    """generates NACA4 coordinates

    Parameters
    ==========

    naca_string: str
        NACA4 string

    num_points: int
        number of points to generate
    """
    M = int(naca_string[0]) / 100
    P = int(naca_string[1]) / 10
    XX = int(naca_string[2:]) / 100

    # camber line
    if use_cosine_sampling:
        beta = np.linspace(0.0,np.pi, num_points)
        xc = 0.5*(1.0-np.cos(beta))
    else:
        xc = np.linspace(0.0, 1.0, num_points)
    # thickness distribution from camber line
    a0 = 0.2969
    a1 = -0.1260
    a2 = -0.3516
    a3 = 0.2843
    a4 = -0.1036
    yt = 5.0*XX*(np.sqrt(xc)*a0 + xc**4*a4 + xc**3*a3 + xc**2*a2 + xc*a1)
    
    # camber line slope
    if P == 0:
        xl = xu = xc
        yu = yt
        yl = -yt

    else:
        yc1 = M*(-xc**2 + 2*xc*P)/P**2
        yc2 = M*(-xc**2 + 2*xc*P - 2*P + 1)/(1 - P)**2
        yc = (np.select([np.logical_and.reduce((np.greater_equal(xc, 0),np.less(xc, P))),np.logical_and.reduce((np.greater_equal(xc, P),np.less_equal(xc, 1))),True], [yc1,yc2,1], default=np.nan))

        dyc1dx = M*(-2*xc + 2*P)/P**2
        dyc2dx = M*(-2*xc + 2*P)/(1 - P)**2
        dycdx = (np.select([np.logical_and.reduce((np.greater_equal(xc, 0),np.less(xc, P))),np.logical_and.reduce((np.greater_equal(xc, P),np.less_equal(xc, 1))),True], [dyc1dx,dyc2dx,1], default=np.nan))
        theta = np.arctan(dycdx)
        
        xu = xc - yt*np.sin(theta)
        yu = yc + yt*np.cos(theta)
        xl = xc + yt*np.sin(theta)
        yl = yc - yt*np.cos(theta)

    # thickness lines
    x = np.concatenate((xu[1:-1], xl[::-1]))
    y = np.concatenate((yu[1:-1], yl[::-1]))

    return np.column_stack((x, y)) # type: ignore
