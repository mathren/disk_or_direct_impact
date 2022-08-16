# Study parameter space for direct accretion or disk in binaries

# Author: Mathieu Renzo <mrenzo@flatironinstitute.org>
# Keywords: files

# Copyright (C) 2019-2022 Mathieu Renzo

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses/.


import numpy as np
from plot_defaults import *


def Rmin_lubow_shu75(sep, Macc, Mdonor):
    """Returns the minimum radial distance from the accretor of the RLOF stream, the
    formula os the fit from Ulrich & Burger 1976 to the simulations of Lubow & Shu 1975.

    if the number returned is smaller than the accretor radius: direct impact
    """
    q = Macc / Mdonor
    return 0.00425 * sep * (q + q ** 2) ** 0.25


def get_R_from_mass_radius_rel(M, exp=0.6):
    """Rough mass-radius R = Rsun (M/Msun)^exp relation for main sequence massive stars
    Note that this assumes polytropic homogeneous stars

    Parameters:
    ----------
    M: `float` or np.array, mass of the star in Msun units
    exp: `float` exponent, default to approximate value for massive stars

    Returns:
    -------
    radius: `float` or np.array in Rsun units
    """
    return M ** exp


def get_R_RLOF(a, q):
    """
    Eggleton 1983 formula for Roche size
    """
    return a * (
        0.49 * q ** (2 / 3.0) / (0.6 * q ** (2 / 3.0) + np.log(1 + q ** (1 / 3.0)))
    )


def plot_mass_radius(fig_name=None):
    # plot mass-radius relation
    fig = plt.figure()
    gs = gridspec.GridSpec(120, 100)
    ax = fig.add_subplot(gs[:, :])

    # low mass stars
    m = np.linspace(0.2, 1.4, 20)
    r = get_R_from_mass_radius_rel(m, exp=0.9)
    ax.plot(np.log10(m), np.log10(r))
    # massive stars
    m = np.linspace(1.4, 30, 20)
    r = get_R_from_mass_radius_rel(m)
    ax.plot(np.log10(m), np.log10(r))

    ax.set_xlabel(r"$\log_{10}(M/M_\odot)$")
    ax.set_ylabel(r"$\log_{10}(R/R_\odot)$")
    if fig_name:
        plt.savefig(fig_name)


def disk_or_hit_no_R2_evol(M1=35, N=50, fig_name=None):
    """
    Plot the parameter space for disk vs. direct accretion using a mass-radius relation for the accretor, so no R2(t) evolution

    Parameters:
    ----------
    N: `int`, resolution
    M1: `float`, primary mass
    fig_name: `str` or None, path to where to save the plot, optional
    """
    fig = plt.figure()
    gs = gridspec.GridSpec(120, 100)
    ax = fig.add_subplot(gs[:, :])
    ax.set_title(r"$M_1 = " + f"{M1:.0f}" + "M_\odot$", size=30)

    mass_ratio = np.linspace(0.001, 1, N)
    sep = np.linspace(0.1, 2000, N)
    direct_hit = -1.0 * np.ones((N, N))

    for i, q in np.ndenumerate(mass_ratio):
        for j, a in np.ndenumerate(sep):
            Rmin = Rmin_lubow_shu75(a, q * M1, M1)
            if q * M1 <= 1.4:
                exp = 0.9
            else:
                exp = 0.6
            R2 = get_R_from_mass_radius_rel(q * M1, exp=exp)
            if Rmin / R2 > 1:
                direct_hit[i, j] = 0  # False
            else:
                direct_hit[i, j] = 1  # True
    cmap = mpl.colors.ListedColormap(["b", "r"])
    ax.pcolormesh(sep, mass_ratio, direct_hit, cmap=cmap)
    # ax.scatter(sep, mass_ratio, c=z, s=500)
    ax.contour(sep, mass_ratio, direct_hit, [0.5])

    ax.set_ylim(0, 1)
    ax.set_xlim(min(sep), max(sep))

    ax.text(
        0.1,
        0.9,
        "Direct impact",
        fontsize=30,
        va="center",
        ha="left",
        bbox=dict(
            facecolor="w", edgecolor="black", alpha=0.75, boxstyle="round,pad=0.1"
        ),
        transform=ax.transAxes,
        zorder=10,
    )
    ax.text(
        0.9,
        0.1,
        "Disk",
        fontsize=30,
        va="center",
        ha="right",
        bbox=dict(
            facecolor="w", edgecolor="black", alpha=0.75, boxstyle="round,pad=0.1"
        ),
        transform=ax.transAxes,
        zorder=10,
    )

    ax.set_xlabel(r"separation [$R_\odot$]")
    ax.set_ylabel(r"mass ratio $q=M_2/M_1$")

    if fig_name:
        plt.savefig(fig_name)  #'disk_or_no_disk_'+f'{M1:.0f}'+'.pdf')


def boundary_multiple_masses(primary_masses, N=50, fig_name=None):
    """plot the direct hit/disk boundary for an array of M1s

    Parameters:
    ----------
    primary_masses: `np.array`, primary masses in Msun
    N: `int`, resolution parameter
    fig_name: `str`, optional, path where to save the plot
    """
    fig = plt.figure()
    gs = gridspec.GridSpec(120, 100)
    ax = fig.add_subplot(gs[:, :])

    mass_ratio = np.linspace(0.001, 1, N)
    sep = np.linspace(0.1, 2000, N)

    colors = plt.cm.viridis(np.linspace(0, 1, len(primary_masses)))

    for k, M1 in np.ndenumerate(primary_masses):
        c = colors[np.argmin(np.absolute(M1 - primary_masses))]
        # reinitialize
        direct_hit = -1.0 * np.ones((N, N))
        for i, q in np.ndenumerate(mass_ratio):
            for j, a in np.ndenumerate(sep):
                Rmin = Rmin_lubow_shu75(a, q * M1, M1)
                if q * M1 <= 1.4:
                    exp = 0.9
                else:
                    exp = 0.6
                R2 = get_R_from_mass_radius_rel(q * M1, exp=exp)
                if Rmin / R2 > 1:
                    direct_hit[i, j] = 0  # False
                else:
                    direct_hit[i, j] = 1  # True
        boundary = ax.contour(sep, mass_ratio, direct_hit, levels=[0.5], colors=[c])
        ax.clabel(boundary, [0.5], fmt=f"{M1:.0f} " + r"$M_\odot$", fontsize=30)

    ax.set_ylim(0, 1)
    ax.set_xlim(min(sep), max(sep))

    ax.text(
        0.1,
        0.9,
        "Direct impact",
        fontsize=30,
        va="center",
        ha="left",
        bbox=dict(
            facecolor="w", edgecolor="black", alpha=0.75, boxstyle="round,pad=0.1"
        ),
        transform=ax.transAxes,
        zorder=10,
    )
    ax.text(
        0.9,
        0.1,
        "Disk",
        fontsize=30,
        va="center",
        ha="right",
        bbox=dict(
            facecolor="w", edgecolor="black", alpha=0.75, boxstyle="round,pad=0.1"
        ),
        transform=ax.transAxes,
        zorder=10,
    )

    ax.set_xlabel(r"separation [$R_\odot$]")
    ax.set_ylabel(r"mass ratio $q=M_2/M_1$")
    ax.set_title(r"variations with M1", color="w")
    if fig_name:
        plt.savefig(fig_name)


if __name__ == "__main__":
    # one mass
    disk_or_hit_no_R2_evol(M1=35, N=400, fig_name="disk_or_no_disk_35.pdf")
    # Multiple primary masses
    primary_masses = [2, 10, 20, 35, 50]
    boundary_multiple_masses(
        primary_masses, N=400, fig_name="disk_or_no_disk_multi_M1.pdf"
    )
