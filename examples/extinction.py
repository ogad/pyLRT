"""
Atmospheric Extinction
======================

This example demonstrates the interpretation of the verbose output from libRadtran
to show the extinction of atmospheric components in different spectral regions.
"""

# %%
from calendar import c
from math import e
from re import T
from pyLRT import RadTran, get_lrt_folder
from pyLRT.misc import planck_function
import matplotlib.pyplot as plt
import copy
import numpy as np

from pathlib import Path

outdir = Path(__file__).parent / "output"


def shade_and_outline(x, y, color="grey"):
    plt.fill_between(x, y, color=color)
    if color == "grey":
        plt.plot(x, y, lw=0.5, c="k")


def plot_planck_functions(temps=[210, 255, 310, 5800], colors=["r", "C1", "y", "b"]):
    for temp, color in zip(temps, colors):
        wvl = swvl_extended if temp == 5800 else twvl
        planck = planck_function(temp, wavelength=wvl * 1e-9)
        planck = planck * wvl
        plt.plot(
            np.log(wvl),
            planck / planck.max(),
            lw=2,
            label=f"{temp} K",
            color=color,
        )


# %% Set up the radiative transfer models
LIBRADTRAN_FOLDER = get_lrt_folder()

slrt = RadTran(LIBRADTRAN_FOLDER)
slrt.options["rte_solver"] = "disort"
slrt.options["source"] = "solar"
slrt.options["wavelength"] = "200 2600"
slrt.options["output_user"] = "lambda eglo eup edn edir"
slrt.options["zout"] = "0 5 TOA"
slrt.options["albedo"] = "0"
slrt.options["umu"] = "-1.0 1.0"
slrt.options["quiet"] = ""
slrt.options["sza"] = "0"

tlrt = copy.deepcopy(slrt)
tlrt.options["rte_solver"] = "disort"
tlrt.options["source"] = "thermal"
tlrt.options["output_user"] = "lambda edir eup uu"
tlrt.options["wavelength"] = "2500 80000"
tlrt.options["mol_abs_param"] = "reptran fine"
tlrt.options["sza"] = "0"

# %% Run the radiative transfer models
print("Initial RT")
sdata, sverb = slrt.run(verbose=True)
tdata, tverb = tlrt.run(verbose=True)
print("Done RT")

# %% Analyse the verbose output

# Get Solar and Thermal wavelengths
swvl = sverb["gases"]["wvl"][::10]
swvl_extended = np.concatenate((swvl, np.linspace(2500, 5000, 100)))

twvl = tverb["gases"]["wvl"][::10]


def extinction_plot(var, toa_index=0, from_alt_index=None, gas_with_rayleigh=False):
    # Integrate optical depth
    slicers = (slice(None, None, 10), slice(toa_index, from_alt_index))

    if var == "total" and gas_with_rayleigh:
        raise ValueError(
            "`gas_with_rayleigh` requires a specific gas to be selected as `var`"
        )

    if var == "total" or gas_with_rayleigh == True:
        s_opt_depth = sverb["gases"]["mol_abs"][*slicers].sum(axis=-1)
        t_opt_depth = tverb["gases"]["mol_abs"][*slicers].sum(axis=-1)
        s_opt_depth += sverb["gases"]["rayleigh_dtau"][*slicers].sum(axis=-1)
        t_opt_depth += tverb["gases"]["rayleigh_dtau"][*slicers].sum(axis=-1)

        if var != "total":
            shade_and_outline(np.log(swvl), 1 - np.exp(-s_opt_depth), color="r")
            shade_and_outline(np.log(twvl), 1 - np.exp(-t_opt_depth), color="r")

    else:
        s_opt_depth = sverb["gases"][var][*slicers].sum(axis=-1)
        t_opt_depth = tverb["gases"][var][*slicers].sum(axis=-1)

    if gas_with_rayleigh:
        s_opt_depth -= sverb["gases"][var][*slicers].sum(axis=-1)
        t_opt_depth -= tverb["gases"][var][*slicers].sum(axis=-1)

    shade_and_outline(np.log(swvl), 1 - np.exp(-s_opt_depth))
    shade_and_outline(np.log(twvl), 1 - np.exp(-t_opt_depth))


# %% Set up plotting parameters
wvlticks = [
    (
        list(range(200, 1000, 100))
        + list(range(1000, 10000, 1000))
        + list(range(10000, 71000, 10000))
    ),
    (["0.2"] + [""] * 7 + ["1"] + [""] * 8 + ["10"] + [""] * 5 + ["70"]),
]
trans_ticks = [[0, 0.25, 0.5, 0.75, 1], [0, 25, 50, 75, 100]]

vars = [
    ["rayleigh_dtau", "rayleigh", "Rayleigh"],
    ["o3", "o3", r"O$_3$"],
    ["o2", "o2", r"O$_2$"],
    ["h2o", "h2o", r"H$_2$O"],
    ["co2", "co2", r"CO$_2$"],
    ["ch4", "ch4", r"CH$_4$"],
]


def format_axes(
    xlabel=True,
    ylabel=True,
    xticklabels=True,
    yticklabels=True,
    ylabel_extra="",
    twin=True,
):
    plt.xticks(
        np.log(wvlticks[0]), wvlticks[1] if xticklabels else [""] * len(wvlticks[1])
    )
    plt.xlim(np.log(wvlticks[0][0]), np.log(wvlticks[0][-1]))
    if xlabel:
        plt.xlabel(r"Wavelength ($\mu$m)")
    plt.yticks(*trans_ticks if yticklabels else [trans_ticks[0], [""] * 4])
    plt.ylim(0, 1)
    if ylabel_extra != "" and ylabel_extra[0] != "\n":
        ylabel_extra = "\n" + ylabel_extra
    if ylabel:
        plt.ylabel(r"Extinction" + ylabel_extra)
    if twin:
        ax2 = plt.gca().twinx()
        plt.ylabel(r"$\lambda$B$_{\lambda}$ (normalised)")
        plt.yticks([], [])


# %% Plot the extinction
fig = plt.gcf()
toa = 0  # Top of atmosphere index

# Multi-plot of extinction and Planck functions, plus contributions by gas
plt.subplot2grid((7, 1), (0, 0), rowspan=2)

extinction_plot("total")
plot_planck_functions()

plt.legend()
format_axes(xticklabels=False, xlabel=False)

# Plot the extinction for a selection of important gases
for v, var in enumerate(vars[:-1]):
    plt.subplot2grid((7, 1), (v + 2, 0))
    extinction_plot(var[0])
    plt.text(np.log(30000), 0.5, var[2], verticalalignment="center")
    is_last_var = v == len(vars) - 2
    format_axes(xlabel=is_last_var, xticklabels=is_last_var, twin=False)

plt.tight_layout(h_pad=0.1)
fig.set_size_inches(6, 5)
fig.savefig(outdir / "as_complete.png", bbox_inches="tight")
fig.clf()
del fig

# %% Total extinction and planck functions
fig = plt.gcf()

extinction_plot("total")
plot_planck_functions()

plt.legend()
format_axes()
plt.tight_layout(h_pad=0)
fig.set_size_inches((8, 3))
fig.savefig(outdir / "as_total.png")
fig.clf()

# %% Total extinction at different heights
fig = plt.gcf()
for k, toa in enumerate([-1, -5, -10]):
    print(k, toa)
    plt.subplot(3, 1, k + 1)

    extinction_plot("total", from_alt_index=toa)
    format_axes(
        xticklabels=k == 2,
        xlabel=k == 2,
        ylabel_extra=["TOA-Surf", "TOA-5km", "TOA-10km"][k],
        twin=False,
    )

plt.tight_layout(h_pad=0)
fig = plt.gcf()
fig.set_size_inches((8, 4))
fig.savefig(outdir / "as_total_heights.png")
fig.clf()


# %% Individual extinction plots
for var in vars:
    extinction_plot(var[0])
    format_axes(twin=False)
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches((8, 3))
    fig.savefig(outdir / "as_{}.png".format(var[1]))
    fig.clf()


# %% Individual extinction plots with Rayleigh
for var in vars:
    plt.subplot(211)
    extinction_plot(var[0], gas_with_rayleigh=True)
    format_axes(twin=False, xticklabels=False, xlabel=False)

    plt.subplot(212)
    extinction_plot(var[0])
    format_axes(twin=False)

    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches((8, 3))
    fig.savefig(outdir / "as2_{}.png".format(var[1]))
    fig.clf()
