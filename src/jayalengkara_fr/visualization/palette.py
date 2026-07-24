"""
Palette and colormaps for jayalengkara.

Stock scientific colormaps carry strong connotations of laboratory output. The
gradients defined here are built for the printed page and the gallery wall
instead: a deep ink ground, warm vellum foreground, and colormaps whose ramps
move through analogous hues rather than through the full spectrum. Perceptual
ordering is preserved, so the maps remain readable as data.

Four gradients are registered with matplotlib under the ``jl_`` prefix and may
be requested by name from any configuration file.

``jl_aurora``   indigo to violet to amber, for hyperbolic tessellation
``jl_nacre``    deep teal to pale cream, for entropy on the simplex
``jl_duality``  diverging teal to vellum to crimson, for signed divergence
``jl_ember``    midnight to rose to pale gold, for diffusion
"""

import numpy as np
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

# Ground and ink.
INK = '#070912'          # figure ground for animations
INK_SOFT = '#101426'     # panel ground
VELLUM = '#ece7dd'       # warm foreground, easier on the eye than pure white
PAPER = '#f7f3ec'        # ground for printed diagnostic panels
GRAPHITE = '#2a2622'     # text on paper
MUTED = '#6f7891'        # secondary rules and annotation

# Accents, chosen to stay legible on both ink and paper.
GOLD = '#e0b356'
TEAL = '#4fb6a8'
ROSE = '#dd6a86'
VIOLET = '#8f7fd1'
JADE = '#7fc8a9'

_SPECS = {
    'jl_aurora': [
        '#0d1636', '#232a63', '#4a3585', '#7b3d84',
        '#b04f74', '#dd7d59', '#f0b96f', '#f7e3b0',
    ],
    'jl_nacre': [
        '#08283a', '#0f5060', '#1f8377', '#5cb494',
        '#a3ddba', '#d8efd7', '#f4efe2',
    ],
    'jl_duality': [
        '#0c3a4c', '#17636f', '#4a9d9c', '#9dc4c0',
        '#ece7dd',
        '#e9c68d', '#dd9a5c', '#c26146', '#7f2f33',
    ],
    'jl_ember': [
        '#080a1e', '#241546', '#4d1f6b', '#7f2d72',
        '#b04168', '#dd6a5c', '#f0a05c', '#f8dca2',
    ],
}


def _register():
    """Register the palette colormaps with matplotlib, idempotently."""
    for name, colors in _SPECS.items():
        if name in matplotlib.colormaps:
            continue
        cmap = LinearSegmentedColormap.from_list(name, colors, N=512)
        try:
            matplotlib.colormaps.register(cmap, name=name)
            matplotlib.colormaps.register(cmap.reversed(), name=name + '_r')
        except (ValueError, AttributeError):  # pragma: no cover
            pass


_register()


def get_cmap(name, fallback='jl_aurora'):
    """
    Resolve a colormap by name, accepting palette names and stock names alike.

    An unknown name falls back to the palette default rather than raising, so a
    configuration file can never break a run on a colormap typo.
    """
    _register()
    try:
        return matplotlib.colormaps[name]
    except (KeyError, ValueError):
        return matplotlib.colormaps[fallback]


def truncate(cmap, lo=0.0, hi=1.0, n=512):
    """
    Restrict a colormap to a sub-range of its ramp.

    Sequential gradients here begin very near the ink ground, so marks carrying
    the lowest values would disappear into the background when drawn on a dark
    figure. Trimming the dark end keeps the low end legible without disturbing
    the perceptual ordering of the remainder.
    """
    if isinstance(cmap, str):
        cmap = get_cmap(cmap)
    colors = cmap(np.linspace(lo, hi, n))
    return LinearSegmentedColormap.from_list(f'{cmap.name}_trunc', colors, N=n)


def figure_style(dark=True):
    """Return the ground, foreground, and muted colors for a figure."""
    if dark:
        return INK, VELLUM, MUTED
    return PAPER, GRAPHITE, '#8a8378'
