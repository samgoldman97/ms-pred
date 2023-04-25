""" plot_utils.py

Set plot utils

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Draw

legend_params = dict(frameon=False, facecolor="none", fancybox=False)
method_colors = {
    # get dark gray color 
    "Random":  "#808080",
    "CFM-ID": "#C2AF53",
    "NEIMS (FFN)": "#B94346",
    "NEIMS (GNN)": "#DB8F76",
    "SCARF": "#479394",
    "ICEBERG": "#3D4A9F",
}

def export_mol(
    mol,
    name,
    width=100,
    height=100,
):
    """Save structure as PDF"""
    from rdkit.Chem.Draw import rdMolDraw2D
    import cairosvg
    import io

    drawer = rdMolDraw2D.MolDraw2DSVG(
        width,
        height,
    )
    opts = drawer.drawOptions()
    opts.bondLineWidth = 1
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    cairosvg.svg2pdf(bytestring=drawer.GetDrawingText().encode(), write_to=str(name))


def export_mol_highlight(
    mol,
    name,
    hatoms,
    hbonds,
    width=100,
    height=100,
    color=(0.925, 0.688, 0.355),
):
    """Save structure as PDF"""
    from rdkit.Chem.Draw import rdMolDraw2D
    import cairosvg
    import io

    d = rdMolDraw2D.MolDraw2DSVG(
        width,
        height,
    )
    rdMolDraw2D.PrepareAndDrawMolecule(
        d,
        mol,
        highlightAtoms=hatoms,
        highlightBonds=hbonds,
        highlightBondColors={i: color for i in hbonds},
        highlightAtomColors={i: color for i in hatoms},
    )
    d.FinishDrawing()
    cairosvg.svg2pdf(bytestring=d.GetDrawingText().encode(), write_to=str(name))


def set_style():
    """set_style"""
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.family"] = "sans-serif"
    sns.set(context="paper", style="ticks")
    mpl.rcParams["text.color"] = "black"
    mpl.rcParams["axes.labelcolor"] = "black"
    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["axes.labelcolor"] = "black"
    mpl.rcParams["xtick.color"] = "black"
    mpl.rcParams["ytick.color"] = "black"
    mpl.rcParams["xtick.major.size"] = 2.5
    mpl.rcParams["ytick.major.size"] = 2.5

    mpl.rcParams["xtick.major.width"] = 0.45
    mpl.rcParams["ytick.major.width"] = 0.45

    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["axes.linewidth"] = 0.45
    mpl.rcParams["font.size"] = 9
    mpl.rcParams["axes.labelsize"] = 9
    mpl.rcParams["axes.titlesize"] = 9
    mpl.rcParams["figure.titlesize"] = 9
    mpl.rcParams["figure.titlesize"] = 9
    mpl.rcParams["legend.fontsize"] = 6
    mpl.rcParams["legend.title_fontsize"] = 9
    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["ytick.labelsize"] = 6


def set_size(w, h, ax=None):
    """w, h: width, height in inches

    Resize the axis to have exactly these dimensions

    """
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)
