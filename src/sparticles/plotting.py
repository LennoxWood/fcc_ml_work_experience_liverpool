"""
plotting.py

Shared plotting utilities for sparticles.dataset_edge_embed.EventsDataset
(hadhad version: tau1 + tau2, no lepton).

Two main entry points:
    plot_kinematics(dataset, object_name)
        pT, eta, phi (pT and phi only for 'met') as subplots in one figure.
    plot_complex_variable(dataset, var_name)
        One derived variable, signal vs background, as a single histogram.
        Works whether the variable lives on the complex node or on an edge.

Reads its structural knowledge directly off the dataset instance
(dataset.active_node_vars, dataset.active_edge_vars, dataset._node_width,
dataset.complex_node_needed, dataset.n_particles) so it stays correct as
you toggle which complex variables are enabled.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sparticles.dataset_edge_embed import (
    TAU1, TAU2, B1, B2, MET, JET, _EDGE_VAR_PAIR
)
from sparticles.complex_variables import NODE_VAR_NAMES, EDGE_VAR_NAMES

# ---------------------------------------------------------------------------
# Object/node configuration
# ---------------------------------------------------------------------------

OBJECT_NODES = {
    'tau1': TAU1,
    'tau2': TAU2,
    'b1':   B1,
    'b2':   B2,
    'met':  MET,
    'jet':  JET,
}

# MET has no eta by construction
NO_ETA_OBJECTS = {'met'}

# The only node that can be absent from an individual event (trash-filtered)
OPTIONAL_OBJECTS = {'jet'}

OBJECT_LABELS = {
    'tau1': r'$\tau_1$',
    'tau2': r'$\tau_2$',
    'b1':   'b-jet 1',
    'b2':   'b-jet 2',
    'met':  r'$E^T_{\mathrm{Miss}}$',
    'jet':  'Jet',
}

PT_XLIM = {
    'tau1': (0, 400), 'tau2': (0, 400),
    'b1':   (0, 750), 'b2':   (0, 300),
    'met':  (0, 300), 'jet':  (0, 700),
}
PT_BINS = {
    'tau1': 200, 'tau2': 200,
    'b1':   300, 'b2':   300,
    'met':  300, 'jet':  200,
}
ETA_PHI_BINS = 50

COMPLEX_VAR_CONFIG = {
    'm_bb':    dict(xlim=(0, 600),  bins=400, label=r'$m_{bb}$ (GeV)'),
    'dR_bb':   dict(xlim=(0, 6),    bins=70,  label=r'$\Delta R_{bb}$'),
    'm_tt':    dict(xlim=(0, 600),  bins=400, label=r'$m_{\tau\tau}$ (GeV)'),
    'dR_tt':   dict(xlim=(0, 6),    bins=70,  label=r'$\Delta R_{\tau\tau}$'),
    'dpT_tt':  dict(xlim=(0, 300),  bins=200, label=r'$\Delta p_T(\tau,\tau)$ (GeV)'),
    'm_T':     dict(xlim=(0, 500),  bins=300, label=r'$m_{T}$ (GeV)'),
    'C_met':   dict(xlim=None,      bins=65,  label=r'$E^T_{\mathrm{Miss}}$ Centrality'),
    'm_HH':    dict(xlim=(0, 2000), bins=500, label=r'$m_{HH}$ (GeV)'),
    'dPhi_HH': dict(xlim=None,      bins=100, label=r'$\Delta\phi_{HH}$'),
}

SIGNAL_COLOR     = '#ea6aea'
BACKGROUND_COLOR = '#93f393'


def set_plot_style():
    plt.rc('font',   size=24)
    plt.rc('axes',   titlesize=24, labelsize=24)
    plt.rc('xtick',  labelsize=22)
    plt.rc('ytick',  labelsize=22)
    plt.rc('legend', fontsize=20)
    plt.rc('figure', titlesize=24)


# ---------------------------------------------------------------------------
# Row-width-aware value extraction (handles MakeHomogeneous doubling)
# ---------------------------------------------------------------------------

def _get_value(row, raw_index, raw_width):
    """
    Read column `raw_index` from `row`, whether raw (length == raw_width)
    or MakeHomogeneous-doubled (length == 2*raw_width, value at 2*raw_index).
    """
    actual = row.shape[0]
    if actual == raw_width:
        return row[raw_index].item()
    elif actual == 2 * raw_width:
        return row[2 * raw_index].item()
    raise ValueError(
        f"Row width {actual} doesn't match expected raw width {raw_width} "
        f"or its doubled form {2 * raw_width}."
    )


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

def _node_rows(dataset, node_index, is_optional, max_nodes):
    """
    Collect dataset[i].x[node_index] for every event, split by label.
    If is_optional, only events with the full node count are included
    (absent nodes are removed by process(), shifting all later indices).
    """
    sig, bkg = [], []
    for g in dataset:
        if is_optional:
            if g.x.shape[0] != max_nodes:
                continue
        elif g.x.shape[0] <= node_index:
            raise IndexError(
                f"Event has {g.x.shape[0]} nodes but node_index={node_index} "
                f"was requested for a non-optional object."
            )
        (sig if g.y.item() == 1 else bkg).append(g.x[node_index])
    return sig, bkg


def plot_kinematics(dataset, object_name,
                    save_dir='Saved Figures/Input Variables', show=True):
    """
    Plot pT, eta, phi (pT and phi only for 'met') for one object,
    signal vs background, as subplots in a single figure.

    Args:
        dataset:     dataset_edge_embed.EventsDataset instance.
        object_name: one of 'tau1', 'tau2', 'b1', 'b2', 'met', 'jet'.
        save_dir:    directory for saved figures (created if missing).
        show:        call plt.show(); set False for batch use.
    """
    if object_name not in OBJECT_NODES:
        raise ValueError(
            f"Unknown object '{object_name}'. "
            f"Known objects: {list(OBJECT_NODES)}"
        )

    set_plot_style()
    node_index = OBJECT_NODES[object_name]
    sig_rows, bkg_rows = _node_rows(
        dataset, node_index,
        is_optional=object_name in OPTIONAL_OBJECTS,
        max_nodes=dataset.n_particles,
    )

    has_eta = object_name not in NO_ETA_OBJECTS
    quantities = [('pT', 0), ('eta', 1), ('phi', 2)] if has_eta \
                 else [('pT', 0), ('phi', 2)]

    fig, axes = plt.subplots(1, len(quantities),
                              figsize=(8 * len(quantities), 6))
    if len(quantities) == 1:
        axes = [axes]

    label     = OBJECT_LABELS.get(object_name, object_name)
    raw_width = dataset._node_width

    for ax, (quantity, col) in zip(axes, quantities):
        if quantity == 'pT':
            xlim   = PT_XLIM.get(object_name)
            bins   = PT_BINS.get(object_name, 100)
            xlabel = f'{label} ' + r'$p_T$ (GeV)'
        elif quantity == 'eta':
            xlim, bins = None, ETA_PHI_BINS
            xlabel = f'{label} ' + r'$\eta$'
        else:
            xlim, bins = None, ETA_PHI_BINS
            xlabel = f'{label} ' + r'$\phi$'

        sig_vals = [_get_value(row, col, raw_width) for row in sig_rows]
        bkg_vals = [_get_value(row, col, raw_width) for row in bkg_rows]
        _draw_histogram(ax, sig_vals, bkg_vals, bins, xlim, xlabel)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'{object_name}_kinematics.png'))
    fig.savefig(os.path.join(save_dir, f'{object_name}_kinematics.pdf'))

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Complex / derived variables -- node-based or edge-based
# ---------------------------------------------------------------------------

def _node_var_values(dataset, var_name):
    """Read a NODE_VAR_NAMES variable from the complex node."""
    if var_name not in dataset.active_node_vars:
        raise ValueError(
            f"'{var_name}' is not enabled for this dataset "
            f"(active_node_vars={dataset.active_node_vars})."
        )
    # Complex node layout: ['start'] + active_node_vars (+ NaN padding)
    raw_index = 1 + dataset.active_node_vars.index(var_name)
    raw_width = dataset._node_width

    sig, bkg = [], []
    for g in dataset:
        if not dataset.complex_node_needed or g.x.shape[0] == 0:
            continue
        value = _get_value(g.x[-1], raw_index, raw_width)
        (sig if g.y.item() == 1 else bkg).append(value)
    return sig, bkg


def _edge_var_values(dataset, var_name):
    """Read an EDGE_VAR_NAMES variable from the appropriate edge."""
    if var_name not in dataset.active_edge_vars:
        raise ValueError(
            f"'{var_name}' is not enabled for this dataset "
            f"(active_edge_vars={dataset.active_edge_vars})."
        )
    slot      = dataset.active_edge_vars.index(var_name)
    raw_width = len(dataset.active_edge_vars)
    i, j      = _EDGE_VAR_PAIR[var_name]   # e.g. (B1,B2) or (TAU2,TAU1)

    sig, bkg = [], []
    for g in dataset:
        if g.edge_index is None or g.edge_attr is None:
            continue
        src, dst = g.edge_index
        match = ((src == i) & (dst == j)) | ((src == j) & (dst == i))
        idx   = match.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        value = _get_value(g.edge_attr[idx[0]], slot, raw_width)
        (sig if g.y.item() == 1 else bkg).append(value)
    return sig, bkg


def _complex_values(dataset, var_name):
    if var_name in NODE_VAR_NAMES:
        return _node_var_values(dataset, var_name)
    elif var_name in EDGE_VAR_NAMES:
        return _edge_var_values(dataset, var_name)
    raise ValueError(
        f"Unknown variable '{var_name}'. "
        f"Known variables: {NODE_VAR_NAMES + EDGE_VAR_NAMES}"
    )


def plot_complex_variable(dataset, var_name,
                           save_dir='Saved Figures/Complex Variables',
                           show=True):
    """
    Plot one derived variable, signal vs background, as a single histogram.
    Works whether the variable lives on the complex node or on an edge.

    Args:
        dataset:  dataset_edge_embed.EventsDataset instance.
        var_name: e.g. 'm_bb', 'dR_tt', 'm_HH'. Must be enabled on the dataset.
        save_dir: directory for saved figures (created if missing).
        show:     call plt.show(); set False for batch use.
    """
    set_plot_style()
    sig_vals, bkg_vals = _complex_values(dataset, var_name)
    config = COMPLEX_VAR_CONFIG.get(
        var_name, dict(xlim=None, bins=100, label=var_name)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_histogram(ax, sig_vals, bkg_vals,
                    config['bins'], config['xlim'], config['label'])
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'{var_name}.png'))
    fig.savefig(os.path.join(save_dir, f'{var_name}.pdf'))

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Shared histogram-drawing helper
# ---------------------------------------------------------------------------

def _draw_histogram(ax, sig_values, bkg_values, bins, xlim, xlabel):
    if len(sig_values) == 0 or len(bkg_values) == 0:
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Event Density')
        ax.legend(handles=[
            plt.Rectangle((0,0),1,1, fc=SIGNAL_COLOR,     label='Signal'),
            plt.Rectangle((0,0),1,1, fc=BACKGROUND_COLOR, label='Background'),
        ])
        return

    edges = np.histogram(np.hstack((sig_values, bkg_values)), bins=bins)[1]
    ax.hist(sig_values, edges, color=SIGNAL_COLOR,     alpha=0.8,
             histtype='step', label='Signal',     density=True,
             linewidth=2, zorder=10)
    ax.hist(bkg_values, edges, color=BACKGROUND_COLOR, alpha=0.8,
             histtype='step', label='Background', density=True, linewidth=2)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Event Density')
    ax.legend()


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def plot_all_kinematics(dataset, objects=None, **kwargs):
    """Plot kinematics for every object (or a given subset)."""
    figs = {}
    for name in (objects or OBJECT_NODES):
        figs[name] = plot_kinematics(dataset, name, **kwargs)
    return figs


def plot_all_complex_variables(dataset, variables=None, **kwargs):
    """
    Plot every enabled derived variable, or a given subset.
    Defaults to dataset.active_node_vars + dataset.active_edge_vars.
    """
    if variables is None:
        variables = dataset.active_node_vars + dataset.active_edge_vars
    figs = {}
    for name in variables:
        figs[name] = plot_complex_variable(dataset, name, **kwargs)
    return figs

def get_event_dataframe(dataset):
    """
    Extract all enabled derived variables for every event into a pandas
    DataFrame with one row per event. Columns are the variable names plus
    'label' (1 = signal, 0 = background). Useful for cut-based analysis.
    """
    import pandas as pd

    rows = []
    for g in dataset:
        row = {'label': g.y.item()}

        # Node-based variables (from complex node)
        if dataset.complex_node_needed and g.x.shape[0] > 0:
            for var in dataset.active_node_vars:
                raw_index = 1 + dataset.active_node_vars.index(var)
                row[var] = _get_value(g.x[-1], raw_index, dataset._node_width)

        # Edge-based variables
        if g.edge_index is not None and g.edge_attr is not None:
            raw_width = len(dataset.active_edge_vars)
            for var in dataset.active_edge_vars:
                slot = dataset.active_edge_vars.index(var)
                i, j = _EDGE_VAR_PAIR[var]
                src, dst = g.edge_index
                match = ((src == i) & (dst == j)) | ((src == j) & (dst == i))
                idx = match.nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    row[var] = _get_value(g.edge_attr[idx[0]], slot, raw_width)
                else:
                    row[var] = float('nan')

        rows.append(row)

    return pd.DataFrame(rows)