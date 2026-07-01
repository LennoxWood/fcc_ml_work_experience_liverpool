import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
import pandas as pd
from torch_geometric.data import Data
from tqdm import tqdm
import os
import glob

from .complex_variables import (
    compute_complex_variables,
    COMPLEX_VAR_NAMES,
    EDGE_VAR_NAMES,
    NODE_VAR_NAMES,
)

# defining function for converting h5 file to have all object features in the correct format.
def convert(x):
    if isinstance(x, dict):
        if len(x) == 1:
            item = list(x.values())[0]
            if isinstance(item, list):
                return item[0]
            else:
                return item
    else:
        return x

convert_vec = np.vectorize(convert)

# Random state for shuffling the dataset.
RANDOM_STATE = 42

# Names of the directories in the raw directory.
RAW_DIR_NAMES = ['signal', 'vjets', 'ttbar']

# Constant labels for signal and background.
SIGNAL_LABEL = 1
BACKGROUND_LABEL = 0

EVENT_LABELS = {
    'signal': SIGNAL_LABEL,
    'vjets':  BACKGROUND_LABEL,
    'ttbar':  BACKGROUND_LABEL,
}

DEFAULT_EVENT_SUBSETS = {
    'signal': 463056,
    'vjets':  242614,
    'ttbar':  6093298,
}

# Fixed node slot order. This MUST match the order blocks are concatenated into
# self.useful_cols, since _build_edge_attr_dense() addresses nodes by these
# fixed positions before any trash-node filtering happens.
TAU1, TAU2, B1, B2, MET, JET, COMPLEX = range(7)

# Which physical node pair each edge variable belongs to.
_EDGE_VAR_PAIR = {
    'm_bb':   (B1, B2),   'dR_bb':  (B1, B2),
    'm_tt':   (TAU2, TAU1), 'dR_tt':  (TAU2, TAU1), 'dpT_tt': (TAU2, TAU1),
}


def _build_edge_attr_dense(edge_value_dict, active_edge_vars, n_nodes):
    """
    Build a dense [n_nodes, n_nodes, len(active_edge_vars)] edge-feature tensor
    with values placed on their specific node pairs and NaN everywhere else.
    Must be built before trash-node filtering since filtering shifts node indices.
    """
    width = len(active_edge_vars)
    e = torch.full((n_nodes, n_nodes, width), float('nan'), dtype=torch.float)

    by_pair = {}
    for slot, name in enumerate(active_edge_vars):
        pair = _EDGE_VAR_PAIR[name]
        by_pair.setdefault(pair, []).append((slot, name))

    for (i, j), slot_name_pairs in by_pair.items():
        idx  = [slot for slot, _ in slot_name_pairs]
        vals = torch.tensor(
            [edge_value_dict[name] for _, name in slot_name_pairs],
            dtype=torch.float
        )
        e[i, j, idx] = vals
        e[j, i, idx] = vals

    return e


def _filter_edge_attr(edge_attr_dense, keep_mask, edge_index):
    """
    After trash-node filtering, extract the edge_attr rows that match the
    surviving edge_index, giving a [num_edges, F] tensor.
    """
    surviving = edge_attr_dense[keep_mask][:, keep_mask]
    src, dst = edge_index
    return surviving[src, dst]


class EventsDataset(InMemoryDataset):
    """
    Dataset of graphs representing particle collision events, read from local
    h5 files. Place your data files in the following structure before creating
    the dataset:

        <root>/raw/signal/<signal_filename>
        <root>/raw/ttbar/<background_filename>
        <root>/raw/vjets/<any_filename>.h5   (optional -- only needed if
                                               vjets events are requested)

    For example, with root='./fcc_hh_data':

        fcc_hh_data/raw/signal/hhbbtata.h5
        fcc_hh_data/raw/ttbar/tt012j.h5

    Processed graph tensors are cached in <root>/processed/ on first run and
    reused automatically on subsequent runs (set delete_processed=True to force
    a rebuild after changing flags or data files).

    Two kinds of derived/higher-order variables can be enabled:
      - Pairwise EDGE features (m_bb, dR_bb on b1<->b2; m_tt, dR_tt, dpT_tt
        on tau2<->tau1) -- stored on the specific edge between that node pair.
      - Whole-event NODE features (m_T, C_met, m_HH, dPhi_HH) -- stored on an
        extra "complex" node connected to every other node.

    Node and edge widths scale with how many variables are actually enabled, so
    disabling all variables gives slim 3-wide nodes with no padding overhead.

    Args:
        root (str): Base directory. Raw files go in <root>/raw/, processed
            cache goes in <root>/processed/.
        event_subsets (dict): How many events to keep per category. Categories
            with a count of 0 are skipped entirely (their h5 file is never
            opened). Default: {'signal': 463056, 'vjets': 242614, 'ttbar': 6093298}.
        add_edge_index (bool): Whether to build a fully-connected edge index.
            Default: True.
        delete_processed (bool): Delete the cached .pt file before deciding
            whether to rerun process(). Use this when you change any flag or
            data file. Default: False.
        signal_filename (str): Name of the h5 file inside <root>/raw/signal/.
        background_filename (str, optional): Name of the h5 file inside
            <root>/raw/ttbar/. If omitted, the first .h5 file found in that
            directory is used.
        m_bb, dR_bb, m_tt, dR_tt, dpT_tt (bool): Pairwise edge variables.
        m_T, C_met, m_HH, dPhi_HH (bool): Whole-event complex-node variables.
        normalize (bool): Min-max normalise each event type independently
            before building graphs. Default: False.
    """

    def __init__(
            self,
            root,
            url,
            event_subsets: dict = DEFAULT_EVENT_SUBSETS,
            add_edge_index: bool = True,
            delete_processed: bool = False,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            delete_raw_archive,
            signal_filename: str = 'hhbbtata.h5',
            background_filename: str = None,
            useful_cols: list = None,
            n_particles: int = None,
            normalize: bool = False,
            m_bb: bool = False,
            dR_bb: bool = False,
            m_tt: bool = False,
            dR_tt: bool = False,
            dpT_tt: bool = False,
            m_T: bool = False,
            C_met: bool = False,
            m_HH: bool = False,
            dPhi_HH: bool = False):

        self.event_subsets      = event_subsets
        self.add_edge_index     = add_edge_index
        self.signal_filename    = signal_filename
        self.background_filename = background_filename
        self.normalize          = normalize

        self.url = url
        self.delete_raw_archive = delete_raw_archive
                
        self.m_bb    = m_bb;  self.dR_bb   = dR_bb
        self.m_tt    = m_tt;  self.dR_tt   = dR_tt;  self.dpT_tt = dpT_tt
        self.m_T     = m_T;   self.C_met   = C_met
        self.m_HH    = m_HH;  self.dPhi_HH = dPhi_HH

        self.active_node_vars = [n for n in NODE_VAR_NAMES if getattr(self, n)]
        self.active_edge_vars = [n for n in EDGE_VAR_NAMES if getattr(self, n)]
        self.complex_node_needed = len(self.active_node_vars) > 0

        if self.complex_node_needed:
            self._node_width = max(3, 1 + len(self.active_node_vars))
        else:
            self._node_width = 3

        if useful_cols is None:
            self.useful_cols = self._build_useful_cols()
        else:
            self.useful_cols = useful_cols

        if n_particles is None:
            self.n_particles = 7 if self.complex_node_needed else 6
        else:
            self.n_particles = n_particles

        event_subset_string = '_'.join(
            [f'{k}_{v}' for k, v in sorted(self.event_subsets.items())]
        )
        flags_string = '_'.join(
            [f'{k}_{int(v)}' for k, v in sorted(self.complex_var_flags.items())]
        )
        self.subset_string = f'{event_subset_string}__{flags_string}'

        if delete_processed:
            for fname in self.processed_file_names:
                processed_path = os.path.join(root, 'processed', fname)
                if os.path.exists(processed_path):
                    os.remove(processed_path)
                    print(f'Deleted cached processed file: {processed_path}')

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _particle_block(self, pt_col, eta_col, phi_col):
        return [pt_col, eta_col, phi_col] + ['nan'] * (self._node_width - 3)

    def _build_useful_cols(self):
        complex_block = (
            ['start'] + self.active_node_vars +
            ['nan'] * (self._node_width - 1 - len(self.active_node_vars))
        ) if self.complex_node_needed else []

        return (
            self._particle_block('pTt1',   'etat1',   'phit1')   +  # tau1
            self._particle_block('pTt2',   'etat2',   'phit2')   +  # tau2
            self._particle_block('pTb1',   'etab1',   'phib1')   +  # b1
            self._particle_block('pTb2',   'etab2',   'phib2')   +  # b2
            ['ETMiss', 'nan', 'ETMissPhi'] + ['nan'] * (self._node_width - 3) +  # MET
            self._particle_block('pTj1',   'etaj1',   'phij1')   +  # jet
            complex_block
        )

    @property
    def complex_var_flags(self):
        return {name: getattr(self, name) for name in COMPLEX_VAR_NAMES}

    @property
    def raw_file_names(self):
        # Return the subdirectory names so PyG can check whether the raw data
        # folders already exist. If they do, download() is not called.
        return RAW_DIR_NAMES

    @property
    def processed_file_names(self):
        return [f'events_{self.subset_string}.pt']

    @property
    def event_structure(self):
        edge_desc = (
            f"{len(self.active_edge_vars)} edge feature(s): {self.active_edge_vars}"
            if self.active_edge_vars else "no edge features"
        )
        node_desc = (
            f"complex node present, {self._node_width} features per node: "
            f"{self.active_node_vars}"
            if self.complex_node_needed else
            f"no complex node, slim {self._node_width}-wide particle nodes"
        )
        return (
            f"This dataset instance: {self.n_particles} nodes, {node_desc}; "
            f"{edge_desc}."
        )

    # -------------------------------------------------------------------------
    # Download -- not used, but must be defined; tells the user what to do
    # -------------------------------------------------------------------------

    def download(self):
        """
        This dataset reads from local files -- there is nothing to download.
        If you see this message, it means the expected raw data folders were
        not found. Please create the following directory structure:

            <root>/raw/signal/<signal_filename>
            <root>/raw/ttbar/<background_filename>
            <root>/raw/vjets/<vjets_filename>.h5  (only if vjets_no > 0)

        For example:
            fcc_hh_data/raw/signal/hhbbtata.h5
            fcc_hh_data/raw/ttbar/tt012j.h5
        """
        print(f'Downloading {self.url} to {self.raw_dir}...')
        print('This may take a while...')
        raw_archive = download_url(self.url, self.raw_dir, filename='events.tar', log=False)

        print('Extracting files...')
        with tarfile.open(raw_archive) as tar:
            members = tar.getmembers()
            for member in members:
                if 'signal' in member.name and self.signal_filename not in member.name:
                    continue
                tar.extract(member, self.raw_dir)

        if self.delete_raw_archive:
            os.remove(raw_archive)

        print('Moving files...')
        for dir in self.raw_file_names:
            dirpath = glob.glob(f'{self.raw_dir}/**/{dir}', recursive=True)[0]
            shutil.move(dirpath, self.raw_dir)
            print(f'Moved {dirpath} to {self.raw_dir}')

        print('Cleaning up...')
        for f in os.listdir(self.raw_dir):
            if f not in self.raw_file_names + ['events.tar']:
                try:
                    shutil.rmtree(os.path.join(self.raw_dir, f))
                except NotADirectoryError:
                    os.remove(os.path.join(self.raw_dir, f))

    # -------------------------------------------------------------------------
    # Process
    # -------------------------------------------------------------------------

    def process(self):
        h5_files = {}

        for d in self.raw_file_names:
            # Skip categories where 0 events are requested -- never open the
            # file at all, so a missing or incompatible h5 causes no error.
            if self.event_subsets.get(d, 0) == 0:
                continue

            dir_path = os.path.join(self.raw_dir, d)

            if d == 'signal':
                path = os.path.join(dir_path, self.signal_filename)
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Signal file not found: {path}\n"
                        f"Expected at: {dir_path}/{self.signal_filename}"
                    )
                h5_files[d] = path

            elif d == 'ttbar' and self.background_filename is not None:
                path = os.path.join(dir_path, self.background_filename)
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Background file not found: {path}\n"
                        f"Expected at: {dir_path}/{self.background_filename}"
                    )
                h5_files[d] = path

            else:
                matches = glob.glob(os.path.join(dir_path, '*.h5'))
                if not matches:
                    raise FileNotFoundError(
                        f"No .h5 file found in {dir_path}"
                    )
                h5_files[d] = matches[0]

        data_list = []

        for event_type, h5_file in h5_files.items():
            label = EVENT_LABELS[event_type]

            graphs = pd.read_hdf(h5_file)
            graphs = graphs.apply(convert_vec)

            graphs = compute_complex_variables(graphs, self.complex_var_flags)

            if self.normalize:
                graphs = (graphs - graphs.min()) / (graphs.max() - graphs.min())

            graphs.drop(
                columns=list(
                    set(graphs.columns)
                    - set(self.useful_cols)
                    - set(self.active_edge_vars)
                ),
                inplace=True,
            )

            graphs['nan']   = torch.nan
            graphs['start'] = 0

            all_cols = ['index'] + self.useful_cols + self.active_edge_vars
            graphs = graphs.reset_index()
            graphs = graphs.sample(
                n=self.event_subsets[event_type],
                random_state=RANDOM_STATE
            )
            graphs = graphs[all_cols]

            node_width = len(self.useful_cols)
            for row in tqdm(
                graphs.values,
                total=graphs.shape[0],
                desc=f'Processing {event_type} events'
            ):
                event_id   = int(row[0])
                node_values = row[1:1 + node_width]
                edge_values = row[1 + node_width:]
                edge_value_dict = dict(zip(self.active_edge_vars, edge_values))

                x = torch.from_numpy(
                    node_values.astype(np.float64)
                ).reshape(self.n_particles, -1).float()

                edge_attr_dense = None
                if self.active_edge_vars:
                    edge_attr_dense = _build_edge_attr_dense(
                        edge_value_dict,
                        self.active_edge_vars,
                        n_nodes=self.n_particles,
                    )

                # Remove trash nodes (-99 sentinel in pT column).
                # The complex node (first feature = 0 sentinel 'start') is kept.
                keep_mask = x[:, 0] > -1
                x = x[keep_mask]

                edge_index = None
                edge_attr  = None
                if self.add_edge_index:
                    directed = torch.combinations(torch.arange(x.shape[0]), 2)
                    edge_index = torch.cat([directed, directed.flip(1)], dim=0).T
                    if edge_attr_dense is not None:
                        edge_attr = _filter_edge_attr(
                            edge_attr_dense, keep_mask, edge_index
                        )

                data_list.append(Data(
                    x=x,
                    event_id=f'{event_type}_{event_id}',
                    y=torch.tensor([label], dtype=torch.long),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                ))

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
