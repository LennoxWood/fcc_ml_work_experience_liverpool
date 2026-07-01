# ============================================================
# Dataset cell -- works on any machine (Windows / Mac / Linux)
# Data files are downloaded automatically on first run (~1-2 GB)
# and cached locally in ./fcc_hh_data/ for all subsequent runs.
# ============================================================

sig_no    = 20000
ttb_no    = 20000
vjets_no  = 0

dataset = EventsDataset(
    root='./fcc_hh_data',   # local folder, created automatically on first run
    url='https://cernbox.cern.ch/s/SpsIy2kRzVZtwnn/download/',
    delete_raw_archive=False,
    delete_processed=True,
    add_edge_index=True,
    event_subsets={'signal': sig_no, 'vjets': vjets_no, 'ttbar': ttb_no},
    # transform=MakeHomogeneous(),  # uncomment when training
    download_type=2,
    signal_filename='hhbbtata.h5',
    background_filename='tt012j.h5',
    m_bb=True,
    dR_bb=True,
    m_tt=True,
    dR_tt=True,
    dpT_tt=True,
    m_HH=True,
    dPhi_HH=True,
)

print(f'Loaded {len(dataset)} events')
print(f'Node features per event: {dataset[0].x.shape}')
print(f'Edge features per event: {dataset[0].edge_attr.shape}')
