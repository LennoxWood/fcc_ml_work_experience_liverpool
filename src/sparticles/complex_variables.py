"""
complex_variables.py

Vectorized (pandas/numpy) helper functions for computing higher-order kinematic
variables -- invariant masses, angular separations, transverse masses, and MET
centrality -- from the basic (pT, eta, phi) measurements of reconstructed objects.

IMPORTANT ASSUMPTIONS (please sanity-check these against whatever paper/thesis
chapter you're replicating -- I'm reconstructing this from variable *names* alone,
since the original module couldn't be located):

1. All objects (jets, b-jets, leptons, taus) are treated as MASSLESS four-vectors
   for the purposes of these formulas. This is the standard simplification used in
   most collider-physics "delta-R / invariant mass from pT-eta-phi" calculations,
   and the error introduced is small for objects with pT >> their rest mass (true
   for jets/leptons at FCC-hh energies; less exactly true for the visible tau decay
   products, but still a common simplification in practice).
2. `m_T` is implemented as the transverse mass of the (lepton, MET) system --
   i.e. the classic "W transverse mass" formula. This is the most common single
   quantity called "mT" in semi-leptonic searches, but if your original analysis
   used (tau, MET) or a combined lepton+tau+MET transverse mass instead, swap which
   columns get passed into `transverse_mass()` below.
3. `C_met` (MET centrality) uses the formula from VBF H->tautau searches (e.g.
   ATLAS-CONF/CMS papers on di-tau resonance searches): MET centrality is positive
   and bounded between 0 and 2 when the MET points "between" the two visible decay
   products in phi, and goes negative/large outside that range. This is a standard
   discriminating variable against backgrounds where MET doesn't point between the
   visible tau-pair legs.
4. `m_HH` and `dPhi_HH` build the bb-candidate and (lepton+tau)-candidate as their
   own four-vectors and combine the two. This ignores MET when forming the
   (lepton+tau) ditau system -- a simplification; if your original used a neutrino
   estimate (e.g. collinear approximation or MMC) to build a full ditau mass before
   combining with the bb system, this will differ somewhat from your original
   values. Worth validating against any reference plots/cutflow you still have.
"""

import numpy as np


def delta_phi(phi1, phi2):
    """Azimuthal angle difference, wrapped into [-pi, pi]."""
    dphi = phi1 - phi2
    return np.arctan2(np.sin(dphi), np.cos(dphi))


def delta_r(eta1, phi1, eta2, phi2):
    """Standard angular separation: dR = sqrt(deta^2 + dphi^2)."""
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


def four_vector(pt, eta, phi, m=0.0):
    """
    Build a (E, px, py, pz) four-vector from (pT, eta, phi[, mass]).
    Assumes m=0 (massless) unless a mass array/scalar is given.
    """
    pz = pt * np.sinh(eta)
    p = pt * np.cosh(eta)
    e = np.sqrt(p**2 + m**2)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    return e, px, py, pz


def invariant_mass(pt1, eta1, phi1, pt2, eta2, phi2, m1=0.0, m2=0.0):
    """
    Invariant mass of a two-body system, given each object's (pT, eta, phi[, mass]).
    Defaults to the massless approximation (m1=m2=0) if no masses are supplied.
    """
    e1, px1, py1, pz1 = four_vector(pt1, eta1, phi1, m1)
    e2, px2, py2, pz2 = four_vector(pt2, eta2, phi2, m2)
    e, px, py, pz = e1 + e2, px1 + px2, py1 + py2, pz1 + pz2
    m_sq = e**2 - (px**2 + py**2 + pz**2)
    # Clip to avoid tiny negative values from floating-point error producing NaN
    # under sqrt.
    return np.sqrt(np.clip(m_sq, 0, None))


def system_four_vector(*objects):
    """
    Sum the four-vectors of any number of (pt, eta, phi[, m]) tuples into one
    combined (E, px, py, pz) system four-vector. Each tuple can be length 3
    (massless) or 4 (with mass).
    """
    e_tot = px_tot = py_tot = pz_tot = 0.0
    for obj in objects:
        pt, eta, phi = obj[0], obj[1], obj[2]
        m = obj[3] if len(obj) > 3 else 0.0
        e, px, py, pz = four_vector(pt, eta, phi, m)
        e_tot = e_tot + e
        px_tot = px_tot + px
        py_tot = py_tot + py
        pz_tot = pz_tot + pz
    return e_tot, px_tot, py_tot, pz_tot


def system_mass(e, px, py, pz):
    """Invariant mass of a combined system, given its summed four-vector."""
    m_sq = e**2 - (px**2 + py**2 + pz**2)
    return np.sqrt(np.clip(m_sq, 0, None))


def system_phi(px, py):
    """Azimuthal angle of a combined system, given its summed transverse momentum."""
    return np.arctan2(py, px)


def transverse_mass(pt1, phi1, pt2, phi2):
    """
    Transverse mass of a two-object system (e.g. lepton + MET), the classic
    "W transverse mass" formula:
        mT = sqrt(2 * pT1 * pT2 * (1 - cos(dphi)))
    """
    dphi = delta_phi(phi1, phi2)
    return np.sqrt(np.clip(2 * pt1 * pt2 * (1 - np.cos(dphi)), 0, None))


def met_centrality(phi_leg1, phi_leg2, phi_met):
    """
    MET centrality: a measure of whether the MET vector points "between" the two
    visible decay-product legs in the transverse plane. Defined as:

        C = sin(dphi(MET, leg2)) / sin(dphi(leg1, leg2))
          + sin(dphi(leg1, MET))  / sin(dphi(leg1, leg2))

    C is positive (and bounded, typically 0-2) when MET falls angularly between
    the two legs, which is the expected signature of neutrinos from a resonance
    decaying to both legs (e.g. H -> tautau). It can become large or change sign
    when the two legs are nearly collinear (sin(dphi(leg1,leg2)) ~ 0) -- guarded
    below by returning NaN in that edge case rather than a huge/unstable number.
    """
    s12 = np.sin(delta_phi(phi_leg1, phi_leg2))
    s12_safe = np.where(np.abs(s12) < 1e-6, np.nan, s12)
    a = np.sin(delta_phi(phi_met, phi_leg2)) / s12_safe
    b = np.sin(delta_phi(phi_leg1, phi_met)) / s12_safe
    return a + b


# All 9 derived variables. Kept as a single list (rather than just splitting them
# inline elsewhere) so there's one source of truth for "every variable this module
# can compute", regardless of whether a given variable ends up on an edge or on
# the complex node downstream.
#
# Naming note: 'tt' here means "tau-tau", in the sense that BOTH legs of this
# pair trace back to a tau decay -- the 'lepton' object is the visible product
# of a leptonically-decaying tau, and the 'tau' object is the visible product of
# a hadronically-decaying tau. m_tt/dR_tt/dpT_tt are therefore the invariant
# mass / angular separation / pT difference between the lepton and tau objects
# (same physical quantities previously named m_lt/dR_lt/dpT_lt).
COMPLEX_VAR_NAMES = [
    'm_bb', 'dR_bb',
    'm_tt', 'dR_tt', 'dpT_tt',
    'm_T',
    'C_met',
    'm_HH', 'dPhi_HH',
]

# These 5 are genuinely two-body quantities (each depends on exactly one specific
# pair of objects) and are represented as EDGE features between that specific
# pair of nodes:
#   - m_bb, dR_bb:            the b1<->b2 edge
#   - m_tt, dR_tt, dpT_tt:    the lepton<->tau edge
EDGE_VAR_NAMES = ['m_bb', 'dR_bb', 'm_tt', 'dR_tt', 'dpT_tt']

# These 4 are always stored on the "complex" node, which is connected to every
# other node and can carry whole-event (or, in m_T's case, a context-dependent)
# information:
#   - m_T:      transverse mass of the (lepton, MET) system -- kept as a NODE
#               feature here rather than an edge, per spec, even though it's
#               technically a pairwise (lepton, MET) quantity.
#   - C_met:    MET centrality, needs lepton, tau, AND MET at once.
#   - m_HH:     needs all four of b1, b2, lepton, tau at once.
#   - dPhi_HH:  needs all four of b1, b2, lepton, tau at once.
NODE_VAR_NAMES = ['m_T', 'C_met', 'm_HH', 'dPhi_HH']


def compute_complex_variables(df, flags):
    """
    Compute all enabled complex variables and add them as new columns on `df`
    (a pandas DataFrame already containing the raw per-object pT/eta/phi columns).

    Args:
        df (pd.DataFrame): event dataframe with columns pTb1/etab1/phib1,
            pTb2/etab2/phib2, pTt2/etat2/phit2, pTt1/etat1/phit1,
            ETMiss/ETMissPhi already present.
        flags (dict): maps each name in COMPLEX_VAR_NAMES to True/False, controlling
            whether it's actually computed (True) or left as NaN (False) -- NaN
            here means "this feature doesn't apply for this configuration", the
            same convention used elsewhere in this package for absent features.

    Returns:
        pd.DataFrame: the same dataframe, with one new column per entry in
            COMPLEX_VAR_NAMES added (either real computed values or NaN).
    """
    # bb system
    if flags.get('m_bb'):
        df['m_bb'] = invariant_mass(df['pTb1'], df['etab1'], df['phib1'],
                                     df['pTb2'], df['etab2'], df['phib2'])
    else:
        df['m_bb'] = np.nan

    if flags.get('dR_bb'):
        df['dR_bb'] = delta_r(df['etab1'], df['phib1'], df['etab2'], df['phib2'])
    else:
        df['dR_bb'] = np.nan

    # tau-tau system (lepton leg = leptonic tau decay, tau leg = hadronic tau decay)
    if flags.get('m_tt'):
        df['m_tt'] = invariant_mass(df['pTt2'], df['etat2'], df['phit2'],
                                     df['pTt1'], df['etat1'], df['phit1'])
    else:
        df['m_tt'] = np.nan

    if flags.get('dR_tt'):
        df['dR_tt'] = delta_r(df['etat2'], df['phit2'], df['etat1'], df['phit1'])
    else:
        df['dR_tt'] = np.nan

    if flags.get('dpT_tt'):
        df['dpT_tt'] = (df['pTt2'] - df['pTt1']).abs()
    else:
        df['dpT_tt'] = np.nan

    # transverse mass: lepton + MET (see module docstring point 2)
    if flags.get('m_T'):
        df['m_T'] = transverse_mass(df['pTt2'], df['phit2'], df['ETMiss'], df['ETMissPhi'])
    else:
        df['m_T'] = np.nan

    # MET centrality relative to the lepton-tau (tau-tau) system
    if flags.get('C_met'):
        df['C_met'] = met_centrality(df['phit2'], df['phit1'], df['ETMissPhi'])
    else:
        df['C_met'] = np.nan

    # di-Higgs system: combine the bb-candidate and the (lepton+tau)-candidate
    if flags.get('m_HH') or flags.get('dPhi_HH'):
        e_bb, px_bb, py_bb, pz_bb = system_four_vector(
            (df['pTb1'], df['etab1'], df['phib1']),
            (df['pTb2'], df['etab2'], df['phib2']),
        )
        e_lt, px_lt, py_lt, pz_lt = system_four_vector(
            (df['pTt2'], df['etat2'], df['phit2']),
            (df['pTt1'], df['etat1'], df['phit1']),
        )

        if flags.get('m_HH'):
            e_hh, px_hh, py_hh, pz_hh = (e_bb + e_lt, px_bb + px_lt, py_bb + py_lt, pz_bb + pz_lt)
            df['m_HH'] = system_mass(e_hh, px_hh, py_hh, pz_hh)
        else:
            df['m_HH'] = np.nan

        if flags.get('dPhi_HH'):
            df['dPhi_HH'] = delta_phi(system_phi(px_bb, py_bb), system_phi(px_lt, py_lt))
        else:
            df['dPhi_HH'] = np.nan
    else:
        df['m_HH'] = np.nan
        df['dPhi_HH'] = np.nan

    return df