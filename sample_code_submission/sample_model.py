import numpy as np
import pandas as pd

class SampleModel:
    """
    This Dummy class implements a decision tree classifier
    change the code in the fit method to implement a decision tree classifier


    """

    def __init__(self):
        pass
    def fit(self, train_data, labels, weights=None):
        pass

    def predict(self, test_data):
        return compute_collinear_mass_vectorized(test_data)

def compute_collinear_mass_vectorized(df):
    # Extract columns as NumPy arrays
    lep_pt = df['PRI_lep_pt'].values
    lep_phi = df['PRI_lep_phi'].values
    had_pt = df['PRI_had_pt'].values
    had_phi = df['PRI_had_phi'].values
    MET = df['PRI_met'].values
    MET_phi = df['PRI_met_phi'].values
    mass_vis = df['DER_mass_vis'].values

    # Compute px and py components
    lep_px = lep_pt * np.cos(lep_phi)
    lep_py = lep_pt * np.sin(lep_phi)
    had_px = had_pt * np.cos(had_phi)
    had_py = had_pt * np.sin(had_phi)
    met_px = MET * np.cos(MET_phi)
    met_py = MET * np.sin(MET_phi)

    # Left-hand side matrix (2x2 system per row)
    det = lep_px * had_py - had_px * lep_py  # determinant of 2x2 matrix

    # Avoid division by zero or near-zero determinant
    safe_det = np.where(np.abs(det) < 1e-6, np.nan, det)

    # Solve using Cramer's Rule
    b1 = lep_px + had_px + met_px
    b2 = lep_py + had_py + met_py

    s_lep = (b1 * had_py - b2 * had_px) / safe_det
    s_had = (lep_px * b2 - lep_py * b1) / safe_det

    # Invert to get x_lep and x_had
    with np.errstate(divide='ignore', invalid='ignore'):
        x_lep = 1.0 / s_lep
        x_had = 1.0 / s_had

        # Only keep physical solutions
        valid = (x_lep > 0) & (x_had > 0)

        collinear_mass = np.full_like(mass_vis, -25)
        collinear_mass[valid] = mass_vis[valid] / np.sqrt(x_lep[valid] * x_had[valid])

    return collinear_mass
