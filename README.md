# ⚡ Silicon Nitride Directional & Ring Coupler Solver

A Python-based 2D Semi-Vectorial Finite Difference (SVFD) Mode Solver and Coupled-Mode Theory (CMT) analysis tool for integrated photonics. Designed specifically for Silicon Nitride ($\text{Si}_3\text{N}_4$) directional couplers and ring-to-bus resonator coupling structures.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌟 Features

* **2D SVFD Mode Solver:** Computes symmetric (even) and antisymmetric (odd) supermode field distributions and effective refractive indices ($n_{\text{eff}}$) across 2D optical cross-sections.
* **Dispersion Analysis:** Includes Sellmeier dispersion equations for both $\text{Si}_3\text{N}_4$ (core) and $\text{SiO}_2$ (cladding).
* **Ring Resonator Coupling Dynamics:**
  * Accounts for straight coupler length ($L$) and curved coupling regions ($R$).
  * Calculates the wavelength-dependent residual effective length $L_{\text{residual}}(\lambda)$ using cladding decay constant $\gamma(\lambda)$.
* **Critical Coupling & Loss Evaluation:**
  * Computes power transfer dynamics ($P_{\text{cross}}$ and $P_{\text{bar}}$).
  * Evaluates round-trip power loss ($1 - a^2$) for multiple attenuation levels ($\alpha = 0.5, 1.5, 5.0\text{ dB/cm}$).
  * Dynamically calculates loaded quality factors ($Q_L$) under critical coupling conditions.
* **Interactive Web Dashboard:** Clean, responsive UI built with Streamlit featuring tabbed visual navigation and real-time metric cards.

---

## 📂 Project Structure

```text
.
├── coupler_engine.py  # Core physics engine (SVFD solver & CMT math)
├── app.py             # Streamlit Web GUI & interactive plotting
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
