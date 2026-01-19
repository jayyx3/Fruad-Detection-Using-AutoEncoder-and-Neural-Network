# Credit Card Fraud Detection (Autoencoder + GNN)

A Streamlit app that detects potential credit card fraud using an Autoencoder for anomaly detection and a Graph Neural Network (GNN) for relational patterns.

## Features
- Interactive fraud prediction form
- Autoencoder reconstruction error analysis
- Optional GNN-based neighbor analysis
- Model insights with plots

## Project Structure
- app.py – Streamlit UI
- models.py – Autoencoder and GCN definitions
- predict.py – Inference helpers
- utils.py – Plot utilities
- *.pth / *.pkl / *.npy – Model artifacts
- *.csv – Sample datasets

## Requirements
- Python 3.10+ (tested with 3.13)

## Quick Start
1. Create and activate a virtual environment.
2. Install dependencies.
3. Run the app:

```
python -m streamlit run app.py
```

## Notes
- Model files must be present in the project root.
- The GNN model is optional; the app will still run without it.

## License
Add a license if you plan to distribute this project.
