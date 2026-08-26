# ChatterDetect – Minimal Dataset and Code (Oxford Package)

This package provides a focused and lightweight set of resources for the Seed Fund project  
**“Control-integrated AI model for chatter detection”**, specifically prepared for Oxford.

It includes a labeled dataset, basic preprocessing and visualization utilities, and an initial notebook for data exploration and model development.

---

## 📁 Folder Structure

```
Oxford/
├── config/                # Preamble for consistent imports and paths
├── data/                 # Processed datasets with chatter annotations
│   └── 2025-05_Oxford_und_Rebecka/
│       └── processed/merge_ext/
├── notebooks/
│   └── 2025-05_Oxford_und_Rebecka/
│       └── Oxford.ipynb
├── results/              # Optional result plots or exports
├── src/                  # Custom Python modules (e.g., loading or plotting)
├── viz/                  # Visualization styles (e.g., IWF template)
└── environment.yml       # Conda environment for reproducibility
```

---

## ⚙️ Setup Instructions

1. Install Conda (if not already installed)
2. Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate chatterdetect
```

3. Launch the notebook interface:

```bash
jupyter lab build
jupyter lab notebooks/2025-05_Oxford_und_Rebecka/Oxford.ipynb
```

---

## 🚀 Getting Started

The notebook includes:
- Signal overview 
- Example visualizations of control and sensor data
- Basis for feature exploration (PCA, variance-based filtering etc.)
- Placeholder for your own model implementation

---

For questions or support, feel free to reach out.

**Prepared by:**  
Martin Heper  
Institute for Machine Tools and Factory Management (IWF), TU Berlin  
📧 heper@iwf.tu-berlin.de
