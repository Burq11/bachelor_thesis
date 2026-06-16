# ChatterDetect – Minimal Dataset and Code (Oxford Package)

This package provides a focused and lightweight set of resources for the Seed Fund project  
**“Control-integrated AI model for chatter detection”**, specifically prepared for Oxford.

It includes a labeled dataset, basic preprocessing and visualization utilities, and an initial notebook for data exploration and model development.

# V02 Demo Version with DB access #

# New files introduced #

**src/loader.py**: This file is responsible for the low-level logic of connecting to the database and reading data. It contains functions that handle the actual data access, such as querying tables or loading datasets from DuckDB.

**src/provider.py**: This acts as a connector. It uses the functions from loader.py and exposes them in a way that is convenient for notebooks or other parts of your codebase. If you add a new function to loader.py, you  make it available by adding a corresponding function or wrapper in provider.py.

**viz/widgets.py**: This file contains custom widgets that can be used in Jupyter notebooks for interactive data exploration and visualization.

---

## 📁 Folder Structure

```
Oxford/
├── config/                       # Preamble for consistent imports and paths
├── data/                         # Processed datasets with chatter annotations
│   └── <Database>                # --------place your database here---------
├── notebooks/
│       └── notebook_demo.ipynb   #     *** demo explaining new functionalities ***
│       └── notebook_visuals.ipynb#     ***  data visualization  ***
│       └── notebook_voila.ipynb  #     *** voila dashboard demo ***       
├── results/                      # Optional folder for result plots or exports
├── src/
│   └── data_processing.py        # analysis
│   └── loader.py                 # handles data fetching & queries 
│   └── provider.py               # gateway     
├── viz/
│   └── IWF_template.py 
│   └── visualizer.py 
│   └── widgets.py                # custom widgets for notebook
└── environment.yml               # Conda environment for reproducibility
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
jupyter lab notebooks/notebook_demo.ipynb
```

---

## About Notebooks

Notebooks in this project are stored **without outputs** in version control to keep the repository clean and focused on code changes.

### Automatic Output Stripping

Outputs are automatically removed before commits via a git pre-commit hook (`nbstripout`). This ensures:
- Clean git history (only code changes are tracked)
- Smaller repository size
- Easier collaboration and merge conflict resolution
---

##  Getting Started

Paste a database file into the `data/` folder. Then, open the `notebook_demo.ipynb` to see how to load and explore the data using the provider and loader functions. The notebook includes examples of querying the database and visualizing data distributions .


### Heatmap workflow

The heatmap workflow now runs directly against the raw DuckDB database. Its optimised queries are fast enough for interactive use without needing a separate cache layer. 

---

For questions or support, feel free to reach out.

**Prepared by:**  
Martin Heper  
Institute for Machine Tools and Factory Management (IWF), TU Berlin  
📧 heper@iwf.tu-berlin.de