# Development of a Scalable ETL Pipeline and Retrieval Interface for Novel Edge Device Data

This repository contains the joint Bachelor's thesis work of **Julia Burczek** and **Dennis Berchert**.

The project consists of two main parts:

1. **Data cleaning and ETL pipeline** – run Dennis Berchert's `data_cleaning` script to process the raw data and generate a DuckDB database.
2. **Data retrieval and analysis** – open the `oxford_notebook` and place the generated database in the `data/` directory before running the notebook.

The complete workflow is:

**Raw data → Data cleaning → DuckDB database → Oxford notebook**