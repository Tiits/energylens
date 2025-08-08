

# EnergyLens

**Regional Electricity Consumption Forecast**  

EnergyLens is a Streamlit-based web application that provides high-resolution (15-minute) forecasts of electricity consumption for individual households. The app leverages Facebook’s Prophet library to model temporal patterns and visualize predictions along with historical data and confidence intervals.

---

## Table of Contents

- [Features](#features)  
- [Data Source](#data-source)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Configuration](#configuration)  
- [Contributing](#contributing)  
- [Acknowledgements](#acknowledgements)  
- [License](#license)

---

## Features

- Interactive selection of date range and household ID  
- Historical consumption visualization with Plotly  
- Customizable forecast horizon (1–30 days)  
- High-frequency (15‑minute) forecasts via Prophet  
- Interactive display of Prophet components (trend, seasonality)  

---

## Data Source

This project uses the HEAPO dataset (Canton of Zurich, 2018–2024), which provides 15-minute electricity consumption records for over 1,400 households.  
- **Source:** HEAPO project, Zurich University of Applied Sciences/ZHAW  
- **Raw data files:** placed under `data/raw/` (not committed; use the download script in `src/utils`)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Tiits/energylens.git
   cd energylens
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Usage

1. Place the HEAPO CSV into `data/raw/` (e.g., `data/raw/heapo.csv`).  
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open the local URL displayed in your browser.  
4. Use the sidebar to select date range, household ID, and forecast horizon.  
5. Click **Run Forecast** to generate predictions and view interactive plots.

---

## Project Structure

```
.
├── assets/                      # Static assets (logos, screenshots)
├── data/
│   ├── raw/                     # Original HEAPO CSV (not committed)
│   └── processed/               # Cleaned and resampled data
├── docs/                        # Optional Sphinx/MkDocs documentation
├── src/
│   ├── utils/                   # Data loading and preprocessing
│   ├── models/                  # Forecasting logic (Prophet)
│   └── viz/                     # Plotting routines (Plotly)
├── app.py                       # Main Streamlit application
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Ignored files
```

---

## Configuration

- **Data path:** Adjust the default path in `app.py` or via the sidebar text input.  
- **Prophet settings:** Modify changepoint prior scale and seasonality toggles under `Forecast Parameters`.

---

## Acknowledgements

- **HEAPO Dataset** (Zurich University of Applied Sciences/ZHAW)  
- **Facebook Prophet** for robust forecasting  
- **Streamlit** for rapid web app prototyping  
- **Plotly** for interactive visualizations  

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.