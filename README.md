# Rooftop Solar Estimator (UNet + NASA POWER)

AI-powered web app to estimate rooftop solar potential by:
1. Segmenting rooftops using a UNet model (image → roof mask)
2. Letting the user select rooftop polygons on top of the predicted mask
3. Fetching irradiance + weather data from **NASA POWER API** (based on city lat/lon)
4. Estimating solar system size (**kW**) and yearly energy output (**kWh/year**)

This repository includes both **Residential** and **Corporate** flows.

---

## Demo / Features

- **Flask web UI**
  - `/` → choose Residential / Corporate
  - `/residential` → run residential analysis
  - `/corporate` → run corporate analysis
- **Two UNet model modes**
  - `custom` → advanced UNet pipeline (`selecthouse.py` / `corporate.py`)
  - `basic` → faster pipeline (`selecthousebasic.py` / `corporatesimple.py`)
- **Interactive polygon selection**
  - OpenCV window lets you click points to define rooftop polygons
  - Keys:
    - `N` → save polygon (requires at least 3 points)
    - `R` → reset current points
    - `Q` → finish selection

---

## Project Structure (high level)

- `app.py` — Flask server and route handlers
- `selecthouse.py` — residential: custom UNet + solar estimation + NASA POWER
- `selecthousebasic.py` — residential: basic UNet + solar estimation
- `corporate.py` — corporate: custom UNet + solar estimation + NASA POWER
- `corporatesimple.py` — corporate: basic/simple UNet + solar estimation
- `templates/`
  - `index.html` — homepage UI
  - `residential.html` — residential results view
  - `corporate.html` — corporate results view
- `static/` — styling/assets

---

## How it works

### 1) Roof segmentation (UNet)
Each analysis script:
- loads a trained UNet model from the local `DataSet/DS/...` path
- reads a sample/input image
- predicts a roof mask

> Note: polygon selection is interactive. The user defines regions (polygons) on top of the visualization.

### 2) Solar estimation
For each selected polygon (one “house” in Residential, one region/building group in Corporate):
- compute an area estimate using:
  - `pixel_resolution` (m/pixel)
  - roof-mask confidence weighting
- estimate an installable system size in **kW** using:
  - `coverage_factor`
  - panel rating constants (e.g., `panel_power = 400` W)
  - confidence adjustment
- fetch NASA POWER data for location and time range:
  - `ALLSKY_SFC_SW_DWN` (solar irradiance)
  - `T2M` (temperature)
  - `RH2M` (humidity)
  - `WS2M` (wind speed)
- adjust efficiency using temperature/wind/humidity factors
- compute yearly energy:
  - `energy = system_kw * irradiance_yearly * system_loss * eff_factor`

---

## Supported Cities

`app.py` contains a mapping from city name → (latitude, longitude). Current list:

- Chennai
- Bangalore
- Delhi
- Mumbai
- Hyderabad
- Palakkad

---

## Setup & Run (local)

### 1) Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2) Install dependencies
Run one of the following depending on your workflow:
- If you already have a `requirements.txt`, use:
  ```bash
  pip install -r requirements.txt
  ```
- Otherwise, install dependencies based on imports used in the codebase:
  - `flask`
  - `opencv-python`
  - `tensorflow`
  - `numpy`
  - `matplotlib`
  - `requests`

### 3) Start the web app
```bash
cd /home/tondamanati-abhinay/Desktop/capstone/Capstone
source venv/bin/activate
python3 app.py
```
Then open the shown Flask URL (usually `http://127.0.0.1:5000`).

---

## Notes / Limitations

- The UNet model loading paths inside scripts currently point to absolute paths like:
  - `/home/tondamanati-abhinay/Desktop/capstone/Capstone/DataSet/...`
- Interactive polygon selection opens an OpenCV window; this is intended for local execution.
- NASA POWER API is called live during analysis.

---

## Contributing

Contributions are welcome—especially around:
- replacing absolute dataset/model paths with relative/project paths
- adding a `requirements.txt`
- adding a proper `DataSet/` folder layout or downloader script

---

## License

(Add your license here.)

