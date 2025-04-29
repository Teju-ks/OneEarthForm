# Waste Nutrient Analysis & Plant Recommendations

A Streamlit application for analyzing waste nutrient content and providing plant-specific recommendations.

## Features
- Upload and analyze waste material datasets
- Predict nutrient content (Nitrogen, Phosphorus, Potassium)
- Get plant-specific recommendations
- Visualize nutrient levels
- Handle missing data with default values

## Deployment Instructions

### Option 1: Deploy on Streamlit Cloud (Recommended)

1. Create a GitHub account if you don't have one
2. Create a new repository on GitHub
3. Upload these files to your repository:
   - app.py
   - requirements.txt
   - plant_database.py
   - utils.py
4. Go to [Streamlit Cloud](https://streamlit.io/cloud)
5. Sign in with your GitHub account
6. Click "New app"
7. Select your repository and branch
8. Set the main file path to `app.py`
9. Click "Deploy"

### Option 2: Run Locally

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## File Structure
- `app.py`: Main application file
- `plant_database.py`: Plant nutrient requirements database
- `utils.py`: Utility functions
- `requirements.txt`: Python dependencies

## Requirements
- Python 3.8 or higher
- Dependencies listed in requirements.txt

## Features

- Upload your own waste material dataset
- Predict nutrient content (Nitrogen, Phosphorus, Potassium)
- Get plant-specific recommendations for:
  - Tomatoes
  - Leafy Greens
  - Root Vegetables
  - Fruit Trees
  - Grains
- Analyze nutrient suitability for different plant types
- Get detailed recommendations for improving nutrient content

## Installation

1. Install Python 3.12 or later
2. Install the required packages:
```bash
py -m pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
py -m streamlit run app.py
```

2. Upload your CSV dataset with the following columns:
   - Required columns:
     - Waste_Type (e.g., Food, Garden, Paper, Mixed, Agricultural)
     - Moisture_Content (%)
     - pH_Level
     - Carbon_Content (%)
     - Particle_Size_mm
     - Age_Days
     - Degradation_Rate
   - Optional columns:
     - Nitrogen_pct
     - Phosphorus_pct
     - Potassium_pct

3. Train the model using your dataset

4. Input waste characteristics to get predictions and recommendations

## Sample Dataset Format

```csv
Waste_Type,Moisture_Content,pH_Level,Carbon_Content,Particle_Size_mm,Age_Days,Degradation_Rate
Food,65.5,6.8,45.2,2.5,30,2.1
Garden,45.2,7.2,35.8,5.0,15,1.5
Paper,15.3,6.5,55.4,1.0,60,0.8
```

## Features Explanation

1. **Dataset Upload**: Upload your waste material dataset in CSV format.

2. **Model Training**: Train machine learning models to predict nutrient content based on waste characteristics.

3. **Nutrient Prediction**: Input waste characteristics to predict:
   - Nitrogen content (%)
   - Phosphorus content (%)
   - Potassium content (%)

4. **Plant Recommendations**: Get detailed analysis for different plant types:
   - Nutrient suitability
   - Optimal ranges
   - Specific recommendations for improvement

5. **General Recommendations**: Additional insights including:
   - C/N ratio analysis
   - Nutrient balance recommendations
   - Potential uses in agriculture

## Note

If your dataset doesn't include nutrient content (N, P, K), the application will generate synthetic values for demonstration purposes. For accurate predictions, include actual nutrient measurements in your dataset. 