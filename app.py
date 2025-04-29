import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from plant_database import plant_requirements
from utils import analyze_nutrient_suitability, generate_synthetic_targets

# Set page configuration
st.set_page_config(
    page_title="Waste Nutrient Analysis & Plant Recommendations",
    page_icon="🌱",
    layout="wide"
)

# Default values for missing columns
DEFAULT_VALUES = {
    'Waste_Type': 'Mixed',
    'Moisture_Content': 50.0,
    'pH_Level': 6.5,
    'Carbon_Content': 35.0,
    'Particle_Size_mm': 10.0,
    'Age_Days': 30,
    'Degradation_Rate': 1.0
}

# Mapping for categorical values to numeric
CATEGORICAL_MAPPINGS = {
    'Moisture_Content': {'Low': 30.0, 'Medium': 50.0, 'High': 70.0},
    'pH_Level': {'Low': 4.5, 'Medium': 6.5, 'High': 8.5},
    'Carbon_Content': {'Low': 25.0, 'Medium': 35.0, 'High': 45.0},
    'Particle_Size_mm': {'Small': 2.0, 'Medium': 10.0, 'Large': 25.0},
    'Age_Days': {'Young': 15, 'Medium': 30, 'Old': 60},
    'Degradation_Rate': {'Slow': 0.5, 'Medium': 1.0, 'Fast': 2.0}
}

def convert_categorical_to_numeric(data):
    """Convert categorical values to numeric using predefined mappings."""
    for col, mapping in CATEGORICAL_MAPPINGS.items():
        if col in data.columns and data[col].dtype == 'object':
            if data[col].isin(mapping.keys()).any():
                data[col] = data[col].map(mapping)
                st.info(f"Converted categorical values in '{col}' to numeric using mapping: {mapping}")
    return data

def handle_missing_data(data):
    """Handle missing columns and values with user guidance."""
    missing_columns = []
    missing_values = {}
    
    # Check for missing columns
    for col, default_value in DEFAULT_VALUES.items():
        if col not in data.columns:
            missing_columns.append(col)
            data[col] = default_value
    
    # Check for missing values in numeric columns
    numeric_columns = ['Moisture_Content', 'pH_Level', 'Carbon_Content', 
                      'Particle_Size_mm', 'Age_Days', 'Degradation_Rate']
    for col in numeric_columns:
        if col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                if data[col].isna().any():
                    missing_values[col] = data[col].isna().sum()
                    data[col] = data[col].fillna(data[col].mean())
            except Exception as e:
                st.error(f"Error converting '{col}' to numeric: {str(e)}")
                data[col] = DEFAULT_VALUES[col]
    
    # Show guidance message
    if missing_columns or missing_values:
        st.warning("""
        ### Data Quality Notice
        
        Your dataset is missing some information. The app will use default values for:
        """)
        
        if missing_columns:
            st.write("**Missing Columns:**")
            for col in missing_columns:
                st.write(f"- {col}: Using default value of {DEFAULT_VALUES[col]}")
        
        if missing_values:
            st.write("**Missing Values:**")
            for col, count in missing_values.items():
                st.write(f"- {col}: {count} missing values filled with mean ({data[col].mean():.2f})")
        
        st.write("""
        For more accurate results, please provide complete data with all columns:
        - Waste_Type
        - Moisture_Content (%)
        - pH_Level
        - Carbon_Content (%)
        - Particle_Size_mm
        - Age_Days
        - Degradation_Rate
        """)
    
    return data

# Main app title
st.title("🌱 Waste Nutrient Analysis & Plant Recommendations")
st.write("""
Upload your waste material dataset to analyze nutrient content and get plant-specific recommendations 
for optimal use in agriculture. The app will work with any available data and use default values for missing information.
""")

# File upload
st.subheader("Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
st.info("""
Your CSV file can include any of these columns (all are optional, defaults will be used for missing data):
- Waste_Type (e.g., Food, Garden, Paper, Mixed, Agricultural)
- Moisture_Content (numeric or categorical: Low/Medium/High)
- pH_Level (numeric or categorical: Low/Medium/High)
- Carbon_Content (numeric or categorical: Low/Medium/High)
- Particle_Size_mm (numeric or categorical: Small/Medium/Large)
- Age_Days (numeric or categorical: Young/Medium/Old)
- Degradation_Rate (numeric or categorical: Slow/Medium/Fast)
- Nitrogen_pct
- Phosphorus_pct
- Potassium_pct
""")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("Dataset successfully loaded!")
        
        # Display dataset overview
        st.subheader("Dataset Overview")
        st.write(data.head())
        
        # Convert categorical values to numeric
        data = convert_categorical_to_numeric(data)
        
        # Handle missing data with user guidance
        data = handle_missing_data(data)
        
        # Model training section
        st.subheader("Model Training")
        
        test_size = st.slider("Test Data Size (%)", 10, 50, 20) / 100
        
        if st.button("Train Model"):
            # Prepare features
            X = data[list(DEFAULT_VALUES.keys())]
            
            # Define target variables
            target_variables = ['Nitrogen_pct', 'Phosphorus_pct', 'Potassium_pct']
            
            # Create synthetic target variables if not present
            if not all(target in data.columns for target in target_variables):
                st.warning("Target variables not found in dataset. Creating synthetic targets for demonstration.")
                data = generate_synthetic_targets(data)
            
            # Train models
            models = {}
            for target in target_variables:
                y = data[target]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Create and train model
                numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
                categorical_features = X.select_dtypes(include=['object']).columns
                
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numeric_features),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                    ])
                
                model = Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
                ])
                
                model.fit(X_train, y_train)
                models[target] = model
            
            st.session_state.models = models
            st.success("Models trained successfully!")
        
        # Prediction section
        if 'models' in st.session_state:
            st.subheader("Nutrient Prediction")
            
            # Input form for prediction
            col1, col2 = st.columns(2)
            
            with col1:
                waste_type = st.selectbox("Waste Type", 
                                        options=['Food', 'Garden', 'Paper', 'Mixed', 'Agricultural'],
                                        index=['Food', 'Garden', 'Paper', 'Mixed', 'Agricultural'].index(data['Waste_Type'].mode()[0]))
                moisture_content = st.slider("Moisture Content (%)", 10.0, 90.0, float(data['Moisture_Content'].mean()))
                ph_level = st.slider("pH Level", 3.5, 9.0, float(data['pH_Level'].mean()))
                carbon_content = st.slider("Carbon Content (%)", 10.0, 60.0, float(data['Carbon_Content'].mean()))
            
            with col2:
                particle_size = st.slider("Particle Size (mm)", 0.5, 50.0, float(data['Particle_Size_mm'].mean()))
                age_days = st.slider("Age (Days)", 1, 365, int(data['Age_Days'].mean()))
                degradation_rate = st.slider("Degradation Rate", 0.1, 5.0, float(data['Degradation_Rate'].mean()))
            
            # Create input data for prediction
            input_data = pd.DataFrame({
                'Waste_Type': [waste_type],
                'Moisture_Content': [moisture_content],
                'pH_Level': [ph_level],
                'Carbon_Content': [carbon_content],
                'Particle_Size_mm': [particle_size],
                'Age_Days': [age_days],
                'Degradation_Rate': [degradation_rate]
            })
            
            if st.button("Predict and Analyze"):
                # Make predictions
                predictions = {}
                for target, model in st.session_state.models.items():
                    predictions[target] = model.predict(input_data)[0]
                
                # Display predictions
                st.subheader("Predicted Nutrient Content")
                
                # Create visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                nutrients = list(predictions.keys())
                values = list(predictions.values())
                
                bars = ax.bar(nutrients, values, color=['#2ecc71', '#3498db', '#e74c3c'])
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.2f}%',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom')
                
                plt.title('Predicted Nutrient Percentages')
                plt.ylabel('Percentage (%)')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Plant recommendations
                st.subheader("Plant Recommendations")
                
                # Analyze suitability for each plant type
                for plant_type, plant_info in plant_requirements.items():
                    st.write(f"### {plant_type}")
                    suitability, recommendations = analyze_nutrient_suitability(predictions, plant_info)
                    
                    # Create three columns for each nutrient
                    cols = st.columns(3)
                    
                    # Display nutrient status
                    for idx, (nutrient, details) in enumerate(suitability.items()):
                        with cols[idx]:
                            st.metric(
                                label=nutrient,
                                value=f"{details['current']:.2f}%",
                                delta=f"{details['current'] - details['optimal']:.2f}% from optimal"
                            )
                    
                    # Display plant-specific description
                    st.info(plant_info['description'])
                    
                    # Display recommendations if any
                    if recommendations:
                        st.write("Recommendations:")
                        for rec in recommendations:
                            st.write(rec)
                    else:
                        st.success("✅ Nutrient levels are optimal for this plant type!")
                    
                    st.markdown("---")
                
                # Additional recommendations
                st.subheader("General Recommendations")
                c_n_ratio = carbon_content / predictions['Nitrogen_pct']
                
                if c_n_ratio > 30:
                    st.warning(f"C/N ratio is high ({c_n_ratio:.1f}). Consider adding nitrogen-rich materials.")
                elif c_n_ratio < 15:
                    st.warning(f"C/N ratio is low ({c_n_ratio:.1f}). Consider adding carbon-rich materials.")
                else:
                    st.success(f"C/N ratio is optimal ({c_n_ratio:.1f}).")
                
    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")
else:
    st.info("Please upload a CSV file to begin analysis.")

# Footer
st.markdown("---")
st.markdown("🌱 Waste Nutrient Analysis & Plant Recommendations Tool | Created with Streamlit")