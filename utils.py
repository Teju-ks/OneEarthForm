import numpy as np
import pandas as pd

def analyze_nutrient_suitability(predictions, plant_info):
    """Analyze how suitable the predicted nutrients are for a specific plant type."""
    suitability = {}
    recommendations = []
    
    for nutrient in ['Nitrogen', 'Phosphorus', 'Potassium']:
        if f'{nutrient}_pct' in predictions:
            value = predictions[f'{nutrient}_pct']
            req = plant_info[nutrient]
            
            if value < req['min']:
                status = 'Low'
                action = f'Increase {nutrient}'
            elif value > req['max']:
                status = 'High'
                action = f'Reduce {nutrient}'
            else:
                status = 'Optimal'
                action = 'Maintain levels'
                
            suitability[nutrient] = {
                'status': status,
                'current': value,
                'optimal': req['optimal'],
                'action': action
            }
            
            if status != 'Optimal':
                if status == 'Low':
                    recommendations.append(f"- {nutrient} is low. Consider supplementing with {nutrient}-rich materials.")
                else:
                    recommendations.append(f"- {nutrient} is high. Consider mixing with lower {nutrient} materials.")
    
    return suitability, recommendations

def generate_synthetic_targets(data):
    """Generate synthetic target variables based on waste characteristics."""
    data['Nitrogen_pct'] = (
        (data['Waste_Type'] == 'Food') * np.random.uniform(1.5, 4.0, len(data)) +
        (data['Waste_Type'] == 'Garden') * np.random.uniform(0.5, 2.0, len(data)) +
        data['Moisture_Content'] * 0.01 - data['pH_Level'] * 0.1
    ).clip(0.05, 5.0)
    
    data['Phosphorus_pct'] = (
        (data['Waste_Type'] == 'Food') * np.random.uniform(0.3, 1.2, len(data)) +
        data['Carbon_Content'] * 0.005
    ).clip(0.01, 2.0)
    
    data['Potassium_pct'] = (
        (data['Waste_Type'] == 'Food') * np.random.uniform(1.0, 2.5, len(data)) +
        data['Moisture_Content'] * 0.008
    ).clip(0.05, 3.0)
    
    return data 