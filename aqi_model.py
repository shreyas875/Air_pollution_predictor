import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
from alerts import AQIAlertSystem
from health_recommendations import HealthRecommendationSystem

def get_aqi_category(aqi):
    """Get AQI category based on value"""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def calculate_aqi(pollutants):
    """Calculate AQI based on EPA standards"""
    try:
        pm25, pm10, no2, so2, co, o3 = pollutants
        
        # Calculate individual AQI values
        aqi_values = []
        
        # PM2.5 (μg/m³)
        if pm25 <= 12.0:
            aqi = (50/12.0) * pm25
        elif pm25 <= 35.4:
            aqi = ((100-51)/(35.4-12.1)) * (pm25-12.1) + 51
        elif pm25 <= 55.4:
            aqi = ((150-101)/(55.4-35.5)) * (pm25-35.5) + 101
        elif pm25 <= 150.4:
            aqi = ((200-151)/(150.4-55.5)) * (pm25-55.5) + 151
        elif pm25 <= 250.4:
            aqi = ((300-201)/(250.4-150.5)) * (pm25-150.5) + 201
        else:
            aqi = ((500-301)/(500.4-250.5)) * (pm25-250.5) + 301
        aqi_values.append(aqi)
        
        # PM10 (μg/m³)
        if pm10 <= 54:
            aqi = (50/54) * pm10
        elif pm10 <= 154:
            aqi = ((100-51)/(154-55)) * (pm10-55) + 51
        elif pm10 <= 254:
            aqi = ((150-101)/(254-155)) * (pm10-155) + 101
        elif pm10 <= 354:
            aqi = ((200-151)/(354-255)) * (pm10-255) + 151
        elif pm10 <= 424:
            aqi = ((300-201)/(424-355)) * (pm10-355) + 201
        else:
            aqi = ((500-301)/(604-425)) * (pm10-425) + 301
        aqi_values.append(aqi)
        
        # NO2 (ppb)
        if no2 <= 53:
            aqi = (50/53) * no2
        elif no2 <= 100:
            aqi = ((100-51)/(100-54)) * (no2-54) + 51
        elif no2 <= 360:
            aqi = ((150-101)/(360-101)) * (no2-101) + 101
        elif no2 <= 649:
            aqi = ((200-151)/(649-361)) * (no2-361) + 151
        elif no2 <= 1249:
            aqi = ((300-201)/(1249-650)) * (no2-650) + 201
        else:
            aqi = ((500-301)/(2049-1250)) * (no2-1250) + 301
        aqi_values.append(aqi)
        
        # SO2 (ppb)
        if so2 <= 35:
            aqi = (50/35) * so2
        elif so2 <= 75:
            aqi = ((100-51)/(75-36)) * (so2-36) + 51
        elif so2 <= 185:
            aqi = ((150-101)/(185-76)) * (so2-76) + 101
        elif so2 <= 304:
            aqi = ((200-151)/(304-186)) * (so2-186) + 151
        else:
            aqi = ((300-201)/(604-305)) * (so2-305) + 201
        aqi_values.append(aqi)
        
        # CO (ppm)
        if co <= 4.4:
            aqi = (50/4.4) * co
        elif co <= 9.4:
            aqi = ((100-51)/(9.4-4.5)) * (co-4.5) + 51
        elif co <= 12.4:
            aqi = ((150-101)/(12.4-9.5)) * (co-9.5) + 101
        elif co <= 15.4:
            aqi = ((200-151)/(15.4-12.5)) * (co-12.5) + 151
        else:
            aqi = ((300-201)/(30.4-15.5)) * (co-15.5) + 201
        aqi_values.append(aqi)
        
        # O3 (ppb)
        if o3 <= 54:
            aqi = (50/54) * o3
        elif o3 <= 70:
            aqi = ((100-51)/(70-55)) * (o3-55) + 51
        elif o3 <= 85:
            aqi = ((150-101)/(85-71)) * (o3-71) + 101
        elif o3 <= 105:
            aqi = ((200-151)/(105-86)) * (o3-86) + 151
        else:
            aqi = ((300-201)/(200-106)) * (o3-106) + 201
        aqi_values.append(aqi)
        
        # Return the maximum AQI value
        max_aqi = max(aqi_values)
        return round(max_aqi)
        
    except Exception as e:
        print(f"Error calculating AQI: {str(e)}")
        return None

# Load and prepare the dataset
data_path = os.path.join('data', 'air_quality_data.csv')
try:
    df = pd.read_csv(data_path)
    print("Successfully loaded the dataset")
except FileNotFoundError:
    print(f"Error: Could not find the dataset at {data_path}")
    print("Please make sure the data directory and air_quality_data.csv file exist")
    raise
except Exception as e:
    print(f"Error loading the dataset: {str(e)}")
    raise

# Drop rows with missing values
df.dropna(inplace=True)

# Encode categorical variables
le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])

# Calculate standard AQI for training data
df['aqi'] = df.apply(lambda row: calculate_aqi([
    row['pm2.5'], row['pm10'], row['no2'], 
    row['so2'], row['co'], row['o3']
]), axis=1)

# Prepare features
feature_columns = [
    'city_encoded',
    'pm2.5', 'pm10', 'no2', 'so2', 'co', 'o3'
]

X = df[feature_columns]
y = df['aqi']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Create and train the model using GridSearchCV
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='neg_mean_squared_error'
)
grid_search.fit(X_train_scaled, y_train)

# Get the best model
model = grid_search.best_estimator_

# Evaluate the model
train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

print("\nModel Performance:")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Train R² score: {r2_score(y_train, train_pred):.4f}")
print(f"Test R² score: {r2_score(y_test, test_pred):.4f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, train_pred)):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, test_pred)):.4f}")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the model, encoder, and scaler
model_path = os.path.join('models', 'aqi_model.joblib')
encoder_path = os.path.join('models', 'label_encoder.joblib')
scaler_path = os.path.join('models', 'scaler.joblib')
joblib.dump(model, model_path)
joblib.dump(le, encoder_path)
joblib.dump(scaler, scaler_path)
print("\nModel, encoder, and scaler saved successfully")

def validate_pollutant_values(pollutants):
    """Validate and clean pollutant values to ensure they are within realistic ranges"""
    pm25, pm10, no2, so2, co, o3 = pollutants
    
    # Define realistic ranges for each pollutant
    ranges = {
        'pm25': (0, 500),    # μg/m³
        'pm10': (0, 600),    # μg/m³
        'no2': (0, 2000),    # ppb
        'so2': (0, 1000),    # ppb
        'co': (0, 50),       # ppm
        'o3': (0, 500)       # ppb
    }
    
    # Clip values to realistic ranges
    cleaned_values = [
        max(0, min(pm25, ranges['pm25'][1])),
        max(0, min(pm10, ranges['pm10'][1])),
        max(0, min(no2, ranges['no2'][1])),
        max(0, min(so2, ranges['so2'][1])),
        max(0, min(co, ranges['co'][1])),
        max(0, min(o3, ranges['o3'][1]))
    ]
    
    return cleaned_values

def get_health_recommendations(aqi):
    """Get health recommendations based on AQI value"""
    recommendations = {
        'outdoor_activities': '',
        'sensitive_groups': '',
        'mask_recommendation': '',
        'ventilation': ''
    }
    
    if aqi <= 50:
        recommendations.update({
            'outdoor_activities': 'Safe for outdoor activities',
            'sensitive_groups': 'No special precautions needed',
            'mask_recommendation': 'No mask required for general public',
            'ventilation': 'Safe to keep windows open'
        })
    elif aqi <= 100:
        recommendations.update({
            'outdoor_activities': 'Moderate activities acceptable',
            'sensitive_groups': 'Unusually sensitive people should consider reducing prolonged outdoor exertion',
            'mask_recommendation': 'Optional for sensitive individuals',
            'ventilation': 'OK to keep windows open, consider closing during peak traffic hours'
        })
    elif aqi <= 150:
        recommendations.update({
            'outdoor_activities': 'Reduce prolonged or heavy outdoor exertion',
            'sensitive_groups': 'People with respiratory issues should limit outdoor activities',
            'mask_recommendation': 'Recommended for sensitive groups when outdoors',
            'ventilation': 'Close windows during peak pollution hours'
        })
    elif aqi <= 200:
        recommendations.update({
            'outdoor_activities': 'Avoid prolonged outdoor activities',
            'sensitive_groups': 'Sensitive groups should avoid outdoor activities',
            'mask_recommendation': 'Wear masks when outdoors',
            'ventilation': 'Keep windows closed, use air purifiers if available'
        })
    elif aqi <= 300:
        recommendations.update({
            'outdoor_activities': 'Avoid all outdoor activities',
            'sensitive_groups': 'Stay indoors and keep activity levels low',
            'mask_recommendation': 'Wear N95 masks if going outdoors is necessary',
            'ventilation': 'Keep windows sealed, use air purifiers'
        })
    else:
        recommendations.update({
            'outdoor_activities': 'Stay indoors',
            'sensitive_groups': 'Medical attention may be needed for sensitive groups',
            'mask_recommendation': 'Wear N95 masks if going outdoors is absolutely necessary',
            'ventilation': 'Seal all windows, use air purifiers, consider temporary relocation'
        })
    
    return recommendations

def save_historical_data(city, prediction_data):
    """Save prediction data for historical tracking"""
    try:
        history_dir = os.path.join('data', 'history')
        os.makedirs(history_dir, exist_ok=True)
        
        history_file = os.path.join(history_dir, f'{city.lower()}_history.csv')
        
        # Prepare data for saving
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data = {
            'timestamp': current_time,
            'aqi': prediction_data['aqi'],
            'category': prediction_data['category'],
            **prediction_data['pollutants']
        }
        
        # Create or append to CSV
        if os.path.exists(history_file):
            df = pd.read_csv(history_file)
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        else:
            df = pd.DataFrame([data])
        
        # Save to CSV
        df.to_csv(history_file, index=False)
        return True
    except Exception as e:
        print(f"Error saving historical data: {str(e)}")
        return False

def get_historical_stats(city):
    """Get historical AQI statistics for a city"""
    try:
        history_file = os.path.join('data', 'history', f'{city.lower()}_history.csv')
        if not os.path.exists(history_file):
            return None
            
        df = pd.read_csv(history_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate statistics
        stats = {
            'last_24h': {
                'mean_aqi': df[df['timestamp'] > datetime.now() - timedelta(days=1)]['aqi'].mean(),
                'max_aqi': df[df['timestamp'] > datetime.now() - timedelta(days=1)]['aqi'].max(),
                'min_aqi': df[df['timestamp'] > datetime.now() - timedelta(days=1)]['aqi'].min(),
            },
            'last_week': {
                'mean_aqi': df[df['timestamp'] > datetime.now() - timedelta(days=7)]['aqi'].mean(),
                'max_aqi': df[df['timestamp'] > datetime.now() - timedelta(days=7)]['aqi'].max(),
                'min_aqi': df[df['timestamp'] > datetime.now() - timedelta(days=7)]['aqi'].min(),
            },
            'most_frequent_category': df['category'].mode().iloc[0] if not df.empty else None,
            'trend': 'improving' if df['aqi'].iloc[-1] < df['aqi'].mean() else 'worsening'
        }
        
        return stats
    except Exception as e:
        print(f"Error getting historical stats: {str(e)}")
        return None

def predict_aqi(city, pollutants):
    """Predict AQI for a given city and pollutant values"""
    try:
        # Validate pollutant values
        pollutants = validate_pollutant_values(pollutants)
        
        # Load the model and label encoder
        model_path = os.path.join('models', 'aqi_model.joblib')
        le_path = os.path.join('models', 'label_encoder.joblib')
        
        try:
            model = joblib.load(model_path)
            le = joblib.load(le_path)
        except FileNotFoundError:
            print("Model files not found. Using standard AQI calculation.")
            return calculate_aqi(pollutants)
        
        # Transform city name
        city_encoded = le.transform([city])[0]
        
        # Create feature vector
        features = np.array([[city_encoded] + pollutants])
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Get prediction
        predicted_aqi = model.predict(features_scaled)[0]
        
        # Calculate standard AQI
        standard_aqi = calculate_aqi(pollutants)
        
        # Use weighted average of both methods
        final_aqi = 0.7 * predicted_aqi + 0.3 * standard_aqi
        
        # Create pollutants dictionary
        pollutants_dict = {
            'pm2.5': pollutants[0],
            'pm10': pollutants[1],
            'no2': pollutants[2],
            'so2': pollutants[3],
            'co': pollutants[4],
            'o3': pollutants[5]
        }
        
        # Get health recommendations
        health_system = HealthRecommendationSystem()
        health_recommendations = health_system.get_detailed_health_recommendations(final_aqi, pollutants_dict)
        
        # Save historical data
        save_historical_data(city, final_aqi, pollutants_dict)
        
        return final_aqi, health_recommendations
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None, None

def predict_24h_aqi(city, current_pollutants):
    """Predict AQI for the next 24 hours"""
    try:
        predictions = []
        health_system = HealthRecommendationSystem()
        
        # Validate initial pollutant values
        current_pollutants = validate_pollutant_values(current_pollutants)
        
        # Get base AQI using standard calculation only
        base_aqi = calculate_aqi(current_pollutants)
        
        # Define maximum allowed variation from base AQI
        max_variation = 0.20  # 20%
        
        # Define hourly patterns with much more conservative factors
        hourly_patterns = {
            'morning_rush': {'hours': range(7, 10), 'factor': {'pm': 1.02, 'gas': 1.01}},
            'midday': {'hours': range(10, 16), 'factor': {'pm': 0.99, 'gas': 1.0}},
            'evening_rush': {'hours': range(16, 20), 'factor': {'pm': 1.03, 'gas': 1.02}},
            'night': {'hours': range(20, 24), 'factor': {'pm': 0.98, 'gas': 0.99}},
            'early_morning': {'hours': range(0, 7), 'factor': {'pm': 0.97, 'gas': 0.98}}
        }
        
        current_hour = datetime.now().hour
        
        for hour in range(24):
            prediction_hour = (current_hour + hour) % 24
            
            # Determine which pattern to use
            pattern = None
            for time_of_day, pattern_data in hourly_patterns.items():
                if prediction_hour in pattern_data['hours']:
                    pattern = pattern_data['factor']
                    break
            if pattern is None:
                pattern = {'pm': 1.0, 'gas': 1.0}
            
            # Apply variations to different pollutants with bounds checking
            varied_pollutants = []
            for i, pollutant in enumerate(current_pollutants):
                # Apply appropriate factor (PM or gas)
                factor = pattern['pm'] if i < 2 else pattern['gas']
                # Calculate new value
                new_value = pollutant * factor
                # Add to list
                varied_pollutants.append(new_value)
            
            # Calculate AQI directly using standard method
            predicted_aqi = calculate_aqi(varied_pollutants)
            
            # Calculate bounds
            min_aqi = max(base_aqi * (1 - max_variation), 0)
            max_aqi = min(base_aqi * (1 + max_variation), 500)
            
            # Adjust AQI if it's outside bounds
            if predicted_aqi < min_aqi:
                predicted_aqi = min_aqi
            elif predicted_aqi > max_aqi:
                predicted_aqi = max_aqi
            
            # Create pollutants dictionary
            pollutants_dict = {
                'pm2.5': varied_pollutants[0],
                'pm10': varied_pollutants[1],
                'no2': varied_pollutants[2],
                'so2': varied_pollutants[3],
                'co': varied_pollutants[4],
                'o3': varied_pollutants[5]
            }
            
            # Get health recommendations
            health_recommendations = health_system.get_detailed_health_recommendations(predicted_aqi, pollutants_dict)
            
            # Add timestamp
            timestamp = (datetime.now() + timedelta(hours=hour)).strftime('%Y-%m-%d %H:%M')
            
            # Add to predictions
            predictions.append({
                'hour': hour,
                'aqi': predicted_aqi,
                'category': get_aqi_category(predicted_aqi),
                'timestamp': timestamp,
                'pollutants': pollutants_dict,
                'health_recommendations': health_recommendations
            })
        
        return predictions
    except Exception as e:
        print(f"Error in 24h prediction: {str(e)}")
        return None

# Initialize the alert system
alert_system = AQIAlertSystem()

# Add subscribers
alert_system.add_subscriber(
    email="user@example.com",
    city="Mumbai",
    threshold=150  # Will be notified when AQI exceeds 150
)

# Example usage
if __name__ == "__main__":
    # Test with more realistic pollutant values
    city = 'Mumbai'
    # More realistic example values (μg/m³ or ppb)
    pollutants = [
        30,    # PM2.5 (μg/m³)
        50,    # PM10 (μg/m³)
        40,    # NO2 (ppb)
        15,    # SO2 (ppb)
        1.5,   # CO (ppm)
        35     # O3 (ppb)
    ]
    
    # Test current AQI prediction with enhanced features
    current_prediction = predict_aqi(city, pollutants)
    if current_prediction:
        print(f"\nCurrent AQI prediction for {city}:")
        print(f"AQI: {current_prediction['aqi']}")
        print(f"Category: {current_prediction['category']}")
        
        print("\nPollutant values used:")
        for pollutant, value in current_prediction['pollutants'].items():
            print(f"{pollutant}: {value}")
        
        print("\nHealth Recommendations:")
        for category, recommendation in current_prediction['health_recommendations'].items():
            print(f"{category.replace('_', ' ').title()}: {recommendation}")
        
        if current_prediction['historical_stats']:
            print("\nHistorical Statistics:")
            stats = current_prediction['historical_stats']
            print("Last 24 Hours:")
            print(f"  Mean AQI: {stats['last_24h']['mean_aqi']:.1f}")
            print(f"  Max AQI: {stats['last_24h']['max_aqi']:.1f}")
            print(f"  Min AQI: {stats['last_24h']['min_aqi']:.1f}")
            print("\nLast Week:")
            print(f"  Mean AQI: {stats['last_week']['mean_aqi']:.1f}")
            print(f"  Max AQI: {stats['last_week']['max_aqi']:.1f}")
            print(f"  Min AQI: {stats['last_week']['min_aqi']:.1f}")
            print(f"\nMost Frequent Category: {stats['most_frequent_category']}")
            print(f"Overall Trend: {stats['trend']}")
    
    # Test 24h predictions
    predictions_24h = predict_24h_aqi(city, pollutants)
    if predictions_24h:
        print("\n24-hour AQI predictions:")
        for pred in predictions_24h:
            print(f"Time: {pred['timestamp']}")
            print(f"AQI: {pred['aqi']} - {pred['category']}")
            print("Pollutant values:")
            for pollutant, value in pred['pollutants'].items():
                print(f"{pollutant}: {value:.1f}")
            print("Health Recommendations:")
            for category, recommendation in pred['health_recommendations'].items():
                print(f"{category.replace('_', ' ').title()}: {recommendation}")
            print("---")
