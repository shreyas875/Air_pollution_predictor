from flask import Flask, render_template, request, jsonify, redirect
import requests
import os
from aqi_model import predict_24h_aqi
from health_recommendations import HealthRecommendationSystem

app = Flask(__name__)

# OpenWeatherMap Configuration
OPENWEATHER_API_KEY = '548db7d3ad6aead433435e4073e5add6'
OPENWEATHER_API_URL = 'http://api.openweathermap.org/data/2.5/air_pollution'
GOOGLE_MAPS_API_KEY = 'AIzaSyDgCcN6ZLZaKlwnmCIgTUcaLKfzNQnRIQg'

def convert_to_standard_aqi(openweather_aqi):
    """Convert OpenWeatherMap's 1-5 AQI scale to standard 0-500 AQI scale"""
    print(f"\nDEBUG - Converting AQI value: {openweather_aqi}")
    aqi_ranges = {
        1: (0, 50),      # Good
        2: (51, 100),    # Fair
        3: (101, 150),   # Moderate
        4: (151, 200),   # Poor
        5: (201, 300)    # Very Poor
    }
    
    if openweather_aqi in aqi_ranges:
        min_val, max_val = aqi_ranges[openweather_aqi]
        # Use the middle value of the range
        converted_aqi = (min_val + max_val) // 2
        print(f"DEBUG - Converted AQI: {converted_aqi} (range: {min_val}-{max_val})")
        return converted_aqi
    print(f"DEBUG - Invalid AQI value: {openweather_aqi}, defaulting to 0")
    return 0  # Default to 0 if invalid value

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/location')
def location():
    return render_template('location.html', google_maps_api_key=GOOGLE_MAPS_API_KEY)

@app.route('/manual_location')
def manual_location():
    return render_template('manual_location.html', google_maps_api_key=GOOGLE_MAPS_API_KEY)

def get_aqi_description(aqi_value):
    """Convert AQI value to description"""
    print(f"\nDEBUG - Getting description for AQI: {aqi_value}")
    if aqi_value <= 50:
        desc = "Good"
    elif aqi_value <= 100:
        desc = "Moderate"
    elif aqi_value <= 150:
        desc = "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        desc = "Unhealthy"
    elif aqi_value <= 300:
        desc = "Very Unhealthy"
    else:
        desc = "Hazardous"
    print(f"DEBUG - Description: {desc}")
    return desc

def get_aqi_color(aqi_value):
    """Get color class based on AQI value"""
    print(f"\nDEBUG - Getting color for AQI: {aqi_value}")
    if aqi_value <= 50:
        color = "text-green-500"
    elif aqi_value <= 100:
        color = "text-yellow-400"
    elif aqi_value <= 150:
        color = "text-orange-500"
    elif aqi_value <= 200:
        color = "text-red-500"
    elif aqi_value <= 300:
        color = "text-purple-600"
    else:
        color = "text-red-700"
    print(f"DEBUG - Color: {color}")
    return color

def convert_units(components):
    """Convert pollutant values from μg/m³ to required units (ppb or ppm)"""
    # Molecular weights (g/mol)
    MW_NO2 = 46.0055
    MW_SO2 = 64.066
    MW_CO = 28.01
    MW_O3 = 48.0

    # Convert from μg/m³ to ppb or ppm
    # ppb = (μg/m³ * 24.45) / MW  at 25°C and 1 atm
    # ppm = ppb / 1000
    return [
        components['pm2_5'],        # PM2.5 stays in μg/m³
        components['pm10'],         # PM10 stays in μg/m³
        (components['no2'] * 24.45) / MW_NO2,    # NO2 to ppb
        (components['so2'] * 24.45) / MW_SO2,    # SO2 to ppb
        ((components['co'] * 24.45) / MW_CO) / 1000,  # CO to ppm
        (components['o3'] * 24.45) / MW_O3       # O3 to ppb
    ]

def get_pollutant_info():
    """Get detailed information about each pollutant"""
    return {
        'pm2_5': {
            'name': 'PM2.5 (Fine Particulate Matter)',
            'description': 'Tiny particles or droplets in the air that are 2.5 microns or less in width.',
            'sources': [
                'Vehicle emissions',
                'Power plants',
                'Industrial processes',
                'Wood burning',
                'Forest fires'
            ],
            'health_effects': [
                'Can penetrate deep into lungs and bloodstream',
                'Respiratory issues',
                'Heart problems',
                'Decreased lung function',
                'Aggravated asthma'
            ],
            'safe_levels': 'EPA standard: 12 μg/m³ (annual average)',
            'protection': [
                'Use air purifiers with HEPA filters',
                'Wear N95 masks when air quality is poor',
                'Keep windows closed during high pollution periods'
            ]
        },
        'pm10': {
            'name': 'PM10 (Coarse Particulate Matter)',
            'description': 'Inhalable particles with diameters of 10 micrometers or less.',
            'sources': [
                'Dust from construction',
                'Agricultural operations',
                'Road dust',
                'Mining operations',
                'Pollen and mold'
            ],
            'health_effects': [
                'Respiratory irritation',
                'Coughing and sneezing',
                'Aggravation of asthma',
                'Decreased lung capacity',
                'Bronchitis'
            ],
            'safe_levels': 'EPA standard: 50 μg/m³ (annual average)',
            'protection': [
                'Stay indoors during dust storms',
                'Use air conditioning with good filters',
                'Regular cleaning to reduce dust'
            ]
        },
        'no2': {
            'name': 'NO₂ (Nitrogen Dioxide)',
            'description': 'Reddish-brown gas with a pungent, acrid odor.',
            'sources': [
                'Vehicle exhaust',
                'Power plants',
                'Industrial emissions',
                'Gas stoves',
                'Home heating'
            ],
            'health_effects': [
                'Respiratory inflammation',
                'Reduced lung function',
                'Increased asthma attacks',
                'Development of respiratory diseases',
                'Eye and throat irritation'
            ],
            'safe_levels': 'EPA standard: 53 ppb (annual average)',
            'protection': [
                'Proper ventilation when using gas appliances',
                'Regular maintenance of combustion appliances',
                'Avoid heavy traffic areas'
            ]
        },
        'so2': {
            'name': 'SO₂ (Sulfur Dioxide)',
            'description': 'Colorless gas with a sharp, pungent odor.',
            'sources': [
                'Fossil fuel combustion',
                'Industrial processes',
                'Volcanic eruptions',
                'Ships and large vessels',
                'Metal smelting'
            ],
            'health_effects': [
                'Breathing difficulties',
                'Eye irritation',
                'Aggravation of asthma',
                'Respiratory tract problems',
                'Contribution to acid rain'
            ],
            'safe_levels': 'EPA standard: 75 ppb (1-hour average)',
            'protection': [
                'Avoid areas with high industrial activity',
                'Use air purifiers with activated carbon',
                'Stay informed about industrial emissions in your area'
            ]
        },
        'co': {
            'name': 'CO (Carbon Monoxide)',
            'description': 'Colorless, odorless gas that can be deadly at high levels.',
            'sources': [
                'Vehicle exhaust',
                'Indoor heating systems',
                'Industrial processes',
                'Cigarette smoke',
                'Blocked chimneys'
            ],
            'health_effects': [
                'Reduced oxygen delivery to organs',
                'Headaches and dizziness',
                'Nausea and vomiting',
                'Confusion and disorientation',
                'Can be fatal at high concentrations'
            ],
            'safe_levels': 'EPA standard: 9 ppm (8-hour average)',
            'protection': [
                'Install CO detectors',
                'Regular maintenance of heating systems',
                'Never run engines in enclosed spaces',
                'Ensure proper ventilation'
            ]
        },
        'o3': {
            'name': 'O₃ (Ozone)',
            'description': 'Reactive gas composed of three oxygen atoms.',
            'sources': [
                'Formed by chemical reactions between NOx and VOCs',
                'Vehicle exhaust',
                'Industrial emissions',
                'Chemical solvents',
                'Natural sources'
            ],
            'health_effects': [
                'Chest pain and coughing',
                'Throat irritation',
                'Airway inflammation',
                'Reduced lung function',
                'Aggravation of asthma'
            ],
            'safe_levels': 'EPA standard: 70 ppb (8-hour average)',
            'protection': [
                'Avoid outdoor activities during peak hours',
                'Stay indoors during high ozone alerts',
                'Use air conditioning with good filtration'
            ]
        }
    }

def get_location_name(lat, lon):
    """Format coordinates into a readable string"""
    try:
        # Convert coordinates to float for proper formatting
        lat_float = float(lat)
        lon_float = float(lon)
        
        # Determine N/S and E/W
        lat_dir = "N" if lat_float >= 0 else "S"
        lon_dir = "E" if lon_float >= 0 else "W"
        
        # Format the coordinates with absolute values and directions
        return f"{abs(lat_float):.6f}°{lat_dir}, {abs(lon_float):.6f}°{lon_dir}"
        
    except Exception as e:
        print(f"DEBUG - Error formatting coordinates: {str(e)}")
        return f"{lat}°, {lon}°"

@app.route('/results')
def results():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    
    if not lat or not lon:
        return "Location data is missing", 400
        
    try:
        # Get location name using geocoding
        location_name = get_location_name(lat, lon)
        print(f"DEBUG - Location name from geocoding: {location_name}")
        
        # Make the API request
        response = requests.get(
            OPENWEATHER_API_URL,
            params={
                'lat': lat,
                'lon': lon,
                'appid': OPENWEATHER_API_KEY
            }
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the JSON response
        data = response.json()
        
        # Validate the response structure
        if not data or 'list' not in data or not data['list']:
            raise ValueError("Invalid response format from OpenWeatherMap API")

        air_data = data['list'][0]
        
        # Get AQI from the API response (1-5 scale) and convert to standard scale
        openweather_aqi = air_data['main']['aqi']
        print(f"\nDEBUG - Raw AQI from API: {openweather_aqi}")
        aqi_value = convert_to_standard_aqi(openweather_aqi)
        print(f"DEBUG - Final AQI value: {aqi_value}")
        
        # Get AQI description and color based on standard scale
        aqi_description = get_aqi_description(aqi_value)
        aqi_color = get_aqi_color(aqi_value)
        
        # Get pollutant components
        components = air_data['components']
        
        # Convert units for display
        pollutants = {
            'co': {'name': 'Carbon Monoxide (CO)', 'value': components['co'], 'unit': 'μg/m³'},
            'no2': {'name': 'Nitrogen Dioxide (NO₂)', 'value': components['no2'], 'unit': 'μg/m³'},
            'o3': {'name': 'Ozone (O₃)', 'value': components['o3'], 'unit': 'μg/m³'},
            'so2': {'name': 'Sulfur Dioxide (SO₂)', 'value': components['so2'], 'unit': 'μg/m³'},
            'pm2_5': {'name': 'PM2.5', 'value': components['pm2_5'], 'unit': 'μg/m³'},
            'pm10': {'name': 'PM10', 'value': components['pm10'], 'unit': 'μg/m³'}
        }

        # Get detailed pollutant information
        pollutant_info = get_pollutant_info()
        
        # Convert pollutant values to appropriate units for AQI calculation
        current_pollutants = convert_units(components)
        print(f"DEBUG - Converted pollutant values: {current_pollutants}")
        
        # Get health recommendations
        health_system = HealthRecommendationSystem()
        pollutants_dict = {
            'pm2_5': components['pm2_5'],
            'pm10': components['pm10'],
            'no2': components['no2'],
            'so2': components['so2'],
            'co': components['co'],
            'o3': components['o3']
        }
        health_recommendations = health_system.get_detailed_health_recommendations(aqi_value, pollutants_dict)
        print(f"DEBUG - Health recommendations: {health_recommendations}")
        
        # Get 24-hour predictions using the location name from geocoding
        predictions_24h = predict_24h_aqi(location_name, current_pollutants)
        print(f"DEBUG - Predictions 24h: {predictions_24h}")

        return render_template(
            'results.html',
            aqi_value=aqi_value,
            aqi_description=aqi_description,
            aqi_color=aqi_color,
            pollutants=pollutants,
            pollutant_info=pollutant_info,
            latitude=lat,
            longitude=lon,
            location_name=location_name,
            predictions_24h=predictions_24h,
            health_recommendations=health_recommendations
        )

    except requests.RequestException as e:
        error_message = f"Error connecting to OpenWeatherMap API: {str(e)}"
        return render_template('error.html', error_message=error_message), 500
        
    except (KeyError, ValueError) as e:
        error_message = f"Error processing API response: {str(e)}"
        return render_template('error.html', error_message=error_message), 500
        
    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        return render_template('error.html', error_message=error_message), 500

@app.route('/get_air_quality', methods=['POST'])
def get_air_quality():
    lat = request.form.get('lat')
    lon = request.form.get('lon')

    if not lat or not lon:
        return jsonify({'error': 'Invalid location data'}), 400

    try:
        response = requests.get(
            OPENWEATHER_API_URL,
            params={
                'lat': lat,
                'lon': lon,
                'appid': OPENWEATHER_API_KEY
            }
        )
        
        response.raise_for_status()
        data = response.json()

        if 'list' in data and len(data['list']) > 0:
            # Convert OpenWeatherMap AQI to standard scale
            openweather_aqi = data['list'][0]['main']['aqi']
            print(f"\nDEBUG - Raw AQI from API: {openweather_aqi}")
            aqi_value = convert_to_standard_aqi(openweather_aqi)
            print(f"DEBUG - Final AQI value: {aqi_value}")
            
            return jsonify({
                'aqi_value': aqi_value,
                'aqi_description': get_aqi_description(aqi_value),
                'aqi_color': get_aqi_color(aqi_value)
            })
        else:
            return jsonify({'error': 'Data not available for this location'}), 404

    except requests.RequestException as e:
        return jsonify({'error': f'Error connecting to OpenWeatherMap API: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Error fetching data: {str(e)}'}), 500

@app.route('/health_calculator')
def health_calculator():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    aqi_value = request.args.get('aqi')
    aqi_color = request.args.get('color')
    
    if not all([lat, lon, aqi_value, aqi_color]):
        return redirect('/location')
        
    return render_template(
        'health_calculator.html',
        latitude=lat,
        longitude=lon,
        aqi_value=float(aqi_value),
        aqi_color=aqi_color
    )

if __name__ == '__main__':
    app.run(debug=True)