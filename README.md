# Air Pollution Predictor

A comprehensive air quality monitoring and prediction system that provides real-time AQI (Air Quality Index) calculations, forecasts, and health recommendations.

## Features

- **Real-time AQI Calculation**: Calculate AQI based on multiple pollutants (PM2.5, PM10, NO2, SO2, CO, O3)
- **24-hour Predictions**: Get hourly AQI forecasts for the next 24 hours
- **Health Recommendations**: Receive personalized health advice based on current AQI levels
- **Historical Tracking**: Monitor air quality trends and statistics
- **Email Alerts**: Get notifications when AQI exceeds specified thresholds
- **Multiple Cities**: Support for various cities with location-specific predictions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Air_Pollution_Predictor.git
cd Air_Pollution_Predictor
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `config` directory and set up email alerts:
```bash
mkdir config
```

2. Create `config/alert_config.json` with your email settings:
```json
{
    "email_settings": {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "sender_email": "your-email@gmail.com",
        "sender_password": "your-app-password"
    },
    "alert_thresholds": {
        "unhealthy": 150,
        "very_unhealthy": 200,
        "hazardous": 300
    },
    "subscribed_users": []
}
```

Note: For Gmail, you'll need to use an App Password. [Learn how to create one](https://support.google.com/accounts/answer/185833?hl=en)

## Usage

1. Train the model:
```bash
python aqi_model.py
```

2. Get AQI predictions:
```python
from aqi_model import predict_aqi

# Example pollutant values (μg/m³ or ppb)
pollutants = [
    30,    # PM2.5 (μg/m³)
    50,    # PM10 (μg/m³)
    40,    # NO2 (ppb)
    15,    # SO2 (ppb)
    1.5,   # CO (ppm)
    35     # O3 (ppb)
]

prediction = predict_aqi("Mumbai", pollutants)
print(f"AQI: {prediction['aqi']}")
print(f"Category: {prediction['category']}")
```

3. Subscribe to alerts:
```python
from alerts import AQIAlertSystem

alert_system = AQIAlertSystem()
alert_system.add_subscriber(
    email="your-email@example.com",
    city="Mumbai",
    threshold=150  # Get alerts when AQI exceeds 150
)
```

## AQI Categories

- 0-50: Good (Green)
- 51-100: Moderate (Yellow)
- 101-150: Unhealthy for Sensitive Groups (Orange)
- 151-200: Unhealthy (Red)
- 201-300: Very Unhealthy (Purple)
- 301+: Hazardous (Maroon)

## Health Recommendations

The system provides recommendations for:
- Outdoor activities
- Sensitive groups
- Mask usage
- Ventilation

## Historical Data

Historical AQI data is stored in `data/history/{city}_history.csv` and includes:
- Timestamp
- AQI value
- Category
- Individual pollutant levels

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- EPA AQI calculation standards
- World Air Quality Index Project
- Various open-source contributors 