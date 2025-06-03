from datetime import datetime, timedelta

class HealthRecommendationSystem:
    def __init__(self):
        # Define thresholds for different pollutants (in μg/m³)
        self.pollutant_thresholds = {
            'pm2_5': {'sensitive': 35, 'unhealthy': 55, 'very_unhealthy': 150, 'hazardous': 250},
            'pm10': {'sensitive': 50, 'unhealthy': 100, 'very_unhealthy': 250, 'hazardous': 350},
            'no2': {'sensitive': 53, 'unhealthy': 100, 'very_unhealthy': 360, 'hazardous': 649},
            'so2': {'sensitive': 35, 'unhealthy': 75, 'very_unhealthy': 185, 'hazardous': 304},
            'co': {'sensitive': 4.4, 'unhealthy': 9.4, 'very_unhealthy': 12.4, 'hazardous': 15.4},
            'o3': {'sensitive': 54, 'unhealthy': 70, 'very_unhealthy': 85, 'hazardous': 105}
        }

    def get_detailed_health_recommendations(self, aqi_value, pollutants=None):
        """Get detailed health recommendations based on AQI and pollutant levels"""
        print(f"\nDEBUG - Health Recommendations - Received AQI: {aqi_value}")
        print(f"DEBUG - Health Recommendations - Pollutants: {pollutants}")
        
        # Ensure AQI is a number and within valid range
        try:
            aqi_value = float(aqi_value)
            if aqi_value < 0:
                print(f"WARNING: Negative AQI value received: {aqi_value}")
                aqi_value = 0
            if aqi_value > 500:
                print(f"WARNING: AQI value exceeds 500: {aqi_value}")
                aqi_value = 500
            print(f"DEBUG - Validated AQI value: {aqi_value}")
        except (TypeError, ValueError) as e:
            print(f"ERROR: Invalid AQI value: {aqi_value}, error: {e}")
            return {'general_advice': []}
        
        # Get general advice based on AQI
        general_advice = self._get_general_advice(aqi_value)
        print(f"DEBUG - Health Recommendations - General Advice: {general_advice}")
        
        # Get pollutant-specific warnings if pollutants are provided
        pollutant_warnings = []
        if pollutants:
            pollutant_warnings = self._get_pollutant_specific_warnings(pollutants)
            print(f"DEBUG - Health Recommendations - Pollutant Warnings: {pollutant_warnings}")
        
        # Combine all recommendations
        recommendations = {
            'general_advice': general_advice,
            'pollutant_warnings': pollutant_warnings
        }
        print(f"DEBUG - Health Recommendations - Final Recommendations: {recommendations}")
        return recommendations

    def _get_general_advice(self, aqi_value):
        """Get general health advice based on AQI value"""
        print(f"\nDEBUG - Getting general advice for AQI: {aqi_value}")
        if aqi_value <= 50:
            advice = {
                'type': 'General',
                'summary': 'Good air quality',
                'details': 'Air quality is satisfactory, and air pollution poses little or no risk.',
                'action_needed': 'No special precautions needed.',
                'color': 'green'
            }
        elif aqi_value <= 100:
            advice = {
                'type': 'General',
                'summary': 'Moderate air quality',
                'details': 'Air quality is acceptable. However, there may be a moderate health concern for a very small number of people who are unusually sensitive to air pollution.',
                'action_needed': 'Active children and adults, and people with respiratory disease, such as asthma, should limit prolonged outdoor exertion.',
                'color': 'yellow'
            }
        elif aqi_value <= 150:
            advice = {
                'type': 'General',
                'summary': 'Unhealthy for Sensitive Groups',
                'details': 'Members of sensitive groups may experience health effects. The general public is not likely to be affected.',
                'action_needed': 'Active children and adults, and people with respiratory disease, such as asthma, should avoid prolonged outdoor exertion; everyone else, especially children, should limit prolonged outdoor exertion.',
                'color': 'orange'
            }
        elif aqi_value <= 200:
            advice = {
                'type': 'General',
                'summary': 'Unhealthy',
                'details': 'Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.',
                'action_needed': 'Active children and adults, and people with respiratory disease, such as asthma, should avoid all outdoor exertion; everyone else, especially children, should limit outdoor exertion.',
                'color': 'red'
            }
        elif aqi_value <= 300:
            advice = {
                'type': 'General',
                'summary': 'Very Unhealthy',
                'details': 'Health alert: The risk of health effects is increased for everyone.',
                'action_needed': 'Active children and adults, and people with respiratory disease, such as asthma, should avoid all outdoor exertion; everyone else, especially children, should limit outdoor exertion.',
                'color': 'purple'
            }
        else:
            advice = {
                'type': 'General',
                'summary': 'Hazardous',
                'details': 'Health warning of emergency conditions: everyone is more likely to be affected.',
                'action_needed': 'Everyone should avoid all outdoor exertion.',
                'color': 'red'
            }
        print(f"DEBUG - General advice determined: {advice}")
        return advice

    def _get_pollutant_specific_warnings(self, pollutants):
        """Get warnings for specific pollutants that exceed thresholds"""
        warnings = []
        for pollutant, value in pollutants.items():
            if pollutant in self.pollutant_thresholds:
                thresholds = self.pollutant_thresholds[pollutant]
                if value > thresholds['hazardous']:
                    warnings.append({
                        'pollutant': pollutant,
                        'level': 'Hazardous',
                        'value': value,
                        'threshold': thresholds['hazardous'],
                        'health_effects': 'Serious health effects for everyone',
                        'precautions': 'Avoid all outdoor activities',
                        'color': 'red'
                    })
                elif value > thresholds['very_unhealthy']:
                    warnings.append({
                        'pollutant': pollutant,
                        'level': 'Very Unhealthy',
                        'value': value,
                        'threshold': thresholds['very_unhealthy'],
                        'health_effects': 'Significant health effects for sensitive groups',
                        'precautions': 'Limit outdoor activities',
                        'color': 'purple'
                    })
                elif value > thresholds['unhealthy']:
                    warnings.append({
                        'pollutant': pollutant,
                        'level': 'Unhealthy',
                        'value': value,
                        'threshold': thresholds['unhealthy'],
                        'health_effects': 'Health effects for sensitive groups',
                        'precautions': 'Reduce outdoor activities',
                        'color': 'red'
                    })
                elif value > thresholds['sensitive']:
                    warnings.append({
                        'pollutant': pollutant,
                        'level': 'Sensitive',
                        'value': value,
                        'threshold': thresholds['sensitive'],
                        'health_effects': 'Possible health effects for sensitive groups',
                        'precautions': 'Monitor health conditions',
                        'color': 'orange'
                    })
        print(f"DEBUG - Pollutant warnings determined: {warnings}")
        return warnings 