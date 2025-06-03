import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
import json

class AQIAlertSystem:
    def __init__(self):
        self.config_file = 'config/alert_config.json'
        self.load_config()
    
    def load_config(self):
        """Load email configuration from JSON file"""
        try:
            if not os.path.exists('config'):
                os.makedirs('config')
            
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = {
                    'email_settings': {
                        'smtp_server': 'smtp.gmail.com',
                        'smtp_port': 587,
                        'sender_email': '',
                        'sender_password': ''
                    },
                    'alert_thresholds': {
                        'unhealthy': 150,
                        'very_unhealthy': 200,
                        'hazardous': 300
                    },
                    'subscribed_users': []
                }
                self.save_config()
        except Exception as e:
            print(f"Error loading config: {str(e)}")
    
    def save_config(self):
        """Save current configuration to JSON file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {str(e)}")
    
    def add_subscriber(self, email, city, threshold=150):
        """Add a new subscriber for AQI alerts"""
        try:
            subscriber = {
                'email': email,
                'city': city,
                'threshold': threshold,
                'subscribed_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            self.config['subscribed_users'].append(subscriber)
            self.save_config()
            return True
        except Exception as e:
            print(f"Error adding subscriber: {str(e)}")
            return False
    
    def remove_subscriber(self, email):
        """Remove a subscriber"""
        try:
            self.config['subscribed_users'] = [
                user for user in self.config['subscribed_users']
                if user['email'] != email
            ]
            self.save_config()
            return True
        except Exception as e:
            print(f"Error removing subscriber: {str(e)}")
            return False
    
    def send_alert(self, user_email, aqi_data):
        """Send email alert to user"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['email_settings']['sender_email']
            msg['To'] = user_email
            msg['Subject'] = f"AQI Alert for {aqi_data['city']} - {aqi_data['category']}"
            
            body = f"""
            Air Quality Alert for {aqi_data['city']}
            
            Current AQI: {aqi_data['aqi']}
            Category: {aqi_data['category']}
            
            Health Recommendations:
            """
            
            for category, recommendation in aqi_data['health_recommendations'].items():
                body += f"\n{category.replace('_', ' ').title()}: {recommendation}"
            
            if aqi_data.get('historical_stats'):
                stats = aqi_data['historical_stats']
                body += f"""
                
                24-Hour Statistics:
                Average AQI: {stats['last_24h']['mean_aqi']:.1f}
                Maximum AQI: {stats['last_24h']['max_aqi']:.1f}
                Minimum AQI: {stats['last_24h']['min_aqi']:.1f}
                
                Overall Trend: {stats['trend']}
                """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Connect to SMTP server
            server = smtplib.SMTP(
                self.config['email_settings']['smtp_server'],
                self.config['email_settings']['smtp_port']
            )
            server.starttls()
            server.login(
                self.config['email_settings']['sender_email'],
                self.config['email_settings']['sender_password']
            )
            
            # Send email
            server.send_message(msg)
            server.quit()
            return True
        except Exception as e:
            print(f"Error sending alert: {str(e)}")
            return False
    
    def check_and_send_alerts(self, aqi_data):
        """Check AQI levels and send alerts to subscribers if needed"""
        try:
            for user in self.config['subscribed_users']:
                if (user['city'].lower() == aqi_data['city'].lower() and 
                    aqi_data['aqi'] >= user['threshold']):
                    self.send_alert(user['email'], aqi_data)
        except Exception as e:
            print(f"Error checking alerts: {str(e)}")

# Example usage
if __name__ == "__main__":
    alert_system = AQIAlertSystem()
    
    # Add a test subscriber
    alert_system.add_subscriber(
        email="test@example.com",
        city="Mumbai",
        threshold=150
    )
    
    # Test alert with sample data
    test_data = {
        'city': 'Mumbai',
        'aqi': 160,
        'category': 'Unhealthy',
        'health_recommendations': {
            'outdoor_activities': 'Avoid prolonged outdoor activities',
            'sensitive_groups': 'Stay indoors if possible',
            'mask_recommendation': 'Wear masks when outdoors',
            'ventilation': 'Keep windows closed'
        },
        'historical_stats': {
            'last_24h': {
                'mean_aqi': 155.0,
                'max_aqi': 170.0,
                'min_aqi': 140.0
            },
            'trend': 'worsening'
        }
    }
    
    alert_system.check_and_send_alerts(test_data) 