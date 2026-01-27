

from garminconnect import Garmin
import getpass
import pandas as pd
from pathlib import Path


token_dir = Path.home() / '.garminconnect'
token_dir.mkdir(exist_ok=True)


# SSL este activ implicit. Dacă ai probleme cu certificatele, rulează scriptul Install Certificates.command din /Applications/Python 3.11/ sau actualizează certifi.

def login():
    email = 'simo.fleanta@gmail.com'
    print("Enter your Garmin password:")
    password = getpass.getpass()
    try:
        client = Garmin(email, password)
        client.login()
        client.garth.dump(str(token_dir))
        print("Login successful!")
        return client
    except Exception as e:
        print(f"Authentication error: {e}")
        return None


def login_with_saved_tokens():
    try:
        client = Garmin('', '')
        client.garth.load(str(token_dir))
        print("Authentication with saved tokens successful!")
        return client
    except:
        return None

print("Trying authentication with saved tokens...")
client = login_with_saved_tokens()
if not client:
    print("No saved tokens found. New authentication required.")
    client = login()

if client:
    try:
        print("Getting activities...")
        activities = client.get_activities(0, 50)
        print(f"Found {len(activities)} activities\n")
        data = []
        for activity in activities:
            data.append({
                'ID': activity.get('activityId'),
                'Name': activity.get('activityName', 'N/A'),
                'Type': activity.get('activityType', {}).get('typeKey', 'N/A'),
                'Date': activity.get('startTimeLocal', 'N/A'),
                'Distance_km': round(activity.get('distance', 0) / 1000, 2),
                'Duration_min': round(activity.get('duration', 0) / 60, 1),
                'Calories': activity.get('calories', 0)
            })
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        csv_file = Path(__file__).parent / 'garmin_activities.csv'
        df.to_csv(csv_file, index=False)
        print(f"Saved to {csv_file}")
        print(df.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")
else:
    print("Authentication failed. Please check your credentials and Garmin Connect status.")
