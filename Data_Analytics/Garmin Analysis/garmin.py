
import gpxpy
import pandas as pd
from pathlib import Path
from datetime import datetime

# Calea către fișierul GPX
gpx_file = Path(__file__).parent.parent.parent / 'activity_21430042156.gpx'

try:
    print(f"Reading GPX file: {gpx_file}")
    
    with open(gpx_file, 'r') as f:
        gpx = gpxpy.parse(f)
    
    data = []
    
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                data.append({
                    'Latitude': point.latitude,
                    'Longitude': point.longitude,
                    'Elevation': point.elevation,
                    'Time': point.time,
                    'Speed': point.speed if hasattr(point, 'speed') else None
                })
    
    df = pd.DataFrame(data)
    
    if not df.empty:
        # Calculează distanța totală
        total_distance = 0
        for track in gpx.tracks:
            total_distance += track.length_3d()
        
        # Calculează durata
        if len(df) > 0 and df['Time'].notna().any():
            start_time = df['Time'].iloc[0]
            end_time = df['Time'].iloc[-1]
            duration = (end_time - start_time).total_seconds() / 60  # în minute
        else:
            duration = 0
        
        # Calculează diferența de elevație
        if df['Elevation'].notna().any():
            elevation_gain = gpx.get_uphill_downhill().uphill
            elevation_loss = gpx.get_uphill_downhill().downhill
        else:
            elevation_gain = 0
            elevation_loss = 0
        
        # Afișează statistici
        print(f"\nGPX Activity Statistics:")
        print(f"Total Points: {len(df)}")
        print(f"Distance: {round(total_distance / 1000, 2)} km")
        print(f"Duration: {round(duration, 1)} min")
        print(f"Elevation Gain: {round(elevation_gain, 1)} m")
        print(f"Elevation Loss: {round(elevation_loss, 1)} m")
        
        # Salvează datele în CSV
        csv_file = Path(__file__).parent / 'gpx_data.csv'
        df.to_csv(csv_file, index=False)
        print(f"\nSaved detailed data to {csv_file}")
        print(f"\nFirst 10 points:")
        print(df.head(10).to_string(index=False))
    else:
        print("No data found in GPX file")
        
except FileNotFoundError:
    print(f"GPX file not found at: {gpx_file}")
except Exception as e:
    print(f"Error reading GPX file: {e}")
