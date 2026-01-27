"""
pipeline_garmin_to_sqlite.py
Script to import activities and sleep data from GPX files into the SQLite database garmin_data.db
"""
import sqlite3
import os
import glob
import gpxpy
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

folder = os.path.dirname(__file__)
db_path = os.path.join(folder, 'garmin_data.db')

def create_table(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS activitati (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            type TEXT,
            start_time TEXT,
            duration REAL,
            distance REAL
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sleep_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            start_time TEXT,
            end_time TEXT,
            duration REAL
        )
    ''')
    conn.commit()



def parse_gpx_file(gpx_file):
    with open(gpx_file, 'r') as f:
        gpx = gpxpy.parse(f)
    activitati = []
    sleep_entries = []
    for track in gpx.tracks:
        name = track.name if track.name else os.path.basename(gpx_file)
        type_ = track.type if track.type else None
        points = []
        for segment in track.segments:
            points.extend(segment.points)
        # Ignore files with too few points (e.g., only heart rate, no route)
        if not points or len(points) < 10:
            continue
        start_time = points[0].time.isoformat() if points[0].time else None
        end_time = points[-1].time.isoformat() if points[-1].time else None
        duration = (points[-1].time - points[0].time).total_seconds() if points[0].time and points[-1].time else None
        # Distance calculation removed
        distance = None
        # Detect sleep activity by type or name
        if (type_ and type_.lower() == 'sleep') or (name and 'sleep' in name.lower()):
            sleep_entries.append({
                'name': name,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration
            })
        else:
            activitati.append({
                'name': name,
                'type': type_,
                'start_time': start_time,
                'duration': duration
            })
    return activitati, sleep_entries

def import_gpx_to_db():
    conn = sqlite3.connect(db_path)
    create_table(conn)
    gpx_files = glob.glob(os.path.join(folder, '*.gpx'))
    for gpx_file in gpx_files:
        activitati, sleep_entries = parse_gpx_file(gpx_file)
        imported = False
        for data in activitati:
            conn.execute('''
                INSERT INTO activitati (name, type, start_time, duration)
                VALUES (?, ?, ?, ?)
            ''', (
                data['name'],
                data['type'],
                data['start_time'],
                data['duration']
            ))
            imported = True
        for sleep in sleep_entries:
            conn.execute('''
                INSERT INTO sleep_data (name, start_time, end_time, duration)
                VALUES (?, ?, ?, ?)
            ''', (
                sleep['name'],
                sleep['start_time'],
                sleep['end_time'],
                sleep['duration']
            ))
            imported = True
        if imported:
            print(f"Imported: {os.path.basename(gpx_file)}")
        else:
            print(f"Parse error: {os.path.basename(gpx_file)}")
    conn.commit()
    conn.close()
    print('GPX import finished!')

if __name__ == '__main__':
    import_gpx_to_db()
