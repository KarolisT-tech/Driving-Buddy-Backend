Driving Buddy — Launch Guide

Frontend Repo: https://github.com/KarolisT-tech/Driving-Buddy-Frontend
Backend WebSocket Port: 8765 | Default Headset: BA HALO 206

Prerequisites
BrainAccess HALO headset (Bluetooth paired, disconnected from BrainAccess Board)
Python 3 (brainaccess, numpy, scipy, websockets)
Android Studio with an Emulator (Google Maps + internet enabled)

How to Launch
1. Start the Android Frontend
Open Driving-Buddy-Frontend in Android Studio.
Boot your Android Emulator (usually emulator-5554), run the app, and grant Location & SMS permissions.

2. Start the Python Backend (main.py)
With Emulator SMS Test (Windows):
Powershell
python main.py --adb-path "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" --sms-target emulator-5554
(If adb is already in PATH, you can omit --adb-path ....)
Without Emulator SMS (EEG + WebSocket only):
Powershell
python main.py --disable-emulator-sms

3. Calibrate
Keep your jaw relaxed and look forward for 8 seconds (natural blinking is fine) until status shows NO REQUEST.
Controls
1 Blink: Read latest SMS message
2 Blinks: Navigate to nearest gas station (Google Maps)
2 Jaw Clenches: Listening mode + inject test SMS ("Clench detected")
Automatic: Drowsiness alert (drowsy_alert)

Quick Fixes:
ADB was not found: Pass --adb-path "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" or use --disable-emulator-sms.
emulator-5556 is not ready: Pass --sms-target emulator-5554 (check adb devices).
Hard to double-blink: Add --blink-double-window 1.2 --blink-k 3.2.
HALO won't connect: Ensure it's disconnected from BrainAccess Board first.