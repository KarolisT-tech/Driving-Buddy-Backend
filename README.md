# Driving Buddy Backend

Python backend for BrainAccess HALO jaw-clench and blink detection.

## Run

Start the selected Android emulators, connect BrainAccess HALO, then run:

```bash
python main.py
```

The backend starts its Android WebSocket server on port `8765` and connects directly to the configured BrainAccess device (`BA HALO 206` by default).

## Emulator SMS test

When a double jaw clench is detected, the backend injects an emulated incoming SMS with the text `Clench detected` into one selected Android Emulator.

Default recipient:

```bash
python main.py --sms-target emulator-5556
```

To select the other emulator:

```bash
python main.py --sms-target emulator-5554
```

The default simulated sender address is the other emulator's console port (`5554` or `5556`). Override it with `--sms-sender` if desired. ADB must be available on `PATH`, and the selected emulator must appear as `device` in `adb devices`. To specify a custom ADB executable, use `--adb-path /path/to/adb`.

This uses the Android Emulator console command `sms send` through ADB. It injects an **incoming test SMS** into the selected emulator; it is not a real carrier SMS and does not execute `SmsManager` on the other emulator.

To run EEG detection without injecting test SMS messages:

```bash
python main.py --disable-emulator-sms
```

## Existing Android WebSocket events

The existing WebSocket server remains on port `8765`. Existing `double_blink` events are unchanged.
