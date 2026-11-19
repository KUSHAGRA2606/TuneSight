# Data

Nothing in this folder is committed (see `.gitignore`); it is created by the scripts.

| Folder | Made by | Contents |
|---|---|---|
| `objectfolder/audio/<object>/audio/<strike>/mic.wav` | `scripts/download_objectfolder.py` | Real impact sounds from ObjectFolder Real (Gao et al., CVPR 2023), 48 kHz, anechoic. Only the microphone files are kept while the archive streams past. |
| `objectfolder/mesh/`, `objectfolder/videos/` | `scripts/download_objectfolder.py --meshes --videos` | Object meshes (true sizes for the scaling law) and lightbox videos. |
| `own/<session>/` | `scripts/record_session.py` | Your recorded session: `taps.csv`, `images/`, `audio/`. |

Fitted calibrations are written to `models/calibration.json` by `scripts/calibrate.py` and by the app's **Save calib** button.
