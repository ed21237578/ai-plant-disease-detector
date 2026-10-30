# PDD · Plant Disease Detector

Team EJMA (OEH-1-015), 2nd OneEarth International Hackathon 2026.

Photograph a leaf, get a diagnosis from an AI that runs on the phone, pin the scan on a field map, and treat only the zones that need it. Works offline.

## Run it
- **Quickest:** double-click `index.html`.
- **Full version (offline install, GPS):** open the GitHub Pages link on a phone. Settings → Pages → Deploy from branch → `main`, `/ (root)`.

## Files
| File | Owner | What it does |
|---|---|---|
| `index.html`, `scan.js`, `style.css` | Person 2 | Scan screen: camera, photo checks, diagnosis, heatmap, treatment, save |
| `storage.js`, `map.js`, `sw.js`, `manifest.json`, `icons/`, `vendor/leaflet/` | Person 3 | GPS + saving on the phone, field map with zones, offline mode, home-screen install |
| `app.js` | shared | Bottom tabs (Scan / Field map) and offline registration |
| `pdd-core.js` | shared | The neural network running in JavaScript, heatmap, leaf and blur checks |
| `model.json` / `model.js`, `treatments.json` / `treatments.js` | Person 1 | Trained model weights and treatment advice (`.js` copies let the app open by double-click) |
| `training/` | Person 1 | Python: prepare data, train, export, check JS = Python |

## Model
CNN (4 conv layers, ~99k parameters) trained by Person 1 on PlantVillage photos of 15 tomato, potato and bell pepper conditions.
Check on 520 separate PlantVillage photos (13 conditions, not in the original training set): 88.1% accuracy.
Answers above the 80% confidence threshold (77% of photos) were right 96.2% of the time; below it the app shows its top 3 and says to ask an expert.

After changing any app file, bump `VERSION` in `sw.js` so phones load the new version.
