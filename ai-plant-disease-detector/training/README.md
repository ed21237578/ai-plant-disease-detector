# PDD · Plant Disease Detector (team EJMA, OEH-1-015)

Prototype for the 2nd OneEarth International Hackathon 2026.

- `prep.py` – loads PlantVillage leaf photos (15 tomato, potato and bell pepper conditions), centre-crops, resizes to 256 px, averages down to 64x64.
- `cnn.py` – the convolutional neural network (4 conv layers, global average pooling, ~99k parameters) with forward and backward passes.
- `train.py` – trains with augmentation (flips, shifts, brightness), Adam and a cosine learning-rate schedule; keeps the best model on validation data.
- `export_model.py` – packs the weights for the browser.
- `pdd-core.js` – the same network running in JavaScript, so the app works offline on a phone or laptop; also the AI heatmap (class activation map), leaf and blur checks.
- `check_equiv.py` – proves the JavaScript model gives the same answers as the Python one.

Results on 937 test photos never used in training: 91.0% accuracy. Answers above the 80% confidence threshold were right 98.2% of the time (82% of photos).
Data: PlantVillage (Hughes & Salathé 2015), https://github.com/spMohanty/PlantVillage-Dataset
