(() => {
  "use strict";

  const CONFIDENCE_THRESHOLD = 0.80;
  const MIN_LEAF_FRACTION = 0.025;
  const MIN_SHARPNESS = 18;

  const $ = id => document.getElementById(id);

  const cameraInput = $("cameraInput");
  const uploadInput = $("uploadInput");
  const photoPreview = $("photoPreview");
  const previewPlaceholder = $("previewPlaceholder");
  const heatmapCanvas = $("heatmapCanvas");
  const statusEl = $("status");
  const resultCard = $("resultCard");
  const saveButton = $("saveButton");

  let model = null;
  let treatments = {};
  let currentScan = null;
  let lastSquareCanvas = null; // [merge] the exact 256x256 square the AI analysed

  // [merge] "Tomato___Early_blight" -> "Tomato · Early blight" (uses treatments.json names when present)
  function niceName(label) {
    const t = treatments[label];
    if (t && t.name) return t.name;
    const [crop, cond] = String(label).split("___");
    const c = (crop || "").replace(/,_bell/i, " (bell)").replace(/_/g, " ");
    const d = (cond || "").replace(/_/g, " ").replace(/\s+/g, " ").trim();
    return d ? `${c} · ${d.charAt(0).toUpperCase()}${d.slice(1)}` : c;
  }

  function setStatus(message, type = "") {
    statusEl.textContent = message;
    statusEl.className = `status ${type}`;
  }

  async function loadJson(path) {
    const response = await fetch(path);

    if (!response.ok) {
      throw new Error(`Could not load ${path}`);
    }

    return response.json();
  }

  async function initialise() {
    if (!window.PDD) {
      setStatus("pdd-core.js was not found.", "error");
      return;
    }

    try {
      const [modelJson, treatmentJson] =
        await Promise.all([
          window.PDD_MODEL_DATA || loadJson("model.json"),
          window.PDD_TREATMENTS_DATA || loadJson("treatments.json")
        ]);

      model = PDD.loadModel(modelJson);
      treatments = treatmentJson;

      setStatus(
        modelJson._placeholder
          ? "Placeholder model loaded. Predictions are not real."
          : "Model loaded successfully.",
        "success"
      );

    } catch (error) {
      setStatus(error.message, "error");
    }
  }

  function readFile(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const image = new Image();

      image.onload = () => {
        URL.revokeObjectURL(url);
        resolve(image);
      };

      image.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error("Could not open this image."));
      };

      image.src = url;
    });
  }

  function makeSquareFrame(image) {
    const side = Math.min(
      image.naturalWidth,
      image.naturalHeight
    );

    const sx = (image.naturalWidth - side) / 2;
    const sy = (image.naturalHeight - side) / 2;

    const canvas = document.createElement("canvas");

    canvas.width = 256;
    canvas.height = 256;

    const ctx = canvas.getContext("2d", {
      willReadFrequently: true
    });

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high"; // [merge] same resize quality the model was tested with
    lastSquareCanvas = canvas;

    ctx.drawImage(
      image,
      sx, sy, side, side,
      0, 0, 256, 256
    );

    return ctx.getImageData(
      0, 0, 256, 256
    ).data;
  }

  function getTopPredictions(probabilities) {
    return Array.from(
      probabilities,
      (confidence, index) => ({
        label: model.classes[index],
        confidence
      })
    )
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 3);
  }

  function drawHeatmap(camResult) {
    const layer = document.createElement("canvas");

    layer.width = camResult.W;
    layer.height = camResult.H;

    const ctx = layer.getContext("2d");

    const pixels = ctx.createImageData(
      camResult.W,
      camResult.H
    );

    for (let i = 0; i < camResult.map.length; i++) {
      const value = Math.max(
        0,
        Math.min(1, camResult.map[i])
      );

      const p = i * 4;

      pixels.data[p] = Math.round(255 * value);
      pixels.data[p + 1] = Math.round(70 * (1 - value));
      pixels.data[p + 2] = Math.round(255 * (1 - value));
      pixels.data[p + 3] = Math.round(150 * value);
    }

    ctx.putImageData(pixels, 0, 0);

    const display = heatmapCanvas.getContext("2d");

    const width = photoPreview.clientWidth;
    const height = photoPreview.clientHeight;

    heatmapCanvas.width = width;
    heatmapCanvas.height = height;

    const scale = Math.min(width / 256, height / 256);

    const dw = 256 * scale;
    const dh = 256 * scale;

    display.globalAlpha = 0.58;

    display.drawImage(
      layer,
      (width - dw) / 2,
      (height - dh) / 2,
      dw,
      dh
    );

    display.globalAlpha = 1;

    heatmapCanvas.hidden = false;
  }

  function showResult(top3, feat) {
    const best = top3[0];

    const confidencePct = best.confidence * 100;

    $("resultTitle").textContent =
      niceName(best.label);

    $("confidenceValue").textContent =
      confidencePct.toFixed(1) + "%";

    $("confidencePill").textContent =
      confidencePct >= 80
        ? "Higher confidence"
        : "Low confidence";

    $("confidenceMeter").style.width =
      confidencePct + "%";

    $("resultDisclaimer").textContent =
      confidencePct < 80
        ? "The model is uncertain. Ask an expert."
        : "Model output only. Confirm the diagnosis before treatment.";

    const treatment = treatments[best.label] || {};
    const lowConfidence = confidencePct < 80;

    $("treatmentTitle").textContent = lowConfidence
      ? "Don't treat yet"
      : (treatment.name || niceName(best.label));

    $("treatmentText").textContent = lowConfidence
      ? "The AI is not sure. Retake the photo in good light, or ask an agronomist to confirm before spraying anything."
      : (treatment.treatment || "Ask a qualified agronomist.");

    $("treatmentNote").textContent = lowConfidence
      ? ""
      : (treatment.note || (treatment.organic ? "Organic option: " + treatment.organic : ""));

    $("topGuesses").hidden = !lowConfidence;

    $("topGuessesList").replaceChildren();

    if (lowConfidence) {
      top3.forEach(guess => {
        const li = document.createElement("li");

        li.textContent =
          `${niceName(guess.label)} — ` +
          `${(guess.confidence * 100).toFixed(1)}%`;

        $("topGuessesList").appendChild(li);
      });
    }

    resultCard.hidden = false;

    const classIndex = model.classes.indexOf(best.label);

    const cam = PDD.cam(model, feat, classIndex);

    drawHeatmap(cam);
  }

  // [merge] sharper copy of the same centred square for the on-screen preview
  function displaySquare(image) {
    const side = Math.min(image.naturalWidth, image.naturalHeight);
    const c = document.createElement("canvas");
    c.width = 640;
    c.height = 640;
    const ctx = c.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(image, (image.naturalWidth - side) / 2, (image.naturalHeight - side) / 2, side, side, 0, 0, 640, 640);
    return c.toDataURL("image/jpeg", 0.9);
  }

  // [merge] 96px JPEG thumbnail stored with the scan (shown in map pins)
  function makeThumb() {
    const c = document.createElement("canvas");
    c.width = 96;
    c.height = 96;
    const ctx = c.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(lastSquareCanvas, 0, 0, 96, 96);
    return c.toDataURL("image/jpeg", 0.75);
  }

  async function processPhoto(file) {
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setStatus("Please choose an image.", "error");
      return;
    }

    if (!model) {
      setStatus("Model is not ready.", "error");
      return;
    }

    saveButton.disabled = true;
    resultCard.hidden = true;
    heatmapCanvas.hidden = true;

    setStatus("Preparing photo...");

    try {
      const image = await readFile(file);

      const rgba = makeSquareFrame(image);

      // [merge] show the same centred square the AI analysed, so the heatmap lines up on any photo shape
      photoPreview.src = displaySquare(image);

      photoPreview.hidden = false;
      previewPlaceholder.hidden = true;

      try { await photoPreview.decode(); } catch (e) { /* draw anyway */ }

      const stats = PDD.leafStats(rgba);

      const sharpness = PDD.sharpness(
        rgba,
        256
      );

      if (stats.leafFrac < MIN_LEAF_FRACTION) {
        throw new Error(
          "Not enough leaf-like colour detected. Try a closer photo."
        );
      }

      if (sharpness < MIN_SHARPNESS) {
        throw new Error(
          "Photo is blurry. Hold your phone steady and try again."
        );
      }

      setStatus("Analysing leaf...");

      const rgb64 = PDD.boxDown(rgba);

      const output = PDD.forward(
        model,
        rgb64
      );

      const probabilities = PDD.softmax(
        output.logits
      );

      const top3 = getTopPredictions(
        probabilities
      );

      showResult(top3, output.feat);

      currentScan = {
        id: crypto.randomUUID
          ? crypto.randomUUID()
          : `scan-${Date.now()}`,

        time: new Date().toISOString(),

        lat: null,
        lng: null,

        label: top3[0].label,

        confidence: top3[0].confidence,

        top3: top3.map(item => ({
          label: item.label,
          confidence: item.confidence
        })),

        thumb: makeThumb()
      };

      saveButton.disabled = false;

      setStatus("Analysis complete.", "success");

    } catch (error) {
      currentScan = null;
      resultCard.hidden = true;

      setStatus(error.message, "error");
    }
  }

  async function saveCurrentScan() {
    if (!currentScan) return;

    saveButton.disabled = true;

    setStatus("Saving scan...");

    const scan = { ...currentScan };

    const position = await new Promise(resolve => {
      if (!navigator.geolocation) {
        return resolve(null);
      }

      navigator.geolocation.getCurrentPosition(
        p => resolve({
          lat: p.coords.latitude,
          lng: p.coords.longitude
        }),
        () => resolve(null),
        {
          enableHighAccuracy: true,
          timeout: 8000
        }
      );
    });

    if (position) {
      scan.lat = position.lat;
      scan.lng = position.lng;
    }

    let saved = scan;

    try {
      if (typeof window.saveScan === "function") {
        saved = (await window.saveScan(scan)) || scan; // [merge] storage.js (Person 3)
      } else {
        const key = "pdd_scans_person2_fallback";

        const existing = JSON.parse(
          localStorage.getItem(key) || "[]"
        );

        existing.push(scan);

        localStorage.setItem(
          key,
          JSON.stringify(existing)
        );
      }

      currentScan = saved;

      setStatus(
        typeof saved.lat === "number"
          ? "Scan saved with GPS. See it on the Field map."
          : "Scan saved without GPS. Turn on location to put it on the map.",
        "success"
      );

    } catch (error) {
      setStatus(
        "Could not save scan: " + error.message,
        "error"
      );

      saveButton.disabled = false;
    }
  }

  function resetScan() {
    cameraInput.value = "";
    uploadInput.value = "";

    photoPreview.hidden = true;
    photoPreview.removeAttribute("src");

    previewPlaceholder.hidden = false;
    heatmapCanvas.hidden = true;

    resultCard.hidden = true;

    currentScan = null;

    saveButton.disabled = true;

    setStatus("Ready for another leaf.");
  }

  cameraInput.addEventListener("change", event => {
    processPhoto(event.target.files[0]);
  });

  uploadInput.addEventListener("change", event => {
    processPhoto(event.target.files[0]);
  });

  saveButton.addEventListener(
    "click",
    saveCurrentScan
  );

  $("newScanButton").addEventListener(
    "click",
    resetScan
  );

  initialise();

})();