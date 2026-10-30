/* PDD app shell: bottom tabs (Scan / Field map) and offline support. */
(() => {
  "use strict";

  const tabs = document.querySelectorAll(".tab");

  function show(viewId) {
    document.querySelectorAll(".view").forEach(view => {
      view.hidden = view.id !== viewId;
    });

    tabs.forEach(tab => {
      if (tab.dataset.view === viewId) tab.setAttribute("aria-current", "page");
      else tab.removeAttribute("aria-current");
    });

    window.scrollTo(0, 0);

    // map.js listens for this so Leaflet can resize once its tab is visible
    window.dispatchEvent(new Event(viewId === "map-view" ? "pdd:show-map" : "pdd:show-scan"));
  }

  tabs.forEach(tab => tab.addEventListener("click", () => show(tab.dataset.view)));

  // Offline mode (sw.js). Not available when index.html is opened as a file.
  if ("serviceWorker" in navigator && location.protocol !== "file:") {
    window.addEventListener("load", () => {
      navigator.serviceWorker.register("sw.js").catch(() => {});
    });
  }
})();
