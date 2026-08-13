(() => {
  const dropzone = document.getElementById("uploadForm");
  const fileInput = document.getElementById("fileInput");
  const scanImageWrap = document.getElementById("scanImageWrap");
  const scanImage = document.getElementById("scanImage");
  const scanline = document.getElementById("scanline");
  const viewTabs = document.getElementById("viewTabs");
  const viewerMeta = document.getElementById("viewerMeta");
  const metaResolution = document.getElementById("metaResolution");
  const metaTumorCount = document.getElementById("metaTumorCount");
  const resetBtn = document.getElementById("resetBtn");
  const consoleBody = document.getElementById("consoleBody");
  const stageIndicator = document.getElementById("stageIndicator");
  const tumorCardTemplate = document.getElementById("tumorCardTemplate");

  let currentImages = null; 

  
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("drag"); })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });

  resetBtn.addEventListener("click", resetAll);

  function resetAll() {
    fileInput.value = "";
    currentImages = null;
    scanImageWrap.hidden = true;
    dropzone.hidden = false;
    viewTabs.hidden = true;
    viewerMeta.hidden = true;
    stageIndicator.querySelectorAll(".stage").forEach((s) => s.classList.remove("live"));
    consoleBody.innerHTML = `<div class="empty-state"><p>Upload a slice to run the segmentation model, then watch the symbolic layer light up its reasoning path — tumor class → associated pattern → suggested treatment → outcome pattern.</p></div>`;
  }

  async function handleFile(file) {
    dropzone.hidden = true;
    scanImageWrap.hidden = false;

    const localUrl = URL.createObjectURL(file);
    scanImage.src = localUrl;
    scanline.hidden = false;

    setStage("neural");
    consoleBody.innerHTML = `<div class="status-line"><span class="pulse"></span> running FPN segmentation…</div>`;

    const formData = new FormData();
    formData.append("scan", file);

    try {
      const res = await fetch("/api/analyze", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok) {
        showError(data.error || "Something went wrong analyzing that scan.");
        return;
      }

      scanline.hidden = true;
      currentImages = { original: data.original, mask: data.mask, overlay: data.overlay };
      setActiveView("overlay");
      viewTabs.hidden = false;

      metaResolution.textContent = data.resolution;
      metaTumorCount.textContent = data.tumor_count === 1 ? "1 region detected" : `${data.tumor_count} regions detected`;
      viewerMeta.hidden = false;

      setStage("symbolic");
      renderTumors(data.tumors);
    } catch (err) {
      showError("Could not reach the analysis service. Is the Flask server running?");
    }
  }

  function setStage(name) {
    stageIndicator.querySelectorAll(".stage").forEach((s) => {
      s.classList.toggle("live", s.dataset.stage === name);
    });
  }

  function showError(message) {
    scanline.hidden = true;
    consoleBody.innerHTML = `<div class="error-line">${escapeHtml(message)}</div>`;
  }

  viewTabs.addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (!btn) return;
    setActiveView(btn.dataset.view);
  });

  function setActiveView(view) {
    if (!currentImages) return;
    scanImage.src = `data:image/png;base64,${currentImages[view]}`;
    viewTabs.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
  }

  
  function renderTumors(tumors) {
    consoleBody.innerHTML = "";

    if (!tumors.length) {
      const note = document.createElement("div");
      note.className = "no-tumor-note";
      note.textContent = "No abnormal region above threshold was found in this slice.";
      consoleBody.appendChild(note);
      return;
    }

    tumors.forEach((t) => {
      const node = tumorCardTemplate.content.cloneNode(true);
      node.querySelector(".tumor-id").textContent = `Region ${t.index}`;
      node.querySelector(".tumor-class-badge").textContent = t.size_class;
      node.querySelector(".m-area").textContent = `${t.area_mm2} mm²`;
      node.querySelector(".m-height").textContent = `${t.height_mm} mm`;
      node.querySelector(".m-width").textContent = `${t.width_mm} mm`;
      node.querySelector(".m-location").textContent = t.location;
      node.querySelector(".m-centroid").textContent = `(${t.centroid_px[0]}, ${t.centroid_px[1]})`;

      node.querySelector(".m-symptoms").textContent =
        t.associated_symptoms.length ? t.associated_symptoms.join(", ") : "—";
      node.querySelector(".m-treatment").textContent = t.suggested_treatment || "—";
      node.querySelector(".m-outcome").textContent = t.likely_outcome_pattern || "—";

      node.querySelector(".kg-image").src = `data:image/png;base64,${t.knowledge_graph_png}`;

      consoleBody.appendChild(node);
    });
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
})();
