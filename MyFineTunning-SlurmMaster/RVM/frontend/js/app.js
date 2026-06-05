/**
 * Visual Evaluation Frontend — Application Logic (Queue-Aware)
 * =============================================================
 * Handles: Upload, Model Selection, API Calls with Queue, Results Rendering, Export
 *
 * QUEUE SYSTEM:
 *   - Backend memproses 1 inferensi pada satu waktu (FIFO GPU queue)
 *   - Saat menunggu, frontend poll /api/queue/status setiap 2 detik
 *   - Menampilkan posisi antrian dan estimasi waktu
 */

(function () {
    "use strict";

    // =========================================================================
    // CONFIG
    // =========================================================================
    const API_BASE = detectApiBase();

    function detectApiBase() {
        if (
            window.location.hostname === "localhost" ||
            window.location.hostname === "127.0.0.1"
        ) {
            return `${window.location.protocol}//${window.location.hostname}:8502`;
        }
        return "https://backend-rvm.penelitian.my.id";
    }

    // =========================================================================
    // STATE
    // =========================================================================
    let selectedModels = new Set();
    let uploadedFile = null;
    let lastResults = null;
    let allModels = [];
    let queuePollInterval = null;

    // =========================================================================
    // DOM REFERENCES
    // =========================================================================
    const $ = (id) => document.getElementById(id);
    const $uploadZone = $("uploadZone");
    const $fileInput = $("fileInput");
    const $previewContainer = $("previewContainer");
    const $previewImage = $("previewImage");
    const $removeImage = $("removeImage");
    const $selectCategory = $("selectCategory");
    const $selectModel = $("selectModel");
    const $selectVariant = $("selectVariant");
    const $btnAddModel = $("btnAddModel");
    const $selectedCount = $("selectedCount");
    const $selectedModelChips = $("selectedModelChips");
    const $btnEvaluate = $("btnEvaluate");
    const $btnText = $("btnText");
    const $spinner = $("spinner");
    const $resultsSection = $("resultsSection");
    const $resultsSummary = $("resultsSummary");
    const $resultsContainer = $("resultsContainer");
    const $confSlider = $("confSlider");
    const $confValue = $("confValue");
    const $iouSlider = $("iouSlider");
    const $iouValue = $("iouValue");
    const $statusText = $("statusText");
    const $themeToggle = $("themeToggle");
    const $toast = $("toast");
    const $exportJson = $("exportJson");
    const $exportCsv = $("exportCsv");

    // =========================================================================
    // INITIALIZATION
    // =========================================================================
    async function init() {
        setupTheme();
        setupUpload();
        setupSliders();
        setupQuickSelect();
        setupExport();
        await fetchModels();
        checkHealth();
    }

    // =========================================================================
    // THEME
    // =========================================================================
    function setupTheme() {
        const saved = localStorage.getItem("theme") || "dark";
        document.documentElement.setAttribute("data-theme", saved);
        updateThemeIcon(saved);

        $themeToggle.addEventListener("click", () => {
            const current = document.documentElement.getAttribute("data-theme");
            const next = current === "dark" ? "light" : "dark";
            document.documentElement.setAttribute("data-theme", next);
            localStorage.setItem("theme", next);
            updateThemeIcon(next);
        });
    }

    function updateThemeIcon(theme) {
        $themeToggle.textContent = theme === "dark" ? "🌙" : "☀️";
    }

    // =========================================================================
    // HEALTH CHECK
    // =========================================================================
    async function checkHealth() {
        try {
            const res = await fetch(`${API_BASE}/api/health`, {
                signal: AbortSignal.timeout(10000),
            });
            if (res.ok) {
                const data = await res.json();
                const gpu = data.gpu?.name || "CPU";
                const qPending = data.queue?.pending || 0;
                $statusText.textContent = `Online • ${data.models_available} models • ${gpu}${qPending > 0 ? ` • Queue: ${qPending}` : ""}`;
                document.querySelector(".status-dot").style.background =
                    "var(--c-success)";
                
                // Update slider secara dinamis dari config_shared (jika tersedia)
                if (data.config) {
                    if (data.config.eval_conf !== undefined) {
                        $confSlider.value = data.config.eval_conf;
                        $confValue.textContent = parseFloat(data.config.eval_conf).toFixed(2);
                    }
                    if (data.config.eval_iou !== undefined) {
                        $iouSlider.value = data.config.eval_iou;
                        $iouValue.textContent = parseFloat(data.config.eval_iou).toFixed(2);
                    }
                }
            } else {
                throw new Error(`HTTP ${res.status}`);
            }
        } catch {
            $statusText.textContent = "Backend Offline";
            document.querySelector(".status-dot").style.background =
                "var(--c-error)";
        }
    }

    // =========================================================================
    // FETCH MODELS
    // =========================================================================
    async function fetchModels() {
        try {
            const res = await fetch(`${API_BASE}/api/models`, {
                signal: AbortSignal.timeout(15000),
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();

            allModels = data.models || [];
            
            // Inisialisasi dropdown cascading
            setupDropdownSelector();

            const defaultModel = allModels.find((m) => m.key === "yolo11l");
            if (defaultModel) {
                selectedModels.add(defaultModel.key);
                renderSelectedChips();
            }
            updateBtnState();
        } catch (err) {
            $selectedModelChips.innerHTML = `
                <div class="error-message">
                    ⚠️ Failed to load models from backend.<br>
                    <small>${escapeHtml(err.message)}</small>
                </div>`;
        }
    }

    // =========================================================================
    // CASCADING DROPDOWN SELECTOR
    // =========================================================================
    function setupDropdownSelector() {
        // Event listener kategori
        $selectCategory.addEventListener("change", () => {
            const cat = $selectCategory.value;
            $selectModel.innerHTML = '<option value="">-- Select Model --</option>';
            $selectVariant.innerHTML = '<option value="">-- Select Variant --</option>';
            $selectModel.disabled = true;
            $selectVariant.disabled = true;
            $btnAddModel.disabled = true;

            if (!cat) return;

            // Dapatkan model family berdasarkan kategori
            const families = new Set();
            allModels.forEach((m) => {
                let match = false;
                if (cat === "yolo") match = m.type === "yolo";
                else if (cat === "hybrid") match = m.type.startsWith("hybrid_") || m.family.includes("Hybrid");
                else if (cat === "detection") match = m.task === "detection";
                else if (cat === "segmentation") match = m.task === "segmentation";

                if (match) families.add(m.family);
            });

            // Urutkan dan masukkan ke dropdown model
            const sortedFamilies = Array.from(families).sort();
            sortedFamilies.forEach((fam) => {
                const opt = document.createElement("option");
                opt.value = fam;
                opt.textContent = fam;
                $selectModel.appendChild(opt);
            });

            $selectModel.disabled = false;
        });

        // Event listener model family
        $selectModel.addEventListener("change", () => {
            const cat = $selectCategory.value;
            const fam = $selectModel.value;
            $selectVariant.innerHTML = '<option value="">-- Select Variant --</option>';
            $selectVariant.disabled = true;
            $btnAddModel.disabled = true;

            if (!fam) return;

            // Dapatkan varian berdasarkan kategori dan family
            const variants = allModels.filter((m) => {
                let matchCat = false;
                if (cat === "yolo") matchCat = m.type === "yolo";
                else if (cat === "hybrid") matchCat = m.type.startsWith("hybrid_") || m.family.includes("Hybrid");
                else if (cat === "detection") matchCat = m.task === "detection";
                else if (cat === "segmentation") matchCat = m.task === "segmentation";

                return matchCat && m.family === fam;
            });

            // Urutkan dan masukkan ke dropdown varian
            variants.sort((a, b) => a.display_name.localeCompare(b.display_name));
            variants.forEach((v) => {
                const opt = document.createElement("option");
                opt.value = v.key;
                const sizeText = v.weights_size_mb !== "N/A" ? ` (${v.weights_size_mb} MB)` : "";
                opt.textContent = `${v.display_name}${sizeText}`;
                $selectVariant.appendChild(opt);
            });

            $selectVariant.disabled = false;
        });

        // Event listener varian
        $selectVariant.addEventListener("change", () => {
            $btnAddModel.disabled = !$selectVariant.value;
        });

        // Event listener tombol Add Model
        $btnAddModel.addEventListener("click", () => {
            const key = $selectVariant.value;
            if (key) {
                if (selectedModels.has(key)) {
                    showToast("Model already selected", "error");
                    return;
                }
                selectedModels.add(key);
                renderSelectedChips();
                updateBtnState();
                showToast("Model added to selection", "success");
            }
        });
    }

    // Render chips untuk model yang dipilih
    function renderSelectedChips() {
        $selectedCount.textContent = selectedModels.size;
        $selectedModelChips.innerHTML = "";

        if (selectedModels.size === 0) {
            $selectedModelChips.innerHTML = '<span class="no-selection-placeholder">No models selected. Use the dropdown or quick select above.</span>';
            return;
        }

        selectedModels.forEach((key) => {
            const m = allModels.find((model) => model.key === key);
            if (!m) return;

            const chip = document.createElement("div");
            chip.className = "selected-model-chip";
            chip.innerHTML = `
                <span>${escapeHtml(m.display_name)}</span>
                <button class="selected-model-chip__remove" data-key="${escapeHtml(key)}" title="Remove ${escapeHtml(m.display_name)}">✕</button>
            `;

            chip.querySelector(".selected-model-chip__remove").addEventListener("click", (e) => {
                e.stopPropagation();
                selectedModels.delete(key);
                renderSelectedChips();
                updateBtnState();
            });

            $selectedModelChips.appendChild(chip);
        });
    }

    // =========================================================================
    // QUICK SELECT
    // =========================================================================
    function setupQuickSelect() {
        document.querySelectorAll(".quick-select__btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                const action = btn.dataset.action;
                switch (action) {
                    case "select-all":
                        allModels.forEach((m) => selectedModels.add(m.key));
                        break;
                    case "deselect-all":
                        selectedModels.clear();
                        break;
                    case "select-yolo":
                        allModels.forEach((m) => {
                            if (m.type === "yolo" && !m.family.includes("RT-DETR"))
                                selectedModels.add(m.key);
                        });
                        break;
                    case "select-hybrid":
                        allModels.forEach((m) => {
                            if (m.family.includes("Hybrid")) selectedModels.add(m.key);
                        });
                        break;
                    case "select-det":
                        selectedModels.clear();
                        allModels.forEach((m) => {
                            if (m.task === "detection") selectedModels.add(m.key);
                        });
                        break;
                    case "select-seg":
                        selectedModels.clear();
                        allModels.forEach((m) => {
                            if (m.task === "segmentation") selectedModels.add(m.key);
                        });
                        break;
                }
                renderSelectedChips();
                updateBtnState();
                showToast(`${selectedModels.size} model(s) selected`, "success");
            });
        });
    }

    // =========================================================================
    // UPLOAD
    // =========================================================================
    function setupUpload() {
        $uploadZone.addEventListener("click", () => $fileInput.click());
        $uploadZone.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $fileInput.click(); }
        });
        $uploadZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            $uploadZone.classList.add("upload-zone--active");
        });
        $uploadZone.addEventListener("dragleave", () => {
            $uploadZone.classList.remove("upload-zone--active");
        });
        $uploadZone.addEventListener("drop", (e) => {
            e.preventDefault();
            $uploadZone.classList.remove("upload-zone--active");
            if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
        });
        $fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) handleFile(e.target.files[0]);
        });
        $removeImage.addEventListener("click", clearUpload);
    }

    function handleFile(file) {
        const validExt = [".jpg", ".jpeg", ".png", ".bmp", ".webp"];
        const ext = "." + file.name.split(".").pop().toLowerCase();

        if (!validExt.includes(ext)) {
            showToast(`Invalid file type: ${ext}`, "error");
            return;
        }
        if (file.size > 16 * 1024 * 1024) {
            showToast("File too large (max 16 MB)", "error");
            return;
        }

        uploadedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            $previewImage.src = e.target.result;
            $previewContainer.classList.add("visible");
            $uploadZone.style.display = "none";
        };
        reader.readAsDataURL(file);
        updateBtnState();
        showToast(`Image loaded: ${file.name}`, "success");
    }

    function clearUpload() {
        uploadedFile = null;
        $previewContainer.classList.remove("visible");
        $previewImage.src = "";
        $uploadZone.style.display = "";
        $fileInput.value = "";
        updateBtnState();
    }

    // =========================================================================
    // SLIDERS
    // =========================================================================
    function setupSliders() {
        $confSlider.addEventListener("input", () => {
            $confValue.textContent = parseFloat($confSlider.value).toFixed(2);
        });
        $iouSlider.addEventListener("input", () => {
            $iouValue.textContent = parseFloat($iouSlider.value).toFixed(2);
        });
    }

    // =========================================================================
    // EVALUATE (Queue-Aware)
    // =========================================================================
    function updateBtnState() {
        const canEval = uploadedFile && selectedModels.size > 0;
        $btnEvaluate.disabled = !canEval;

        if (!$btnEvaluate._bound) {
            $btnEvaluate.addEventListener("click", runEvaluation);
            $btnEvaluate._bound = true;
        }
    }

    async function runEvaluation() {
        if (!uploadedFile || selectedModels.size === 0) return;

        // UI: Loading state
        $btnEvaluate.disabled = true;
        $btnEvaluate.classList.add("btn-evaluate--loading");
        $btnText.textContent = `⏳ Submitting to GPU queue...`;
        $spinner.classList.add("visible");
        $resultsSection.classList.remove("visible");

        // Start queue polling
        startQueuePolling();

        const formData = new FormData();
        formData.append("image", uploadedFile);
        formData.append("models", Array.from(selectedModels).join(","));
        formData.append("conf", $confSlider.value);
        formData.append("iou", $iouSlider.value);

        try {
            const res = await fetch(`${API_BASE}/api/evaluate`, {
                method: "POST",
                body: formData,
            });

            stopQueuePolling();

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.message || `Server error: HTTP ${res.status}`);
            }

            const data = await res.json();

            if (data.status === "timeout") {
                showToast(
                    `Request timeout. Job ID: ${data.job_id}. Check queue status.`,
                    "error"
                );
                return;
            }

            lastResults = data;
            renderResults(data);

            const waitInfo = data.queue_wait_ms
                ? ` (waited ${(data.queue_wait_ms / 1000).toFixed(1)}s in queue)`
                : "";
            showToast(
                `Evaluation complete! ${data.total_models_evaluated} models in ${(data.total_time_ms / 1000).toFixed(1)}s${waitInfo}`,
                "success"
            );
        } catch (err) {
            stopQueuePolling();
            showToast(`Evaluation failed: ${err.message}`, "error");
        } finally {
            $btnEvaluate.disabled = false;
            $btnEvaluate.classList.remove("btn-evaluate--loading");
            $btnText.textContent = "🚀 Run Evaluation";
            $spinner.classList.remove("visible");
        }
    }

    // =========================================================================
    // QUEUE POLLING — Monitor antrian saat menunggu
    // =========================================================================
    function startQueuePolling() {
        stopQueuePolling();
        queuePollInterval = setInterval(async () => {
            try {
                const res = await fetch(`${API_BASE}/api/queue/status`, {
                    signal: AbortSignal.timeout(5000),
                });
                if (res.ok) {
                    const data = await res.json();
                    const pending = data.queue_size || 0;
                    const processing = data.currently_processing ? "🔥 Processing" : "";

                    if (pending > 0) {
                        $btnText.textContent = `⏳ Queue: ${pending} ahead • ${processing}`;
                    } else if (processing) {
                        $btnText.textContent = `🔥 Your request is being processed...`;
                    } else {
                        $btnText.textContent = `⏳ Waiting for GPU...`;
                    }
                }
            } catch {
                // Silent fail — polling is best-effort
            }
        }, 2000);
    }

    function stopQueuePolling() {
        if (queuePollInterval) {
            clearInterval(queuePollInterval);
            queuePollInterval = null;
        }
    }

    // =========================================================================
    // RESULTS RENDERING
    // =========================================================================
    function renderResults(data) {
        const totalDets = data.results.reduce(
            (sum, r) => sum + (r.detections?.length || 0), 0
        );
        const avgTime = data.results.length > 0
            ? data.results.reduce((sum, r) => sum + (r.inference_time_ms || 0), 0) / data.results.length
            : 0;
        const errCount = data.results.filter((r) => r.error).length;

        let summaryHtml = `
            <div class="stat-card">
                <div class="stat-card__value">${data.total_models_evaluated}</div>
                <div class="stat-card__label">Models Evaluated</div>
            </div>
            <div class="stat-card">
                <div class="stat-card__value">${totalDets}</div>
                <div class="stat-card__label">Total Detections</div>
            </div>
            <div class="stat-card">
                <div class="stat-card__value">${(data.total_time_ms / 1000).toFixed(1)}s</div>
                <div class="stat-card__label">Total Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-card__value">${avgTime.toFixed(0)}ms</div>
                <div class="stat-card__label">Avg per Model</div>
            </div>`;

        if (data.queue_wait_ms && data.queue_wait_ms > 100) {
            summaryHtml += `
            <div class="stat-card">
                <div class="stat-card__value">${(data.queue_wait_ms / 1000).toFixed(1)}s</div>
                <div class="stat-card__label">Queue Wait</div>
            </div>`;
        }

        if (errCount > 0) {
            summaryHtml += `
            <div class="stat-card">
                <div class="stat-card__value" style="color:var(--c-error);">${errCount}</div>
                <div class="stat-card__label">Errors</div>
            </div>`;
        }

        $resultsSummary.innerHTML = summaryHtml;

        // Result Cards
        let cardsHtml = "";
        const sorted = [...data.results].sort(
            (a, b) => (b.detections?.length || 0) - (a.detections?.length || 0)
        );

        for (const r of sorted) {
            const hasError = !!r.error;
            const detCount = r.detections?.length || 0;
            const cardId = `card-${r.model_key || Math.random().toString(36).slice(2)}`;

            cardsHtml += `
            <div class="result-card ${hasError ? "result-card--error" : ""}" id="${cardId}">
                <div class="result-card__header" onclick="window.__toggleCard('${cardId}')">
                    <div>
                        <span class="result-card__model-name">${escapeHtml(r.model || r.model_key)}</span>
                        <span style="font-size:0.75rem;color:var(--c-text-dim);margin-left:8px;">${escapeHtml(r.family || r.model_type || "")}</span>
                    </div>
                    <div class="result-card__meta">
                        ${r.inference_time_ms ? `<span class="result-card__time">${r.inference_time_ms.toFixed(0)} ms</span>` : ""}
                        <span class="result-card__count">${detCount} det${detCount !== 1 ? "s" : ""}</span>
                        <span class="result-card__chevron" id="${cardId}-chevron">▼</span>
                    </div>
                </div>
                <div class="result-card__body" id="${cardId}-body">
                    ${hasError ? `<div class="error-message">⚠️ ${escapeHtml(r.error)}</div>` : ""}
                    ${detCount > 0 ? renderDetTable(r.detections) : '<p style="color:var(--c-text-dim);font-size:0.85rem;">No detections at this confidence threshold.</p>'}
                </div>
            </div>`;
        }

        $resultsContainer.innerHTML = cardsHtml;
        $resultsSection.classList.add("visible");
        $resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function renderDetTable(detections) {
        let html = `
            <table class="det-table"><thead><tr>
                <th>#</th><th>Class</th><th>Confidence</th><th>BBox</th><th>Mask Area</th>
            </tr></thead><tbody>`;

        for (let i = 0; i < detections.length; i++) {
            const d = detections[i];
            const confPct = ((d.confidence || 0) * 100).toFixed(1);
            const barWidth = Math.max(4, Math.min(80, (d.confidence || 0) * 80));
            html += `
                <tr>
                    <td>${i + 1}</td>
                    <td><strong>${escapeHtml(d.class || "unknown")}</strong></td>
                    <td><span class="conf-bar" style="width:${barWidth}px;"></span>${confPct}%</td>
                    <td style="font-family:var(--font-mono);font-size:0.78rem;">
                        ${d.bbox ? d.bbox.map((v) => v.toFixed(0)).join(", ") : "N/A"}
                    </td>
                    <td style="font-family:var(--font-mono);">
                        ${d.mask_area != null ? d.mask_area.toLocaleString() + " px" : "—"}
                    </td>
                </tr>`;
        }
        html += `</tbody></table>`;
        return html;
    }

    window.__toggleCard = function (cardId) {
        const body = document.getElementById(`${cardId}-body`);
        const chevron = document.getElementById(`${cardId}-chevron`);
        if (body && chevron) {
            body.classList.toggle("expanded");
            chevron.classList.toggle("rotated");
        }
    };

    // =========================================================================
    // EXPORT
    // =========================================================================
    function setupExport() {
        $exportJson.addEventListener("click", () => {
            if (!lastResults) { showToast("No results to export", "error"); return; }
            downloadBlob(JSON.stringify(lastResults, null, 2), "evaluation_results.json", "application/json");
            showToast("JSON exported!", "success");
        });

        $exportCsv.addEventListener("click", () => {
            if (!lastResults) { showToast("No results to export", "error"); return; }
            downloadBlob(resultsToCsv(lastResults), "evaluation_results.csv", "text/csv");
            showToast("CSV exported!", "success");
        });
    }

    function resultsToCsv(data) {
        const headers = ["Model", "Family", "Type", "Inference (ms)", "Detection #", "Class", "Confidence", "BBox", "Mask Area"];
        const rows = [headers.join(",")];

        for (const r of data.results) {
            if (r.detections && r.detections.length > 0) {
                for (let i = 0; i < r.detections.length; i++) {
                    const d = r.detections[i];
                    rows.push([
                        csvEscape(r.model), csvEscape(r.family || ""), csvEscape(r.model_type || ""),
                        r.inference_time_ms || "", i + 1, csvEscape(d.class || ""),
                        d.confidence || "", d.bbox ? `"${d.bbox.join(", ")}"` : "",
                        d.mask_area != null ? d.mask_area : "",
                    ].join(","));
                }
            } else {
                rows.push([csvEscape(r.model), csvEscape(r.family || ""), csvEscape(r.model_type || ""),
                    r.inference_time_ms || "", 0, "", "", "", ""].join(","));
            }
        }
        return rows.join("\n");
    }

    function downloadBlob(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url; a.download = filename; a.click();
        URL.revokeObjectURL(url);
    }

    // =========================================================================
    // UTILITIES
    // =========================================================================
    function escapeHtml(str) {
        if (!str) return "";
        return str.replace(/&/g, "&amp;").replace(/</g, "&lt;")
            .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }

    function csvEscape(str) {
        if (!str) return "";
        if (str.includes(",") || str.includes('"') || str.includes("\n"))
            return `"${str.replace(/"/g, '""')}"`;
        return str;
    }

    let toastTimer = null;
    function showToast(message, type = "success") {
        $toast.textContent = message;
        $toast.className = `toast toast--${type} visible`;
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => { $toast.classList.remove("visible"); }, 3000);
    }

    // =========================================================================
    // BOOT
    // =========================================================================
    document.addEventListener("DOMContentLoaded", init);
})();
