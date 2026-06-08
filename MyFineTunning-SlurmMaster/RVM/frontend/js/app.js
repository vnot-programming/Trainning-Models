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

    // Cloudflare Zero Trust Service Token untuk koneksi API eksternal (diinject oleh backend /env.js)
    const CF_HEADERS = window.RVM_ENV || {
        "CF-Access-Client-Id": "",
        "CF-Access-Client-Secret": ""
    };

    function detectApiBase() {
        if (
            window.location.hostname === "localhost" ||
            window.location.hostname === "127.0.0.1"
        ) {
            return `${window.location.protocol}//${window.location.hostname}:8502`;
        }
        // Gunakan relative path agar API request masuk ke server frontend (port 8501)
        // yang akan mem-proxy-kannya ke backend lokal (port 8502).
        // Ini menghindari blokir CORS OPTIONS dari Cloudflare Access.
        return "";
    }

    // =========================================================================
    // STATE
    // =========================================================================
    const MAX_COMPARISON_MODELS = 5;  // Batas max model dalam Image Comparison Mode

    let selectedModels = new Set();
    let uploadedFile = null;
    let lastResults = null;
    let allModels = [];
    let queuePollInterval = null;
    let imageComparisonMode = false;  // Flag mode perbandingan gambar

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
    // --- Elemen baru untuk Image Comparison Mode ---
    const $imageComparisonCheckbox = $("imageComparisonCheckbox");
    const $resultPanel = $("resultPanel");
    const $resultPanelPlaceholder = $("resultPanelPlaceholder");
    const $comparisonGrid = $("comparisonGrid");
    const $comparisonModeBanner = $("comparisonModeBanner");
    const $comparisonModelCount = $("comparisonModelCount");
    const $comparisonLimitHint = $("comparisonLimitHint");
    const $btnDownloadGrid = $("btnDownloadGrid");

    // =========================================================================
    // INITIALIZATION
    // =========================================================================
    async function init() {
        setupTheme();
        setupUpload();
        setupSliders();
        setupQuickSelect();
        setupExport();
        setupImageComparison();
        setupGridDownload();
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
                headers: CF_HEADERS,
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
                headers: CF_HEADERS,
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

        // Event listener tombol Add Model (dengan validasi batas comparison mode)
        $btnAddModel.addEventListener("click", () => {
            const key = $selectVariant.value;
            if (key) {
                if (selectedModels.has(key)) {
                    showToast("Model already selected", "error");
                    return;
                }
                // Cek batas maks 5 model jika comparison mode aktif
                if (imageComparisonMode && selectedModels.size >= MAX_COMPARISON_MODELS) {
                    showToast(`⚠️ Image Comparison Mode: Maks. ${MAX_COMPARISON_MODELS} model yang dapat dipilih.`, "error");
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

        // Update badge dan hint saat comparison mode aktif
        if (imageComparisonMode) {
            const count = selectedModels.size;
            $comparisonModelCount.textContent = `${count}/${MAX_COMPARISON_MODELS}`;
            $comparisonModelCount.classList.toggle("limit-reached", count >= MAX_COMPARISON_MODELS);
        }

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

                // Fungsi helper: tambah model dengan tetap mematuhi batas comparison mode
                const addWithLimit = (key) => {
                    if (imageComparisonMode && selectedModels.size >= MAX_COMPARISON_MODELS) return;
                    selectedModels.add(key);
                };

                switch (action) {
                    case "select-all":
                        if (imageComparisonMode) {
                            showToast(`⚠️ Comparison Mode aktif: Maks. ${MAX_COMPARISON_MODELS} model. Gunakan Add Model satu per satu.`, "error");
                            return;
                        }
                        allModels.forEach((m) => selectedModels.add(m.key));
                        break;
                    case "deselect-all":
                        selectedModels.clear();
                        break;
                    case "select-yolo":
                        allModels.forEach((m) => {
                            if (m.type === "yolo" && !m.family.includes("RT-DETR"))
                                addWithLimit(m.key);
                        });
                        break;
                    case "select-hybrid":
                        allModels.forEach((m) => {
                            if (m.family.includes("Hybrid")) addWithLimit(m.key);
                        });
                        break;
                    case "select-det":
                        selectedModels.clear();
                        allModels.forEach((m) => {
                            if (m.task === "detection") addWithLimit(m.key);
                        });
                        break;
                    case "select-seg":
                        selectedModels.clear();
                        allModels.forEach((m) => {
                            if (m.task === "segmentation") addWithLimit(m.key);
                        });
                        break;
                }
                renderSelectedChips();
                updateBtnState();
                const limitNote = imageComparisonMode ? ` (Comparison Mode: maks. ${MAX_COMPARISON_MODELS})` : "";
                showToast(`${selectedModels.size} model(s) selected${limitNote}`, "success");
            });
        });
    }

    // =========================================================================
    // IMAGE COMPARISON MODE
    // =========================================================================
    function setupImageComparison() {
        $imageComparisonCheckbox.addEventListener("change", () => {
            imageComparisonMode = $imageComparisonCheckbox.checked;

            // Toggle: banner di Select Models
            $comparisonModeBanner.style.display = imageComparisonMode ? "flex" : "none";

            // Toggle: hint batas di Selected Models header
            $comparisonLimitHint.style.display = imageComparisonMode ? "inline-flex" : "none";

            // Toggle: class enabled di result panel
            if (imageComparisonMode) {
                $resultPanel.classList.add("comparison-enabled");
                $resultPanelPlaceholder.querySelector(".result-panel-placeholder__icon").textContent = "📊";
                $resultPanelPlaceholder.querySelector(".result-panel-placeholder__text").innerHTML =
                    "Jalankan evaluasi untuk melihat <strong>visualisasi perbandingan</strong> model di sini.";
            } else {
                $resultPanel.classList.remove("comparison-enabled");
                $resultPanelPlaceholder.querySelector(".result-panel-placeholder__icon").textContent = "🔒";
                $resultPanelPlaceholder.querySelector(".result-panel-placeholder__text").innerHTML =
                    "Aktifkan <strong>Image Comparison</strong> untuk melihat visualisasi di sini.";
                // Sembunyikan grid, tampilkan placeholder kembali
                $comparisonGrid.style.display = "none";
                $resultPanelPlaceholder.style.display = "flex";
                $btnDownloadGrid.style.display = "none";
            }

            // Jika comparison mode aktif dan model sudah lebih dari batas, potong seleksi
            if (imageComparisonMode && selectedModels.size > MAX_COMPARISON_MODELS) {
                const keys = Array.from(selectedModels).slice(0, MAX_COMPARISON_MODELS);
                selectedModels.clear();
                keys.forEach((k) => selectedModels.add(k));
                showToast(`Comparison Mode: Seleksi dipotong ke ${MAX_COMPARISON_MODELS} model pertama.`, "error");
            }

            // Update badge count
            $comparisonModelCount.textContent = `${selectedModels.size}/${MAX_COMPARISON_MODELS}`;
            $comparisonModelCount.classList.toggle("limit-reached", selectedModels.size >= MAX_COMPARISON_MODELS);

            renderSelectedChips();
            updateBtnState();
        });
    }

    // DOWNLOAD GRID IMAGE
    // =========================================================================
    function setupGridDownload() {
        $btnDownloadGrid.addEventListener("click", () => {
            const cells = $comparisonGrid.querySelectorAll(".comparison-cell");
            if (cells.length === 0) return;

            // Cari ukuran gambar natural dari gambar pertama yang di-load
            const firstCanvas = cells[0].querySelector("canvas");
            if (!firstCanvas) return;

            const cellW = firstCanvas.width;
            const cellH = firstCanvas.height;

            if (cellW === 0 || cellH === 0) {
                showToast("Gambar belum selesai dimuat, silakan coba lagi.", "error");
                return;
            }

            // Buat canvas gabungan besar
            const gridCanvas = document.createElement("canvas");
            // Grid 3x2
            gridCanvas.width = cellW * 3;
            gridCanvas.height = cellH * 2;

            const gridCtx = gridCanvas.getContext("2d");
            // Isi background sesuai tema aktif
            const isLight = document.documentElement.getAttribute("data-theme") === "light";
            gridCtx.fillStyle = isLight ? "#f3f4f6" : "#0f1115";
            gridCtx.fillRect(0, 0, gridCanvas.width, gridCanvas.height);

            cells.forEach((cell, idx) => {
                const col = idx % 3;
                const row = Math.floor(idx / 3);
                const x = col * cellW;
                const y = row * cellH;

                const cellCanvas = cell.querySelector("canvas");
                if (cellCanvas) {
                    gridCtx.drawImage(cellCanvas, x, y, cellW, cellH);

                    // Gambar teks header model name di pojok kiri atas setiap sel
                    const labelSpan = cell.querySelector(".comparison-cell__header span");
                    const labelText = labelSpan ? labelSpan.textContent : (idx === 0 ? "Ground Truth" : `Model ${idx}`);
                    
                    gridCtx.save();
                    // Bikin banner gelap transparan untuk teks header
                    const bannerH = Math.max(32, cellH * 0.06);
                    gridCtx.fillStyle = "rgba(0, 0, 0, 0.7)";
                    gridCtx.fillRect(x, y, cellW, bannerH);

                    // Tulis teks model
                    gridCtx.fillStyle = "#ffffff";
                    const fontS = Math.max(12, cellH * 0.026);
                    gridCtx.font = `bold ${fontS}px Inter, -apple-system, sans-serif`;
                    gridCtx.textBaseline = "middle";
                    gridCtx.fillText(labelText, x + 15, y + bannerH / 2);
                    gridCtx.restore();
                }
            });

            // Trigger download
            try {
                const link = document.createElement("a");
                const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
                link.download = `rvm_comparison_grid_${timestamp}.png`;
                link.href = gridCanvas.toDataURL("image/png");
                link.click();
                showToast("Grid comparison image downloaded successfully!", "success");
            } catch (err) {
                showToast(`Failed to download grid: ${err.message}`, "error");
            }
        });
    }

    /**
     * Render grid perbandingan 3×2: Ground Truth + hingga 5 model.
     * Setiap cell menampilkan gambar original + bounding box dari hasil deteksi.
     * @param {string} originalSrc   - Data URL gambar original (dari FileReader)
     * @param {Array}  results       - Array hasil evaluasi dari API (maks. 5)
     */
    function renderComparisonGrid(originalSrc, results) {
        $comparisonGrid.innerHTML = "";

        // === Cell 0: Ground Truth (gambar original tanpa anotasi) ===
        const gtCell = buildComparisonCell({
            label: "Ground Truth",
            badge: "Original",
            isGroundTruth: true,
            imageSrc: originalSrc,
            detections: [],
            inferenceMs: null,
            hasError: false,
        });
        $comparisonGrid.appendChild(gtCell);

        // === Cell 1–5: Satu cell per model ===
        const topResults = results.slice(0, MAX_COMPARISON_MODELS);
        topResults.forEach((r, idx) => {
            const cell = buildComparisonCell({
                label: r.model || r.model_key || `Model ${idx + 1}`,
                badge: r.family || r.model_type || "N/A",
                isGroundTruth: false,
                imageSrc: originalSrc,
                detections: r.detections || [],
                inferenceMs: r.inference_time_ms || null,
                hasError: !!r.error,
                errorMsg: r.error || null,
            });
            $comparisonGrid.appendChild(cell);
        });

        // Sembunyikan placeholder, tampilkan grid
        $resultPanelPlaceholder.style.display = "none";
        $comparisonGrid.style.display = "grid";
        $btnDownloadGrid.style.display = "inline-flex";
    }

    /**
     * Bangun satu cell comparison (div + canvas + anotasi bbox).
     * Menggunakan Canvas API untuk overlay bounding box di atas gambar.
     */
    function buildComparisonCell({ label, badge, isGroundTruth, imageSrc, detections, inferenceMs, hasError, errorMsg }) {
        const cell = document.createElement("div");
        cell.className = `comparison-cell${isGroundTruth ? " comparison-cell--ground-truth" : ""}${hasError ? " comparison-cell--error" : ""}`;

        // Header
        const header = document.createElement("div");
        header.className = "comparison-cell__header";
        header.innerHTML = `
            <span title="${escapeHtml(label)}" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:70%;">${escapeHtml(label)}</span>
            <span class="comparison-cell__badge">${escapeHtml(badge)}</span>
        `;
        cell.appendChild(header);

        // Image wrapper + Canvas
        const wrapper = document.createElement("div");
        wrapper.className = "comparison-cell__img-wrapper";

        const canvas = document.createElement("canvas");
        canvas.className = "comparison-cell__canvas";
        canvas.setAttribute("role", "img");
        canvas.setAttribute("aria-label", `${label} — ${detections.length} deteksi`);
        wrapper.appendChild(canvas);
        cell.appendChild(wrapper);

        // Footer
        const footer = document.createElement("div");
        footer.className = "comparison-cell__footer";
        const detCount = hasError ? "Error" : `${detections.length} det`;
        const timeText = inferenceMs !== null ? `${inferenceMs.toFixed(0)} ms` : (isGroundTruth ? "Original" : "N/A");
        footer.innerHTML = `<span>${detCount}</span><span>${timeText}</span>`;
        cell.appendChild(footer);

        // Gambar di canvas + overlay bbox setelah gambar dimuat
        const img = new Image();
        img.onload = () => {
            // Dimensi canvas mengikuti natural image (resolusi tinggi)
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;

            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0);

            if (!hasError && !isGroundTruth && detections.length > 0) {
                _drawBoundingBoxes(ctx, detections, img.naturalWidth, img.naturalHeight);
            }

            if (hasError && errorMsg) {
                // Overlay teks error di atas gambar
                ctx.fillStyle = "rgba(220, 50, 50, 0.75)";
                ctx.fillRect(0, img.naturalHeight - 40, img.naturalWidth, 40);
                ctx.fillStyle = "white";
                ctx.font = `bold ${Math.max(12, img.naturalWidth * 0.03)}px sans-serif`;
                ctx.fillText(`⚠ ${errorMsg}`.slice(0, 60), 8, img.naturalHeight - 12);
            }
        };
        img.onerror = () => {
            const ctx = canvas.getContext("2d");
            canvas.width = 400;
            canvas.height = 300;
            ctx.fillStyle = "#1a1a2e";
            ctx.fillRect(0, 0, 400, 300);
            ctx.fillStyle = "#888";
            ctx.font = "14px sans-serif";
            ctx.fillText("Gagal memuat gambar", 140, 150);
        };
        img.src = imageSrc;

        return cell;
    }

    /**
     * Gambar bounding box di atas canvas dengan label class dan confidence.
     * Warna berbeda per class agar mudah dibedakan.
     */
    function _drawBoundingBoxes(ctx, detections, imgW, imgH) {
        // Palet warna untuk class yang berbeda
        const colorPalette = [
            "hsl(168, 76%, 52%)",  // teal (primary)
            "hsl(263, 70%, 68%)",  // ungu (accent)
            "hsl(38, 92%, 56%)",   // oranye (warning)
            "hsl(210, 78%, 56%)",  // biru (info)
            "hsl(152, 68%, 46%)",  // hijau (success)
            "hsl(0, 72%, 56%)",    // merah (error)
            "hsl(300, 60%, 60%)",  // pink
            "hsl(60, 80%, 55%)",   // kuning
        ];
        const classColorMap = {};
        let colorIdx = 0;

        const lineW = Math.max(2, imgW * 0.003);
        const fontSize = Math.max(11, imgW * 0.022);
        const labelPad = Math.max(4, imgW * 0.005);

        ctx.lineWidth = lineW;
        ctx.font = `bold ${fontSize}px Inter, sans-serif`;

        detections.forEach((det) => {
            if (!det.bbox || det.bbox.length < 4) return;

            const cls = det.class || "unknown";
            if (!classColorMap[cls]) {
                classColorMap[cls] = colorPalette[colorIdx % colorPalette.length];
                colorIdx++;
            }
            const color = classColorMap[cls];

            // 1. Gambar mask segmentasi (jika model merupakan segmentasi & mengembalikan segmen poligon)
            if (det.segment && Array.isArray(det.segment) && det.segment.length > 0) {
                ctx.save();
                const fillMaskColor = color.replace(")", ", 0.35)").replace("hsl", "hsla");
                ctx.fillStyle = fillMaskColor;
                ctx.beginPath();
                ctx.moveTo(det.segment[0][0], det.segment[0][1]);
                for (let j = 1; j < det.segment.length; j++) {
                    ctx.lineTo(det.segment[j][0], det.segment[j][1]);
                }
                ctx.closePath();
                ctx.fill();
                
                // Outline mask yang lebih halus
                ctx.strokeStyle = color;
                ctx.lineWidth = Math.max(1.5, lineW * 0.4);
                ctx.stroke();
                ctx.restore();
            }

            // 2. Gambar bounding box
            const [x1, y1, x2, y2] = det.bbox;
            const w = x2 - x1;
            const h = y2 - y1;

            ctx.strokeStyle = color;
            ctx.lineWidth = lineW;
            ctx.strokeRect(x1, y1, w, h);

            // Label background
            const confPct = ((det.confidence || 0) * 100).toFixed(0);
            const labelText = `${cls} ${confPct}%`;
            const textMetrics = ctx.measureText(labelText);
            const labelW = textMetrics.width + labelPad * 2;
            const labelH = fontSize + labelPad * 2;

            ctx.fillStyle = color;
            // Posisi label: di atas bbox, jika tidak muat geser ke dalam
            const labelY = y1 - labelH >= 0 ? y1 - labelH : y1;
            ctx.fillRect(x1, labelY, labelW, labelH);

            // Teks label
            ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
            ctx.fillText(labelText, x1 + labelPad, labelY + fontSize + labelPad * 0.5);
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
                headers: CF_HEADERS,
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

            // === Jika Image Comparison Mode aktif, render grid 3x2 di panel kanan ===
            if (imageComparisonMode && uploadedFile) {
                // Baca ulang gambar sebagai Data URL untuk di-render di canvas
                const reader = new FileReader();
                reader.onload = (e) => {
                    renderComparisonGrid(e.target.result, data.results || []);
                };
                reader.readAsDataURL(uploadedFile);
            }

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
                    headers: CF_HEADERS,
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
