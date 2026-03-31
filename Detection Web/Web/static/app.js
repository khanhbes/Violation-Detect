// =====================================================================
// TAB NAVIGATION
// =====================================================================
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const tabId = link.dataset.tab;
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        link.classList.add('active');
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        const panel = document.getElementById('tab-' + tabId);
        if (panel) {
            panel.classList.add('active');
            if (tabId === 'home') triggerAnimations(panel);
        }
    });
});

// Hero CTA buttons (navigate to tabs)
document.querySelectorAll('.hero-cta .btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        e.preventDefault();
        const tabId = btn.dataset.tab;
        if (tabId) {
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            const navLink = document.querySelector(`.nav-link[data-tab="${tabId}"]`);
            if (navLink) navLink.classList.add('active');
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            const panel = document.getElementById('tab-' + tabId);
            if (panel) panel.classList.add('active');
        }
    });
});

// =====================================================================
// SCROLL ANIMATIONS (Intersection Observer)
// =====================================================================
function triggerAnimations(container) {
    container.querySelectorAll('.animate-in').forEach((el, i) => {
        el.style.transitionDelay = (i * 0.08) + 's';
        el.classList.remove('visible');
        void el.offsetWidth;
        el.classList.add('visible');
    });
}

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) entry.target.classList.add('visible');
    });
}, { threshold: 0.1 });

document.querySelectorAll('.animate-in').forEach(el => observer.observe(el));
window.addEventListener('load', () => triggerAnimations(document.getElementById('tab-home')));

// =====================================================================
// ELEMENTS
// =====================================================================
const canvas = document.getElementById('videoCanvas');
const ctx = canvas.getContext('2d');
const videoOverlay = document.getElementById('videoOverlay');
const videoStats = document.getElementById('videoStats');
const progressBar = document.getElementById('progressBar');
const progressFill = document.getElementById('progressFill');
const videoSelect = document.getElementById('videoSelect');
const modelSelect = document.getElementById('modelSelect');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const debugStatus = document.getElementById('debugStatus');
const frameNum = document.getElementById('frameNum');
const fpsValue = document.getElementById('fpsValue');
const progressValue = document.getElementById('progressValue');
const helmetCount = document.getElementById('helmetCount');
const redlightCount = document.getElementById('redlightCount');
const sidewalkCount = document.getElementById('sidewalkCount');
const wrongwayCount = document.getElementById('wrongwayCount');
const wronglaneCount = document.getElementById('wronglaneCount');
const totalCount = document.getElementById('totalCount');
const snapshotsGallery = document.getElementById('snapshotsGallery');
const clearSnapshotsBtn = document.getElementById('clearSnapshotsBtn');
const toast = document.getElementById('toast');
const confRange = document.getElementById('confRange');
const debugToggle = document.getElementById('debugToggle');
const confBubble = document.getElementById('confBubble');
const rangeFill = document.getElementById('rangeFill');
const globalPageLoader = document.getElementById('globalPageLoader');
const globalPageLoaderMessage = document.getElementById('globalPageLoaderMessage');

let ws = null;
let selectedDetectors = ['helmet'];
let isRunning = false;
let currentSnapshotFilter = 'all';
let _globalLoaderCount = 0;
let _globalLoaderDelayTimer = null;

// =====================================================================
// UTILITIES
// =====================================================================
function showToast(message, type = 'success') {
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    setTimeout(() => toast.classList.remove('show'), 3000);
}
function formatSize(bytes) { return (bytes / 1024 / 1024).toFixed(1) + ' MB'; }

function showGlobalPageLoader(message = 'Vui lòng chờ trong giây lát...') {
    if (!globalPageLoader) return () => {};

    _globalLoaderCount += 1;
    if (globalPageLoaderMessage && message) {
        globalPageLoaderMessage.textContent = message;
    }

    if (!_globalLoaderDelayTimer && !globalPageLoader.classList.contains('active')) {
        _globalLoaderDelayTimer = setTimeout(() => {
            if (_globalLoaderCount > 0) {
                globalPageLoader.classList.add('active');
                globalPageLoader.setAttribute('aria-hidden', 'false');
            }
            _globalLoaderDelayTimer = null;
        }, 120);
    }

    let released = false;
    return () => {
        if (released) return;
        released = true;

        _globalLoaderCount = Math.max(0, _globalLoaderCount - 1);
        if (_globalLoaderCount > 0) return;

        if (_globalLoaderDelayTimer) {
            clearTimeout(_globalLoaderDelayTimer);
            _globalLoaderDelayTimer = null;
        }
        globalPageLoader.classList.remove('active');
        globalPageLoader.setAttribute('aria-hidden', 'true');
    };
}

function resetUI() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    videoOverlay.classList.remove('hidden');
    videoStats.style.display = 'none';
    progressBar.style.display = 'none';
    progressFill.style.width = '0%';
    progressValue.textContent = '0%';
    frameNum.textContent = '0';
    fpsValue.textContent = '0';
    helmetCount.textContent = '0';
    redlightCount.textContent = '0';
    sidewalkCount.textContent = '0';
    wrongwayCount.textContent = '0';
    wronglaneCount.textContent = '0';
    totalCount.textContent = '0';
    isRunning = false;
}

// =====================================================================
// LOAD DATA
// =====================================================================
async function loadVideos() {
    try {
        const resp = await fetch('/api/videos');
        const videos = await resp.json();
        videoSelect.innerHTML = videos.map(v => `<option value="${v.path}">${v.name} (${formatSize(v.size)})</option>`).join('');
    } catch (e) { videoSelect.innerHTML = '<option value="">No videos found</option>'; }
}
async function loadModels() {
    try {
        const resp = await fetch('/api/models');
        const models = await resp.json();
        modelSelect.innerHTML = models.map(m => `<option value="${m.path}">${m.name} (${formatSize(m.size)})</option>`).join('');
    } catch (e) { modelSelect.innerHTML = '<option value="">No models found</option>'; }
}
async function loadSnapshots(filter = 'all') {
    currentSnapshotFilter = filter || 'all';
    try {
        const resp = await fetch('/api/snapshots');
        let snapshots = await resp.json();
        if (filter !== 'all') snapshots = snapshots.filter(s => s.type === filter);
        if (snapshots.length === 0) { snapshotsGallery.innerHTML = '<p class="no-items">No violations detected yet.</p>'; return; }
        snapshotsGallery.innerHTML = snapshots.slice(0, 12).map(s => `
            <div class="gallery-item" data-violation-type="${s.type}" data-filename="${s.filename}">
                <img src="${s.path}" alt="${s.type}" loading="lazy" onclick="window.open('${s.path}', '_blank')">
                <div class="gallery-item-info">
                    <span class="gallery-item-type ${s.type}">${s.type}</span>
                    <span class="gallery-item-name">${s.filename}</span>
                </div>
                <button class="gallery-item-delete" title="Xoa anh vi pham" onclick="deleteSnapshot(event, '${s.type}', '${s.filename}')">🗑️</button>
            </div>
        `).join('');
    } catch (e) { console.error('Failed to load snapshots:', e); }
}

async function deleteSnapshot(event, violationType, filename) {
    event.stopPropagation();
    if (!confirm('Xoa anh vi pham "' + filename + '"?')) return;
    try {
        const resp = await fetch('/api/snapshots/' + encodeURIComponent(violationType) + '/' + encodeURIComponent(filename), {
            method: 'DELETE'
        });
        const data = await resp.json();
        if (data.error) { showToast('Loi xoa anh: ' + data.error, 'error'); return; }
        const card = event.currentTarget.closest('.gallery-item');
        if (card) card.remove();
        if (snapshotsGallery.querySelectorAll('.gallery-item').length === 0) {
            snapshotsGallery.innerHTML = '<p class="no-items">No violations detected yet.</p>';
        }
        showToast('🗑️ Da xoa anh vi pham', 'success');
    } catch (e) { showToast('Loi xoa anh: ' + e.message, 'error'); }
}

async function clearSnapshots() {
    const targetType = currentSnapshotFilter === 'all' ? 'all' : currentSnapshotFilter;
    const scopeText = targetType === 'all' ? 'tat ca anh vi pham' : `tat ca anh loai "${targetType}"`;
    if (!confirm(`Xoa ${scopeText}?`)) return;

    const finishLoading = showGlobalPageLoader('Dang xoa anh vi pham...');
    if (clearSnapshotsBtn) clearSnapshotsBtn.disabled = true;
    try {
        const query = targetType === 'all' ? '' : `?type=${encodeURIComponent(targetType)}`;
        const resp = await fetch(`/api/snapshots${query}`, { method: 'DELETE' });
        const data = await resp.json();
        if (!resp.ok || data.error) {
            throw new Error(data.error || `HTTP ${resp.status}`);
        }
        await loadSnapshots(currentSnapshotFilter);
        showToast(`🧹 Da xoa ${data.deleted_count || 0} anh`, 'success');
    } catch (e) {
        showToast(`Loi xoa hang loat: ${e.message}`, 'error');
    } finally {
        finishLoading();
        if (clearSnapshotsBtn) clearSnapshotsBtn.disabled = false;
    }
}

// =====================================================================
// DETECTOR SELECTION
// =====================================================================
document.querySelectorAll('.detector-btn').forEach(btn => {
    btn.addEventListener('click', () => { btn.classList.toggle('active'); updateSelectedDetectors(); });
});
function updateSelectedDetectors() {
    selectedDetectors = [];
    document.querySelectorAll('.detector-btn.active').forEach(btn => selectedDetectors.push(btn.dataset.detector));
}

// =====================================================================
// GALLERY TABS
// =====================================================================
document.querySelectorAll('.gallery-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.gallery-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        loadSnapshots(tab.dataset.tab);
    });
});

if (clearSnapshotsBtn) {
    clearSnapshotsBtn.addEventListener('click', clearSnapshots);
}

// =====================================================================
// REAL-TIME WEBSOCKET DETECTION
// =====================================================================
function sendSettingsUpdate() {
    if (ws && ws.readyState === WebSocket.OPEN && isRunning) {
        ws.send(JSON.stringify({
            action: 'update_settings',
            conf: parseInt(confRange.value) / 100,
            debug: debugToggle.checked
        }));
    }
}

function startDetection() {
    // Guard: prevent double-start
    if (isRunning || (ws && ws.readyState === WebSocket.OPEN)) return;

    const video = videoSelect.value;
    const model = modelSelect.value;
    const conf = parseInt(confRange.value) / 100;
    const debug = debugToggle.checked;
    if (!video || !model) { showToast('Please select video and model', 'error'); return; }
    if (selectedDetectors.length === 0) { showToast('Please select at least one detector', 'error'); return; }

    startBtn.disabled = true;

    ws = new WebSocket(`${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/detect`);

    ws.onopen = () => {
        statusIndicator.className = 'status-indicator running';
        statusText.textContent = 'Connecting...';
        ws.send(JSON.stringify({ action: 'start', video, model, detectors: selectedDetectors, conf, debug }));
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        switch (data.type) {
            case 'started':
                videoOverlay.classList.add('hidden');
                videoStats.style.display = 'flex';
                progressBar.style.display = 'block';
                startBtn.disabled = true;
                stopBtn.disabled = false;
                isRunning = true;
                statusText.textContent = `Processing: ${data.video}`;
                debugStatus.textContent = `Debug: ${data.debug ? 'ON' : 'OFF'}`;
                showToast(`Started detection with ${data.detectors.join(', ')}`, 'success');
                break;
            case 'frame':
                const img = new Image();
                img.onload = () => { canvas.width = img.width; canvas.height = img.height; ctx.drawImage(img, 0, 0); };
                img.src = 'data:image/jpeg;base64,' + data.image;
                if (data.stats) {
                    frameNum.textContent = data.stats.frame_idx;
                    fpsValue.textContent = data.stats.fps;
                    const v = data.stats.violations || {};
                    helmetCount.textContent = v.helmet || 0;
                    redlightCount.textContent = v.redlight || 0;
                    sidewalkCount.textContent = v.sidewalk || 0;
                    wrongwayCount.textContent = v.wrong_way || 0;
                    wronglaneCount.textContent = v.wrong_lane || 0;
                    totalCount.textContent = data.stats.total || 0;
                }
                if (data.progress) { progressFill.style.width = data.progress + '%'; progressValue.textContent = data.progress.toFixed(1) + '%'; }
                break;
            case 'violation':
                showToast(`⚠️ ${data.data.type} violation detected!`, 'warning');
                loadSnapshots();
                break;
            case 'settings_updated':
                showToast(`Settings updated: conf=${(data.conf * 100).toFixed(0)}%, debug=${data.debug ? 'ON' : 'OFF'}`, 'success');
                debugStatus.textContent = `Debug: ${data.debug ? 'ON' : 'OFF'}`;
                break;
            case 'finished':
                statusIndicator.className = 'status-indicator idle';
                statusText.textContent = 'Detection finished';
                startBtn.disabled = false;
                stopBtn.disabled = true;
                showToast('Detection completed!', 'success');
                loadSnapshots();
                resetUI();
                break;
            case 'stopped':
                statusIndicator.className = 'status-indicator idle';
                statusText.textContent = 'Stopped';
                startBtn.disabled = false;
                stopBtn.disabled = true;
                resetUI();
                break;
            case 'error':
                statusIndicator.className = 'status-indicator error';
                statusText.textContent = 'Error: ' + data.message;
                showToast('Error: ' + data.message, 'error');
                startBtn.disabled = false;
                stopBtn.disabled = true;
                isRunning = false;
                if (data.code !== 'busy') resetUI();
                break;
        }
    };
    ws.onerror = () => { statusIndicator.className = 'status-indicator error'; statusText.textContent = 'Connection error'; showToast('Connection error', 'error'); isRunning = false; startBtn.disabled = false; stopBtn.disabled = true; };
    ws.onclose = () => { startBtn.disabled = false; stopBtn.disabled = true; isRunning = false; };
}

function stopDetection() {
    if (ws && ws.readyState === WebSocket.OPEN) { ws.send(JSON.stringify({ action: 'stop' })); ws.close(); }
}

// =====================================================================
// CONFIDENCE & DEBUG — LIVE UPDATE
// =====================================================================
function updateConfUI() {
    const val = parseInt(confRange.value);
    const pct = (val - 5) / (90 - 5) * 100;
    confBubble.textContent = val + '%';
    rangeFill.style.width = pct + '%';
    if (val <= 30) confBubble.style.background = 'linear-gradient(135deg, #10b981, #34d399)';
    else if (val <= 60) confBubble.style.background = 'linear-gradient(135deg, #f59e0b, #fbbf24)';
    else confBubble.style.background = 'linear-gradient(135deg, #ef4444, #f87171)';
}
confRange.addEventListener('input', () => { updateConfUI(); sendSettingsUpdate(); });
debugToggle.addEventListener('change', () => {
    debugStatus.textContent = `Debug: ${debugToggle.checked ? 'ON' : 'OFF'}`;
    sendSettingsUpdate();
});
startBtn.addEventListener('click', startDetection);
stopBtn.addEventListener('click', stopDetection);

// =====================================================================
// IMAGE DETECTION (Enhanced UI/UX)
// =====================================================================
const imageInput = document.getElementById('imageInput');
const imageUploadArea = document.getElementById('imageUploadArea');
const imageUploadContent = document.getElementById('imageUploadContent');
const imagePreview = document.getElementById('imagePreview');
const imagePreviewWrapper = document.getElementById('imagePreviewWrapper');
const imageDetectBtn = document.getElementById('imageDetectBtn');
const imageConf = document.getElementById('imageConf');
const imageConfBubbleEl = document.getElementById('imageConfBubble');
const imageConfTrack = document.getElementById('imageConfTrack');
const imageClearBtn = document.getElementById('imageClearBtn');
const imageCompareBtn = document.getElementById('imageCompareBtn');
const imageFullscreenBtn = document.getElementById('imageFullscreenBtn');
const imageDownloadBtn = document.getElementById('imageDownloadBtn');
const imageFullscreenOverlay = document.getElementById('imageFullscreenOverlay');
const imageFullscreenClose = document.getElementById('imageFullscreenClose');
const imageFullscreenImg = document.getElementById('imageFullscreenImg');
const imageSkeletonCard = document.getElementById('imageSkeletonCard');
const imageModelSelect = document.getElementById('imageModelSelect');
let imageFile = null;
let imageOriginalDataUrl = null;
let imageDetectedDataUrl = null;
let imageCompareActive = false;
let imageDetectStartTime = 0;

// Load available models into dropdown
(async function loadImageModels() {
    try {
        const resp = await fetch('/api/models');
        const data = await resp.json();
        if (data.models && imageModelSelect) {
            data.models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m;
                opt.textContent = m;
                imageModelSelect.appendChild(opt);
            });
        }
    } catch (_) {}
})();

// Toggle chip click handlers
document.querySelectorAll('.img-toggle-chip').forEach(chip => {
    chip.addEventListener('click', () => {
        const cb = chip.querySelector('input[type="checkbox"]');
        cb.checked = !cb.checked;
        chip.classList.toggle('active', cb.checked);
    });
});

// Confidence slider with visual track + bubble
function updateImageConfUI() {
    const val = parseInt(imageConf.value);
    const min = parseInt(imageConf.min);
    const max = parseInt(imageConf.max);
    const pct = ((val - min) / (max - min)) * 100;
    imageConfBubbleEl.textContent = val + '%';
    imageConfBubbleEl.style.left = pct + '%';
    if (imageConfTrack) imageConfTrack.style.width = pct + '%';
}
imageConf.addEventListener('input', updateImageConfUI);
updateImageConfUI();

// Upload area events
imageUploadArea.addEventListener('click', (e) => {
    if (e.target.closest('.img-clear-btn')) return;
    imageInput.click();
});
imageUploadArea.addEventListener('dragover', e => { e.preventDefault(); imageUploadArea.classList.add('drag-over'); });
imageUploadArea.addEventListener('dragleave', () => imageUploadArea.classList.remove('drag-over'));
imageUploadArea.addEventListener('drop', e => {
    e.preventDefault(); imageUploadArea.classList.remove('drag-over');
    if (e.dataTransfer.files.length) { imageFile = e.dataTransfer.files[0]; showImagePreview(); }
});
imageInput.addEventListener('change', () => { if (imageInput.files.length) { imageFile = imageInput.files[0]; showImagePreview(); } });

// Clear button
imageClearBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    imageFile = null;
    imageOriginalDataUrl = null;
    imageInput.value = '';
    imagePreviewWrapper.style.display = 'none';
    imageUploadContent.style.display = '';
    imageDetectBtn.disabled = true;
    document.getElementById('imageResultCard').style.display = 'none';
    document.getElementById('imageStatsCard').style.display = 'none';
    document.getElementById('imageDetectionMeta').style.display = 'none';
});

function showImagePreview() {
    if (!imageFile) return;
    const url = URL.createObjectURL(imageFile);
    imagePreview.src = url;
    imagePreviewWrapper.style.display = '';
    imageUploadContent.style.display = 'none';
    imageDetectBtn.disabled = false;

    // Show file info
    const sizeKB = (imageFile.size / 1024).toFixed(1);
    const sizeStr = imageFile.size > 1048576 ? (imageFile.size / 1048576).toFixed(1) + ' MB' : sizeKB + ' KB';
    document.getElementById('imageFileInfo').innerHTML =
        `<span>📄 ${imageFile.name}</span><span>📦 ${sizeStr}</span>`;

    // Store original for comparison
    const reader = new FileReader();
    reader.onload = (ev) => { imageOriginalDataUrl = ev.target.result; };
    reader.readAsDataURL(imageFile);
}

// Detect
imageDetectBtn.addEventListener('click', async () => {
    if (!imageFile) return;
    imageDetectBtn.disabled = true;
    imageDetectBtn.querySelector('.img-btn-text').textContent = 'Detecting...';
    imageDetectBtn.querySelector('.img-btn-icon').textContent = '⏳';
    imageDetectBtn.classList.add('detecting');
    imageSkeletonCard.style.display = 'block';
    document.getElementById('imageResultCard').style.display = 'none';
    document.getElementById('imageStatsCard').style.display = 'none';
    imageDetectStartTime = performance.now();

    const finishLoading = showGlobalPageLoader(
        currentLang === 'en' ? 'Analyzing image with AI...' : 'Đang phân tích ảnh bằng AI...',
    );
    const fd = new FormData();
    fd.append('file', imageFile);
    fd.append('conf', parseInt(imageConf.value) / 100);
    // Model selection
    const selectedModel = imageModelSelect ? imageModelSelect.value : '';
    if (selectedModel) fd.append('model', selectedModel);
    // Display toggles
    const showMasks = document.querySelector('#toggleMask input')?.checked ?? true;
    const showBoxes = document.querySelector('#toggleBoxes input')?.checked ?? true;
    const showLabels = document.querySelector('#toggleLabels input')?.checked ?? true;
    fd.append('show_masks', showMasks);
    fd.append('show_labels', showLabels);
    fd.append('show_boxes', showBoxes);
    try {
        const resp = await fetch('/api/detect/image', { method: 'POST', body: fd });
        const data = await resp.json();
        if (data.error) { showToast('Error: ' + data.error, 'error'); return; }

        const elapsed = ((performance.now() - imageDetectStartTime) / 1000).toFixed(2);
        imageDetectedDataUrl = 'data:image/jpeg;base64,' + data.image;

        // Show result
        imageSkeletonCard.style.display = 'none';
        document.getElementById('imageResultCard').style.display = 'block';
        document.getElementById('imageResult').src = imageDetectedDataUrl;

        // Reset compare view
        imageCompareActive = false;
        imageCompareBtn.classList.remove('active');
        document.getElementById('imageResultSingle').style.display = '';
        document.getElementById('imageCompareView').style.display = 'none';

        // Stats
        const statsCard = document.getElementById('imageStatsCard');
        statsCard.style.display = 'block';
        document.getElementById('imageTotalCount').innerHTML =
            `<span class="img-total-number">${data.total}</span><span class="img-total-label">objects detected</span>`;

        // Class list with progress bars
        const classList = document.getElementById('imageClassList');
        const maxCount = Math.max(...Object.values(data.class_summary).map(v => v.count), 1);
        classList.innerHTML = Object.entries(data.class_summary).map(([name, info]) => {
            const barPct = (info.count / maxCount * 100).toFixed(0);
            const confPct = (info.max_conf * 100).toFixed(1);
            return `<div class="class-item">
                <div style="flex:1">
                    <div style="display:flex;align-items:center;justify-content:space-between;gap:8px">
                        <span class="class-name">${name}</span>
                        <span class="class-count">${info.count}</span>
                    </div>
                    <div class="img-class-item-conf-bar"><div class="img-class-item-conf-fill" style="width:${confPct}%"></div></div>
                </div>
                <span class="class-conf">${confPct}%</span>
                <div class="img-class-item-bar" style="width:${barPct}%"></div>
            </div>`;
        }).join('');

        // Detection meta
        const metaEl = document.getElementById('imageDetectionMeta');
        metaEl.style.display = '';
        document.getElementById('imageDetectionTime').textContent = elapsed + 's';
        // Get image dimensions from preview
        const img = new Image();
        img.onload = () => {
            document.getElementById('imageDetectionSize').textContent = img.naturalWidth + ' × ' + img.naturalHeight + 'px';
        };
        img.src = imageDetectedDataUrl;

        showToast(`Detected ${data.total} objects!`, 'success');
    } catch (e) { showToast('Detection failed: ' + e.message, 'error'); }
    finally {
        finishLoading();
        imageSkeletonCard.style.display = 'none';
        imageDetectBtn.disabled = false;
        imageDetectBtn.querySelector('.img-btn-text').textContent = 'Detect Objects';
        imageDetectBtn.querySelector('.img-btn-icon').textContent = '🔍';
        imageDetectBtn.classList.remove('detecting');
    }
});

// Compare toggle
imageCompareBtn.addEventListener('click', () => {
    if (!imageOriginalDataUrl || !imageDetectedDataUrl) return;
    imageCompareActive = !imageCompareActive;
    imageCompareBtn.classList.toggle('active', imageCompareActive);
    document.getElementById('imageResultSingle').style.display = imageCompareActive ? 'none' : '';
    document.getElementById('imageCompareView').style.display = imageCompareActive ? '' : 'none';

    if (imageCompareActive) {
        const slider = document.getElementById('imageCompareSlider');
        const beforeImg = document.getElementById('imageCompareBefore');
        const afterImg = document.getElementById('imageCompareAfterImg');
        const afterDiv = document.getElementById('imageCompareAfter');
        const handle = document.getElementById('imageCompareHandle');

        beforeImg.src = imageOriginalDataUrl;
        afterImg.src = imageDetectedDataUrl;

        // Wait for images to load then set size
        afterImg.onload = () => {
            const w = slider.offsetWidth;
            afterDiv.style.width = '50%';
            handle.style.left = '50%';
            afterImg.style.width = w + 'px';
        };

        // Mouse/touch drag for compare slider
        let isDragging = false;
        const onMove = (clientX) => {
            const rect = slider.getBoundingClientRect();
            let pct = ((clientX - rect.left) / rect.width) * 100;
            pct = Math.max(2, Math.min(98, pct));
            afterDiv.style.width = pct + '%';
            handle.style.left = pct + '%';
        };
        slider.onmousedown = (e) => { isDragging = true; onMove(e.clientX); };
        document.addEventListener('mousemove', (e) => { if (isDragging) onMove(e.clientX); });
        document.addEventListener('mouseup', () => { isDragging = false; });
        slider.ontouchstart = (e) => { isDragging = true; onMove(e.touches[0].clientX); };
        document.addEventListener('touchmove', (e) => { if (isDragging) onMove(e.touches[0].clientX); });
        document.addEventListener('touchend', () => { isDragging = false; });
    }
});

// Fullscreen
imageFullscreenBtn.addEventListener('click', () => {
    if (!imageDetectedDataUrl) return;
    imageFullscreenImg.src = imageDetectedDataUrl;
    imageFullscreenOverlay.classList.add('active');
});
imageFullscreenClose.addEventListener('click', () => { imageFullscreenOverlay.classList.remove('active'); });
imageFullscreenOverlay.addEventListener('click', (e) => {
    if (e.target === imageFullscreenOverlay) imageFullscreenOverlay.classList.remove('active');
});
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && imageFullscreenOverlay.classList.contains('active')) {
        imageFullscreenOverlay.classList.remove('active');
    }
});

// Result image click to fullscreen
document.getElementById('imageResult').addEventListener('click', () => {
    if (imageDetectedDataUrl) {
        imageFullscreenImg.src = imageDetectedDataUrl;
        imageFullscreenOverlay.classList.add('active');
    }
});

// Download
imageDownloadBtn.addEventListener('click', () => {
    if (!imageDetectedDataUrl) return;
    const link = document.createElement('a');
    link.href = imageDetectedDataUrl;
    link.download = 'detection_result_' + new Date().toISOString().slice(0,19).replace(/[:T]/g, '-') + '.jpg';
    link.click();
});

// =====================================================================
// VIDEO DETECTION
// =====================================================================
const videoFileInput = document.getElementById('videoFileInput');
const videoUploadArea = document.getElementById('videoUploadArea');
const videoUploadContent = document.getElementById('videoUploadContent');
const videoPreviewEl = document.getElementById('videoPreview');
const videoDetectBtn = document.getElementById('videoDetectBtn');
const videoConfEl = document.getElementById('videoConf');
const videoConfLabel = document.getElementById('videoConfLabel');
let videoFile = null;

videoConfEl.addEventListener('input', () => { videoConfLabel.textContent = videoConfEl.value + '%'; });

videoUploadArea.addEventListener('click', () => videoFileInput.click());
videoUploadArea.addEventListener('dragover', e => { e.preventDefault(); videoUploadArea.classList.add('drag-over'); });
videoUploadArea.addEventListener('dragleave', () => videoUploadArea.classList.remove('drag-over'));
videoUploadArea.addEventListener('drop', e => {
    e.preventDefault(); videoUploadArea.classList.remove('drag-over');
    if (e.dataTransfer.files.length) { videoFile = e.dataTransfer.files[0]; showVideoPreview(); }
});
videoFileInput.addEventListener('change', () => { if (videoFileInput.files.length) { videoFile = videoFileInput.files[0]; showVideoPreview(); } });

function showVideoPreview() {
    const url = URL.createObjectURL(videoFile);
    videoPreviewEl.src = url; videoPreviewEl.style.display = 'block';
    videoUploadContent.style.display = 'none';
    videoDetectBtn.disabled = false;
}

videoDetectBtn.addEventListener('click', async () => {
    if (!videoFile) return;
    videoDetectBtn.disabled = true; videoDetectBtn.textContent = '⏳ Uploading...';
    document.getElementById('videoProcessCard').style.display = 'block';
    document.getElementById('videoResultCard').style.display = 'none';
    const finishLoading = showGlobalPageLoader(
        currentLang === 'en' ? 'Uploading video for processing...' : 'Đang tải video lên để xử lý...',
    );

    const fd = new FormData();
    fd.append('file', videoFile);
    fd.append('conf', parseInt(videoConfEl.value) / 100);
    try {
        const resp = await fetch('/api/detect/video', { method: 'POST', body: fd });
        const data = await resp.json();
        if (data.error) { showToast('Error: ' + data.error, 'error'); return; }
        pollVideoStatus(data.task_id);
    } catch (e) { showToast('Upload failed: ' + e.message, 'error'); videoDetectBtn.disabled = false; videoDetectBtn.textContent = '🎬 Process Video'; }
    finally {
        finishLoading();
    }
});

function pollVideoStatus(taskId) {
    const interval = setInterval(async () => {
        try {
            const resp = await fetch(`/api/detect/video/status/${taskId}`);
            const data = await resp.json();
            document.getElementById('videoProgressFill').style.width = data.progress + '%';
            document.getElementById('videoProgressText').textContent = data.progress.toFixed(1) + '%';
            document.getElementById('videoProcessText').textContent = data.status === 'processing' ? `Đang xử lý... ${data.progress.toFixed(1)}%` : data.status;

            if (data.status === 'done') {
                clearInterval(interval);
                document.getElementById('videoProcessCard').style.display = 'none';
                document.getElementById('videoResultCard').style.display = 'block';
                document.getElementById('videoResult').src = data.output;
                document.getElementById('videoStatsCard').style.display = 'block';
                document.getElementById('videoTotalCount').textContent = data.total_detections + ' total detections';
                const vList = document.getElementById('videoClassList');
                vList.innerHTML = Object.entries(data.class_summary || {}).map(([name, info]) =>
                    `<div class="class-item"><span class="class-name">${name}</span><span class="class-count">${info.count}</span><span class="class-conf">${(info.max_conf * 100).toFixed(1)}%</span></div>`
                ).join('');
                videoDetectBtn.disabled = false; videoDetectBtn.textContent = '🎬 Process Video';
                showToast('Video processing complete!', 'success');
            } else if (data.status === 'error') {
                clearInterval(interval);
                showToast('Error: ' + data.error, 'error');
                videoDetectBtn.disabled = false; videoDetectBtn.textContent = '🎬 Process Video';
            }
        } catch (e) { clearInterval(interval); }
    }, 2000);
}

// =====================================================================
// LOOKUP
// =====================================================================
document.getElementById('lookupBtn').addEventListener('click', async () => {
    const plate = document.getElementById('lookupPlate').value;
    const cccd = document.getElementById('lookupCCCD').value;
    const phone = document.getElementById('lookupPhone').value;
    if (!plate && !cccd && !phone) { showToast('Vui lòng nhập ít nhất một thông tin', 'error'); return; }

    const btn = document.getElementById('lookupBtn');
    btn.disabled = true; btn.textContent = '⏳ Đang tra cứu...';
    const finishLoading = showGlobalPageLoader(
        currentLang === 'en' ? 'Searching violation records...' : 'Đang tra cứu dữ liệu vi phạm...',
    );
    const fd = new FormData();
    fd.append('license_plate', plate); fd.append('cccd', cccd); fd.append('phone', phone);
    try {
        const resp = await fetch('/api/lookup', { method: 'POST', body: fd });
        const data = await resp.json();
        const container = document.getElementById('lookupResults');
        if (data.results.length === 0) {
            container.innerHTML = '<p class="no-items">Không tìm thấy vi phạm nào phù hợp.</p>';
        } else {
            container.innerHTML = data.results.map(r => `
                <div class="lookup-item ${r.related ? 'related' : ''}" onclick="window.open('${r.path}','_blank')">
                    <img src="${r.path}" alt="${r.type}" loading="lazy">
                    <div class="lookup-item-info">
                        <span class="gallery-item-type ${r.type}">${r.type}</span>
                        <span>${r.filename}</span>
                        ${r.related ? '<span class="related-badge">Liên quan</span>' : ''}
                    </div>
                </div>
            `).join('');
        }
        showToast(`Tìm thấy ${data.total} kết quả`, 'success');
    } catch (e) { showToast('Lỗi tra cứu: ' + e.message, 'error'); }
    finally {
        finishLoading();
        btn.disabled = false;
        btn.textContent = '🔍 Tra cứu';
    }
});

// =====================================================================
// INIT
// =====================================================================
(async function bootstrapWebPageData() {
    const startupLang = localStorage.getItem('lang') || 'vi';
    const finishLoading = showGlobalPageLoader(
        startupLang === 'en' ? 'Loading initial data...' : 'Đang tải dữ liệu ban đầu...',
    );
    try {
        await Promise.all([loadVideos(), loadModels(), loadSnapshots()]);
    } finally {
        finishLoading();
    }
})();
updateConfUI();
debugStatus.textContent = `Debug: ${debugToggle.checked ? 'ON' : 'OFF'}`;

// =====================================================================
// SERVER IP DISPLAY
// =====================================================================
(async function loadServerInfo() {
    const ipValue = document.getElementById('serverIpValue');
    const ipCopy = document.getElementById('serverIpCopy');
    if (!ipValue || !ipCopy) return;

    try {
        const resp = await fetch('/api/server-info');
        const data = await resp.json();
        const preferredIp = (data.preferred_ip || '').toString().trim();
        const ip = preferredIp || (data.ips && data.ips.length > 0 ? data.ips[0] : 'localhost');
        const port = data.port || 8000;
        const fullAddr = `${ip}:${port}`;
        ipValue.textContent = fullAddr;

        ipCopy.addEventListener('click', (e) => {
            e.stopPropagation();
            navigator.clipboard.writeText(fullAddr).then(() => {
                ipCopy.classList.add('copied');
                setTimeout(() => ipCopy.classList.remove('copied'), 1500);
            }).catch(() => {
                // Fallback for older browsers
                const ta = document.createElement('textarea');
                ta.value = fullAddr;
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                document.body.removeChild(ta);
                ipCopy.classList.add('copied');
                setTimeout(() => ipCopy.classList.remove('copied'), 1500);
            });
        });
    } catch (e) {
        ipValue.textContent = 'Unavailable';
        console.warn('Failed to fetch server info:', e);
    }
})();

// =====================================================================
// I18N — Vietnamese / English Language Toggle
// =====================================================================
const i18n = {
    vi: {
        // Hero
        hero_desc: 'Hệ thống phát hiện vi phạm giao thông thời gian thực',
        hero_cta_start: '🚀 Bắt đầu ngay',
        hero_cta_image: '🖼️ Thử với ảnh',
        stat_classes: 'Object Classes',
        stat_violations: 'Violation Types',
        stat_speed: 'Detection Speed',
        stat_model: 'AI Model',
        // About
        about_title: 'Giới thiệu dự án',
        about_desc: 'Dự án nghiên cứu khoa học ứng dụng Deep Learning vào giám sát giao thông đô thị, phát hiện tự động các hành vi vi phạm luật giao thông thông qua camera giám sát.',
        feat_helmet: 'Phát hiện người điều khiển xe máy không đội mũ bảo hiểm',
        feat_redlight: 'Phát hiện phương tiện vượt đèn đỏ tại ngã tư',
        feat_sidewalk: 'Phát hiện phương tiện chạy lên vỉa hè, dải phân cách',
        feat_wrongway: 'Phát hiện phương tiện chạy ngược chiều',
        feat_wronglane: 'Phát hiện vi phạm sai làn đường, đè vạch kẻ đường',
        feat_sign: 'Phát hiện vi phạm biển báo giao thông',
        // Tech
        tech_title: 'Mô hình & Công nghệ',
        tech_yolo: 'Object Detection + Instance Segmentation, nhận diện 40 classes bao gồm phương tiện, đèn giao thông, vạch kẻ, biển báo',
        tech_bytetrack: 'Multi-object tracking cho theo dõi phương tiện liên tục qua nhiều frame, hỗ trợ phân tích hành vi',
        tech_fastapi: 'Backend hiệu suất cao với streaming real-time, xử lý đồng thời nhiều client',
        tech_calibration: 'Tự động hiệu chỉnh vùng vi phạm (vỉa hè, vạch dừng, làn đường) trong 5-10 giây đầu',
        // Team
        team_title: 'Thành viên nhóm',
        team_a_desc: 'Phụ trách kiến trúc hệ thống & training model',
        team_b_desc: 'Phát triển các module phát hiện vi phạm',
        team_c_desc: 'Thu thập, gán nhãn và xử lý dữ liệu huấn luyện',
        team_d_desc: 'Thiết kế giao diện web & trải nghiệm người dùng',
        // Contact
        contact_title: 'Liên hệ',
        contact_phone: 'Điện thoại',
        contact_org: 'Đơn vị',
        contact_org_name: 'Khoa Công nghệ Thông tin',
        // Image tab
        image_subtitle: 'Tải lên hình ảnh để nhận diện tất cả các vật thể',
        image_drop: 'Kéo thả ảnh vào đây hoặc <span class="upload-link">chọn file</span>',
        // Video tab
        video_subtitle: 'Tải lên video để nhận diện và theo dõi vật thể',
        video_drop: 'Kéo thả video vào đây hoặc <span class="upload-link">chọn file</span>',
        video_processing: 'Đang xử lý video...',
        // Real-time tab
        realtime_subtitle: 'Phát hiện vi phạm giao thông theo thời gian thực',
        realtime_overlay: 'Chọn video và nhấn Start Detection',
        // Lookup tab
        lookup_title: 'Tra cứu vi phạm',
        lookup_subtitle: 'Tìm kiếm thông tin vi phạm theo biển số xe, CCCD hoặc số điện thoại',
        lookup_form_title: 'Thông tin tra cứu',
        lookup_plate: '🚗 Biển số xe',
        lookup_cccd: '🪪 Số CCCD',
        lookup_phone: '📱 Số điện thoại',
        lookup_btn: '🔍 Tra cứu',
        lookup_results_title: 'Kết quả tra cứu',
        lookup_empty: 'Nhập thông tin và nhấn Tra cứu để tìm kiếm vi phạm.',
        // Navigation + Manage + Complaints
        nav_home: 'Trang chủ',
        nav_image: 'Ảnh',
        nav_video: 'Video',
        nav_realtime: 'Thời gian thực',
        nav_lookup: 'Tra cứu',
        nav_manage: 'Quản lý dữ liệu',
        nav_complaints: 'Khiếu nại',
        manage_title: 'Quản lý Dữ liệu',
        manage_subtitle: 'Tiếp nhận thông tin người dùng, phương tiện, vi phạm, khiếu nại từ App',
        manage_stat_users: 'Người dùng',
        manage_stat_vehicles: 'Phương tiện',
        manage_stat_violations: 'Vi phạm',
        manage_stat_pending_fines: 'Tổng nợ phạt',
        manage_stat_complaints: 'Khiếu nại',
        manage_pending_updates_prefix: 'Có',
        manage_pending_updates_suffix: 'yêu cầu thay đổi thông tin đang chờ duyệt từ người dùng App',
        manage_pending_updates_cta: 'Xem ngay',
        manage_overview_title: '👥 Tổng quan người dùng',
        manage_search_placeholder: 'Tìm kiếm theo tên, email, biển số...',
        manage_auto_sync_badge: '⚡ Tự động đồng bộ',
        auto_sync_on: 'Auto Sync: Bật',
        auto_sync_off: 'Auto Sync: Tắt',
        manage_initial_hint: 'Dữ liệu sẽ tự động đồng bộ khi có thay đổi từ App/Web.',
        manage_row_count_initial: '0 người dùng',
        complaints_title: 'Quản lý khiếu nại',
        complaints_subtitle: 'Theo dõi, lọc, duyệt khiếu nại và kiểm tra bằng chứng từ App theo thời gian thực',
        complaints_search_placeholder: 'Tìm theo người dùng, lý do, mã vi phạm, mô tả...',
        complaints_filter_all: 'Tất cả trạng thái',
        complaints_filter_pending: 'Đang xử lý',
        complaints_filter_approved: 'Đã duyệt',
        complaints_filter_rejected: 'Đã từ chối',
        complaints_empty_title: 'Chưa có dữ liệu khiếu nại',
        complaints_empty_subtitle: 'Dữ liệu sẽ xuất hiện khi app gửi khiếu nại.',
        complaints_total: 'Tổng khiếu nại',
        complaints_pending: 'Đang xử lý',
        complaints_approved: 'Đã duyệt',
        complaints_rejected: 'Đã từ chối',
        complaints_not_found_title: 'Không tìm thấy khiếu nại phù hợp',
        complaints_not_found_subtitle: 'Hãy đổi từ khóa tìm kiếm hoặc bộ lọc trạng thái.',
        complaints_fallback_reason: 'Khiếu nại',
        complaints_fallback_desc: 'Không có mô tả chi tiết',
        complaints_no_evidence: 'Không có ảnh bằng chứng',
        complaints_image_loading: 'Đang tải ảnh...',
        complaints_image_error: 'Không tải được ảnh',
        complaints_meta_id: 'Mã khiếu nại',
        complaints_meta_violation_id: 'Mã vi phạm',
        complaints_meta_type: 'Loại vi phạm',
        complaints_meta_fine: 'Mức phạt',
        complaints_admin_note: '💬 Ghi chú admin',
        complaints_btn_detail: '👁️ Chi tiết',
        complaints_btn_profile: '👤 Hồ sơ',
        complaints_btn_approve: '✅ Chấp nhận',
        complaints_btn_reject: '❌ Từ chối',
        complaints_btn_delete: '🗑️ Xóa',
        complaints_btn_processing: 'Đang xử lý...',
        complaints_pagination_info: 'Hiển thị {from}-{to} / {total}',
        complaints_pagination_prev: '← Trước',
        complaints_pagination_next: 'Tiếp →',
        complaints_pagination_page: 'Trang {page}/{total}',
        complaints_delete_confirm: 'Bạn có chắc muốn xóa khiếu nại này khỏi hệ thống?',
        complaints_delete_success: '🗑️ Đã xóa khiếu nại',
        complaints_delete_error: 'Lỗi khi xóa khiếu nại',
        complaints_delete_loading: 'Đang xóa khiếu nại...',
        complaint_status_pending: '⏳ Đang xử lý',
        complaint_status_approved: '✅ Đã duyệt',
        complaint_status_rejected: '❌ Đã từ chối',
        global_loader_title: 'Đang xử lý',
        global_loader_message: 'Vui lòng chờ trong giây lát...',
        admin_sync_success: 'Dữ liệu đã được đồng bộ tự động',
        user_count_suffix: 'người dùng',
        // Image tab controls
        image_model_label: 'Model',
        image_model_default: 'Mặc định',
        image_display_label: 'Hiển thị',
        image_toggle_mask: 'Mask',
        image_toggle_boxes: 'BBox',
        image_toggle_labels: 'Labels',
        image_detect_btn: 'Nhận diện',
        // Data Management table headers
        manage_th_name: 'Họ và tên',
        manage_th_cccd: 'Số CCCD',
        manage_th_phone: 'Số điện thoại',
        manage_th_violations: 'Vi phạm / Thanh toán',
        manage_th_complaints: 'Khiếu nại / Thông báo',
        manage_th_actions: 'Hành động',
        // Data Management dynamic strings
        manage_not_updated: 'Chưa cập nhật',
        manage_unpaid: 'Chưa nộp',
        manage_paid: 'Đã nộp',
        manage_debt: 'Nợ',
        manage_no_debt: 'Không nợ',
        manage_complaints_count: 'khiếu nại',
        manage_notifications_count: 'thông báo',
        manage_btn_review: 'Duyệt sửa đổi',
        manage_btn_detail: 'Chi tiết',
        manage_btn_delete: 'Xóa TK',
        manage_no_result: 'Không tìm thấy kết quả phù hợp.',
        manage_no_data: 'Hệ thống chưa có dữ liệu người dùng.',
        // Quota
        quota_title: 'Firestore Quota Monitor',
        quota_reads: 'Đọc',
        quota_writes: 'Ghi',
        quota_deletes: 'Xóa',
        quota_uptime: 'Thời gian hoạt động',
        quota_note: 'Dữ liệu quota được theo dõi từ khi server khởi động và cập nhật mỗi 10 giây.',
        // Reject modal
        reject_modal_title: 'Từ chối khiếu nại',
        reject_modal_subtitle: 'Vui lòng nhập lý do từ chối',
        reject_modal_placeholder: 'Nhập lý do từ chối...',
        reject_modal_cancel: 'Hủy',
        reject_modal_confirm: 'Xác nhận từ chối',
        // Typed.js
        typed_strings: [
            'trí tuệ nhân tạo YOLOv26-seg',
            'nhận diện 40 loại đối tượng',
            'xử lý real-time với độ chính xác cao',
            'theo dõi phương tiện liên tục qua ByteTrack',
            'phân tích hành vi vi phạm tự động'
        ],
    },
    en: {
        // Hero
        hero_desc: 'Real-time traffic violation detection system',
        hero_cta_start: '🚀 Get Started',
        hero_cta_image: '🖼️ Try with Image',
        stat_classes: 'Object Classes',
        stat_violations: 'Violation Types',
        stat_speed: 'Detection Speed',
        stat_model: 'AI Model',
        // About
        about_title: 'Project Introduction',
        about_desc: 'A scientific research project applying Deep Learning to urban traffic monitoring, automatically detecting traffic law violations through surveillance cameras.',
        feat_helmet: 'Detect motorcycle riders not wearing helmets',
        feat_redlight: 'Detect vehicles running red lights at intersections',
        feat_sidewalk: 'Detect vehicles driving on sidewalks and medians',
        feat_wrongway: 'Detect vehicles going the wrong way',
        feat_wronglane: 'Detect wrong lane violations and lane line crossing',
        feat_sign: 'Detect traffic sign violations',
        // Tech
        tech_title: 'Models & Technology',
        tech_yolo: 'Object Detection + Instance Segmentation, recognizing 40 classes including vehicles, traffic lights, lane markings, and signs',
        tech_bytetrack: 'Multi-object tracking for continuous vehicle monitoring across frames, supporting behavior analysis',
        tech_fastapi: 'High-performance backend with real-time streaming, handling multiple concurrent clients',
        tech_calibration: 'Auto-calibration of violation zones (sidewalks, stop lines, lanes) within the first 5-10 seconds',
        // Team
        team_title: 'Team Members',
        team_a_desc: 'System architecture & model training lead',
        team_b_desc: 'Violation detection module development',
        team_c_desc: 'Data collection, labeling & preprocessing',
        team_d_desc: 'Web interface design & user experience',
        // Contact
        contact_title: 'Contact',
        contact_phone: 'Phone',
        contact_org: 'Department',
        contact_org_name: 'Faculty of Information Technology',
        // Image tab
        image_subtitle: 'Upload an image to detect all objects',
        image_drop: 'Drag & drop image here or <span class="upload-link">browse file</span>',
        // Video tab
        video_subtitle: 'Upload a video to detect and track objects',
        video_drop: 'Drag & drop video here or <span class="upload-link">browse file</span>',
        video_processing: 'Processing video...',
        // Real-time tab
        realtime_subtitle: 'Detect traffic violations in real-time',
        realtime_overlay: 'Select options and click Start Detection',
        // Lookup tab
        lookup_title: 'Violation Lookup',
        lookup_subtitle: 'Search violations by license plate, ID card or phone number',
        lookup_form_title: 'Search Information',
        lookup_plate: '🚗 License Plate',
        lookup_cccd: '🪪 ID Card Number',
        lookup_phone: '📱 Phone Number',
        lookup_btn: '🔍 Search',
        lookup_results_title: 'Search Results',
        lookup_empty: 'Enter information and click Search to find violations.',
        // Navigation + Manage + Complaints
        nav_home: 'Home',
        nav_image: 'Image',
        nav_video: 'Video',
        nav_realtime: 'Real-Time',
        nav_lookup: 'Look Up',
        nav_manage: 'Data Management',
        nav_complaints: 'Complaints',
        manage_title: 'Data Management',
        manage_subtitle: 'Receive user, vehicle, violation, and complaint information from the App',
        manage_stat_users: 'Users',
        manage_stat_vehicles: 'Vehicles',
        manage_stat_violations: 'Violations',
        manage_stat_pending_fines: 'Pending Fines',
        manage_stat_complaints: 'Complaints',
        manage_pending_updates_prefix: 'There are',
        manage_pending_updates_suffix: 'profile change requests waiting for admin review from the app',
        manage_pending_updates_cta: 'Review now',
        manage_overview_title: '👥 User Overview',
        manage_search_placeholder: 'Search by name, email, plate number...',
        manage_auto_sync_badge: '⚡ Auto Sync Enabled',
        auto_sync_on: 'Auto Sync: On',
        auto_sync_off: 'Auto Sync: Off',
        manage_initial_hint: 'Data will auto-sync whenever App/Web changes are detected.',
        manage_row_count_initial: '0 users',
        complaints_title: 'Complaint Management',
        complaints_subtitle: 'Track, filter, review complaints and verify app evidence in realtime',
        complaints_search_placeholder: 'Search by user, reason, violation ID, description...',
        complaints_filter_all: 'All statuses',
        complaints_filter_pending: 'Pending',
        complaints_filter_approved: 'Approved',
        complaints_filter_rejected: 'Rejected',
        complaints_empty_title: 'No complaint data yet',
        complaints_empty_subtitle: 'Items will appear when complaints are submitted from the app.',
        complaints_total: 'Total complaints',
        complaints_pending: 'Pending',
        complaints_approved: 'Approved',
        complaints_rejected: 'Rejected',
        complaints_not_found_title: 'No matching complaints found',
        complaints_not_found_subtitle: 'Try another keyword or status filter.',
        complaints_fallback_reason: 'Complaint',
        complaints_fallback_desc: 'No detailed description',
        complaints_no_evidence: 'No evidence image',
        complaints_image_loading: 'Loading image...',
        complaints_image_error: 'Failed to load image',
        complaints_meta_id: 'Complaint ID',
        complaints_meta_violation_id: 'Violation ID',
        complaints_meta_type: 'Violation type',
        complaints_meta_fine: 'Fine amount',
        complaints_admin_note: '💬 Admin note',
        complaints_btn_detail: '👁️ Detail',
        complaints_btn_profile: '👤 Profile',
        complaints_btn_approve: '✅ Approve',
        complaints_btn_reject: '❌ Reject',
        complaints_btn_delete: '🗑️ Delete',
        complaints_btn_processing: 'Processing...',
        complaints_pagination_info: 'Showing {from}-{to} / {total}',
        complaints_pagination_prev: '← Prev',
        complaints_pagination_next: 'Next →',
        complaints_pagination_page: 'Page {page}/{total}',
        complaints_delete_confirm: 'Delete this complaint from the system?',
        complaints_delete_success: '🗑️ Complaint deleted',
        complaints_delete_error: 'Failed to delete complaint',
        complaints_delete_loading: 'Deleting complaint...',
        complaint_status_pending: '⏳ Pending',
        complaint_status_approved: '✅ Approved',
        complaint_status_rejected: '❌ Rejected',
        global_loader_title: 'Processing',
        global_loader_message: 'Please wait for a moment...',
        admin_sync_success: 'Data auto-synced successfully',
        user_count_suffix: 'users',
        // Image tab controls
        image_model_label: 'Model',
        image_model_default: 'Default',
        image_display_label: 'Display',
        image_toggle_mask: 'Mask',
        image_toggle_boxes: 'BBox',
        image_toggle_labels: 'Labels',
        image_detect_btn: 'Detect Objects',
        // Data Management table headers
        manage_th_name: 'Full Name',
        manage_th_cccd: 'ID Card',
        manage_th_phone: 'Phone',
        manage_th_violations: 'Violations / Payment',
        manage_th_complaints: 'Complaints / Notifications',
        manage_th_actions: 'Actions',
        // Data Management dynamic strings
        manage_not_updated: 'Not updated',
        manage_unpaid: 'Unpaid',
        manage_paid: 'Paid',
        manage_debt: 'Owes',
        manage_no_debt: 'No debt',
        manage_complaints_count: 'complaints',
        manage_notifications_count: 'notifications',
        manage_btn_review: 'Review changes',
        manage_btn_detail: 'Details',
        manage_btn_delete: 'Delete',
        manage_no_result: 'No matching result found.',
        manage_no_data: 'No user data available yet.',
        // Quota
        quota_title: 'Firestore Quota Monitor',
        quota_reads: 'Reads',
        quota_writes: 'Writes',
        quota_deletes: 'Deletes',
        quota_uptime: 'Uptime',
        quota_note: 'Quota data is tracked since server start and refreshed every 10 seconds.',
        // Reject modal
        reject_modal_title: 'Reject Complaint',
        reject_modal_subtitle: 'Please enter rejection reason',
        reject_modal_placeholder: 'Enter reason for rejection...',
        reject_modal_cancel: 'Cancel',
        reject_modal_confirm: 'Confirm rejection',
        // Typed.js
        typed_strings: [
            'powered by YOLOv26-seg AI',
            'recognizing 40 object classes',
            'real-time processing with high accuracy',
            'continuous tracking via ByteTrack',
            'automatic violation behavior analysis'
        ],
    }
};

let currentLang = localStorage.getItem('lang') || 'vi';
let typedInstance = null;

function setLanguage(lang) {
    currentLang = lang;
    localStorage.setItem('lang', lang);
    const t = i18n[lang];

    // Update all data-i18n elements
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.dataset.i18n;
        const value = t[key];
        if (value === undefined || value === null) return;
        if (typeof value === 'string' && value.includes('<')) {
            el.innerHTML = value;
        } else if (typeof value === 'string') {
            el.textContent = value;
        } else {
            el.textContent = String(value);
        }
    });

    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
        const key = el.dataset.i18nPlaceholder;
        const value = t[key];
        if (typeof value === 'string') {
            el.setAttribute('placeholder', value);
        }
    });

    document.querySelectorAll('[data-i18n-title]').forEach(el => {
        const key = el.dataset.i18nTitle;
        const value = t[key];
        if (typeof value === 'string') {
            el.setAttribute('title', value);
        }
    });

    // Update toggle button
    const langFlag = document.getElementById('langFlag');
    const langCode = document.getElementById('langCode');
    if (langFlag) langFlag.textContent = lang === 'vi' ? '🇻🇳' : '🇬🇧';
    if (langCode) langCode.textContent = lang === 'vi' ? 'VI' : 'EN';

    // Update Typed.js with new language strings
    if (typeof Typed !== 'undefined' && document.querySelector('.typing-text')) {
        if (typedInstance) typedInstance.destroy();
        typedInstance = new Typed('.typing-text', {
            strings: t.typed_strings,
            typeSpeed: 40,
            backSpeed: 20,
            backDelay: 2000,
            loop: true,
            showCursor: true,
            cursorChar: '|'
        });
    }

    // Update page title
    document.title = lang === 'vi'
        ? '🚦 Phát hiện vi phạm giao thông'
        : '🚦 Traffic Violation Detection';

    try {
        if (typeof renderAutoSyncToggleState === 'function') {
            renderAutoSyncToggleState();
        }
        if (_adminCachedData) {
            renderAdminTable(_adminCachedData, manageSearchInput ? manageSearchInput.value : '');
            renderComplaintBoard(_adminCachedData);
            const detailPanel = document.getElementById('tab-user-detail');
            if (detailPanel?.classList.contains('active') && detailPanel.dataset.uid) {
                showUserDetails(detailPanel.dataset.uid);
            }
        }
    } catch (_) {}
}

function tr(key, fallback = '') {
    const langDict = i18n[currentLang] || {};
    const value = langDict[key];
    if (typeof value === 'string') return value;
    return fallback || key;
}

function trTemplate(key, params = {}, fallback = '') {
    let template = tr(key, fallback);
    Object.entries(params).forEach(([name, value]) => {
        template = template.replace(new RegExp(`\\{${name}\\}`, 'g'), String(value));
    });
    return template;
}

function getCurrentLocale() {
    return currentLang === 'en' ? 'en-US' : 'vi-VN';
}

function formatVnd(value) {
    return new Intl.NumberFormat(getCurrentLocale(), {
        style: 'currency',
        currency: 'VND',
    }).format(Number(value || 0));
}

// Language toggle button
const langToggle = document.getElementById('langToggle');
if (langToggle) {
    langToggle.addEventListener('click', () => {
        const newLang = currentLang === 'vi' ? 'en' : 'vi';
        setLanguage(newLang);
    });
}

// Apply saved language on load
setLanguage(currentLang);

// =====================================================================
// PARTICLES.JS — Interactive particle network in hero
// =====================================================================
if (typeof particlesJS !== 'undefined' && document.getElementById('particles-js')) {
    particlesJS('particles-js', {
        particles: {
            number: { value: 60, density: { enable: true, value_area: 900 } },
            color: { value: '#6366f1' },
            shape: { type: 'circle' },
            opacity: { value: 0.5, random: true, anim: { enable: true, speed: 0.8, opacity_min: 0.1 } },
            size: { value: 3, random: true, anim: { enable: true, speed: 2, size_min: 0.3 } },
            line_linked: { enable: true, distance: 150, color: '#6366f1', opacity: 0.2, width: 1 },
            move: { enable: true, speed: 1.5, direction: 'none', random: true, out_mode: 'out' }
        },
        interactivity: {
            detect_on: 'canvas',
            events: {
                onhover: { enable: true, mode: 'grab' },
                onclick: { enable: true, mode: 'push' },
                resize: true
            },
            modes: {
                grab: { distance: 200, line_linked: { opacity: 0.5 } },
                push: { particles_nb: 3 }
            }
        },
        retina_detect: true
    });
}


// (Typed.js is now initialized by the i18n system above)


// =====================================================================
// SCROLLREVEAL — Staggered reveal animations
// =====================================================================
if (typeof ScrollReveal !== 'undefined') {
    const sr = ScrollReveal({
        origin: 'bottom',
        distance: '80px',
        duration: 1000,
        easing: 'cubic-bezier(0.5, 0, 0, 1)',
        reset: false
    });

    sr.reveal('.hero-badge', { delay: 100, origin: 'top' });
    sr.reveal('.hero-title', { delay: 200, origin: 'top' });
    sr.reveal('.hero-desc', { delay: 300 });
    sr.reveal('.hero-stats', { delay: 400 });
    sr.reveal('.hero-cta', { delay: 500 });
}

// =====================================================================
// VANILLA TILT — 3D tilt effect on cards
// =====================================================================
if (typeof VanillaTilt !== 'undefined') {
    VanillaTilt.init(document.querySelectorAll('.tilt-card'), {
        max: 8,
        speed: 400,
        glare: true,
        'max-glare': 0.12,
        perspective: 1000
    });
}

// =====================================================================
// COUNT-UP ANIMATION — Hero stat values
// =====================================================================
function animateCountUp() {
    document.querySelectorAll('.hero-stat-value[data-count]').forEach(el => {
        const target = parseInt(el.dataset.count);
        const duration = 2000;
        const start = performance.now();
        const step = (now) => {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3); // easeOutCubic
            el.textContent = Math.round(target * eased);
            if (progress < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
    });
    document.querySelectorAll('.hero-stat-value[data-text]').forEach(el => {
        const text = el.dataset.text;
        setTimeout(() => { el.textContent = text; }, 800);
    });
}

// Trigger count-up when hero becomes visible
const heroObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            animateCountUp();
            heroObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.3 });
const heroSection = document.querySelector('.hero');
if (heroSection) heroObserver.observe(heroSection);

// =====================================================================
// SCROLL TO TOP BUTTON
// =====================================================================
const scrollTopBtn = document.getElementById('scrollTop');
if (scrollTopBtn) {
    window.addEventListener('scroll', () => {
        if (window.scrollY > 300) scrollTopBtn.classList.add('visible');
        else scrollTopBtn.classList.remove('visible');
    });
    scrollTopBtn.addEventListener('click', (e) => {
        e.preventDefault();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
}

// =====================================================================
// FCM — Firebase Cloud Messaging (Web Push Notifications)
// =====================================================================
// ⚠️ REPLACE the firebaseConfig values with YOUR Firebase project config
// Get them from: Firebase Console → Project Settings → General → Your apps
(function initFCM() {
    // Check if Firebase is loaded
    if (typeof firebase === 'undefined') {
        console.warn('[FCM] Firebase SDK not loaded, skipping FCM init');
        return;
    }

    const firebaseConfig = typeof FIREBASE_CONFIG !== 'undefined' ? FIREBASE_CONFIG : null;
    if (!firebaseConfig) {
        console.warn('[FCM] Firebase config not found. Load firebase-config.js before app.js');
        return;
    }

    try {
        // Initialize Firebase (only if not already initialized)
        if (!firebase.apps.length) {
            firebase.initializeApp(firebaseConfig);
        }

        const messaging = firebase.messaging();

        // Register Service Worker for background notifications
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/firebase-messaging-sw.js')
                .then((registration) => {
                    console.log('[FCM] Service Worker registered:', registration.scope);
                    messaging.useServiceWorker(registration);
                    requestNotificationPermission(messaging);
                })
                .catch((err) => {
                    console.error('[FCM] Service Worker registration failed:', err);
                    // Still try to request permission without explicit SW
                    requestNotificationPermission(messaging);
                });
        } else {
            console.warn('[FCM] Service Workers not supported');
        }

        // Handle foreground messages (show toast)
        messaging.onMessage((payload) => {
            console.log('[FCM] Foreground message:', payload);

            const title = payload.notification?.title || '🚨 Vi phạm giao thông';
            const body = payload.notification?.body || 'Có vi phạm mới được phát hiện';

            // Show in-app toast
            showToast(`📩 ${title}: ${body}`, 'warning');

            // Also show browser notification (if permitted)
            if (Notification.permission === 'granted') {
                new Notification(title, {
                    body: body,
                    icon: '/static/favicon.png',
                    tag: 'violation-' + (payload.data?.violation_id || Date.now()),
                });
            }

            // Refresh snapshots gallery if on realtime tab
            loadSnapshots();
        });

        // Listen for messages from service worker (notification click)
        navigator.serviceWorker?.addEventListener('message', (event) => {
            if (event.data?.type === 'NOTIFICATION_CLICK') {
                console.log('[FCM] Notification clicked:', event.data.data);
                // Navigate to realtime tab to show detection results
                const realtimeTab = document.querySelector('.nav-link[data-tab="realtime"]');
                if (realtimeTab) realtimeTab.click();
            }
        });

    } catch (e) {
        console.error('[FCM] Initialization error:', e);
    }
})();

// Request notification permission and get FCM token
async function requestNotificationPermission(messaging) {
    try {
        const permission = await Notification.requestPermission();
        console.log('[FCM] Permission:', permission);

        if (permission !== 'granted') {
            console.warn('[FCM] Notification permission denied');
            return;
        }

        // Get FCM token
        const token = await messaging.getToken({
            // vapidKey: 'YOUR_VAPID_KEY_HERE'  // Uncomment and set for production
        });

        if (token) {
            console.log('[FCM] Token:', token.substring(0, 20) + '...');

            // Register token with backend
            try {
                const resp = await fetch('/api/fcm/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_id: 'default_user',
                        fcm_token: token,
                        platform: 'web',
                        device_info: navigator.userAgent.substring(0, 100),
                    })
                });
                const data = await resp.json();
                console.log('[FCM] Token registered:', data);
            } catch (e) {
                console.warn('[FCM] Token registration failed (backend may be offline):', e);
            }
        }
    } catch (e) {
        console.error('[FCM] Permission/token error:', e);
    }
}

// =====================================================================
// DATA MANAGEMENT (Admin Dashboard)
// =====================================================================
const adminDataTableBody = document.getElementById('adminDataTableBody');
const manageSearchInput = document.getElementById('manageSearchInput');

// Stat elements
const statTotalUsers = document.getElementById('statTotalUsers');
const statTotalVehicles = document.getElementById('statTotalVehicles');
const statTotalViolations = document.getElementById('statTotalViolations');
const statPendingFines = document.getElementById('statPendingFines');
const statTotalComplaints = document.getElementById('statTotalComplaints');
const manageRowCount = document.getElementById('manageRowCount');
const complaintBoardList = document.getElementById('complaintBoardList');
const complaintBoardStats = document.getElementById('complaintBoardStats');
const complaintSearchInput = document.getElementById('complaintSearchInput');
const complaintStatusFilter = document.getElementById('complaintStatusFilter');
const manageAutoSyncToggle = document.getElementById('manageAutoSyncToggle');
const complaintAutoSyncToggle = document.getElementById('complaintAutoSyncToggle');

// Cache data for search
let _adminCachedData = null;
let _complaintBoardQuery = '';
let _complaintBoardStatus = 'all';
let _complaintBoardPage = 1;
const COMPLAINT_PAGE_SIZE = 10;
const _complaintReviewLoading = new Set();
const _complaintDeleteLoading = new Set();
let _adminAutoSyncEnabled = localStorage.getItem('adminAutoSyncEnabled') === '1'; // P0: default OFF
// In-flight guard: prevent overlapping loadAdminData requests
let _adminLoadInFlight = false;
// WS event debounce: coalesce burst events into one refresh
let _adminWsDebounceTimer = null;
const _ADMIN_WS_DEBOUNCE_MS = 1500;
// Pending scope from WS events (accumulate during debounce window)
let _adminPendingScope = null;
// Fallback polling interval (P0: raised from 10s to 60s)
const _ADMIN_POLL_INTERVAL_MS = 60000;
// Track active manage/complaint tab visibility
let _adminTabActive = false;
if (complaintStatusFilter) {
    _complaintBoardStatus = complaintStatusFilter.value || 'all';
}

function normalizeDriverLicenses(user) {
    if (!user) return [];

    if (Array.isArray(user.driverLicenses) && user.driverLicenses.length > 0) {
        return user.driverLicenses.map(item => ({
            class: (item?.class || '').toString(),
            vehicleType: (item?.vehicleType || '').toString(),
            issueDate: (item?.issueDate || '').toString(),
            expiryDate: (item?.expiryDate || '').toString(),
            licenseNumber: (item?.licenseNumber || '').toString(),
        }));
    }

    const fallback = [];
    if (user.carLicenseClass || user.licenseNumber || user.licenseIssueDate || user.licenseExpiryDate) {
        fallback.push({
            class: (user.carLicenseClass || user.licenseClass || '').toString(),
            vehicleType: 'Ô tô',
            issueDate: (user.licenseIssueDate || '').toString(),
            expiryDate: (user.licenseExpiryDate || '').toString(),
            licenseNumber: (user.licenseNumber || '').toString(),
        });
    }
    if (user.motoLicenseClass) {
        fallback.push({
            class: user.motoLicenseClass.toString(),
            vehicleType: 'Xe máy',
            issueDate: (user.licenseIssueDate || '').toString(),
            expiryDate: (user.licenseExpiryDate || '').toString(),
            licenseNumber: (user.licenseNumber || '').toString(),
        });
    }
    return fallback;
}

function isMotoLicense(license) {
    const vehicleType = (license?.vehicleType || '').toString().toLowerCase();
    const cls = (license?.class || '').toString().toUpperCase();
    return vehicleType.includes('xe máy') || vehicleType.includes('motor') || cls.startsWith('A');
}

function isCarLicense(license) {
    const vehicleType = (license?.vehicleType || '').toString().toLowerCase();
    const cls = (license?.class || '').toString().toUpperCase();
    return vehicleType.includes('ô tô') || vehicleType.includes('o to') || vehicleType.includes('car') || /^[BCDEF]/.test(cls);
}

function getLicensePointState(user, licenses) {
    const fallbackPoints = Number.isFinite(Number(user?.points)) ? Number(user.points) : 12;
    const motoPoints = Number.isFinite(Number(user?.motoPoints))
        ? Number(user.motoPoints)
        : (Number.isFinite(Number(user?.motoLicensePoints)) ? Number(user.motoLicensePoints) : fallbackPoints);
    const carPoints = Number.isFinite(Number(user?.carPoints))
        ? Number(user.carPoints)
        : (Number.isFinite(Number(user?.carLicensePoints)) ? Number(user.carLicensePoints) : fallbackPoints);

    const motoLicenses = (licenses || []).filter(isMotoLicense);
    const carLicenses = (licenses || []).filter(isCarLicense);

    return {
        moto: {
            exists: motoLicenses.length > 0,
            points: Math.max(0, Math.min(12, Math.round(motoPoints))),
            licenses: motoLicenses,
        },
        car: {
            exists: carLicenses.length > 0,
            points: Math.max(0, Math.min(12, Math.round(carPoints))),
            licenses: carLicenses,
        },
    };
}

function getPointColor(points) {
    if (points >= 8) return '#10b981';
    if (points >= 4) return '#f59e0b';
    return '#ef4444';
}

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function toEpochMs(value) {
    if (value === null || value === undefined || value === '') return null;
    if (typeof value === 'number') {
        return value > 1e12 ? value : value * 1000;
    }
    if (value && typeof value.toDate === 'function') {
        try { return value.toDate().getTime(); } catch (_) {}
    }
    const parsed = Date.parse(String(value));
    return Number.isNaN(parsed) ? null : parsed;
}

function formatDateTime(value) {
    const ms = toEpochMs(value);
    if (!ms) return '—';
    return new Date(ms).toLocaleString(getCurrentLocale(), {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    });
}

function normalizeComplaintStatus(status) {
    const normalized = String(status || '').toLowerCase().trim();
    if (normalized === 'approved' || normalized === 'rejected' || normalized === 'pending') {
        return normalized;
    }
    return 'pending';
}

function getComplaintStatusMeta(status) {
    const normalized = normalizeComplaintStatus(status);
    if (normalized === 'approved') {
        return { label: tr('complaint_status_approved', '✅ Đã duyệt'), tagClass: 'tag-success', cardClass: 'approved' };
    }
    if (normalized === 'rejected') {
        return { label: tr('complaint_status_rejected', '❌ Đã từ chối'), tagClass: 'tag-danger', cardClass: 'rejected' };
    }
    return { label: tr('complaint_status_pending', '⏳ Đang xử lý'), tagClass: 'tag-warning', cardClass: 'pending' };
}

function getComplaintEvidenceUrl(complaint) {
    if (!complaint) return '';
    const complaintId = String(complaint.id || '').trim();
    const candidates = [
        complaint.evidenceUrlResolved,
        complaint.evidenceUrl,
        complaint.evidenceURL,
        complaint.evidenceDownloadUrl,
        complaint.evidence_download_url,
        complaint.evidenceImageUrl,
        complaint.evidence_image_url,
        complaint.downloadUrl,
        complaint.downloadURL,
        typeof complaint.evidence === 'string' ? complaint.evidence : '',
        complaint.proofUrl,
        complaint.imageUrl,
        complaint.image_url,
        complaint.evidencePath,
        complaint.evidence_path,
        complaint.storagePath,
        complaint.storage_path,
        complaint.fileUrl,
        complaint.file_url,
    ];

    if (complaint.evidence && typeof complaint.evidence === 'object') {
        candidates.push(complaint.evidence.url);
        candidates.push(complaint.evidence.downloadUrl);
        candidates.push(complaint.evidence.downloadURL);
        candidates.push(complaint.evidence.download_url);
        candidates.push(complaint.evidence.path);
        candidates.push(complaint.evidence.storagePath);
        candidates.push(complaint.evidence.storage_path);
        candidates.push(complaint.evidence.fullPath);
        candidates.push(complaint.evidence.full_path);
    }

    const found = candidates.find(v => typeof v === 'string' && v.trim().length > 0);
    if (!found) return '';

    const value = found.trim();
    if (!value) return '';

    if (value.startsWith('/api/admin/complaints/')) return value;
    if (value.startsWith('/')) return value;

    const looksLikeStoragePath =
        value.startsWith('complaints/') ||
        value.startsWith('gs://') ||
        value.includes('/o/') ||
        value.includes('firebasestorage.googleapis.com') ||
        value.includes('storage.googleapis.com');

    if (looksLikeStoragePath && complaintId) {
        return `/api/admin/complaints/${encodeURIComponent(complaintId)}/evidence?path=${encodeURIComponent(value)}`;
    }

    return value;
}

function setActionButtonLoading(button, isLoading, loadingText) {
    if (!button) return;
    if (isLoading) {
        button.dataset.originalHtml = button.innerHTML;
        button.classList.add('btn-loading');
        button.disabled = true;
        button.innerHTML = `<span class="spin"></span><span>${escapeHtml(loadingText || tr('complaints_btn_processing', 'Đang xử lý...'))}</span>`;
        return;
    }
    const original = button.dataset.originalHtml;
    if (original) button.innerHTML = original;
    delete button.dataset.originalHtml;
    button.classList.remove('btn-loading');
    button.disabled = false;
}

function updateManageStats(data) {
    const users = data.users || [];
    const vehicles = data.vehicles || [];
    const violations = data.violations || [];
    const complaints = data.complaints || [];

    let totalPending = 0;
    violations.forEach(v => {
        if (v.status === 'pending') totalPending += (v.fineAmount || 0);
    });

    const formatter = { format: formatVnd };

    if (statTotalUsers) statTotalUsers.textContent = users.length;
    if (statTotalVehicles) statTotalVehicles.textContent = vehicles.length;
    if (statTotalViolations) statTotalViolations.textContent = violations.length;
    if (statPendingFines) statPendingFines.textContent = formatter.format(totalPending);
    if (statTotalComplaints) statTotalComplaints.textContent = complaints.length;

    // Show pending updates banner
    const profileUpdates = data.profile_updates || [];
    const banner = document.getElementById('pendingUpdatesBanner');
    const countEl = document.getElementById('pendingUpdatesCount');
    if (banner && countEl) {
        if (profileUpdates.length > 0) {
            countEl.textContent = profileUpdates.length;
            banner.style.display = 'block';
        } else {
            banner.style.display = 'none';
        }
    }
}

function scrollToPendingUpdates() {
    // Scroll to first row that has a pending update button
    const pendingBtn = adminDataTableBody && adminDataTableBody.querySelector('.pending-update-btn');
    if (pendingBtn) {
        pendingBtn.closest('tr').scrollIntoView({ behavior: 'smooth', block: 'center' });
        pendingBtn.closest('tr').style.outline = '2px solid #f59e0b';
        pendingBtn.closest('tr').style.outlineOffset = '-2px';
        setTimeout(() => { pendingBtn.closest('tr').style.outline = ''; }, 2000);
    }
}

function renderAdminTable(data, searchQuery = '') {
    if (!adminDataTableBody) return;

    const users = data.users || [];
    const vehicles = data.vehicles || [];
    const violations = data.violations || [];
    const complaints = data.complaints || [];
    const notifications = data.notifications || [];
    const profileUpdates = data.profile_updates || [];

    const query = searchQuery.toLowerCase().trim();
    const formatter = { format: formatVnd };

    let html = '';
    let rowIndex = 0;

    users.forEach(u => {
        const uid = u.id;
        const uVehicles = vehicles.filter(v => v.ownerId === uid);
        const targetPlates = uVehicles.map(v => v.licensePlate);
        const uViolations = violations.filter(v => v.userId === uid || (v.licensePlate && targetPlates.includes(v.licensePlate)));
        const uComplaints = complaints.filter(c => c.userId === uid);
        const uNotifications = notifications.filter(n => n.userId === uid);
        const isPendingUpdate = profileUpdates.find(r => r.id === uid);
        const licenses = normalizeDriverLicenses(u);
        const licensePointState = getLicensePointState(u, licenses);
        const licensePointQuick = [
            licensePointState.moto.exists ? `🏍️ ${licensePointState.moto.points}/12` : '',
            licensePointState.car.exists ? `🚗 ${licensePointState.car.points}/12` : '',
        ].filter(Boolean).join(' • ');

        // Search filter
        if (query) {
            const searchable = [
                u.email, u.fullName, u.idCard, u.phone, uid,
                ...targetPlates,
                ...uVehicles.map(v => `${v.brand || ''} ${v.model || ''}`),
                ...licenses.map(l => `${l.class || ''} ${l.licenseNumber || ''} ${l.vehicleType || ''}`),
            ].join(' ').toLowerCase();
            if (!searchable.includes(query)) return;
        }

        rowIndex++;

        let pendingCount = 0;
        let paidCount = 0;
        let totalPendingFine = 0;

        uViolations.forEach(v => {
            if (v.status === 'pending' || v.status === 'pending_payment') {
                pendingCount++;
                totalPendingFine += (v.fineAmount || 0);
            } else if (v.status === 'paid') {
                paidCount++;
            }
        });

        const nameInitial = (u.fullName || '?').charAt(0).toUpperCase();

        html += `
            <tr id="row-${uid}">
                <td>${rowIndex}</td>
                <td>
                    <div style="display:flex; align-items:center; gap:10px;">
                        <div style="width:34px;height:34px;border-radius:50%;background:var(--accent-gradient);display:flex;align-items:center;justify-content:center;font-weight:800;font-size:0.85rem;flex-shrink:0;">${nameInitial}</div>
                        <div>
                            <div style="font-weight:700;color:var(--text-primary);">${u.fullName || `<span style="color:var(--text-muted);font-style:italic;">${tr('manage_not_updated')}</span>`}</div>
                            <div style="font-size:0.72rem;color:var(--text-muted);">${u.email || ''}</div>
                        </div>
                    </div>
                </td>
                <td><div style="color:var(--text-secondary);font-family:monospace;">🪪 ${u.idCard || '—'}</div></td>
                <td><div style="color:var(--text-secondary);">📱 ${u.phone || '—'}</div></td>
                <td>
                    <span class="data-tag tag-danger">${tr('manage_unpaid')}: ${pendingCount}</span>
                    <span class="data-tag tag-success">${tr('manage_paid')}: ${paidCount}</span>
                    ${totalPendingFine > 0 ? `<div style="font-weight:700;color:var(--danger);margin-top:6px;font-size:0.8rem;">${tr('manage_debt')}: ${formatter.format(totalPendingFine)}</div>` : `<div style="color:var(--success);margin-top:4px;font-weight:600;font-size:0.78rem;">✓ ${tr('manage_no_debt')}</div>`}
                    ${licensePointQuick ? `<div style="margin-top:6px;font-size:0.78rem;color:var(--text-secondary);font-weight:600;">${licensePointQuick}</div>` : ''}
                </td>
                <td>
                    ${uComplaints.length > 0
                        ? `<span class="data-tag tag-warning">📝 ${uComplaints.length} ${tr('manage_complaints_count')}</span><br>`
                        : `<span class="data-tag tag-secondary">0 ${tr('manage_complaints_count')}</span><br>`}
                    <span class="data-tag tag-secondary">🔔 ${uNotifications.length} ${tr('manage_notifications_count')}</span>
                </td>
                <td>
                    <div style="display:flex;flex-direction:column;gap:5px;min-width:110px;">
                        ${isPendingUpdate ? `<button class="btn btn-warning pending-update-btn" style="padding:5px 10px;font-size:0.78rem;color:#fff;border-radius:7px;" onclick="reviewUserUpdate('${uid}')">🔔 ${tr('manage_btn_review')}</button>` : ''}
                        <button class="btn btn-primary" style="padding:5px 10px;font-size:0.78rem;border-radius:7px;" onclick="showUserDetailPage('${uid}')">👁️ ${tr('manage_btn_detail')}</button>
                        <button class="btn btn-danger" style="padding:5px 10px;font-size:0.78rem;border-radius:7px;" onclick="confirmDeleteUser('${uid}', '${(u.fullName||'').replace(/'/g,'')}')" >🗑️ ${tr('manage_btn_delete')}</button>
                    </div>
                </td>
            </tr>
        `;
    });

    if (rowIndex === 0) {
        html = `<tr><td colspan="7" style="text-align:center;color:var(--text-muted);padding:50px 40px;">
            <div style="font-size:2rem;margin-bottom:8px;">🔍</div>
            ${query ? tr('manage_no_result') : tr('manage_no_data')}
        </td></tr>`;
    }

    adminDataTableBody.innerHTML = html;
    if (manageRowCount) manageRowCount.textContent = `${rowIndex} ${tr('user_count_suffix', currentLang === 'en' ? 'users' : 'người dùng')}`;
}

function renderManageLoadingRow(message) {
    return `<tr><td colspan="7" style="text-align:center;padding:60px 40px;">
        <div class="manage-loading-wrap">
            <div class="manage-loading manage-loading-spinner"></div>
            <div class="manage-loading-text">${escapeHtml(message || (currentLang === 'en' ? 'Loading data from Firebase...' : 'Đang tải dữ liệu từ Firebase...'))}</div>
            <div class="manage-loading-dots"><span></span><span></span><span></span></div>
        </div>
    </td></tr>`;
}

function renderComplaintBoardLoading(message) {
    if (!complaintBoardList) return;
    complaintBoardList.innerHTML = `
        <div class="complaint-board-empty">
            <div class="manage-loading-wrap">
                <div class="manage-loading manage-loading-spinner"></div>
                <div class="manage-loading-text">${escapeHtml(message || (currentLang === 'en' ? 'Loading complaint list...' : 'Đang tải danh sách khiếu nại...'))}</div>
                <div class="manage-loading-dots"><span></span><span></span><span></span></div>
            </div>
        </div>
    `;
}

function renderComplaintBoard(data) {
    if (!complaintBoardList) return;

    const users = data?.users || [];
    const violations = data?.violations || [];
    const complaints = data?.complaints || [];

    const usersById = new Map(users.map(u => [String(u.id || ''), u]));
    const violationsById = new Map(violations.map(v => [String(v.id || '').toUpperCase(), v]));

    const normalized = complaints.map(c => {
        const complaintId = String(c.id || '');
        const userId = String(c.userId || '');
        const violationId = String(c.violationId || '');
        const user = usersById.get(userId);
        const relatedViolation = violationsById.get(violationId.toUpperCase()) || null;
        const status = normalizeComplaintStatus(c.status);
        const meta = getComplaintStatusMeta(status);
        const evidenceUrl = getComplaintEvidenceUrl(c);
        const createdMs = toEpochMs(c.createdAt ?? c.updatedAt ?? c.reviewedAt ?? c.timestamp) || 0;
        return {
            ...c,
            id: complaintId,
            userId,
            violationId,
            status,
            statusMeta: meta,
            evidenceUrl,
            createdMs,
            userName: user?.fullName || user?.email || userId || (currentLang === 'en' ? 'Unknown' : 'Không rõ'),
            userEmail: user?.email || '—',
            relatedViolation,
        };
    }).sort((a, b) => b.createdMs - a.createdMs);

    const pendingCount = normalized.filter(c => c.status === 'pending').length;
    const approvedCount = normalized.filter(c => c.status === 'approved').length;
    const rejectedCount = normalized.filter(c => c.status === 'rejected').length;

    if (complaintBoardStats) {
        complaintBoardStats.innerHTML = `
            <div class="complaint-board-stat"><div class="stat-icon icon-total">📋</div><div class="stat-content"><div class="value">${normalized.length}</div><div class="label">${escapeHtml(tr('complaints_total', 'Tổng khiếu nại'))}</div></div></div>
            <div class="complaint-board-stat"><div class="stat-icon icon-pending">⏳</div><div class="stat-content"><div class="value" style="color:#f59e0b;">${pendingCount}</div><div class="label">${escapeHtml(tr('complaints_pending', 'Đang xử lý'))}</div></div></div>
            <div class="complaint-board-stat"><div class="stat-icon icon-approved">✅</div><div class="stat-content"><div class="value" style="color:#10b981;">${approvedCount}</div><div class="label">${escapeHtml(tr('complaints_approved', 'Đã duyệt'))}</div></div></div>
            <div class="complaint-board-stat"><div class="stat-icon icon-rejected">❌</div><div class="stat-content"><div class="value" style="color:#ef4444;">${rejectedCount}</div><div class="label">${escapeHtml(tr('complaints_rejected', 'Đã từ chối'))}</div></div></div>
        `;
    }

    const query = (_complaintBoardQuery || '').toLowerCase().trim();
    const statusFilter = (_complaintBoardStatus || 'all').toLowerCase().trim();
    const filtered = normalized.filter(c => {
        if (statusFilter !== 'all' && c.status !== statusFilter) return false;
        if (!query) return true;

        const searchable = [
            c.id,
            c.userId,
            c.userName,
            c.userEmail,
            c.violationId,
            c.reason,
            c.description,
            c.adminNote,
            c.relatedViolation?.violationType,
            c.relatedViolation?.type,
        ].map(v => (v || '').toString().toLowerCase()).join(' ');

        return searchable.includes(query);
    });

    if (filtered.length === 0) {
        complaintBoardList.innerHTML = `
            <div class="complaint-board-empty">
                <div style="font-size:2rem;">🔍</div>
                <div style="margin-top:8px;font-weight:700;">${escapeHtml(tr('complaints_not_found_title', 'Không tìm thấy khiếu nại phù hợp'))}</div>
                <div style="margin-top:4px;font-size:0.85rem;color:var(--text-muted);">${escapeHtml(tr('complaints_not_found_subtitle', 'Hãy đổi từ khóa tìm kiếm hoặc bộ lọc trạng thái.'))}</div>
            </div>
        `;
        return;
    }

    const totalPages = Math.max(1, Math.ceil(filtered.length / COMPLAINT_PAGE_SIZE));
    _complaintBoardPage = Math.max(1, Math.min(_complaintBoardPage, totalPages));

    const startIndex = (_complaintBoardPage - 1) * COMPLAINT_PAGE_SIZE;
    const endIndex = Math.min(startIndex + COMPLAINT_PAGE_SIZE, filtered.length);
    const pagedItems = filtered.slice(startIndex, endIndex);

    const cardsHtml = pagedItems.map(c => {
        const relatedViolation = c.relatedViolation;
        const fineText = relatedViolation ? formatVnd(relatedViolation.fineAmount || 0) : '—';
        const actionBusy = _complaintReviewLoading.has(c.id) || _complaintDeleteLoading.has(c.id);
        const loadingText = escapeHtml(tr('complaints_image_loading', 'Đang tải ảnh...'));
        const imageErrorText = escapeHtml(tr('complaints_image_error', 'Không tải được ảnh'));

        const evidenceHtml = c.evidenceUrl
            ? `
                <div class="complaint-card-evidence">
                    <div class="image-loading-mask">${loadingText}</div>
                    <img
                        src="${escapeHtml(c.evidenceUrl)}"
                        alt="Complaint evidence"
                        loading="lazy"
                        style="cursor:zoom-in;"
                        onclick="openImageLightbox(this.src)"
                        onload="if(this.previousElementSibling){this.previousElementSibling.style.display='none';}"
                        onerror="this.style.display='none'; if(this.previousElementSibling){this.previousElementSibling.style.display='flex'; this.previousElementSibling.style.animation='none'; this.previousElementSibling.textContent='${imageErrorText}';}"
                    />
                </div>
            `
            : `<div class="complaint-card-evidence"><div class="complaint-card-evidence-fallback">${escapeHtml(tr('complaints_no_evidence', 'Không có ảnh bằng chứng'))}</div></div>`;

        return `
            <article class="complaint-card ${c.statusMeta.cardClass}">
                <div class="complaint-card-main">
                    <div class="complaint-card-top">
                        <div>
                            <div class="complaint-card-title">${escapeHtml(c.reason || tr('complaints_fallback_reason', 'Khiếu nại'))}</div>
                            <div class="complaint-card-subtitle">👤 ${escapeHtml(c.userName)} • ${formatDateTime(c.createdAt)}</div>
                        </div>
                        <span class="data-tag ${c.statusMeta.tagClass}" style="white-space:nowrap;">${c.statusMeta.label}</span>
                    </div>

                    <div class="complaint-card-desc">${escapeHtml(c.description || tr('complaints_fallback_desc', 'Không có mô tả chi tiết'))}</div>

                    <div class="complaint-card-meta">
                        <div><span class="meta-label">${escapeHtml(tr('complaints_meta_id', 'Mã khiếu nại'))}:</span> <span class="meta-value">${escapeHtml(c.id || '—')}</span></div>
                        <div><span class="meta-label">${escapeHtml(tr('complaints_meta_violation_id', 'Mã vi phạm'))}:</span> <span class="meta-value">${escapeHtml(c.violationId || '—')}</span></div>
                        <div><span class="meta-label">${escapeHtml(tr('complaints_meta_type', 'Loại vi phạm'))}:</span> <span class="meta-value">${escapeHtml(relatedViolation?.violationType || relatedViolation?.type || '—')}</span></div>
                        <div><span class="meta-label">${escapeHtml(tr('complaints_meta_fine', 'Mức phạt'))}:</span> <span class="meta-value">${escapeHtml(fineText)}</span></div>
                    </div>

                    ${c.adminNote ? `<div style="font-size:0.78rem;color:#fca5a5;">${escapeHtml(tr('complaints_admin_note', '💬 Ghi chú admin'))}: ${escapeHtml(c.adminNote)}</div>` : ''}
                </div>

                <div class="complaint-card-side">
                    ${evidenceHtml}
                    <div class="complaint-card-actions">
                        <button class="complaint-card-btn detail" onclick="showComplaintDetailModal('${c.id}')">${escapeHtml(tr('complaints_btn_detail', '👁️ Chi tiết'))}</button>
                        <button class="complaint-card-btn profile" onclick="showUserDetailPage('${c.userId}')">${escapeHtml(tr('complaints_btn_profile', '👤 Hồ sơ'))}</button>
                        ${c.status === 'pending' ? `
                            <button class="complaint-card-btn approve ${actionBusy ? 'btn-loading' : ''}" ${actionBusy ? 'disabled' : ''} onclick="reviewComplaint('${c.id}', 'approve', '', this)">
                                ${actionBusy ? `<span class="spin"></span><span>${escapeHtml(tr('complaints_btn_processing', 'Đang xử lý...'))}</span>` : escapeHtml(tr('complaints_btn_approve', '✅ Chấp nhận'))}
                            </button>
                            <button class="complaint-card-btn reject ${actionBusy ? 'btn-loading' : ''}" ${actionBusy ? 'disabled' : ''} onclick="showRejectComplaintModal('${c.id}')">
                                ${actionBusy ? `<span class="spin"></span><span>${escapeHtml(tr('complaints_btn_processing', 'Đang xử lý...'))}</span>` : escapeHtml(tr('complaints_btn_reject', '❌ Từ chối'))}
                            </button>
                        ` : `
                            <button class="complaint-card-btn remove ${actionBusy ? 'btn-loading' : ''}" ${actionBusy ? 'disabled' : ''} onclick="confirmDeleteComplaint('${c.id}', this)">
                                ${actionBusy ? `<span class="spin"></span><span>${escapeHtml(tr('complaints_btn_processing', 'Đang xử lý...'))}</span>` : escapeHtml(tr('complaints_btn_delete', '🗑️ Xóa'))}
                            </button>
                        `}
                    </div>
                </div>
            </article>
        `;
    }).join('');

    const paginationHtml = `
        <div class="complaint-board-pagination">
            <div class="info">${escapeHtml(trTemplate('complaints_pagination_info', {
                from: startIndex + 1,
                to: endIndex,
                total: filtered.length,
            }, `Hiển thị ${startIndex + 1}-${endIndex} / ${filtered.length}`))}</div>
            <div class="complaint-board-page-controls">
                <button class="complaint-board-page-btn" ${_complaintBoardPage <= 1 ? 'disabled' : ''} onclick="setComplaintBoardPage(${_complaintBoardPage - 1})">${escapeHtml(tr('complaints_pagination_prev', '← Trước'))}</button>
                <span class="complaint-board-page-number">${escapeHtml(trTemplate('complaints_pagination_page', {
                    page: _complaintBoardPage,
                    total: totalPages,
                }, `Trang ${_complaintBoardPage}/${totalPages}`))}</span>
                <button class="complaint-board-page-btn" ${_complaintBoardPage >= totalPages ? 'disabled' : ''} onclick="setComplaintBoardPage(${_complaintBoardPage + 1})">${escapeHtml(tr('complaints_pagination_next', 'Tiếp →'))}</button>
            </div>
        </div>
    `;

    complaintBoardList.innerHTML = cardsHtml + paginationHtml;
}

function setComplaintBoardPage(nextPage) {
    const safePage = Number(nextPage);
    if (!Number.isFinite(safePage)) return;
    _complaintBoardPage = Math.max(1, Math.floor(safePage));
    if (_adminCachedData) renderComplaintBoard(_adminCachedData);
}

async function loadAdminData(options = {}) {
    if (!adminDataTableBody) return;
    // P0: in-flight guard — skip if a request is already running
    if (_adminLoadInFlight) return;
    _adminLoadInFlight = true;

    const silent = options.silent === true;
    const scope = options.scope || null; // P1: partial scope (CSV string)
    const force = options.force ? 1 : 0;
    const finishLoading = !silent
        ? showGlobalPageLoader(currentLang === 'en'
            ? 'Syncing Data Management...'
            : 'Đang đồng bộ dữ liệu Data Management...')
        : null;

    if (!silent) {
        adminDataTableBody.innerHTML = renderManageLoadingRow(currentLang === 'en'
            ? 'Loading data from Firebase...'
            : 'Đang tải dữ liệu từ Firebase...');
        renderComplaintBoardLoading(currentLang === 'en'
            ? 'Loading complaint list...'
            : 'Đang tải danh sách khiếu nại...');
    }

    try {
        // P1: build URL with scope/force params
        let url = '/api/admin/data';
        const params = [];
        if (force) params.push('force=1');
        if (scope) params.push('scope=' + encodeURIComponent(scope));
        if (params.length) url += '?' + params.join('&');

        const resp = await fetch(url);
        const json = await resp.json();

        if (json.status !== 'ok') {
            showToast('Lỗi tải dữ liệu: ' + json.message, 'error');
            adminDataTableBody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:var(--danger);padding:50px;">
                <div style="font-size:2rem;margin-bottom:8px;">⚠️</div>${json.message}
            </td></tr>`;
            renderComplaintBoardLoading(json.message || 'Không tải được dữ liệu khiếu nại');
            return;
        }

        // P1: partial response — merge into existing cache
        if (json.partial && _adminCachedData) {
            Object.assign(_adminCachedData, json.data);
        } else {
            _adminCachedData = json.data;
        }
        updateManageStats(_adminCachedData);
        renderAdminTable(_adminCachedData, manageSearchInput ? manageSearchInput.value : '');
        renderComplaintBoard(_adminCachedData);

    } catch (e) {
        console.error(e);
        if (!silent) {
            adminDataTableBody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:var(--danger);padding:50px;">
                <div style="font-size:2rem;margin-bottom:8px;">❌</div>Lỗi kết nối máy chủ.
            </td></tr>`;
            renderComplaintBoardLoading('Lỗi kết nối khi tải khiếu nại');
        }
    } finally {
        _adminLoadInFlight = false;
        if (finishLoading) finishLoading();
    }
}

// ── Delete user with styled confirm ─────────────────────────────────
function confirmDeleteUser(uid, name) {
    const existing = document.getElementById('deleteConfirmOverlay');
    if (existing) existing.remove();

    const overlay = document.createElement('div');
    overlay.id = 'deleteConfirmOverlay';
    overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.75);display:flex;align-items:center;justify-content:center;z-index:10002;padding:20px;backdrop-filter:blur(4px);';
    overlay.innerHTML = `
        <div style="background:var(--bg-secondary);border:1px solid rgba(239,68,68,0.35);border-radius:20px;max-width:420px;width:100%;padding:28px;box-shadow:0 24px 64px rgba(0,0,0,0.7);animation:fadeIn 0.25s ease;">
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;">
                <div style="width:48px;height:48px;border-radius:50%;background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.35);display:flex;align-items:center;justify-content:center;font-size:1.4rem;flex-shrink:0;">🗑️</div>
                <div>
                    <div style="font-size:1.1rem;font-weight:700;color:#fff;">Xóa tài khoản</div>
                    <div style="font-size:0.82rem;color:var(--text-muted);margin-top:2px;">Hành động này không thể hoàn tác</div>
                </div>
            </div>
            <p style="font-size:0.9rem;color:var(--text-secondary);line-height:1.6;margin-bottom:24px;">
                Bạn có chắc chắn muốn xóa tài khoản <strong style="color:#fff;">${name || uid}</strong>?<br>
                <span style="color:#ef4444;">Toàn bộ vi phạm, phương tiện, thông báo và khiếu nại của người dùng này sẽ bị xóa vĩnh viễn.</span>
            </p>
            <div style="display:flex;gap:10px;justify-content:flex-end;">
                <button onclick="document.getElementById('deleteConfirmOverlay').remove()" style="padding:10px 20px;border-radius:10px;border:1px solid var(--border-color);background:rgba(255,255,255,0.06);color:var(--text-primary);font-weight:600;cursor:pointer;font-size:0.88rem;font-family:inherit;">Hủy</button>
                <button onclick="deleteUser('${uid}')" id="confirmDeleteBtn" style="padding:10px 20px;border-radius:10px;border:none;background:linear-gradient(135deg,#ef4444,#dc2626);color:#fff;font-weight:700;cursor:pointer;font-size:0.88rem;font-family:inherit;">🗑️ Xóa tài khoản</button>
            </div>
        </div>
    `;
    document.body.appendChild(overlay);
    overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
}

async function deleteUser(uid) {
    const overlay = document.getElementById('deleteConfirmOverlay');
    if (overlay) overlay.remove();
    const finishLoading = showGlobalPageLoader(currentLang === 'en'
        ? 'Deleting account and related data...'
        : 'Đang xóa tài khoản và dữ liệu liên quan...');
    try {
        const resp = await fetch(`/api/admin/users/${uid}`, { method: 'DELETE' });
        const json = await resp.json();
        if (json.status === 'ok') {
            showToast(
                currentLang === 'en'
                    ? '✅ User account deleted successfully'
                    : '✅ Đã xóa tài khoản người dùng thành công',
                'success',
            );
            await loadAdminData({ silent: true });
        } else {
            showToast((currentLang === 'en' ? 'Error deleting user: ' : 'Lỗi khi xóa người dùng: ') + json.message, 'error');
        }
    } catch (e) {
        console.error(e);
        showToast(currentLang === 'en' ? 'Server connection error while deleting user' : 'Lỗi kết nối máy chủ khi xóa người dùng', 'error');
    } finally {
        finishLoading();
    }
}

function confirmDeleteComplaint(complaintId, triggerButton = null) {
    const normalizedId = String(complaintId || '').trim();
    if (!normalizedId) return;
    if (_complaintReviewLoading.has(normalizedId) || _complaintDeleteLoading.has(normalizedId)) return;

    const message = tr(
        'complaints_delete_confirm',
        'Bạn có chắc muốn xóa khiếu nại này khỏi hệ thống?',
    );
    if (!confirm(message)) return;
    deleteComplaint(normalizedId, triggerButton);
}

async function deleteComplaint(complaintId, triggerButton = null) {
    const normalizedId = String(complaintId || '').trim();
    if (!normalizedId) return;
    if (_complaintDeleteLoading.has(normalizedId)) return;

    _complaintDeleteLoading.add(normalizedId);
    setActionButtonLoading(triggerButton, true, tr('complaints_delete_loading', 'Đang xóa khiếu nại...'));
    const finishLoading = showGlobalPageLoader(tr('complaints_delete_loading', 'Đang xóa khiếu nại...'));

    try {
        const res = await fetch(`/api/admin/complaints/${encodeURIComponent(normalizedId)}`, {
            method: 'DELETE',
        });
        const data = await res.json();

        if (res.ok && data.status === 'ok') {
            if (_adminCachedData) {
                _adminCachedData.complaints = (_adminCachedData.complaints || []).filter(
                    c => String(c.id || '') !== normalizedId,
                );
                updateManageStats(_adminCachedData);
                renderAdminTable(_adminCachedData, manageSearchInput ? manageSearchInput.value : '');
                renderComplaintBoard(_adminCachedData);
            }

            const detailModal = document.getElementById('complaintDetailModal');
            if (detailModal) detailModal.remove();

            await loadAdminData({ silent: true });
            const detailPanel = document.getElementById('tab-user-detail');
            if (detailPanel?.classList.contains('active') && detailPanel.dataset.uid) {
                showUserDetails(detailPanel.dataset.uid);
            }

            showToast(data.message || tr('complaints_delete_success', '🗑️ Đã xóa khiếu nại'), 'success');
        } else {
            showToast(data.message || tr('complaints_delete_error', 'Lỗi khi xóa khiếu nại'), 'error');
        }
    } catch (e) {
        showToast(tr('complaints_delete_error', 'Lỗi khi xóa khiếu nại'), 'error');
    } finally {
        _complaintDeleteLoading.delete(normalizedId);
        setActionButtonLoading(triggerButton, false);
        finishLoading();
        if (_adminCachedData) renderComplaintBoard(_adminCachedData);
    }
}

// Search input
if (manageSearchInput) {
    let searchTimeout;
    manageSearchInput.addEventListener('input', () => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            if (_adminCachedData) {
                renderAdminTable(_adminCachedData, manageSearchInput.value);
            }
        }, 250);
    });
}

if (complaintSearchInput) {
    let complaintSearchTimeout;
    complaintSearchInput.addEventListener('input', () => {
        clearTimeout(complaintSearchTimeout);
        complaintSearchTimeout = setTimeout(() => {
            _complaintBoardQuery = complaintSearchInput.value || '';
            _complaintBoardPage = 1;
            if (_adminCachedData) renderComplaintBoard(_adminCachedData);
        }, 220);
    });
}

if (complaintStatusFilter) {
    complaintStatusFilter.addEventListener('change', () => {
        _complaintBoardStatus = complaintStatusFilter.value || 'all';
        _complaintBoardPage = 1;
        if (_adminCachedData) renderComplaintBoard(_adminCachedData);
    });
}

// Auto-load when switching to the tab
document.querySelectorAll('.nav-link[data-tab="manage"], .nav-link[data-tab="complaints"]').forEach(link => {
    link.addEventListener('click', () => {
        loadAdminData({ silent: true });
    });
});

// ── Realtime auto-refresh + WebSocket channel ─────────────────────
let _adminAutoRefreshTimer = null;
let _adminRealtimeWs = null;
let _adminRealtimeReconnectTimer = null;
let _adminWsPingTimer = null;

function renderAutoSyncToggleState() {
    const label = _adminAutoSyncEnabled
        ? tr('auto_sync_on', 'Auto Sync')
        : tr('auto_sync_off', 'Auto Sync');
    [manageAutoSyncToggle, complaintAutoSyncToggle].forEach(btn => {
        if (!btn) return;
        btn.innerHTML = `<span class="sync-switch"></span><span>${label}</span>`;
        btn.classList.toggle('is-on', _adminAutoSyncEnabled);
        btn.classList.toggle('is-off', !_adminAutoSyncEnabled);
        btn.setAttribute('aria-pressed', _adminAutoSyncEnabled ? 'true' : 'false');
    });
}

function stopAdminRealtimeChannel() {
    if (_adminWsPingTimer) {
        clearInterval(_adminWsPingTimer);
        _adminWsPingTimer = null;
    }
    if (_adminRealtimeReconnectTimer) {
        clearTimeout(_adminRealtimeReconnectTimer);
        _adminRealtimeReconnectTimer = null;
    }
    if (_adminRealtimeWs) {
        try { _adminRealtimeWs.close(); } catch (_) {}
        _adminRealtimeWs = null;
    }
}

function connectAdminRealtimeChannel() {
    // Luôn kết nối WS (cho quota realtime); admin_data_changed sẽ tự check _adminAutoSyncEnabled
    stopAdminRealtimeChannel();
    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${wsProtocol}://${window.location.host}/ws/admin`;

    try {
        const ws = new WebSocket(wsUrl);
        _adminRealtimeWs = ws;

        ws.onopen = () => {
            _adminWsPingTimer = setInterval(() => {
                if (_adminRealtimeWs && _adminRealtimeWs.readyState === WebSocket.OPEN) {
                    _adminRealtimeWs.send(JSON.stringify({ action: 'ping' }));
                }
            }, 15000);
        };

        ws.onmessage = async (event) => {
            let msg = null;
            try { msg = JSON.parse(event.data); } catch (_) { return; }
            if (!msg) return;

            // Realtime quota update — xử lý luôn bất kể auto-sync
            if (msg.type === 'quota_update') {
                _applyQuotaData(msg);
                return;
            }

            if (!_adminAutoSyncEnabled) return;
            if (msg.type !== 'admin_data_changed') return;
            // P0: coalesce burst WS events via debounce
            const scope = msg.scope || null;
            if (scope && scope !== 'all') {
                _adminPendingScope = _adminPendingScope
                    ? _adminPendingScope + ',' + scope
                    : scope;
            } else {
                _adminPendingScope = null; // full refresh
            }
            clearTimeout(_adminWsDebounceTimer);
            _adminWsDebounceTimer = setTimeout(() => {
                const s = _adminPendingScope;
                _adminPendingScope = null;
                // Deduplicate scope parts
                const scopeParam = s ? [...new Set(s.split(',').map(x => x.trim()).filter(Boolean))].join(',') : null;
                loadAdminData({ silent: true, scope: scopeParam });
            }, _ADMIN_WS_DEBOUNCE_MS);
        };

        ws.onclose = () => {
            if (_adminWsPingTimer) {
                clearInterval(_adminWsPingTimer);
                _adminWsPingTimer = null;
            }
            _adminRealtimeWs = null;
            // Luôn reconnect — WS cần cho quota realtime
            _adminRealtimeReconnectTimer = setTimeout(connectAdminRealtimeChannel, 2000);
        };

        ws.onerror = () => {
            try { ws.close(); } catch (_) {}
        };
    } catch (_) {}
}

function startAdminAutoRefresh() {
    if (!_adminAutoSyncEnabled) return;
    stopAdminAutoRefresh();
    _adminAutoRefreshTimer = setInterval(() => {
        // P0: only poll when admin tab is active AND page is visible
        if (!_adminTabActive || document.hidden) return;
        loadAdminData({ silent: true });
    }, _ADMIN_POLL_INTERVAL_MS); // P0: 60s fallback instead of 10s
}

function stopAdminAutoRefresh() {
    if (_adminAutoRefreshTimer) {
        clearInterval(_adminAutoRefreshTimer);
        _adminAutoRefreshTimer = null;
    }
}

function setAdminAutoSyncEnabled(enabled, options = {}) {
    _adminAutoSyncEnabled = !!enabled;
    localStorage.setItem('adminAutoSyncEnabled', _adminAutoSyncEnabled ? '1' : '0');
    renderAutoSyncToggleState();

    if (_adminAutoSyncEnabled) {
        startAdminAutoRefresh();
        connectAdminRealtimeChannel();
        if (options.loadNow !== false) {
            loadAdminData({ silent: true });
        }
        return;
    }

    stopAdminAutoRefresh();
    // Không đóng WS channel — vẫn cần cho quota realtime
}

// Start realtime sync when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Luôn kết nối admin WS cho quota realtime push
    connectAdminRealtimeChannel();

    if (adminDataTableBody || complaintBoardList) {
        renderAutoSyncToggleState();
        setAdminAutoSyncEnabled(_adminAutoSyncEnabled, { loadNow: false });
    }

    [manageAutoSyncToggle, complaintAutoSyncToggle].forEach(btn => {
        if (!btn) return;
        btn.addEventListener('click', () => {
            setAdminAutoSyncEnabled(!_adminAutoSyncEnabled);
        });
    });

    if (_adminAutoSyncEnabled && (adminDataTableBody || complaintBoardList)) {
        loadAdminData({ silent: true });
    }
});

// P0: Track which tab is active — only poll when manage/complaints tab visible
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => {
        const tabId = link.dataset.tab;
        _adminTabActive = (tabId === 'manage' || tabId === 'complaints');
    });
});

// P0: Pause sync when page is hidden (minimized / other tab)
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Stop polling while page is hidden
        stopAdminAutoRefresh();
    } else if (_adminAutoSyncEnabled && _adminTabActive) {
        // Resume polling when page becomes visible AND admin tab is active
        startAdminAutoRefresh();
    }
});

// =====================================================================
// USER DETAIL — FULL-PAGE VIEW
// =====================================================================
function showUserDetailPage(uid) {
    if (!_adminCachedData) return;

    // Store uid for realtime refresh
    const detailPanelRef = document.getElementById('tab-user-detail');
    if (detailPanelRef) detailPanelRef.dataset.uid = uid;

    showUserDetails(uid);
}

function showUserDetails(uid) {
    if (!_adminCachedData) return;

    const users = _adminCachedData.users || [];
    const vehicles = _adminCachedData.vehicles || [];
    const violations = _adminCachedData.violations || [];
    const complaints = _adminCachedData.complaints || [];
    const profileUpdates = _adminCachedData.profile_updates || [];

    const u = users.find(x => x.id === uid);
    if (!u) return;

    const uVehicles = vehicles.filter(v => v.ownerId === uid);
    const targetPlates = uVehicles.map(v => v.licensePlate);
    const uViolations = violations.filter(v => v.userId === uid || (v.licensePlate && targetPlates.includes(v.licensePlate)));
    const uComplaints = complaints.filter(c => c.userId === uid);
    const isPendingUpdate = profileUpdates.find(r => r.id === uid);

    const formatter = { format: formatVnd };

    let pendingCount = 0, paidCount = 0, totalPendingFine = 0;
    uViolations.forEach(v => {
        if (v.status === 'pending' || v.status === 'pending_payment') { pendingCount++; totalPendingFine += (v.fineAmount || 0); }
        else if (v.status === 'paid') paidCount++;
    });

    const driverLicenses = normalizeDriverLicenses(u);
    const licensePointState = getLicensePointState(u, driverLicenses);
    const licenseStatsHtml = [
        licensePointState.moto.exists
            ? `
                <div class="user-detail-stat">
                    <span class="user-detail-stat-val amber">${licensePointState.moto.points}/12</span>
                    <div class="user-detail-stat-lbl">GPLX xe máy</div>
                </div>
            `
            : '',
        licensePointState.car.exists
            ? `
                <div class="user-detail-stat">
                    <span class="user-detail-stat-val amber">${licensePointState.car.points}/12</span>
                    <div class="user-detail-stat-lbl">GPLX ô tô</div>
                </div>
            `
            : '',
    ].join('');
    const nameInitial = (u.fullName || '?').charAt(0).toUpperCase();

    // ── HEADER ──
    const headerEl = document.getElementById('userDetailHeader');
    if (headerEl) {
        headerEl.innerHTML = `
            <div class="user-detail-avatar">${nameInitial}</div>
            <div class="user-detail-name-block">
                <div class="user-detail-name">${u.fullName || 'Chưa cập nhật tên'}</div>
                <div class="user-detail-email">📧 ${u.email || '—'} &nbsp;|&nbsp; 📱 ${u.phone || '—'}</div>
                <div class="user-detail-badges">
                    <span class="user-detail-badge verified">✅ Đã xác minh</span>
                    ${isPendingUpdate ? `<span class="user-detail-badge pending-update">🔔 Đang chờ duyệt sửa đổi</span>` : ''}
                </div>
            </div>
            <div class="user-detail-stats-row">
                <div class="user-detail-stat">
                    <span class="user-detail-stat-val ${pendingCount > 0 ? 'danger' : 'success'}">${pendingCount}</span>
                    <div class="user-detail-stat-lbl">Chưa nộp</div>
                </div>
                <div class="user-detail-stat">
                    <span class="user-detail-stat-val success">${paidCount}</span>
                    <div class="user-detail-stat-lbl">Đã nộp</div>
                </div>
                ${licenseStatsHtml}
                <div class="user-detail-stat">
                    <span class="user-detail-stat-val">${uVehicles.length}</span>
                    <div class="user-detail-stat-lbl">Phương tiện</div>
                </div>
            </div>
        `;
    }

    // ── TOP ACTIONS ──
    const actionsTop = document.getElementById('userDetailActionsTop');
    if (actionsTop) {
        actionsTop.innerHTML = `
            ${isPendingUpdate ? `<button class="btn btn-warning" style="padding:10px 18px;font-size:0.85rem;border-radius:10px;font-weight:700;color:#fff;" onclick="reviewUserUpdate('${uid}')">🔔 Duyệt sửa đổi</button>` : ''}
            <button class="btn btn-danger" style="padding:10px 18px;font-size:0.85rem;border-radius:10px;font-weight:700;" onclick="confirmDeleteUser('${uid}', '${(u.fullName||'').replace(/'/g,'')}')">🗑️ Xóa tài khoản</button>
        `;
    }

    // ── GRID CONTENT ──
    const gridEl = document.getElementById('userDetailGrid');
    if (gridEl) {
        // Personal info
        const personalHtml = `
            <div class="user-detail-section">
                <div class="user-detail-section-title">👤 Thông tin cá nhân</div>
                ${row('Họ và tên', u.fullName || '—')}
                ${row('CCCD/CMND', u.idCard || '—')}
                ${row('Ngày cấp CCCD', u.idCardIssueDate || '—')}
                ${row('Số điện thoại', u.phone || '—')}
                ${row('Email', u.email || '—')}
                ${row('Địa chỉ', u.address || '—')}
                ${row('Ngày sinh', u.dateOfBirth || '—')}
                ${row('Giới tính', u.gender || '—')}
                ${row('Nghề nghiệp', u.occupation || '—')}
            </div>`;

        // License info (show separately for motorcycle/car and hide if none)
        const renderLicenseBlock = (title, icon, licenses, points) => {
            if (!licenses || licenses.length === 0) return '';
            const pointColor = getPointColor(points);
            const pointPct = Math.max(0, Math.round((points / 12) * 100));
            const licenseListHtml = licenses.map((l, idx) => `
                <div style="padding:10px 12px;border-radius:10px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);margin-bottom:10px;">
                    <div style="font-weight:700;color:var(--text-primary);margin-bottom:6px;">GPLX ${idx + 1}: Hạng ${escapeHtml(l.class || '—')}</div>
                    ${row('Loại xe', escapeHtml(l.vehicleType || '—'))}
                    ${row('Số GPLX', escapeHtml(l.licenseNumber || '—'))}
                    ${row('Ngày cấp', escapeHtml(l.issueDate || '—'))}
                    ${row('Ngày hết hạn', escapeHtml(l.expiryDate || '—'))}
                </div>
            `).join('');

            return `
                <div style="padding:12px;border-radius:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);margin-bottom:12px;">
                    <div style="font-weight:800;color:var(--text-primary);margin-bottom:8px;">${icon} ${title}</div>
                    ${licenseListHtml}
                    <div class="user-detail-info-row" style="margin-top:8px;">
                        <span class="user-detail-info-label">Điểm GPLX</span>
                        <span class="user-detail-info-value" style="color:${pointColor};font-weight:800;">${points}/12 điểm</span>
                    </div>
                    <div class="points-bar-wrap">
                        <div class="points-bar-track">
                            <div class="points-bar-fill" style="width:${pointPct}%;background:${pointColor};"></div>
                        </div>
                    </div>
                    ${points === 0 ? `<div style="margin-top:8px;font-size:0.78rem;color:#ef4444;font-weight:700;">🚫 GPLX đang bị vô hiệu hóa do 0 điểm</div>` : ''}
                </div>
            `;
        };

        const licenseBlocksHtml = [
            renderLicenseBlock('Giấy phép lái xe máy', '🏍️', licensePointState.moto.licenses, licensePointState.moto.points),
            renderLicenseBlock('Giấy phép lái xe ô tô', '🚗', licensePointState.car.licenses, licensePointState.car.points),
        ].join('');

        const hasAnyLicense = licensePointState.moto.exists || licensePointState.car.exists;
        const licenseHtml = hasAnyLicense ? `
            <div class="user-detail-section" style="position:relative;">
                <div class="user-detail-section-title">🪪 Giấy phép lái xe</div>
                ${licenseBlocksHtml}
                ${row('Nơi cấp', u.licenseIssuedBy || '—')}
                ${(licensePointState.moto.points < 12 || licensePointState.car.points < 12)
                    ? `<div style="margin-top:10px;">
                        <button onclick="restoreUserPoints('${uid}')" style="padding:7px 14px;border-radius:8px;border:none;background:linear-gradient(135deg,#10b981,#059669);color:#fff;font-weight:700;cursor:pointer;font-size:0.78rem;font-family:inherit;">🔄 Admin phục hồi điểm</button>
                       </div>`
                    : ''}
            </div>
        ` : '';

        // Vehicles
        const vehiclesHtml = `
            <div class="user-detail-section full-width">
                <div class="user-detail-section-title">🚗 Phương tiện sở hữu (${uVehicles.length})</div>
                ${uVehicles.length === 0 ? '<p style="color:var(--text-muted);font-size:0.88rem;">Chưa có phương tiện nào</p>' : `
                    <div class="vehicle-cards-grid">
                        ${uVehicles.map(v => {
                            const isMoto = (v.vehicleType || v.type || '').includes('máy');
                            return `
                                <div class="vehicle-card-item">
                                    <div class="vehicle-card-icon ${isMoto ? 'moto' : 'car'}">${isMoto ? '🏍️' : '🚗'}</div>
                                    <div style="flex:1;">
                                        <div class="vehicle-card-plate">${v.licensePlate || '—'}</div>
                                        <div class="vehicle-card-info">${v.vehicleType || v.type || ''} • ${v.brand || ''} ${v.model || ''} • ${v.color || '—'}</div>
                                    </div>
                                    <div style="text-align:right;">
                                        <div style="font-size:0.75rem;color:var(--text-muted);">Chủ: ${v.ownerName || u.fullName || '—'}</div>
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                `}
            </div>`;

        // Violations
        const violationsHtml = `
            <div class="user-detail-section full-width">
                <div class="user-detail-section-title">🚨 Lịch sử vi phạm (${uViolations.length})</div>
                ${uViolations.length === 0 ? '<p style="color:var(--text-muted);font-size:0.88rem;">Không có vi phạm nào</p>' : `
                    <div class="violation-history-list">
                        ${uViolations.map(v => {
                            const statusClass = v.status === 'paid' ? 'paid' : 'pending';
                            const fineClass = v.status === 'paid' ? 'paid' : 'unpaid';
                            const dateStr = v.timestamp ? new Date(v.status === 'pending' ? v.timestamp : (v.createdAt ? v.createdAt * 1000 : Date.now())).toLocaleDateString('vi-VN') : '—';
                            return `
                                <div class="violation-history-item">
                                    <div class="violation-history-status ${statusClass}"></div>
                                    <div class="violation-history-name">${v.violationType || v.type || '—'}<br><span style="font-size:0.72rem;color:var(--text-muted);">${v.location || '—'}</span></div>
                                    <div class="violation-history-date">${dateStr}</div>
                                    <div class="violation-history-fine ${fineClass}">${formatter.format(v.fineAmount || 0)}</div>
                                    <span class="data-tag ${v.status === 'paid' ? 'tag-success' : 'tag-danger'}" style="margin:0;">${v.status === 'paid' ? 'Đã nộp' : 'Chưa nộp'}</span>
                                </div>
                            `;
                        }).join('')}
                    </div>
                `}
            </div>`;

        // Complaints
        const complaintsHtml = `
            <div class="user-detail-section full-width">
                <div class="user-detail-section-title">📝 Khiếu nại (${uComplaints.length})</div>
                ${uComplaints.length === 0 ? '<p style="color:var(--text-muted);font-size:0.88rem;">Chưa có khiếu nại nào</p>' : `
                    <div class="complaint-board-list">
                        ${uComplaints.map(c => {
                            const statusMeta = getComplaintStatusMeta(c.status);
                            const evidenceUrl = getComplaintEvidenceUrl(c);
                            const relatedViolation = (uViolations || []).find(v => String(v.id || '').toUpperCase() === String(c.violationId || '').toUpperCase()) || null;
                            const actionBusy = _complaintReviewLoading.has(c.id) || _complaintDeleteLoading.has(c.id);

                            const evidenceHtml = evidenceUrl
                                ? `
                                    <div class="complaint-card-evidence">
                                        <div class="image-loading-mask">${escapeHtml(tr('complaints_image_loading', 'Đang tải ảnh...'))}</div>
                                        <img
                                            src="${escapeHtml(evidenceUrl)}"
                                            alt="Complaint evidence"
                                            loading="lazy"
                                            onload="if(this.previousElementSibling){this.previousElementSibling.style.display='none';}"
                                            onerror="this.style.display='none'; if(this.previousElementSibling){this.previousElementSibling.style.display='flex'; this.previousElementSibling.style.animation='none'; this.previousElementSibling.textContent='${escapeHtml(tr('complaints_image_error', 'Không tải được ảnh'))}';}"
                                        />
                                    </div>
                                `
                                : `<div class="complaint-card-evidence"><div class="complaint-card-evidence-fallback">${escapeHtml(tr('complaints_no_evidence', 'Không có ảnh bằng chứng'))}</div></div>`;

                            return `
                                <article class="complaint-card ${statusMeta.cardClass}">
                                    <div class="complaint-card-main">
                                        <div class="complaint-card-top">
                                            <div>
                                                <div class="complaint-card-title">${escapeHtml(c.reason || tr('complaints_fallback_reason', 'Khiếu nại'))}</div>
                                                <div class="complaint-card-subtitle">${formatDateTime(c.createdAt || c.updatedAt || c.reviewedAt)}</div>
                                            </div>
                                            <span class="data-tag ${statusMeta.tagClass}" style="white-space:nowrap;">${statusMeta.label}</span>
                                        </div>
                                        <div class="complaint-card-desc">${escapeHtml(c.description || tr('complaints_fallback_desc', 'Không có mô tả'))}</div>
                                        <div class="complaint-card-meta">
                                            <div><span class="meta-label">${escapeHtml(tr('complaints_meta_id', 'Mã khiếu nại'))}:</span> <span class="meta-value">${escapeHtml(c.id || '—')}</span></div>
                                            <div><span class="meta-label">${escapeHtml(tr('complaints_meta_violation_id', 'Mã vi phạm'))}:</span> <span class="meta-value">${escapeHtml(c.violationId || '—')}</span></div>
                                            <div><span class="meta-label">${escapeHtml(tr('complaints_meta_type', 'Loại vi phạm'))}:</span> <span class="meta-value">${escapeHtml(relatedViolation?.violationType || relatedViolation?.type || '—')}</span></div>
                                            <div><span class="meta-label">${escapeHtml(tr('complaints_meta_fine', 'Mức phạt'))}:</span> <span class="meta-value">${escapeHtml(formatVnd(relatedViolation?.fineAmount || 0))}</span></div>
                                        </div>
                                        ${c.adminNote ? `<div style="font-size:0.78rem;color:#fca5a5;">${escapeHtml(tr('complaints_admin_note', '💬 Ghi chú admin'))}: ${escapeHtml(c.adminNote)}</div>` : ''}
                                    </div>
                                    <div class="complaint-card-side">
                                        ${evidenceHtml}
                                        <div class="complaint-card-actions">
                                            <button class="complaint-card-btn detail" onclick="showComplaintDetailModal('${c.id}')">${escapeHtml(tr('complaints_btn_detail', '👁️ Chi tiết'))}</button>
                                            ${c.status === 'pending' ? `
                                                <button class="complaint-card-btn approve ${actionBusy ? 'btn-loading' : ''}" ${actionBusy ? 'disabled' : ''} onclick="reviewComplaint('${c.id}', 'approve', '', this)">
                                                    ${actionBusy ? `<span class="spin"></span><span>${escapeHtml(tr('complaints_btn_processing', 'Đang xử lý...'))}</span>` : escapeHtml(tr('complaints_btn_approve', '✅ Chấp nhận'))}
                                                </button>
                                                <button class="complaint-card-btn reject ${actionBusy ? 'btn-loading' : ''}" ${actionBusy ? 'disabled' : ''} onclick="showRejectComplaintModal('${c.id}')">
                                                    ${actionBusy ? `<span class="spin"></span><span>${escapeHtml(tr('complaints_btn_processing', 'Đang xử lý...'))}</span>` : escapeHtml(tr('complaints_btn_reject', '❌ Từ chối'))}
                                                </button>
                                            ` : `
                                                <button class="complaint-card-btn remove ${actionBusy ? 'btn-loading' : ''}" ${actionBusy ? 'disabled' : ''} onclick="confirmDeleteComplaint('${c.id}', this)">
                                                    ${actionBusy ? `<span class="spin"></span><span>${escapeHtml(tr('complaints_btn_processing', 'Đang xử lý...'))}</span>` : escapeHtml(tr('complaints_btn_delete', '🗑️ Xóa'))}
                                                </button>
                                            `}
                                        </div>
                                    </div>
                                </article>
                            `;
                        }).join('')}
                    </div>
                `}
            </div>`;

        gridEl.innerHTML = personalHtml + licenseHtml + vehiclesHtml + violationsHtml + complaintsHtml;
    }

    // Switch to user detail tab
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    const detailPanel = document.getElementById('tab-user-detail');
    if (detailPanel) {
        detailPanel.classList.add('active');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

function closeUserDetailPage() {
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    const manageLink = document.querySelector('.nav-link[data-tab="manage"]');
    if (manageLink) manageLink.classList.add('active');
    const managePanel = document.getElementById('tab-manage');
    if (managePanel) managePanel.classList.add('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Helper for info rows
function row(label, value) {
    return `<div class="user-detail-info-row">
        <span class="user-detail-info-label">${label}</span>
        <span class="user-detail-info-value">${value}</span>
    </div>`;
}

/* ── Image Lightbox ────────────────────────────────────── */
function openImageLightbox(src) {
    if (!src) return;
    // Remove existing lightbox if any
    const existing = document.getElementById('image-lightbox-overlay');
    if (existing) existing.remove();

    const overlay = document.createElement('div');
    overlay.id = 'image-lightbox-overlay';
    overlay.className = 'image-lightbox-overlay';
    overlay.innerHTML = `<img src="${src.replace(/"/g, '&quot;')}" alt="Zoomed evidence" />`;
    overlay.addEventListener('click', () => overlay.remove());
    document.addEventListener('keydown', function _esc(e) {
        if (e.key === 'Escape') {
            overlay.remove();
            document.removeEventListener('keydown', _esc);
        }
    });
    document.body.appendChild(overlay);
}

function showComplaintDetailModal(complaintId) {
    if (!_adminCachedData) return;
    const complaint = (_adminCachedData.complaints || []).find(c => c.id === complaintId);
    if (!complaint) return;

    const relatedViolation = (_adminCachedData.violations || []).find(v => String(v.id || '').toUpperCase() === String(complaint.violationId || '').toUpperCase());
    const evidenceUrl = getComplaintEvidenceUrl(complaint);
    const statusMeta = getComplaintStatusMeta(complaint.status);
    const actionBusy = _complaintReviewLoading.has(complaint.id) || _complaintDeleteLoading.has(complaint.id);
    const existing = document.getElementById('complaintDetailModal');
    if (existing) existing.remove();

    const modal = document.createElement('div');
    modal.id = 'complaintDetailModal';
    modal.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.78);display:flex;align-items:center;justify-content:center;z-index:10006;padding:18px;backdrop-filter:blur(4px);';
    modal.innerHTML = `
        <div style="background:var(--bg-secondary);border:1px solid rgba(96,165,250,0.35);border-radius:18px;max-width:560px;width:100%;padding:20px;max-height:85vh;overflow:auto;box-shadow:0 24px 64px rgba(0,0,0,0.7);">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:14px;">
                <div>
                    <div style="font-size:1.05rem;font-weight:800;color:#fff;">📝 Chi tiết khiếu nại</div>
                    <div style="font-size:0.8rem;color:var(--text-muted);margin-top:2px;">${statusMeta.label} • ${formatDateTime(complaint.createdAt || complaint.updatedAt || complaint.reviewedAt)}</div>
                </div>
                <button onclick="document.getElementById('complaintDetailModal').remove()" style="padding:6px 10px;border-radius:8px;border:1px solid rgba(255,255,255,0.18);background:rgba(255,255,255,0.06);color:#fff;font-weight:700;cursor:pointer;">✕</button>
            </div>
            <div style="font-size:0.85rem;color:var(--text-secondary);line-height:1.6;">
                <div><strong style="color:#fff;">Mã khiếu nại:</strong> ${escapeHtml(complaint.id || '—')}</div>
                <div><strong style="color:#fff;">Mã vi phạm:</strong> ${escapeHtml(complaint.violationId || '—')}</div>
                <div><strong style="color:#fff;">Người gửi:</strong> ${escapeHtml(((_adminCachedData.users || []).find(u => u.id === complaint.userId)?.fullName) || complaint.userId || '—')}</div>
                <div><strong style="color:#fff;">Lý do:</strong> ${escapeHtml(complaint.reason || '—')}</div>
                <div style="margin-top:8px;"><strong style="color:#fff;">Mô tả chi tiết:</strong></div>
                <div style="margin-top:4px;padding:10px;border-radius:10px;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);white-space:pre-wrap;">${escapeHtml(complaint.description || '—')}</div>
                ${complaint.adminNote ? `<div style="margin-top:10px;"><strong style="color:#fff;">Ghi chú admin:</strong> <span style="color:#fca5a5;">${escapeHtml(complaint.adminNote)}</span></div>` : ''}
                ${relatedViolation ? `
                    <div style="margin-top:12px;padding:10px;border-radius:10px;background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);">
                        <div><strong style="color:#fff;">Lỗi vi phạm:</strong> ${escapeHtml(relatedViolation.violationType || relatedViolation.type || '—')}</div>
                        <div><strong style="color:#fff;">Mức phạt:</strong> ${formatVnd(relatedViolation.fineAmount || 0)}</div>
                        <div><strong style="color:#fff;">Trừ điểm:</strong> ${escapeHtml(String(relatedViolation.deductedPoints ?? '—'))}</div>
                    </div>
                ` : ''}
                ${evidenceUrl ? `
                    <div style="margin-top:12px;">
                        <div style="margin-bottom:6px;"><strong style="color:#fff;">Ảnh bằng chứng:</strong></div>
                        <a href="${escapeHtml(evidenceUrl)}" target="_blank" style="display:inline-block;margin-bottom:8px;color:#93c5fd;">🔗 Mở ảnh gốc</a>
                        <div class="complaint-card-evidence" style="min-height:150px;">
                            <div class="image-loading-mask">Đang tải ảnh...</div>
                            <img
                                src="${escapeHtml(evidenceUrl)}"
                                alt="evidence"
                                style="width:100%;max-height:260px;height:auto;object-fit:cover;border-radius:10px;border:1px solid rgba(255,255,255,0.12);"
                                onload="if(this.previousElementSibling){this.previousElementSibling.style.display='none';}"
                                onerror="this.style.display='none'; if(this.previousElementSibling){this.previousElementSibling.style.display='flex'; this.previousElementSibling.style.animation='none'; this.previousElementSibling.textContent='Không tải được ảnh';}"
                            />
                        </div>
                    </div>
                ` : '<div style="margin-top:10px;color:var(--text-muted);">Không có ảnh bằng chứng.</div>'}
            </div>
            ${complaint.status === 'pending' ? `
                <div style="display:flex;gap:8px;margin-top:14px;">
                    <button ${actionBusy ? 'disabled' : ''} class="${actionBusy ? 'btn-loading' : ''}" onclick="reviewComplaint('${complaint.id}', 'approve', '', this)" style="flex:1;padding:9px 12px;border-radius:9px;border:none;background:linear-gradient(135deg,#10b981,#059669);color:#fff;font-weight:700;cursor:pointer;font-family:inherit;display:flex;align-items:center;justify-content:center;gap:6px;">
                        ${actionBusy ? '<span class="spin"></span><span>Đang xử lý...</span>' : '✅ Chấp nhận'}
                    </button>
                    <button ${actionBusy ? 'disabled' : ''} class="${actionBusy ? 'btn-loading' : ''}" onclick="document.getElementById('complaintDetailModal').remove(); showRejectComplaintModal('${complaint.id}');" style="flex:1;padding:9px 12px;border-radius:9px;border:none;background:linear-gradient(135deg,#ef4444,#dc2626);color:#fff;font-weight:700;cursor:pointer;font-family:inherit;display:flex;align-items:center;justify-content:center;gap:6px;">
                        ${actionBusy ? '<span class="spin"></span><span>Đang xử lý...</span>' : '❌ Từ chối'}
                    </button>
                </div>
            ` : `
                <div style="display:flex;gap:8px;margin-top:14px;">
                    <button ${actionBusy ? 'disabled' : ''} class="${actionBusy ? 'btn-loading' : ''}" onclick="confirmDeleteComplaint('${complaint.id}', this)" style="flex:1;padding:9px 12px;border-radius:9px;border:1px solid rgba(239,68,68,0.35);background:rgba(239,68,68,0.15);color:#fecaca;font-weight:700;cursor:pointer;font-family:inherit;display:flex;align-items:center;justify-content:center;gap:6px;">
                        ${actionBusy ? `<span class="spin"></span><span>${escapeHtml(tr('complaints_btn_processing', 'Đang xử lý...'))}</span>` : escapeHtml(tr('complaints_btn_delete', '🗑️ Xóa'))}
                    </button>
                </div>
            `}
        </div>
    `;
    document.body.appendChild(modal);
    modal.addEventListener('click', (e) => { if (e.target === modal) modal.remove(); });
}

// ── Review pending update request ───────────────────────────────────
function reviewUserUpdate(uid) {
    if (!_adminCachedData || !_adminCachedData.profile_updates) return;
    const req = _adminCachedData.profile_updates.find(x => x.id === uid);
    if (!req) return;

    const users = _adminCachedData.users || [];
    const u = users.find(x => x.id === uid);

    const fieldLabels = {
        fullName: 'Họ và tên',
        phone: 'Số điện thoại',
        idCard: 'CCCD/CMND',
        idCardExpiryDate: 'Ngày hết hạn CCCD',
        address: 'Địa chỉ',
        email: 'Email',
        dateOfBirth: 'Ngày sinh',
        gender: 'Giới tính',
        idCardIssueDate: 'Ngày cấp CCCD',
        occupation: 'Nghề nghiệp',
        driverLicenses: 'Giấy phép lái xe',
        licenseIssuedBy: 'Nơi cấp GPLX',
    };

    const ignoreKeys = [
        'userId',
        'status',
        'createdAt',
        'updatedAt',
        'reviewedAt',
        'requestSection',
        'requestType',
        'requestSource',
        'requestedAt',
        'id',
    ];

    const formatValue = (value) => {
        if (value === null || value === undefined || value === '') return '—';
        if (typeof value === 'object') {
            return `<pre style="white-space:pre-wrap;margin:0;font-size:0.75rem;line-height:1.4;">${escapeHtml(JSON.stringify(value, null, 2))}</pre>`;
        }
        return escapeHtml(String(value));
    };

    const formatLicenses = (licenses) => {
        if (!Array.isArray(licenses) || licenses.length === 0) return '—';
        return licenses.map((l, i) => {
            return [
                `GPLX ${i + 1}: Hạng ${escapeHtml(l?.class || '—')}`,
                `Số: ${escapeHtml(l?.licenseNumber || '—')}`,
                `Loại xe: ${escapeHtml(l?.vehicleType || '—')}`,
                `Ngày cấp: ${escapeHtml(l?.issueDate || '—')}`,
                `Hạn: ${escapeHtml(l?.expiryDate || '—')}`,
            ].join('<br>');
        }).join('<hr style="border-color:rgba(255,255,255,0.12);margin:8px 0;">');
    };

    let changesHtml = '';
    for (const [key, value] of Object.entries(req)) {
        if (ignoreKeys.includes(key)) continue;
        const label = fieldLabels[key] || key;
        let currentVal = u ? (u[key] || '—') : '—';
        let nextVal = value;

        if (key === 'driverLicenses') {
            currentVal = formatLicenses(normalizeDriverLicenses(u || {}));
            nextVal = formatLicenses(Array.isArray(value) ? value : []);
        } else {
            currentVal = formatValue(currentVal);
            nextVal = formatValue(value);
        }

        changesHtml += `
            <div class="update-change-row">
                <div class="update-change-label">${label}</div>
                <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:8px;align-items:center;font-size:0.82rem;margin-top:4px;">
                    <div class="update-change-old" style="padding:8px 10px;background:rgba(255,255,255,0.04);border-radius:6px;color:var(--text-muted);text-decoration:line-through;">${currentVal}</div>
                    <span style="color:var(--accent-primary);font-size:1.1rem;">→</span>
                    <div class="update-change-value">${nextVal}</div>
                </div>
            </div>
        `;
    }

    const existing = document.getElementById('reviewUpdateModal');
    if (existing) existing.remove();

    const overlay = document.createElement('div');
    overlay.id = 'reviewUpdateModal';
    overlay.className = 'review-update-modal-overlay';
    overlay.innerHTML = `
        <div class="review-update-modal-box">
            <div class="review-update-modal-title">🔔 Yêu cầu thay đổi thông tin</div>
            <div class="review-update-modal-subtitle">Người dùng <strong>${u ? u.fullName || u.email : uid}</strong> đã yêu cầu thay đổi các thông tin sau:</div>
            ${changesHtml}
            <div style="display:flex;gap:10px;margin-top:24px;justify-content:flex-end;">
                <button onclick="document.getElementById('reviewUpdateModal').remove()" style="padding:10px 20px;border-radius:10px;border:1px solid var(--border-color);background:rgba(255,255,255,0.06);color:var(--text-primary);font-weight:600;cursor:pointer;font-size:0.88rem;font-family:inherit;">Hủy</button>
                <button onclick="submitReviewUpdate('${uid}','reject')" style="padding:10px 20px;border-radius:10px;border:none;background:rgba(239,68,68,0.2);color:#ef4444;font-weight:700;cursor:pointer;font-size:0.88rem;font-family:inherit;border:1px solid rgba(239,68,68,0.3);">❌ Từ chối</button>
                <button onclick="submitReviewUpdate('${uid}','approve')" style="padding:10px 20px;border-radius:10px;border:none;background:linear-gradient(135deg,#10b981,#059669);color:#fff;font-weight:700;cursor:pointer;font-size:0.88rem;font-family:inherit;">✅ Chấp nhận</button>
            </div>
        </div>
    `;
    overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
    document.body.appendChild(overlay);
}

async function submitReviewUpdate(uid, action) {
    const modal = document.getElementById('reviewUpdateModal');
    const btns = modal ? modal.querySelectorAll('button') : [];
    btns.forEach(b => { b.disabled = true; });
    const finishLoading = showGlobalPageLoader(
        action === 'approve'
            ? (currentLang === 'en' ? 'Reviewing profile updates...' : 'Đang duyệt thay đổi thông tin người dùng...')
            : (currentLang === 'en' ? 'Rejecting profile update request...' : 'Đang từ chối yêu cầu thay đổi...'),
    );

    try {
        const res = await fetch(`/api/admin/users/${uid}/approve_update`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ action })
        });
        const data = await res.json();
        if (res.ok && data.status === 'ok') {
            showToast(
                action === 'approve'
                    ? (currentLang === 'en' ? '✅ Profile update approved and applied' : '✅ Đã chấp nhận và cập nhật thông tin!')
                    : (currentLang === 'en' ? '❌ Profile update request rejected' : '❌ Đã từ chối yêu cầu thay đổi'),
                action === 'approve' ? 'success' : 'warning',
            );
            if (modal) modal.remove();
            await loadAdminData({ silent: true });
            if (document.getElementById('tab-user-detail')?.classList.contains('active')) {
                showUserDetails(uid); // Refresh detail page in real-time
            }
        } else {
            showToast(data.message || (currentLang === 'en' ? 'Request processing failed' : 'Lỗi khi xử lý'), 'error');
            btns.forEach(b => { b.disabled = false; });
        }
    } catch(e) {
        showToast(currentLang === 'en' ? 'Network error' : 'Lỗi mạng', 'error');
        btns.forEach(b => { b.disabled = false; });
    } finally {
        finishLoading();
    }
}

// ── Restore Points ───────────────────────────────────
async function restoreUserPoints(uid) {
    const confirmText = currentLang === 'en'
        ? 'Are you sure you want to restore motorcycle and car license points to 12/12 for this user?'
        : 'Bạn có chắc chắn muốn phục hồi 12/12 điểm GPLX xe máy và ô tô cho người dùng này?';
    if (!confirm(confirmText)) return;
    const finishLoading = showGlobalPageLoader(
        currentLang === 'en' ? 'Restoring driver license points...' : 'Đang phục hồi điểm GPLX cho người dùng...',
    );
    try {
        const res = await fetch(`/api/admin/users/${uid}/restore_points`, {
            method: 'POST'
        });
        const data = await res.json();
        if (res.ok && data.status === 'ok') {
            showToast(data.message || (currentLang === 'en' ? '✅ Points restored successfully' : '✅ Đã phục hồi điểm'), 'success');
            await loadAdminData({ silent: true });
            if (document.getElementById('tab-user-detail')?.classList.contains('active')) {
                showUserDetails(uid);
            }
        } else {
            showToast(data.message || (currentLang === 'en' ? 'Failed to restore points' : 'Lỗi khi phục hồi điểm'), 'error');
        }
    } catch(e) {
        showToast(currentLang === 'en' ? 'Network error' : 'Lỗi mạng', 'error');
    } finally {
        finishLoading();
    }
}

// ── Review Complaint (Approve / Reject) ────────────────────────────
async function reviewComplaint(complaintId, action, adminNote, triggerButton = null) {
    const normalizedId = String(complaintId || '').trim();
    if (!normalizedId) return;
    if (_complaintReviewLoading.has(normalizedId)) return;

    _complaintReviewLoading.add(normalizedId);
    setActionButtonLoading(
        triggerButton,
        true,
        action === 'approve'
            ? (currentLang === 'en' ? 'Approving...' : 'Đang chấp nhận...')
            : (currentLang === 'en' ? 'Rejecting...' : 'Đang từ chối...'),
    );
    const finishLoading = showGlobalPageLoader(
        action === 'approve'
            ? (currentLang === 'en' ? 'Reviewing complaint and updating system...' : 'Đang duyệt khiếu nại và cập nhật hệ thống...')
            : (currentLang === 'en' ? 'Rejecting complaint...' : 'Đang từ chối khiếu nại...'),
    );

    try {
        const res = await fetch(`/api/admin/complaints/${normalizedId}/review`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action, adminNote: adminNote || '' })
        });
        const data = await res.json();

        if (res.ok && data.status === 'ok') {
            // Optimistic local update for instant UI feedback
            if (_adminCachedData) {
                const complaints = _adminCachedData.complaints || [];
                const target = complaints.find(c => String(c.id || '') === normalizedId);
                if (target) {
                    target.status = action === 'approve' ? 'approved' : 'rejected';
                    target.adminNote = adminNote || '';
                    target.reviewedAt = new Date().toISOString();

                    if (action === 'approve') {
                        const violationId = String(target.violationId || '').toUpperCase();
                        if (violationId) {
                            _adminCachedData.violations = (_adminCachedData.violations || []).filter(
                                v => String(v.id || '').toUpperCase() !== violationId,
                            );
                        }
                    }
                }

                updateManageStats(_adminCachedData);
                renderAdminTable(_adminCachedData, manageSearchInput ? manageSearchInput.value : '');
                renderComplaintBoard(_adminCachedData);
            }

            const rejectModal = document.getElementById('rejectComplaintModal');
            if (rejectModal) rejectModal.remove();
            const detailModal = document.getElementById('complaintDetailModal');
            if (detailModal) detailModal.remove();

            // Sync latest from backend immediately after optimistic update
            await loadAdminData({ silent: true });
            const detailPanel = document.getElementById('tab-user-detail');
            if (detailPanel?.classList.contains('active') && detailPanel.dataset.uid) {
                showUserDetails(detailPanel.dataset.uid);
            }

            showToast(
                action === 'approve'
                    ? (currentLang === 'en' ? '✅ Complaint approved!' : '✅ Đã chấp nhận khiếu nại!')
                    : (currentLang === 'en' ? '❌ Complaint rejected' : '❌ Đã từ chối khiếu nại'),
                action === 'approve' ? 'success' : 'warning',
            );
        } else {
            showToast(data.message || (currentLang === 'en' ? 'Error while processing complaint' : 'Lỗi khi xử lý khiếu nại'), 'error');
        }
    } catch (e) {
        showToast(currentLang === 'en' ? 'Network error' : 'Lỗi mạng', 'error');
    } finally {
        _complaintReviewLoading.delete(normalizedId);
        setActionButtonLoading(triggerButton, false);
        finishLoading();
        if (_adminCachedData) {
            renderComplaintBoard(_adminCachedData);
        }
    }
}

function showRejectComplaintModal(complaintId) {
    if (_complaintReviewLoading.has(String(complaintId || ''))) return;
    const existing = document.getElementById('rejectComplaintModal');
    if (existing) existing.remove();
    const overlay = document.createElement('div');
    overlay.id = 'rejectComplaintModal';
    overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.75);display:flex;align-items:center;justify-content:center;z-index:10005;padding:20px;backdrop-filter:blur(4px);';
    overlay.innerHTML = `
        <div style="background:var(--bg-secondary);border:1px solid rgba(239,68,68,0.3);border-radius:20px;max-width:440px;width:100%;padding:28px;box-shadow:0 24px 64px rgba(0,0,0,0.7);">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:18px;">
                <div style="width:44px;height:44px;border-radius:50%;background:rgba(239,68,68,0.15);display:flex;align-items:center;justify-content:center;font-size:1.3rem;">❌</div>
                <div>
                    <div style="font-size:1.05rem;font-weight:700;color:#fff;">${tr('reject_modal_title', 'Từ chối khiếu nại')}</div>
                    <div style="font-size:0.8rem;color:var(--text-muted);">${tr('reject_modal_subtitle', 'Vui lòng nhập lý do từ chối')}</div>
                </div>
            </div>
            <textarea id="rejectNoteInput" placeholder="${tr('reject_modal_placeholder', 'Nhập lý do từ chối...')}" rows="4" style="width:100%;padding:12px;border-radius:10px;border:1px solid rgba(255,255,255,0.12);background:rgba(255,255,255,0.06);color:#fff;font-family:inherit;font-size:0.9rem;resize:vertical;box-sizing:border-box;margin-bottom:16px;"></textarea>
            <div style="display:flex;gap:10px;justify-content:flex-end;">
                <button onclick="document.getElementById('rejectComplaintModal').remove()" style="padding:10px 18px;border-radius:10px;border:1px solid var(--border-color);background:rgba(255,255,255,0.06);color:var(--text-primary);font-weight:600;cursor:pointer;font-family:inherit;">${tr('reject_modal_cancel', 'Hủy')}</button>
                <button onclick="reviewComplaint('${complaintId}', 'reject', document.getElementById('rejectNoteInput').value, this)" style="padding:10px 18px;border-radius:10px;border:none;background:linear-gradient(135deg,#ef4444,#dc2626);color:#fff;font-weight:700;cursor:pointer;font-family:inherit;">❌ ${tr('reject_modal_confirm', 'Xác nhận từ chối')}</button>
            </div>
        </div>
    `;
    document.body.appendChild(overlay);
    overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
}

// =====================================================================
// FIRESTORE QUOTA — REALTIME VIA WS + INITIAL FETCH FALLBACK
// =====================================================================
const _QUOTA_LIMITS_DEFAULT = { reads: 50000, writes: 20000, deletes: 20000 };

function _formatUptime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
}

function _applyQuotaData(d) {
    const qr = document.getElementById('quotaReads');
    const qw = document.getElementById('quotaWrites');
    const qd = document.getElementById('quotaDeletes');
    const qu = document.getElementById('quotaUptime');
    if (qr) qr.textContent = Number(d.reads || 0).toLocaleString();
    if (qw) qw.textContent = Number(d.writes || 0).toLocaleString();
    if (qd) qd.textContent = Number(d.deletes || 0).toLocaleString();
    if (qu) qu.textContent = _formatUptime(d.uptime_seconds || 0);

    // Date display
    const qDate = document.getElementById('quotaDate');
    if (qDate && d.date) qDate.textContent = d.date;

    // Progress bars
    const limits = d.limits || _QUOTA_LIMITS_DEFAULT;
    const br = document.getElementById('quotaBarRead');
    const bw = document.getElementById('quotaBarWrite');
    const bd = document.getElementById('quotaBarDelete');
    if (br) br.style.width = Math.min((d.reads / limits.reads) * 100, 100).toFixed(2) + '%';
    if (bw) bw.style.width = Math.min((d.writes / limits.writes) * 100, 100).toFixed(2) + '%';
    if (bd) bd.style.width = Math.min((d.deletes / limits.deletes) * 100, 100).toFixed(2) + '%';

    // Visual warning
    const grid = document.getElementById('quotaGrid');
    if (grid) {
        grid.classList.toggle('quota-warn', d.quota_status === 'warn');
        grid.classList.toggle('quota-critical', d.quota_status === 'critical');
    }
}

// Initial fetch on page load — sau đó WS sẽ push realtime
(async function _initQuota() {
    try {
        const resp = await fetch('/api/firestore-quota');
        const d = await resp.json();
        _applyQuotaData(d);
    } catch (_) {}
})();



