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

// =====================================================================
// SCROLL ANIMATIONS (Intersection Observer)
// =====================================================================
function triggerAnimations(container) {
    container.querySelectorAll('.animate-in').forEach((el, i) => {
        el.style.animationDelay = (i * 0.08) + 's';
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
const toast = document.getElementById('toast');
const confRange = document.getElementById('confRange');
const debugToggle = document.getElementById('debugToggle');
const confBubble = document.getElementById('confBubble');
const rangeFill = document.getElementById('rangeFill');

let ws = null;
let selectedDetectors = ['helmet'];
let isRunning = false;

// =====================================================================
// UTILITIES
// =====================================================================
function showToast(message, type = 'success') {
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    setTimeout(() => toast.classList.remove('show'), 3000);
}
function formatSize(bytes) { return (bytes / 1024 / 1024).toFixed(1) + ' MB'; }

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
    try {
        const resp = await fetch('/api/snapshots');
        let snapshots = await resp.json();
        if (filter !== 'all') snapshots = snapshots.filter(s => s.type === filter);
        if (snapshots.length === 0) { snapshotsGallery.innerHTML = '<p class="no-items">No violations detected yet.</p>'; return; }
        snapshotsGallery.innerHTML = snapshots.slice(0, 12).map(s => `
            <div class="gallery-item" onclick="window.open('${s.path}', '_blank')">
                <img src="${s.path}" alt="${s.type}" loading="lazy">
                <div class="gallery-item-info">
                    <span class="gallery-item-type ${s.type}">${s.type}</span>
                    <span class="gallery-item-name">${s.filename}</span>
                </div>
            </div>
        `).join('');
    } catch (e) { console.error('Failed to load snapshots:', e); }
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
    const video = videoSelect.value;
    const model = modelSelect.value;
    const conf = parseInt(confRange.value) / 100;
    const debug = debugToggle.checked;
    if (!video || !model) { showToast('Please select video and model', 'error'); return; }
    if (selectedDetectors.length === 0) { showToast('Please select at least one detector', 'error'); return; }

    ws = new WebSocket(`ws://${window.location.host}/ws/detect`);

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
                break;
        }
    };
    ws.onerror = () => { statusIndicator.className = 'status-indicator error'; statusText.textContent = 'Connection error'; showToast('Connection error', 'error'); };
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
// IMAGE DETECTION
// =====================================================================
const imageInput = document.getElementById('imageInput');
const imageUploadArea = document.getElementById('imageUploadArea');
const imageUploadContent = document.getElementById('imageUploadContent');
const imagePreview = document.getElementById('imagePreview');
const imageDetectBtn = document.getElementById('imageDetectBtn');
const imageConf = document.getElementById('imageConf');
const imageConfLabel = document.getElementById('imageConfLabel');
let imageFile = null;

imageConf.addEventListener('input', () => { imageConfLabel.textContent = imageConf.value + '%'; });

imageUploadArea.addEventListener('click', () => imageInput.click());
imageUploadArea.addEventListener('dragover', e => { e.preventDefault(); imageUploadArea.classList.add('drag-over'); });
imageUploadArea.addEventListener('dragleave', () => imageUploadArea.classList.remove('drag-over'));
imageUploadArea.addEventListener('drop', e => {
    e.preventDefault(); imageUploadArea.classList.remove('drag-over');
    if (e.dataTransfer.files.length) { imageFile = e.dataTransfer.files[0]; showImagePreview(); }
});
imageInput.addEventListener('change', () => { if (imageInput.files.length) { imageFile = imageInput.files[0]; showImagePreview(); } });

function showImagePreview() {
    const url = URL.createObjectURL(imageFile);
    imagePreview.src = url; imagePreview.style.display = 'block';
    imageUploadContent.style.display = 'none';
    imageDetectBtn.disabled = false;
}

imageDetectBtn.addEventListener('click', async () => {
    if (!imageFile) return;
    imageDetectBtn.disabled = true; imageDetectBtn.textContent = '⏳ Detecting...';
    const fd = new FormData();
    fd.append('file', imageFile);
    fd.append('conf', parseInt(imageConf.value) / 100);
    try {
        const resp = await fetch('/api/detect/image', { method: 'POST', body: fd });
        const data = await resp.json();
        if (data.error) { showToast('Error: ' + data.error, 'error'); return; }
        document.getElementById('imageResultCard').style.display = 'block';
        document.getElementById('imageResult').src = 'data:image/jpeg;base64,' + data.image;
        document.getElementById('imageStatsCard').style.display = 'block';
        document.getElementById('imageTotalCount').textContent = data.total + ' objects detected';
        const classList = document.getElementById('imageClassList');
        classList.innerHTML = Object.entries(data.class_summary).map(([name, info]) =>
            `<div class="class-item"><span class="class-name">${name}</span><span class="class-count">${info.count}</span><span class="class-conf">${(info.max_conf * 100).toFixed(1)}%</span></div>`
        ).join('');
        showToast(`Detected ${data.total} objects!`, 'success');
    } catch (e) { showToast('Detection failed: ' + e.message, 'error'); }
    finally { imageDetectBtn.disabled = false; imageDetectBtn.textContent = '🔍 Detect Objects'; }
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

    const fd = new FormData();
    fd.append('file', videoFile);
    fd.append('conf', parseInt(videoConfEl.value) / 100);
    try {
        const resp = await fetch('/api/detect/video', { method: 'POST', body: fd });
        const data = await resp.json();
        if (data.error) { showToast('Error: ' + data.error, 'error'); return; }
        pollVideoStatus(data.task_id);
    } catch (e) { showToast('Upload failed: ' + e.message, 'error'); videoDetectBtn.disabled = false; videoDetectBtn.textContent = '🎬 Process Video'; }
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
    finally { btn.disabled = false; btn.textContent = '🔍 Tra cứu'; }
});

// =====================================================================
// INIT
// =====================================================================
loadVideos();
loadModels();
loadSnapshots();
updateConfUI();
debugStatus.textContent = `Debug: ${debugToggle.checked ? 'ON' : 'OFF'}`;
