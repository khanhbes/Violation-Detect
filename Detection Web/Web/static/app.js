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
        const ip = data.ips && data.ips.length > 0 ? data.ips[0] : 'localhost';
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
        // Typed.js
        typed_strings: [
            'trí tuệ nhân tạo YOLOv12-Seg',
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
        // Typed.js
        typed_strings: [
            'powered by YOLOv12-Seg AI',
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
        if (t[key] !== undefined) {
            // Use innerHTML for keys that contain HTML (like upload-link spans)
            if (t[key].includes('<')) {
                el.innerHTML = t[key];
            } else {
                el.textContent = t[key];
            }
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
            navigator.serviceWorker.register('/static/firebase-messaging-sw.js')
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
const refreshManageBtn = document.getElementById('refreshManageBtn');
const adminDataTableBody = document.getElementById('adminDataTableBody');
const manageSearchInput = document.getElementById('manageSearchInput');

// Stat elements
const statTotalUsers = document.getElementById('statTotalUsers');
const statTotalVehicles = document.getElementById('statTotalVehicles');
const statTotalViolations = document.getElementById('statTotalViolations');
const statPendingFines = document.getElementById('statPendingFines');
const statTotalComplaints = document.getElementById('statTotalComplaints');
const manageRowCount = document.getElementById('manageRowCount');

// Cache data for search
let _adminCachedData = null;

function updateManageStats(data) {
    const users = data.users || [];
    const vehicles = data.vehicles || [];
    const violations = data.violations || [];
    const complaints = data.complaints || [];

    let totalPending = 0;
    violations.forEach(v => {
        if (v.status === 'pending') totalPending += (v.fineAmount || 0);
    });

    const formatter = new Intl.NumberFormat('vi-VN', { style: 'currency', currency: 'VND' });

    if (statTotalUsers) statTotalUsers.textContent = users.length;
    if (statTotalVehicles) statTotalVehicles.textContent = vehicles.length;
    if (statTotalViolations) statTotalViolations.textContent = violations.length;
    if (statPendingFines) statPendingFines.textContent = formatter.format(totalPending);
    if (statTotalComplaints) statTotalComplaints.textContent = complaints.length;
}

function renderAdminTable(data, searchQuery = '') {
    if (!adminDataTableBody) return;

    const users = data.users || [];
    const vehicles = data.vehicles || [];
    const violations = data.violations || [];
    const complaints = data.complaints || [];
    const notifications = data.notifications || [];

    const query = searchQuery.toLowerCase().trim();
    const formatter = new Intl.NumberFormat('vi-VN', { style: 'currency', currency: 'VND' });

    let html = '';
    let rowIndex = 0;

    users.forEach(u => {
        const uid = u.id;
        const uVehicles = vehicles.filter(v => v.ownerId === uid);
        const targetPlates = uVehicles.map(v => v.licensePlate);
        const uViolations = violations.filter(v => v.userId === uid || (v.licensePlate && targetPlates.includes(v.licensePlate)));
        const uComplaints = complaints.filter(c => c.userId === uid);
        const uNotifications = notifications.filter(n => n.userId === uid);

        // Search filter
        if (query) {
            const searchable = [
                u.email, u.fullName, u.idCard, u.phone, uid,
                ...targetPlates,
                ...uVehicles.map(v => `${v.brand || ''} ${v.model || ''}`),
            ].join(' ').toLowerCase();
            if (!searchable.includes(query)) return;
        }

        rowIndex++;

        let pendingCount = 0;
        let paidCount = 0;
        let totalPendingFine = 0;

        uViolations.forEach(v => {
            if (v.status === 'pending') {
                pendingCount++;
                totalPendingFine += (v.fineAmount || 0);
            } else if (v.status === 'paid') {
                paidCount++;
            }
        });

        html += `
            <tr>
                <td>${rowIndex}</td>
                <td>
                    <div style="font-weight: 600; color: var(--text-primary);">${u.email || '-'}</div>
                    <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 2px; word-break: break-all;">${uid}</div>
                </td>
                <td>
                    <div style="font-weight: 600;">${u.fullName || '<span style="color: var(--text-muted); font-style: italic;">Chưa cập nhật</span>'}</div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 4px;">🪪 ${u.idCard || '-'}</div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary);">📱 ${u.phone || '-'}</div>
                </td>
                <td>
                    Hạng: <span class="data-tag tag-info">${u.licenseClass || '-'}</span><br>
                    Số: <span style="font-weight: 500;">${u.licenseNumber || '-'}</span><br>
                    <span style="font-weight: 700; color: ${(u.points ?? 12) < 12 ? 'var(--warning)' : 'var(--success)'};">Điểm: ${u.points ?? 12}/12</span>
                </td>
                <td>
                    ${uVehicles.length > 0
                        ? uVehicles.map(v => `<div style="margin-bottom: 6px;"><span class="data-tag tag-info">${v.licensePlate || ''}</span><div style="font-size: 0.72rem; color: var(--text-muted);">${v.brand || ''} ${v.model || ''} — ${v.type || ''}</div></div>`).join('')
                        : '<span style="color: var(--text-muted); font-style: italic;">Chưa có</span>'}
                </td>
                <td>
                    <span class="data-tag tag-danger">Chưa nộp: ${pendingCount}</span>
                    <span class="data-tag tag-success">Đã nộp: ${paidCount}</span>
                    ${totalPendingFine > 0 ? `<div style="font-weight: 700; color: var(--danger); margin-top: 6px;">Nợ: ${formatter.format(totalPendingFine)}</div>` : '<div style="color: var(--success); margin-top: 4px; font-weight: 600; font-size: 0.8rem;">✓ Không nợ</div>'}
                </td>
                <td>
                    ${uComplaints.length > 0 
                        ? `<div style="margin-bottom: 6px;"><span class="data-tag tag-warning">📝 ${uComplaints.length} kh.nại</span></div>` + 
                          uComplaints.map(c => `<div style="font-size: 0.72rem; color: var(--warning); border: 1px solid var(--warning); border-radius: 4px; padding: 2px 4px; margin-bottom: 4px; background: rgba(255, 152, 0, 0.1);">Lý do: ${c.reason || 'Khác'} - ${c.status === 'pending' ? 'Đang chờ' : (c.status === 'approved' ? 'Đã duyệt' : 'Đã từ chối')}</div>`).join('')
                        : '<span class="data-tag tag-secondary" style="margin-bottom: 6px; display: inline-block;">0 kh.nại</span><br>'}
                    <span class="data-tag tag-secondary">🔔 ${uNotifications.length}</span>
                </td>
                <td>
                    <button class="btn btn-danger" style="padding: 4px 8px; font-size: 0.8rem;" onclick="deleteUser('${uid}')">🗑️ Xóa</button>
                </td>
            </tr>
        `;
    });

    if (rowIndex === 0) {
        html = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 50px 40px;">
            <div style="font-size: 2rem; margin-bottom: 8px;">🔍</div>
            ${query ? 'Không tìm thấy kết quả phù hợp.' : 'Hệ thống chưa có dữ liệu người dùng.'}
        </td></tr>`;
    }

    adminDataTableBody.innerHTML = html;
    if (manageRowCount) manageRowCount.textContent = `${rowIndex} người dùng`;
}

async function loadAdminData() {
    if (!adminDataTableBody) return;

    adminDataTableBody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-secondary); padding: 60px 40px;">
        <div class="manage-loading" style="font-size: 2.5rem; margin-bottom: 12px;">⏳</div>
        <div style="font-weight: 500;">Đang tải dữ liệu từ Firebase...</div>
    </td></tr>`;

    try {
        const resp = await fetch('/api/admin/data');
        const json = await resp.json();

        if (json.status !== 'ok') {
            showToast('Lỗi tải dữ liệu: ' + json.message, 'error');
            adminDataTableBody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--danger); padding: 50px;">
                <div style="font-size: 2rem; margin-bottom: 8px;">⚠️</div>${json.message}
            </td></tr>`;
            return;
        }

        _adminCachedData = json.data;
        updateManageStats(_adminCachedData);
        renderAdminTable(_adminCachedData, manageSearchInput ? manageSearchInput.value : '');
        showToast('Đã làm mới dữ liệu hệ thống!', 'success');

    } catch (e) {
        console.error(e);
        adminDataTableBody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--danger); padding: 50px;">
            <div style="font-size: 2rem; margin-bottom: 8px;">❌</div>Lỗi kết nối máy chủ.
        </td></tr>`;
    }
}

async function deleteUser(uid) {
    if (!confirm('Bạn có chắc chắn muốn xóa tài khoản này và các dữ liệu liên quan? Hành động này không thể hoàn tác!')) {
        return;
    }
    try {
        const resp = await fetch(`/api/admin/users/${uid}`, { method: 'DELETE' });
        const json = await resp.json();
        if (json.status === 'ok') {
            showToast('Đã xóa người dùng thành công', 'success');
            loadAdminData(); // Refresh the table
        } else {
            showToast('Lỗi khi xóa người dùng: ' + json.message, 'error');
        }
    } catch (e) {
        console.error(e);
        showToast('Lỗi kết nối máy chủ khi xóa người dùng', 'error');
    }
}

// Refresh button
if (refreshManageBtn) {
    refreshManageBtn.addEventListener('click', loadAdminData);
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

// Auto-load when switching to the tab
document.querySelectorAll('.nav-link[data-tab="manage"]').forEach(link => {
    link.addEventListener('click', () => {
        loadAdminData();
    });
});
