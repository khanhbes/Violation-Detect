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
        hero_desc: 'Hệ thống phát hiện vi phạm giao thông thời gian thực',
        hero_cta_start: '🚀 Bắt đầu ngay',
        hero_cta_image: '🖼️ Thử với ảnh',
        about_title: 'Giới thiệu dự án',
        about_desc: 'Dự án nghiên cứu khoa học ứng dụng Deep Learning vào giám sát giao thông đô thị, phát hiện tự động các hành vi vi phạm luật giao thông thông qua camera giám sát.',
        feat_helmet: 'Phát hiện người điều khiển xe máy không đội mũ bảo hiểm',
        feat_redlight: 'Phát hiện phương tiện vượt đèn đỏ tại ngã tư',
        feat_sidewalk: 'Phát hiện phương tiện chạy lên vỉa hè, dải phân cách',
        feat_wrongway: 'Phát hiện phương tiện chạy ngược chiều',
        feat_wronglane: 'Phát hiện vi phạm sai làn đường, đè vạch kẻ đường',
        feat_sign: 'Phát hiện vi phạm biển báo giao thông',
        tech_title: 'Mô hình & Công nghệ',
        tech_yolo: 'Object Detection + Instance Segmentation, nhận diện 40 classes bao gồm phương tiện, đèn giao thông, vạch kẻ, biển báo',
        tech_bytetrack: 'Multi-object tracking cho theo dõi phương tiện liên tục qua nhiều frame, hỗ trợ phân tích hành vi',
        tech_fastapi: 'Backend hiệu suất cao với streaming real-time, xử lý đồng thời nhiều client',
        tech_calibration: 'Tự động hiệu chỉnh vùng vi phạm (vỉa hè, vạch dừng, làn đường) trong 5-10 giây đầu',
        team_title: 'Thành viên nhóm',
        team_a_desc: 'Phụ trách kiến trúc hệ thống & training model',
        team_b_desc: 'Phát triển các module phát hiện vi phạm',
        team_c_desc: 'Thu thập, gán nhãn và xử lý dữ liệu huấn luyện',
        team_d_desc: 'Thiết kế giao diện web & trải nghiệm người dùng',
        contact_title: 'Liên hệ',
        contact_phone: 'Điện thoại',
        contact_org: 'Đơn vị',
        contact_org_name: 'Khoa Công nghệ Thông tin',
        image_subtitle: 'Tải lên hình ảnh để nhận diện tất cả các vật thể',
        video_subtitle: 'Tải lên video để nhận diện và theo dõi vật thể',
        video_processing: 'Đang xử lý video...',
        realtime_subtitle: 'Phát hiện vi phạm giao thông theo thời gian thực',
        lookup_title: 'Tra cứu vi phạm',
        lookup_subtitle: 'Tìm kiếm thông tin vi phạm theo biển số xe, CCCD hoặc số điện thoại',
        lookup_form_title: 'Thông tin tra cứu',
        lookup_plate: '🚗 Biển số xe',
        lookup_cccd: '🪪 Số CCCD',
        lookup_phone: '📱 Số điện thoại',
        lookup_results_title: 'Kết quả tra cứu',
        lookup_empty: 'Nhập thông tin và nhấn Tra cứu để tìm kiếm vi phạm.',
        typed_strings: [
            'trí tuệ nhân tạo YOLOv12-Seg',
            'nhận diện 40 loại đối tượng',
            'xử lý real-time với độ chính xác cao',
            'theo dõi phương tiện liên tục qua ByteTrack',
            'phân tích hành vi vi phạm tự động'
        ],
    },
    en: {
        hero_desc: 'Real-time traffic violation detection system',
        hero_cta_start: '🚀 Get Started',
        hero_cta_image: '🖼️ Try with Image',
        about_title: 'Project Introduction',
        about_desc: 'A scientific research project applying Deep Learning to urban traffic monitoring, automatically detecting traffic law violations through surveillance cameras.',
        feat_helmet: 'Detect motorcycle riders not wearing helmets',
        feat_redlight: 'Detect vehicles running red lights at intersections',
        feat_sidewalk: 'Detect vehicles driving on sidewalks and medians',
        feat_wrongway: 'Detect vehicles going the wrong way',
        feat_wronglane: 'Detect wrong lane violations and lane line crossing',
        feat_sign: 'Detect traffic sign violations',
        tech_title: 'Models & Technology',
        tech_yolo: 'Object Detection + Instance Segmentation, recognizing 40 classes including vehicles, traffic lights, lane markings, and signs',
        tech_bytetrack: 'Multi-object tracking for continuous vehicle monitoring across frames, supporting behavior analysis',
        tech_fastapi: 'High-performance backend with real-time streaming, handling multiple concurrent clients',
        tech_calibration: 'Auto-calibration of violation zones (sidewalks, stop lines, lanes) within the first 5-10 seconds',
        team_title: 'Team Members',
        team_a_desc: 'System architecture & model training lead',
        team_b_desc: 'Violation detection module development',
        team_c_desc: 'Data collection, labeling & preprocessing',
        team_d_desc: 'Web interface design & user experience',
        contact_title: 'Contact',
        contact_phone: 'Phone',
        contact_org: 'Department',
        contact_org_name: 'Faculty of Information Technology',
        image_subtitle: 'Upload an image to detect all objects',
        video_subtitle: 'Upload a video to detect and track objects',
        video_processing: 'Processing video...',
        realtime_subtitle: 'Detect traffic violations in real-time',
        lookup_title: 'Violation Lookup',
        lookup_subtitle: 'Search violations by license plate, ID card or phone number',
        lookup_form_title: 'Search Information',
        lookup_plate: '🚗 License Plate',
        lookup_cccd: '🪪 ID Card Number',
        lookup_phone: '📱 Phone Number',
        lookup_results_title: 'Search Results',
        lookup_empty: 'Enter information and click Search to find violations.',
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

// Apply saved language on load (wait for other scripts)
window.addEventListener('load', () => {
    setTimeout(() => setLanguage(currentLang), 500);
});
