/**
 * Polyphonic AI - Global Sidebar Navigation Manager
 * Handles active state, mobile drawer toggle, and user display
 */

function initSidebar(activePage) {
    // 1. Determine active page name if not provided
    if (!activePage) {
        const path = window.location.pathname;
        const page = path.substring(path.lastIndexOf('/') + 1) || 'dashboard.html';
        activePage = page.replace('.html', '');
    }

    // 2. Mark active sidebar item
    document.querySelectorAll('.sidebar-item').forEach(item => {
        item.classList.remove('active');
        const href = item.getAttribute('href');
        if (href && (href.includes(activePage + '.html') || href === activePage)) {
            item.classList.add('active');
        }
    });

    // 3. Populate user information
    try {
        const user = typeof getCurrentUser === 'function' ? getCurrentUser() : JSON.parse(localStorage.getItem('user'));
        const name = user ? (user.name || user.email.split('@')[0]) : 'Musician';
        const initial = name.charAt(0).toUpperCase();

        const nameEls = document.querySelectorAll('.sidebar-user-name, .topbar-user-name');
        nameEls.forEach(el => el.textContent = name);

        const avatarEls = document.querySelectorAll('.topbar-user-avatar');
        avatarEls.forEach(el => el.textContent = initial);
    } catch (e) {}

    // 4. Mobile sidebar toggle
    const toggleBtn = document.querySelector('.mobile-toggle');
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.querySelector('.sidebar-overlay');

    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', () => {
            sidebar.classList.toggle('open');
            if (overlay) overlay.classList.toggle('active');
        });
    }

    if (overlay && sidebar) {
        overlay.addEventListener('click', () => {
            sidebar.classList.remove('open');
            overlay.classList.remove('active');
        });
    }
}

// Generate the standard sidebar HTML snippet
function getSidebarHTML(activePage) {
    const items = [
        { id: 'dashboard', href: 'dashboard.html', icon: 'fa-table-columns', label: 'Dashboard' },
        { id: 'upload', href: 'upload.html', icon: 'fa-cloud-arrow-up', label: 'Upload Music' },
        { id: 'separation', href: 'separation.html', icon: 'fa-layer-group', label: 'Instrument Separation' },
        { id: 'transcription', href: 'transcription.html', icon: 'fa-wand-magic-sparkles', label: 'Music Transcription' },
        { id: 'performance', href: 'performance.html', icon: 'fa-guitar', label: 'Performance Analysis' },
        { id: 'level', href: 'level.html', icon: 'fa-ranking-star', label: 'Skill Level' },
        { id: 'confidence', href: 'confidence.html', icon: 'fa-shield-halved', label: 'Confidence Score' },
        { id: 'progress', href: 'progress.html', icon: 'fa-chart-line', label: 'Progress Tracking' }
    ];

    return `
    <div class="sidebar-overlay"></div>
    <aside class="sidebar">
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">
                <i class="fa-solid fa-music"></i>
            </div>
            <div class="sidebar-brand-text">
                <h2>Polyphonic AI</h2>
                <span>Music Studio</span>
            </div>
        </div>

        <nav class="sidebar-menu">
            <div class="sidebar-section-label">WORKSPACE</div>
            ${items.slice(0, 4).map(item => `
                <a href="${item.href}" class="sidebar-item ${activePage === item.id ? 'active' : ''}">
                    <i class="fa-solid ${item.icon}"></i>
                    <span>${item.label}</span>
                </a>
            `).join('')}
            <div class="sidebar-section-label performance-label">PERFORMANCE</div>
            ${items.slice(4).map(item => `
                <a href="${item.href}" class="sidebar-item ${activePage === item.id ? 'active' : ''}">
                    <i class="fa-solid ${item.icon}"></i>
                    <span>${item.label}</span>
                </a>
            `).join('')}
        </nav>

        <div class="sidebar-footer">
            <a href="profile.html" class="sidebar-item ${activePage === 'profile' ? 'active' : ''}">
                <i class="fa-solid fa-user"></i>
                <span>Profile</span>
            </a>
            <button onclick="logout()" class="sidebar-item logout-item">
                <i class="fa-solid fa-right-from-bracket"></i>
                <span>Logout</span>
            </button>
        </div>
    </aside>
    `;
}

document.addEventListener('DOMContentLoaded', () => {
    // Replace legacy page-specific sidebars with the shared navigation.
    const container = document.getElementById('sidebarContainer');
    if (container) {
        const pageKey = container.getAttribute('data-active') || '';
        container.innerHTML = getSidebarHTML(pageKey);
        if (!container.closest('.app-layout')) {
            document.body.classList.add('has-global-sidebar');
        }
    } else {
        const legacySidebar = document.querySelector('.sidebar');
        if (legacySidebar) {
            const page = window.location.pathname.split('/').pop().replace('.html', '');
            const replacement = document.createElement('div');
            replacement.id = 'sidebarContainer';
            replacement.setAttribute('data-active', page);
            replacement.innerHTML = getSidebarHTML(page);
            legacySidebar.replaceWith(replacement);
            document.body.classList.add('has-global-sidebar');
        }
    }
    const canonicalSidebar = document.querySelector('#sidebarContainer .sidebar');
    document.querySelectorAll('.sidebar').forEach(sidebar => {
        if (sidebar !== canonicalSidebar) sidebar.remove();
    });
    initSidebar();
});

// Global logout fallback if not implemented by page
if (typeof window.logout !== 'function') {
    window.logout = function() {
        if (window.PolyphonicAPI && typeof window.PolyphonicAPI.logout === 'function') {
            window.PolyphonicAPI.logout();
            return;
        }
        localStorage.removeItem("token");
        localStorage.removeItem("user");
        window.location.href = "login.html";
    };
}
