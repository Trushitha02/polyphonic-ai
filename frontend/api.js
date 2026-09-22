/**
 * Polyphonic AI - Frontend API & Session Manager
 * Connects HTML frontend seamlessly to Flask REST API backend.
 */

const API_BASE = "/api";

// User Session Management
function getCurrentUser() {
    try {
        const stored = localStorage.getItem("polyphonic_user");
        return stored ? JSON.parse(stored) : null;
    } catch (e) {
        return null;
    }
}

function setCurrentUser(user) {
    localStorage.setItem("polyphonic_user", JSON.stringify(user));
    if (user && user.id) {
        localStorage.setItem("user_id", user.id);
    }
    if (user && user.name) {
        localStorage.setItem("user_name", user.name);
    }
}

function clearUser() {
    localStorage.removeItem("polyphonic_user");
    localStorage.removeItem("user_id");
    localStorage.removeItem("user_name");
}

function requireAuth() {
    const user = getCurrentUser();
    if (!user) {
        window.location.href = "index.html";
    }
    return user;
}

function logout() {
    if (confirm("Are you sure you want to logout?")) {
        fetch(`${API_BASE}/auth/logout`, { method: "POST" }).catch(() => {});
        clearUser();
        window.location.href = "index.html";
    }
}

// Global UI Initialization (Navbar user & logout)
document.addEventListener("DOMContentLoaded", () => {
    const user = getCurrentUser();
    const userDisplay = document.querySelector(".welcome-user") ||
                        document.getElementById("navUserName") ||
                        document.querySelector(".user-name");

    if (userDisplay && user && user.name) {
        userDisplay.innerHTML = `<i class="fa-solid fa-user-check" style="margin-right:6px;"></i> Welcome, <strong>${user.name}</strong>`;
    }

    // Attach logout buttons
    const logoutBtns = document.querySelectorAll(".logout-btn, .logout");
    logoutBtns.forEach(btn => {
        btn.onclick = (e) => {
            e.preventDefault();
            logout();
        };
    });
});

// Toast notification helper
function showToast(message, type = "info") {
    let container = document.getElementById("toast-container");
    if (!container) {
        container = document.createElement("div");
        container.id = "toast-container";
        container.style.cssText = `
            position: fixed;
            top: 25px;
            right: 25px;
            z-index: 999999;
            display: flex;
            flex-direction: column;
            gap: 12px;
            pointer-events: none;
        `;
        document.body.appendChild(container);
    }

    const toast = document.createElement("div");
    const bgColors = {
        success: "linear-gradient(135deg, #10b981, #059669)",
        error: "linear-gradient(135deg, #ef4444, #dc2626)",
        warning: "linear-gradient(135deg, #f59e0b, #d97706)",
        info: "linear-gradient(135deg, #6a11cb, #2575fc)"
    };

    const icons = {
        success: "fa-check-circle",
        error: "fa-triangle-exclamation",
        warning: "fa-circle-exclamation",
        info: "fa-circle-info"
    };

    toast.style.cssText = `
        min-width: 280px;
        max-width: 420px;
        padding: 14px 20px;
        background: ${bgColors[type] || bgColors.info};
        color: white;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.18);
        display: flex;
        align-items: center;
        gap: 12px;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 14px;
        font-weight: 500;
        pointer-events: auto;
        animation: toastSlideIn 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        transition: all 0.3s ease;
    `;

    toast.innerHTML = `<i class="fa-solid ${icons[type] || icons.info}" style="font-size:18px;"></i> <span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateY(-15px)";
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// Ensure keyframe exists
if (!document.getElementById("toast-keyframes")) {
    const style = document.createElement("style");
    style.id = "toast-keyframes";
    style.textContent = `
        @keyframes toastSlideIn {
            from { transform: translateX(100px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
    `;
    document.head.appendChild(style);
}
