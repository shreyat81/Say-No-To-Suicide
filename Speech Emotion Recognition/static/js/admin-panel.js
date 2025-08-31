// Admin Panel JavaScript
class AdminPanel {
    constructor() {
        this.baseUrl = window.location.origin;
        this.users = [];
        this.filteredUsers = [];
        this.init();
    }

    init() {
        this.loadUsers();
        this.loadRecentActivity();
        
        // Auto-refresh every 30 seconds
        setInterval(() => {
            this.loadUsers();
            this.loadRecentActivity();
        }, 30000);
    }

    async loadUsers() {
        try {
            const response = await fetch(`${this.baseUrl}/admin/users`);
            if (!response.ok) throw new Error('Failed to load users');

            const data = await response.json();
            this.users = data.users;
            this.filteredUsers = [...this.users];
            
            this.updateOverview();
            this.renderUsersTable();
            
        } catch (error) {
            console.error('Failed to load users:', error);
            this.showError('Failed to load user data');
        }
    }

    updateOverview() {
        const total = this.users.length;
        const active = this.users.filter(u => u.risk_score < 20 && !u.is_banned).length;
        const warning = this.users.filter(u => u.risk_score >= 20 && u.risk_score < 50).length;
        const alert = this.users.filter(u => u.risk_score >= 50 && u.risk_score < 80).length;
        const banned = this.users.filter(u => u.is_banned).length;

        document.getElementById('totalUsersAdmin').textContent = total;
        document.getElementById('activeUsers').textContent = active;
        document.getElementById('warningUsersAdmin').textContent = warning;
        document.getElementById('alertUsersAdmin').textContent = alert;
        document.getElementById('bannedUsersAdmin').textContent = banned;
    }

    renderUsersTable() {
        const tbody = document.getElementById('usersTableBody');
        if (!tbody) return;

        tbody.innerHTML = '';

        this.filteredUsers.forEach(user => {
            const row = document.createElement('tr');
            
            // Determine status
            let status = 'active';
            let statusText = 'Active';
            
            if (user.is_banned) {
                status = 'banned';
                statusText = 'Banned';
            } else if (user.risk_score >= 50) {
                status = 'alert';
                statusText = 'Alert';
            } else if (user.risk_score >= 20) {
                status = 'warning';
                statusText = 'Warning';
            }

            row.innerHTML = `
                <td><strong>${user.user_id}</strong></td>
                <td>${user.email || 'Not provided'}</td>
                <td>
                    <span class="risk-score-badge" style="color: ${this.getRiskColor(user.risk_score)}">
                        ${Math.round(user.risk_score)}
                    </span>
                </td>
                <td><span class="status-badge ${status}">${statusText}</span></td>
                <td>${user.last_update ? new Date(user.last_update).toLocaleString() : 'Never'}</td>
                <td>
                    <div class="user-actions">
                        <button class="user-action-btn reset-user-btn" onclick="resetUserScore('${user.user_id}')">
                            <i class="fas fa-undo"></i> Reset
                        </button>
                        ${user.is_banned ? 
                            `<button class="user-action-btn unban-user-btn" onclick="unbanUser('${user.user_id}')">
                                <i class="fas fa-check"></i> Unban
                            </button>` :
                            `<button class="user-action-btn ban-user-btn" onclick="banUser('${user.user_id}')">
                                <i class="fas fa-ban"></i> Ban
                            </button>`
                        }
                    </div>
                </td>
            `;
            
            tbody.appendChild(row);
        });
    }

    getRiskColor(score) {
        if (score >= 80) return '#dc2626';
        if (score >= 50) return '#ea580c';
        if (score >= 20) return '#d97706';
        return '#059669';
    }

    async loadRecentActivity() {
        // Simulate recent activity - in real implementation, this would come from logs
        const activityLog = document.getElementById('activityLog');
        if (!activityLog) return;

        // Mock activity data
        const activities = [
            {
                type: 'alert',
                icon: 'fas fa-bell',
                title: 'High Risk Alert',
                description: 'User test_user_2 reached alert threshold',
                time: '2 minutes ago'
            },
            {
                type: 'warn',
                icon: 'fas fa-exclamation-triangle',
                title: 'Warning Triggered',
                description: 'User test_user_1 risk score increased',
                time: '5 minutes ago'
            },
            {
                type: 'reset',
                icon: 'fas fa-undo',
                title: 'User Reset',
                description: 'Admin reset user high_risk_user',
                time: '10 minutes ago'
            }
        ];

        activityLog.innerHTML = activities.map(activity => `
            <div class="activity-item">
                <div class="activity-icon ${activity.type}">
                    <i class="${activity.icon}"></i>
                </div>
                <div class="activity-content">
                    <h4>${activity.title}</h4>
                    <p>${activity.description}</p>
                </div>
                <div class="activity-time">${activity.time}</div>
            </div>
        `).join('');
    }

    showError(message) {
        // Simple error display - could be enhanced with toast notifications
        alert(`Error: ${message}`);
    }

    showSuccess(message) {
        // Simple success display - could be enhanced with toast notifications
        alert(`Success: ${message}`);
    }
}

// Global functions for user actions
async function resetUserScore(userId) {
    if (!confirm(`Are you sure you want to reset the risk score for user "${userId}"?`)) {
        return;
    }

    try {
        const response = await fetch(`/reset_user/${userId}`, {
            method: 'POST'
        });

        if (!response.ok) throw new Error('Failed to reset user');

        const result = await response.json();
        adminPanel.showSuccess(result.message);
        adminPanel.loadUsers();
        
    } catch (error) {
        console.error('Reset error:', error);
        adminPanel.showError('Failed to reset user score');
    }
}

async function banUser(userId) {
    if (!confirm(`Are you sure you want to ban user "${userId}"?`)) {
        return;
    }

    // Note: This would need to be implemented in the backend
    adminPanel.showError('Ban functionality not yet implemented in backend');
}

async function unbanUser(userId) {
    if (!confirm(`Are you sure you want to unban user "${userId}"?`)) {
        return;
    }

    // Note: This would need to be implemented in the backend
    adminPanel.showError('Unban functionality not yet implemented in backend');
}

// Filter functions
function filterUsers() {
    const statusFilter = document.getElementById('statusFilter').value;
    const searchTerm = document.getElementById('userSearch').value.toLowerCase();

    adminPanel.filteredUsers = adminPanel.users.filter(user => {
        // Status filter
        let statusMatch = true;
        if (statusFilter !== 'all') {
            switch (statusFilter) {
                case 'active':
                    statusMatch = user.risk_score < 20 && !user.is_banned;
                    break;
                case 'warning':
                    statusMatch = user.risk_score >= 20 && user.risk_score < 50;
                    break;
                case 'alert':
                    statusMatch = user.risk_score >= 50 && user.risk_score < 80;
                    break;
                case 'banned':
                    statusMatch = user.is_banned;
                    break;
            }
        }

        // Search filter
        const searchMatch = searchTerm === '' || 
            user.user_id.toLowerCase().includes(searchTerm) ||
            (user.email && user.email.toLowerCase().includes(searchTerm));

        return statusMatch && searchMatch;
    });

    adminPanel.renderUsersTable();
}

// System action functions
async function triggerDecay() {
    const inactivityDays = parseInt(document.getElementById('inactivityDays').value);
    const decayAmount = parseInt(document.getElementById('decayAmount').value);

    if (!confirm(`Trigger score decay for users inactive for ${inactivityDays} days (reduce by ${decayAmount} points)?`)) {
        return;
    }

    try {
        const response = await fetch('/decay_scores', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                inactivity_days: inactivityDays,
                decay_amount: decayAmount
            })
        });

        if (!response.ok) throw new Error('Failed to trigger decay');

        const result = await response.json();
        adminPanel.showSuccess(result.message);
        adminPanel.loadUsers();
        
    } catch (error) {
        console.error('Decay error:', error);
        adminPanel.showError('Failed to trigger score decay');
    }
}

function createBackup() {
    // This would typically trigger a backend endpoint to create a database backup
    adminPanel.showError('Backup functionality not yet implemented');
}

function generateReport() {
    const reportType = document.getElementById('reportType').value;
    
    // This would typically generate and download a report
    adminPanel.showError(`${reportType} report generation not yet implemented`);
}

function refreshUsers() {
    adminPanel.loadUsers();
    adminPanel.showSuccess('User data refreshed');
}

function exportUsers() {
    // Export users to CSV
    const csvContent = [
        ['User ID', 'Email', 'Risk Score', 'Status', 'Last Update'],
        ...adminPanel.users.map(user => [
            user.user_id,
            user.email || '',
            user.risk_score,
            user.is_banned ? 'Banned' : 'Active',
            user.last_update || ''
        ])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `users_export_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
}

// Initialize admin panel when page loads
let adminPanel;
document.addEventListener('DOMContentLoaded', () => {
    adminPanel = new AdminPanel();
});

// Add additional styles
const style = document.createElement('style');
style.textContent = `
    .risk-score-badge {
        font-weight: 700;
        font-size: 16px;
    }
    
    .user-action-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
    }
    
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 4px solid #f3f4f6;
        border-top: 4px solid #6366f1;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
`;
document.head.appendChild(style);
