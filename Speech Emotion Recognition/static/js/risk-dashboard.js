// Risk Dashboard JavaScript
class RiskDashboard {
    constructor() {
        this.baseUrl = window.location.origin;
        this.currentUser = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadStats();
        this.setupFileUpload();
    }

    setupEventListeners() {
        const form = document.getElementById('emotionForm');
        if (form) {
            form.addEventListener('submit', (e) => this.handleEmotionAnalysis(e));
        }

        // Auto-refresh stats every 30 seconds
        setInterval(() => this.loadStats(), 30000);
    }

    setupFileUpload() {
        const fileInput = document.getElementById('audioFile');
        const uploadArea = document.getElementById('fileUploadArea');

        if (fileInput && uploadArea) {
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('drag-over');
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('drag-over');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('drag-over');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    fileInput.files = files;
                    this.updateFileDisplay(files[0]);
                }
            });

            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.updateFileDisplay(e.target.files[0]);
                }
            });
        }
    }

    updateFileDisplay(file) {
        const placeholder = document.querySelector('.upload-placeholder');
        if (placeholder) {
            placeholder.innerHTML = `
                <i class="fas fa-file-audio"></i>
                <p><strong>${file.name}</strong></p>
                <small>Size: ${(file.size / 1024 / 1024).toFixed(2)} MB</small>
            `;
        }
    }

    async handleEmotionAnalysis(e) {
        e.preventDefault();
        
        const btn = document.getElementById('analyzeBtn');
        const resultsPanel = document.getElementById('resultsPanel');
        
        // Show loading state
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        
        try {
            const formData = new FormData(e.target);
            
            // Use test endpoint for now (can switch to /predict when TensorFlow is working)
            const response = await fetch(`${this.baseUrl}/predict_test`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.displayResults(result);
            this.loadStats(); // Refresh stats after analysis
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError('Analysis failed. Please try again.');
        } finally {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-brain"></i> Analyze Emotion & Risk';
        }
    }

    displayResults(result) {
        const resultsPanel = document.getElementById('resultsPanel');
        resultsPanel.style.display = 'block';

        // Update timestamp
        document.getElementById('resultTimestamp').textContent = 
            new Date().toLocaleString();

        // Update emotion display
        this.updateEmotionDisplay(result.emotion, result.confidence);
        
        // Update risk assessment
        this.updateRiskAssessment(result.risk_info);
        
        // Show alert if needed
        this.updateAlertSection(result);

        // Smooth scroll to results
        resultsPanel.scrollIntoView({ behavior: 'smooth' });
    }

    updateEmotionDisplay(emotion, confidence) {
        const emotionIcon = document.getElementById('emotionIcon');
        const emotionLabel = document.getElementById('emotionLabel');
        const confidenceFill = document.getElementById('confidenceFill');
        const confidenceText = document.getElementById('confidenceText');

        // Emotion icons mapping
        const emotionIcons = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'fearful': '😨',
            'disgust': '🤢',
            'surprised': '😲',
            'neutral': '😐',
            'calm': '😌'
        };

        emotionIcon.textContent = emotionIcons[emotion.toLowerCase()] || '😐';
        emotionLabel.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
        
        const confidencePercent = Math.round(confidence * 100);
        confidenceFill.style.width = `${confidencePercent}%`;
        confidenceText.textContent = `${confidencePercent}%`;
    }

    updateRiskAssessment(riskInfo) {
        const riskScore = document.getElementById('riskScore');
        const oldScore = document.getElementById('oldScore');
        const scoreChange = document.getElementById('scoreChange');
        const keywordRisk = document.getElementById('keywordRisk');
        const riskAction = document.getElementById('riskAction');
        const riskNeedle = document.getElementById('riskNeedle');

        // Update values
        riskScore.textContent = Math.round(riskInfo.new_score);
        oldScore.textContent = Math.round(riskInfo.old_score);
        
        const change = riskInfo.increment;
        scoreChange.textContent = change >= 0 ? `+${Math.round(change)}` : Math.round(change);
        scoreChange.className = `risk-value ${change >= 0 ? 'positive' : 'negative'}`;
        
        keywordRisk.textContent = `+${riskInfo.keyword_risk || 0}`;
        
        // Update action display
        riskAction.textContent = riskInfo.action.charAt(0).toUpperCase() + riskInfo.action.slice(1);
        riskAction.className = `risk-action ${riskInfo.action}`;

        // Update gauge needle (0-100 scale)
        const needleAngle = Math.min((riskInfo.new_score / 100) * 180, 180);
        riskNeedle.style.transform = `translateX(-50%) rotate(${needleAngle}deg)`;

        // Update risk score color based on level
        if (riskInfo.new_score >= 80) {
            riskScore.style.color = '#dc2626';
        } else if (riskInfo.new_score >= 50) {
            riskScore.style.color = '#ea580c';
        } else if (riskInfo.new_score >= 20) {
            riskScore.style.color = '#d97706';
        } else {
            riskScore.style.color = '#059669';
        }
    }

    updateAlertSection(result) {
        const alertSection = document.getElementById('alertSection');
        const alertMessage = document.getElementById('alertMessage');

        if (result.warning || result.risk_info.is_banned) {
            alertSection.style.display = 'block';
            alertMessage.innerHTML = `
                <i class="fas fa-exclamation-triangle"></i>
                ${result.warning || 'Account has been flagged for concerning behavior.'}
            `;
        } else {
            alertSection.style.display = 'none';
        }
    }

    async loadStats() {
        try {
            const response = await fetch(`${this.baseUrl}/admin/users`);
            if (!response.ok) return;

            const data = await response.json();
            this.updateStats(data.users);
        } catch (error) {
            console.error('Failed to load stats:', error);
        }
    }

    updateStats(users) {
        const totalUsers = users.length;
        const warningUsers = users.filter(u => u.risk_score >= 20 && u.risk_score < 50).length;
        const alertUsers = users.filter(u => u.risk_score >= 50 && u.risk_score < 80).length;
        const bannedUsers = users.filter(u => u.is_banned).length;

        document.getElementById('totalUsers').textContent = totalUsers;
        document.getElementById('warningUsers').textContent = warningUsers;
        document.getElementById('alertUsers').textContent = alertUsers;
        document.getElementById('bannedUsers').textContent = bannedUsers;
    }

    showError(message) {
        const resultsPanel = document.getElementById('resultsPanel');
        resultsPanel.style.display = 'block';
        resultsPanel.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-circle"></i>
                <h3>Error</h3>
                <p>${message}</p>
            </div>
        `;
    }
}

// User lookup functions
async function lookupUser() {
    const userId = document.getElementById('lookupUserId').value.trim();
    const resultsDiv = document.getElementById('lookupResults');
    
    if (!userId) {
        alert('Please enter a User ID');
        return;
    }

    try {
        const response = await fetch(`/user_status/${userId}`);
        
        if (!response.ok) {
            if (response.status === 404) {
                resultsDiv.style.display = 'none';
                alert('User not found');
                return;
            }
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const userData = await response.json();
        displayUserStatus(userData);
        
    } catch (error) {
        console.error('Lookup error:', error);
        alert('Failed to lookup user. Please try again.');
    }
}

function displayUserStatus(userData) {
    const resultsDiv = document.getElementById('lookupResults');
    
    document.getElementById('statusUserId').textContent = userData.user_id;
    document.getElementById('statusRiskScore').textContent = Math.round(userData.risk_score);
    document.getElementById('statusLastUpdate').textContent = 
        userData.last_update ? new Date(userData.last_update).toLocaleString() : 'Never';
    document.getElementById('statusEmail').textContent = userData.email || 'Not provided';
    
    const statusBadge = document.getElementById('statusBadge');
    if (userData.is_banned) {
        statusBadge.textContent = 'Banned';
        statusBadge.className = 'status-badge banned';
    } else {
        statusBadge.textContent = 'Active';
        statusBadge.className = 'status-badge active';
    }
    
    resultsDiv.style.display = 'block';
    
    // Store current user for reset function
    window.currentLookupUser = userData.user_id;
}

async function resetUser() {
    const userId = window.currentLookupUser;
    if (!userId) return;

    if (!confirm(`Are you sure you want to reset the risk score for user "${userId}"?`)) {
        return;
    }

    try {
        const response = await fetch(`/reset_user/${userId}`, {
            method: 'POST'
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        alert(result.message);
        
        // Refresh the user status
        lookupUser();
        
    } catch (error) {
        console.error('Reset error:', error);
        alert('Failed to reset user. Please try again.');
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    new RiskDashboard();
});

// Add some CSS for dynamic elements
const style = document.createElement('style');
style.textContent = `
    .drag-over {
        border-color: #6366f1 !important;
        background: #f0f9ff !important;
    }
    
    .risk-value.positive {
        color: #dc2626;
    }
    
    .risk-value.negative {
        color: #059669;
    }
    
    .error-message {
        text-align: center;
        padding: 40px;
        color: #dc2626;
    }
    
    .error-message i {
        font-size: 48px;
        margin-bottom: 20px;
    }
    
    .loading {
        opacity: 0.6;
        pointer-events: none;
    }
`;
document.head.appendChild(style);
