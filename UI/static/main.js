// Global variables for charts
let pieChart, barChart;

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
});

function initializeEventListeners() {
    // Single text form
    document.getElementById('sentimentForm').addEventListener('submit', handleSingleAnalysis);
    
    // File upload
    const fileUploadArea = document.getElementById('fileUploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');

    fileUploadArea.addEventListener('click', () => fileInput.click());
    fileUploadArea.addEventListener('dragover', handleDragOver);
    fileUploadArea.addEventListener('dragleave', handleDragLeave);
    fileUploadArea.addEventListener('drop', handleFileDrop);
    
    fileInput.addEventListener('change', handleFileSelect);
    uploadBtn.addEventListener('click', handleFileUpload);
}

async function handleSingleAnalysis(e) {
    e.preventDefault();
    
    const textInput = document.getElementById('textInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const text = textInput.value.trim();
    
    if (!text) {
        showToast('Please enter some text to analyze.', 'error');
        return;
    }

    // Show loading state
    analyzeBtn.innerHTML = '<span class="loading me-2"></span>Analyzing...';
    analyzeBtn.disabled = true;

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        if (data.success) {
            displaySingleResult(data.result);
            showToast('Analysis completed successfully!', 'success');
        } else {
            showToast(data.error || 'Analysis failed', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('Network error occurred', 'error');
    } finally {
        // Reset button
        analyzeBtn.innerHTML = '<i class="fas fa-search me-2"></i>Analyze Sentiment';
        analyzeBtn.disabled = false;
    }
}

function displaySingleResult(result) {
    const resultDiv = document.getElementById('singleResult');
    const sentimentDisplay = document.getElementById('sentimentDisplay');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');
    const probabilitiesDisplay = document.getElementById('probabilitiesDisplay');

    // Show result section
    resultDiv.style.display = 'block';
    resultDiv.classList.add('result-animation');

    // Display sentiment
    const sentiment = result.predicted_sentiment;
    const confidence = (result.confidence * 100).toFixed(1);
    
    let sentimentClass = 'sentiment-neutral';
    let sentimentIcon = 'fas fa-meh';
    
    if (sentiment.toLowerCase().includes('positive')) {
        sentimentClass = 'sentiment-positive';
        sentimentIcon = 'fas fa-smile';
    } else if (sentiment.toLowerCase().includes('negative')) {
        sentimentClass = 'sentiment-negative';
        sentimentIcon = 'fas fa-frown';
    }

    sentimentDisplay.innerHTML = `
        <div class="sentiment-badge ${sentimentClass}">
            <i class="${sentimentIcon} me-2"></i>
            ${sentiment.toUpperCase()}
        </div>
    `;

    // Update confidence bar
    confidenceBar.style.width = confidence + '%';
    confidenceText.textContent = `${confidence}% confident`;

    // Display all probabilities
    let probHtml = '<strong>All Sentiment Probabilities:</strong><br>';
    for (const [label, prob] of Object.entries(result.all_probabilities)) {
        const percentage = (prob * 100).toFixed(1);
        probHtml += `<small class="text-muted">${label}: ${percentage}%</small><br>`;
    }
    probabilitiesDisplay.innerHTML = probHtml;

    // Scroll to result
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('dragover');
}

function handleFileDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelection(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFileSelection(files[0]);
    }
}

function handleFileSelection(file) {
    const uploadBtn = document.getElementById('uploadBtn');
    
    if (file && (file.name.endsWith('.csv') || file.name.endsWith('.txt'))) {
        uploadBtn.style.display = 'block';
        uploadBtn.innerHTML = `<i class="fas fa-upload me-2"></i>Upload & Analyze "${file.name}"`;
        showToast(`File "${file.name}" selected. Click upload to analyze.`, 'success');
    } else {
        showToast('Please select a valid CSV or TXT file.', 'error');
        uploadBtn.style.display = 'none';
    }
}

async function handleFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const uploadProgress = document.getElementById('uploadProgress');
    const uploadBtn = document.getElementById('uploadBtn');
    
    if (!fileInput.files || fileInput.files.length === 0) {
        showToast('Please select a file first.', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    // Show progress
    uploadProgress.style.display = 'block';
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<span class="loading me-2"></span>Processing...';

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayBatchResults(data);
            showToast(`Analysis completed! Processed ${data.total_analyzed} texts.`, 'success');
        } else {
            showToast(data.error || 'Upload failed', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('Network error occurred during upload', 'error');
    } finally {
        // Reset UI
        uploadProgress.style.display = 'none';
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = '<i class="fas fa-upload me-2"></i>Upload & Analyze File';
        fileInput.value = '';
        uploadBtn.style.display = 'none';
    }
}

function displayBatchResults(data) {
    const resultsSection = document.getElementById('results');
    const statsCards = document.getElementById('statsCards');
    const resultsTableBody = document.getElementById('resultsTableBody');

    // Show results section
    resultsSection.style.display = 'block';

    // Create statistics cards
    let statsHtml = '';
    for (const [sentiment, count] of Object.entries(data.statistics)) {
        const percentage = ((count / data.total_analyzed) * 100).toFixed(1);
        let cardClass = 'stats-card';
        let icon = 'fas fa-chart-bar';
        
        if (sentiment.toLowerCase().includes('positive')) {
            icon = 'fas fa-smile';
        } else if (sentiment.toLowerCase().includes('negative')) {
            icon = 'fas fa-frown';
        } else {
            icon = 'fas fa-meh';
        }

        statsHtml += `
            <div class="col-md-4">
                <div class="${cardClass}">
                    <i class="${icon} fa-2x mb-2"></i>
                    <span class="stats-number">${count}</span>
                    <small>${sentiment.toUpperCase()}</small>
                    <small class="d-block">${percentage}%</small>
                </div>
            </div>
        `;
    }
    statsCards.innerHTML = statsHtml;

    // Create charts
    createCharts(data.statistics);

    // Populate results table
    let tableHtml = '';
    data.results.slice(0, 100).forEach((result, index) => { // Show first 100 results
        const confidence = (result.confidence * 100).toFixed(1);
        const truncatedText = result.text.length > 100 ? 
            result.text.substring(0, 100) + '...' : result.text;
        
        let sentimentClass = 'text-muted';
        let sentimentIcon = 'fas fa-meh';
        
        if (result.predicted_sentiment.toLowerCase().includes('positive')) {
            sentimentClass = 'text-success';
            sentimentIcon = 'fas fa-smile';
        } else if (result.predicted_sentiment.toLowerCase().includes('negative')) {
            sentimentClass = 'text-danger';
            sentimentIcon = 'fas fa-frown';
        }

        tableHtml += `
            <tr>
                <td>${index + 1}</td>
                <td title="${result.text}">${truncatedText}</td>
                <td>
                    <span class="${sentimentClass}">
                        <i class="${sentimentIcon} me-1"></i>
                        ${result.predicted_sentiment}
                    </span>
                </td>
                <td>
                    <div class="progress" style="height: 20px;">
                        <div class="progress-bar" role="progressbar" 
                             style="width: ${confidence}%" 
                             aria-valuenow="${confidence}" 
                             aria-valuemin="0" 
                             aria-valuemax="100">
                            ${confidence}%
                        </div>
                    </div>
                </td>
            </tr>
        `;
    });
    resultsTableBody.innerHTML = tableHtml;

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function createCharts(statistics) {
    const labels = Object.keys(statistics);
    const data = Object.values(statistics);
    const colors = labels.map(label => {
        if (label.toLowerCase().includes('positive')) return '#28a745';
        if (label.toLowerCase().includes('negative')) return '#dc3545';
        return '#6c757d';
    });

    // Destroy existing charts
    if (pieChart) pieChart.destroy();
    if (barChart) barChart.destroy();

    // Pie Chart
    const pieCtx = document.getElementById('pieChart').getContext('2d');
    pieChart = new Chart(pieCtx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors,
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((value / total) * 100).toFixed(1);
                            return `${label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });

    // Bar Chart
    const barCtx = document.getElementById('barChart').getContext('2d');
    barChart = new Chart(barCtx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Count',
                data: data,
                backgroundColor: colors,
                borderColor: colors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Count: ${context.parsed.y}`;
                        }
                    }
                }
            }
        }
    });
}

function showToast(message, type) {
    const toastId = type === 'error' ? 'errorToast' : 'successToast';
    const toast = document.getElementById(toastId);
    const toastBody = toast.querySelector('.toast-body');
    
    toastBody.textContent = message;
    
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
}