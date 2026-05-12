// =============================
// DOM Elements
// =============================
const analyzeBtn = document.getElementById('analyze-btn');
const medicalText = document.getElementById('medical-text');
const loading = document.getElementById('loading');
const outputSection = document.getElementById('output-section');

// Output elements
const categoryBadge = document.getElementById('category-badge');
const categoryText = document.getElementById('category-text');
const expandedTerms = document.getElementById('expanded-terms');
const retrievedDocs = document.getElementById('retrieved-docs');
const perplexityValue = document.getElementById('perplexity-value');
const accuracyValue = document.getElementById('accuracy-value');
const f1Value = document.getElementById('f1-value');
const confusionMatrix = document.getElementById('confusion-matrix');

// =============================
// Event Listeners
// =============================
analyzeBtn.addEventListener('click', analyzeText);
medicalText.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        analyzeText();
    }
});

// =============================
// Main Analyze Function
// =============================
async function analyzeText() {
    const text = medicalText.value.trim();
    
    if (!text) {
        showError('Please enter some medical text to analyze.');
        return;
    }
    
    // Show loading, hide output
    loading.classList.remove('hidden');
    outputSection.classList.add('hidden');
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            throw new Error('Failed to analyze text');
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        showError('An error occurred while analyzing the text. Please try again.');
    } finally {
        loading.classList.add('hidden');
    }
}

// =============================
// Display Results
// =============================
function displayResults(data) {
    // Display category
    displayCategory(data.category);
    
    // Display expanded terms
    displayExpandedTerms(data.expanded_terms);
    
    // Display retrieved documents
    displayRetrievedDocs(data.results);
    
    // Display metrics
    displayMetrics(data.metrics);
    
    // Display confusion matrix
    displayConfusionMatrix(data.confusion_matrix);
    
    // Show output section
    outputSection.classList.remove('hidden');
    
    // Scroll to output
    outputSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// =============================
// Display Category
// =============================
function displayCategory(category) {
    categoryBadge.className = 'category-badge';
    categoryText.textContent = category;
    
    // Add appropriate class based on category
    if (category.toLowerCase() === 'cardiology') {
        categoryBadge.classList.add('cardiology');
    } else if (category.toLowerCase() === 'neurology') {
        categoryBadge.classList.add('neurology');
    } else {
        categoryBadge.classList.add('general');
    }
}

// =============================
// Display Expanded Terms
// =============================
function displayExpandedTerms(terms) {
    expandedTerms.innerHTML = '';
    
    if (!terms || terms.length === 0) {
        expandedTerms.innerHTML = '<p class="no-results">No expanded terms found.</p>';
        return;
    }
    
    terms.forEach(term => {
        const pill = document.createElement('span');
        pill.className = 'term-pill';
        pill.textContent = term.replace(/_/g, ' ');
        expandedTerms.appendChild(pill);
    });
}

// =============================
// Display Retrieved Documents
// =============================
function displayRetrievedDocs(results) {
    retrievedDocs.innerHTML = '';
    
    if (!results || results.length === 0) {
        retrievedDocs.innerHTML = '<p class="no-results">No relevant documents found.</p>';
        return;
    }
    
    results.forEach((doc, index) => {
        const docCard = document.createElement('div');
        docCard.className = 'doc-card';
        docCard.style.animationDelay = `${index * 0.1}s`;
        
        docCard.innerHTML = `
            <div class="doc-header">
                <span class="doc-id">Doc #${doc.doc_id}</span>
                <span class="doc-score">
                    <i class="fas fa-star"></i>
                    Score: ${doc.score}
                </span>
            </div>
            <div class="doc-preview">${escapeHtml(doc.preview)}</div>
        `;
        
        retrievedDocs.appendChild(docCard);
    });
}

// =============================
// Display Metrics
// =============================
function displayMetrics(metrics) {
    perplexityValue.textContent = metrics.perplexity.toFixed(2);
    accuracyValue.textContent = (metrics.accuracy * 100).toFixed(0) + '%';
    f1Value.textContent = (metrics.f1 * 100).toFixed(0) + '%';
}

// =============================
// Display Confusion Matrix
// =============================
function displayConfusionMatrix(matrix) {
    if (!matrix || matrix.length === 0) {
        confusionMatrix.innerHTML = '<p class="no-results">Confusion matrix not available.</p>';
        return;
    }
    
    const labels = ['Cardiology', 'Neurology', 'General'];
    const maxVal = Math.max(...matrix.flat());
    
    let tableHTML = '<table class="matrix-table">';
    
    // Header row
    tableHTML += '<tr><th class="matrix-cell header"></th>';
    labels.forEach(label => {
        tableHTML += `<th class="matrix-cell header">${label}</th>`;
    });
    tableHTML += '</tr>';
    
    // Data rows
    matrix.forEach((row, i) => {
        tableHTML += `<tr><th class="matrix-cell header">${labels[i]}</th>`;
        row.forEach(val => {
            let cellClass = 'value-0';
            if (val > 0) {
                const ratio = val / maxVal;
                if (ratio < 0.33) cellClass = 'value-low';
                else if (ratio < 0.66) cellClass = 'value-medium';
                else cellClass = 'value-high';
            }
            tableHTML += `<td class="matrix-cell ${cellClass}">${val}</td>`;
        });
        tableHTML += '</tr>';
    });
    
    tableHTML += '</table>';
    confusionMatrix.innerHTML = tableHTML;
}

// =============================
// Utility Functions
// =============================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showError(message) {
    // Create a simple error toast
    const existingToast = document.querySelector('.error-toast');
    if (existingToast) {
        existingToast.remove();
    }
    
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    toast.innerHTML = `
        <i class="fas fa-exclamation-circle"></i>
        <span>${message}</span>
    `;
    
    // Add styles
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: #dc2626;
        color: white;
        padding: 15px 25px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        gap: 10px;
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.3);
        z-index: 1000;
        font-weight: 500;
        animation: slideDown 0.3s ease;
    `;
    
    document.body.appendChild(toast);
    
    // Remove after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'slideUp 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Add animations to style
const style = document.createElement('style');
style.textContent = `
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateX(-50%) translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
        }
    }
    
    @keyframes slideUp {
        from {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
        }
        to {
            opacity: 0;
            transform: translateX(-50%) translateY(-20px);
        }
    }
`;
document.head.appendChild(style);
