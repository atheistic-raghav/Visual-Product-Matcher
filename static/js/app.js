document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const fileInput = document.getElementById('fileInput');
    const urlInput = document.getElementById('urlInput');
    const searchBtn = document.getElementById('searchBtn');
    const imagePreview = document.getElementById('imagePreview');
    const loading = document.getElementById('loading');
    const resultsSection = document.getElementById('resultsSection');
    const results = document.getElementById('results');
    const errorAlert = document.getElementById('errorAlert');
    const errorMessage = document.getElementById('errorMessage');
    const filterSection = document.getElementById('filterSection');
    const similarityRange = document.getElementById('similarityRange');
    const similarityValue = document.getElementById('similarityValue');
    const resultCount = document.getElementById('resultCount');
    const fileInputLabel = document.querySelector('.file-input-label');

    // State
    let currentResults = [];
    let searchInProgress = false;

    // File Input Handling
    fileInput.addEventListener('change', handleFileSelect);
    fileInput.addEventListener('dragenter', preventDefaults);
    fileInput.addEventListener('dragover', preventDefaults);
    fileInput.addEventListener('dragleave', preventDefaults);
    fileInput.addEventListener('drop', handleDrop);

    // URL Input Handling
    urlInput.addEventListener('input', handleUrlInput);

    // Search Button
    searchBtn.addEventListener('click', handleSearch);

    // Similarity Range
    similarityRange.addEventListener('input', handleSimilarityChange);

    // Drag and Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        fileInputLabel.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        fileInputLabel.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        fileInputLabel.addEventListener(eventName, unhighlight, false);
    });

    fileInputLabel.addEventListener('drop', handleDrop, false);

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        fileInputLabel.classList.add('dragover');
    }

    function unhighlight() {
        fileInputLabel.classList.remove('dragover');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect({ target: { files } });
        }
    }

    function handleFileSelect(e) {
        if (e.target.files?.[0]) {
            const file = e.target.files[0];
            
            // Validate file size (16MB limit)
            if (file.size > 16 * 1024 * 1024) {
                showError('File size must be less than 16MB');
                return;
            }
            
            // Validate file type
            if (!file.type.startsWith('image/')) {
                showError('Please select a valid image file');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = ev => {
                showImagePreview(ev.target.result, file.name);
                urlInput.value = '';
                hideError();
            };
            reader.readAsDataURL(file);
        }
    }

    function handleUrlInput(e) {
        const url = e.target.value.trim();
        if (url) {
            // Basic URL validation
            try {
                new URL(url);
                showImagePreview(url, 'Image from URL');
                fileInput.value = '';
                hideError();
            } catch {
                imagePreview.style.display = 'none';
            }
        } else {
            imagePreview.style.display = 'none';
        }
    }

    function handleSearch() {
        if (searchInProgress) return;
        
        if (!fileInput.files?.[0] && !urlInput.value.trim()) {
            showError('Please select an image file or enter an image URL');
            return;
        }
        
        uploadAndSearch();
    }

    function handleSimilarityChange(e) {
        const value = parseInt(e.target.value);
        similarityValue.textContent = value;
        filterResults(value / 100);
    }

    function showImagePreview(src, title) {
        imagePreview.innerHTML = `
            <div class="preview-card fade-in">
                <div class="preview-img-container">
                    <img src="${src}" alt="Preview Image" loading="lazy">
                </div>
                <div class="preview-card-body">
                    <h3 class="preview-title">📤 ${escapeHtml(title)}</h3>
                </div>
            </div>
        `;
        imagePreview.style.display = 'block';
    }

    async function uploadAndSearch() {
        if (searchInProgress) return;
        
        searchInProgress = true;
        showLoading();
        hideError();
        
        try {
            const uploadRes = await uploadImage();
            if (!uploadRes.success) {
                throw new Error(uploadRes.message);
            }
            
            const searchRes = await searchSimilar(uploadRes.image_path);
            if (!searchRes.success) {
                throw new Error(searchRes.message);
            }
            
            currentResults = searchRes.results || [];
            
            // FIXED: Reset filter to 0 and show all results initially
            similarityRange.value = 0;
            similarityValue.textContent = '0';
            
            // FIXED: Pass true to enable scrolling for initial search
            displayResults(currentResults, true);
            
            if (currentResults.length > 0) {
                filterSection.style.display = 'block';
                filterSection.classList.add('fade-in');
            }
            
        } catch (err) {
            showError(err.message || 'An error occurred during search');
            console.error('Search error:', err);
        } finally {
            hideLoading();
            searchInProgress = false;
        }
    }

    async function uploadImage() {
        if (fileInput.files?.[0]) {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
            
        } else if (urlInput.value.trim()) {
            const response = await fetch('/upload', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    image_url: urlInput.value.trim() 
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        }
        
        throw new Error('No image provided');
    }

    async function searchSimilar(imagePath) {
        const response = await fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                image_path: imagePath 
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }

    // FIXED: Added shouldScroll parameter to control scrolling behavior
    function displayResults(products, shouldScroll = false) {
        results.innerHTML = '';
        
        if (!products || products.length === 0) {
            results.innerHTML = `
                <div style="grid-column: 1 / -1; text-align: center; padding: 3rem; background: var(--bg-primary); border-radius: var(--radius-lg); border: 1px solid var(--border-color);">
                    <h3 style="color: var(--text-primary); margin-bottom: 1rem;">🔍 No similar products found</h3>
                    <p style="color: var(--text-secondary);">Try uploading a different image or adjusting the similarity threshold.</p>
                </div>
            `;
            resultsSection.style.display = 'block';
            resultCount.textContent = '0';
            return;
        }
        
        products.forEach((product, index) => {
            const similarity = Math.round((product.similarity || 0) * 100);
            const cardHtml = `
                <div class="product-card hover-effect fade-in" style="animation-delay: ${index * 0.1}s">
                    <div class="card-img-container">
                        <img src="${escapeHtml(product.image)}" 
                             alt="${escapeHtml(product.name)}" 
                             loading="lazy"
                             onerror="this.src='data:image/svg+xml,<svg xmlns=\\"http://www.w3.org/2000/svg\\" width=\\"200\\" height=\\"200\\"><rect width=\\"200\\" height=\\"200\\" fill=\\"%23f1f5f9\\"/></svg>
                    </div>
                    <div class="card-body">
                        <div class="card-content">
                            <h3 class="card-title">${escapeHtml(product.name || 'Unknown Product')}</h3>
                            <p class="card-category">${escapeHtml(product.category || 'Uncategorized')}</p>
                        </div>
                        <div class="similarity-badge">✨ ${similarity}% Match</div>
                    </div>
                </div>
            `;
            results.innerHTML += cardHtml;
        });
        
        resultsSection.style.display = 'block';
        resultsSection.classList.add('fade-in');
        resultCount.textContent = products.length.toString();
        
        // FIXED: Only scroll if shouldScroll is true (initial search, not filtering)
        if (shouldScroll) {
            setTimeout(() => {
                resultsSection.scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'start' 
                });
            }, 100);
        }
    }

    // FIXED: Improved filter logic - no scrolling during filtering
    function filterResults(minSimilarity) {
        if (!currentResults || currentResults.length === 0) {
            return;
        }
        
        const filtered = currentResults.filter(product => {
            const similarity = product.similarity || 0;
            return similarity >= minSimilarity;
        });
        
        // FIXED: Pass false to disable scrolling during filtering
        displayResults(filtered, false);
    }

    function showLoading() {
        loading.style.display = 'block';
        resultsSection.style.display = 'none';
        filterSection.style.display = 'none';
        searchBtn.disabled = true;
        searchBtn.innerHTML = '<span>🔍 Searching...</span>';
    }

    function hideLoading() {
        loading.style.display = 'none';
        searchBtn.disabled = false;
        searchBtn.innerHTML = '<span>🔍 Find Similar Products</span>';
    }

    function showError(message) {
        errorMessage.textContent = message || 'An error occurred';
        errorAlert.style.display = 'block';
        errorAlert.classList.add('fade-in');
        
        // Auto-hide error after 5 seconds
        setTimeout(hideError, 5000);
    }

    function hideError() {
        errorAlert.style.display = 'none';
        errorAlert.classList.remove('fade-in');
    }

    function escapeHtml(text) {
        if (typeof text !== 'string') return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Initialize
    hideError();
    hideLoading();
});
