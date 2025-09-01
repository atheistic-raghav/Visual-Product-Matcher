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
                    <img src="${src}" alt="Preview" onerror="this.style.display='none';">
                </div>
                <div class="preview-card-body">
                    <h3 class="preview-title">${escapeHtml(title)}</h3>
                </div>
            </div>
        `;
        imagePreview.style.display = 'block';
    }

    async function uploadAndSearch() {
        if (searchInProgress) return;
        
        setSearchInProgress(true);
        hideError();
        showLoading();
        hideResults();

        try {
            let uploadData;

            // Handle file upload
            if (fileInput.files?.[0]) {
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                console.log('📤 Uploading file...');
                const uploadResponse = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!uploadResponse.ok) {
                    throw new Error(`Upload failed: ${uploadResponse.status}`);
                }

                uploadData = await uploadResponse.json();

            } 
            // Handle URL upload
            else if (urlInput.value.trim()) {
                console.log('🌐 Processing URL...');
                const uploadResponse = await fetch('/upload', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image_url: urlInput.value.trim()
                    })
                });

                if (!uploadResponse.ok) {
                    throw new Error(`URL processing failed: ${uploadResponse.status}`);
                }

                uploadData = await uploadResponse.json();
            }

            // Check upload success
            if (!uploadData?.success) {
                throw new Error(uploadData?.message || 'Upload failed');
            }

            console.log('✅ Upload successful:', uploadData);

            // Perform search
            console.log('🔍 Starting similarity search...');
            const searchResponse = await fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_path: uploadData.image_path
                })
            });

            if (!searchResponse.ok) {
                throw new Error(`Search failed: ${searchResponse.status}`);
            }

            const searchData = await searchResponse.json();

            if (!searchData.success) {
                throw new Error(searchData.message || 'Search failed');
            }

            console.log('✅ Search completed:', searchData);

            // Display results
            currentResults = searchData.results || [];
            displayResults(currentResults);

            if (currentResults.length > 0) {
                filterSection.style.display = 'block';
            }

        } catch (error) {
            console.error('❌ Search error:', error);
            showError(`Search failed: ${error.message}`);
        } finally {
            setSearchInProgress(false);
            hideLoading();
        }
    }

    function displayResults(products) {
        if (!products || products.length === 0) {
            results.innerHTML = `
                <div style="grid-column: 1/-1; text-align: center; padding: 3rem; color: var(--text-secondary);">
                    <h3>No similar products found</h3>
                    <p>Try uploading a different image or adjusting the similarity threshold.</p>
                </div>
            `;
            resultCount.textContent = '0 results';
        } else {
            results.innerHTML = products.map(product => `
                <div class="product-card fade-in">
                    <div class="card-img-container">
                        <img src="${product.image}" 
                             alt="${escapeHtml(product.name)}"
                             loading="lazy"
                             onerror="this.style.display='none'; this.parentNode.innerHTML='<div style=\\'display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-muted);\\'>🖼️ Image not available</div>';">
                    </div>
                    <div class="card-body">
                        <div class="card-content">
                            <h3 class="card-title">${escapeHtml(product.name)}</h3>
                            <p class="card-category">${escapeHtml(product.category || 'Uncategorized')}</p>
                        </div>
                        <div class="similarity-badge">
                            ${Math.round(product.similarity * 100)}% Match
                        </div>
                    </div>
                </div>
            `).join('');
            
            resultCount.textContent = `${products.length} result${products.length !== 1 ? 's' : ''}`;
        }
        
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function filterResults(threshold) {
        if (!currentResults || currentResults.length === 0) return;

        const filteredResults = currentResults.filter(product => 
            product.similarity >= threshold
        );

        displayResults(filteredResults);
    }

    function setSearchInProgress(inProgress) {
        searchInProgress = inProgress;
        searchBtn.disabled = inProgress;
        searchBtn.textContent = inProgress ? '⏳ Searching...' : '🔍 Find Similar Products';
    }

    function showLoading() {
        loading.style.display = 'block';
        loading.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function hideLoading() {
        loading.style.display = 'none';
    }

    function showResults() {
        resultsSection.style.display = 'block';
    }

    function hideResults() {
        resultsSection.style.display = 'none';
        filterSection.style.display = 'none';
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorAlert.style.display = 'block';
        errorAlert.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function hideError() {
        errorAlert.style.display = 'none';
    }

    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});