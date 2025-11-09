class ImageHub {
    constructor(config) {
        this.endpoints = config;
        this.sessionId = null;
        this.images = [];
        this.selected = new Set();
        this.processedMap = new Map();
        this.currentPreviewIndex = 0;
        this.totalBytes = 0;
        this.overlay = document.getElementById('loading-overlay');
        this.overlayMessage = document.getElementById('loading-message');
        this.init();
    }

    init() {
        this.fileInput = document.getElementById('file-input');
        this.uploadZone = document.getElementById('upload-zone');
        this.imageGrid = document.getElementById('image-grid');
        this.fileCountEl = document.getElementById('file-count');
        this.totalSizeEl = document.getElementById('total-size');
        this.originalPreview = document.getElementById('original-preview');
        this.processedPreview = document.getElementById('processed-preview');
        this.previewIndex = document.getElementById('preview-index');
        this.metricPsnr = document.getElementById('metric-psnr');
        this.metricSsim = document.getElementById('metric-ssim');
        this.autoButton = document.getElementById('auto-mode');
        this.autoInput = document.getElementById('auto-count');
        this.autoResult = document.getElementById('auto-result');
        this.autoResultText = document.getElementById('auto-result-text');
        this.autoResultLink = document.getElementById('auto-result-link');

        document.getElementById('upload-trigger').addEventListener('click', () => this.fileInput.click());
        document.getElementById('process-selected').addEventListener('click', () => this.processSelection());
        document.getElementById('apply-all').addEventListener('click', () => this.processSelection(true));
        document.getElementById('reset-config').addEventListener('click', () => this.resetConfig());
        document.getElementById('clear-images').addEventListener('click', () => this.clearImages());
        document.getElementById('download-zip').addEventListener('click', () => this.downloadZip());
        document.getElementById('download-selected').addEventListener('click', () => this.downloadSelected());
        document.getElementById('prev-image').addEventListener('click', () => this.showPreview(this.currentPreviewIndex - 1));
        document.getElementById('next-image').addEventListener('click', () => this.showPreview(this.currentPreviewIndex + 1));
        if (this.autoButton) {
            this.autoButton.addEventListener('click', () => this.runAutoMode());
        }

        this.fileInput.addEventListener('change', (evt) => {
            const files = Array.from(evt.target.files);
            if (files.length) {
                this.uploadImages(files);
            }
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            this.uploadZone.addEventListener(eventName, (evt) => {
                evt.preventDefault();
                evt.stopPropagation();
                this.uploadZone.classList.add('drag-over');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            this.uploadZone.addEventListener(eventName, (evt) => {
                evt.preventDefault();
                evt.stopPropagation();
                this.uploadZone.classList.remove('drag-over');
            });
        });

        this.uploadZone.addEventListener('drop', (evt) => {
            const files = Array.from(evt.dataTransfer.files).filter(file => file.type.startsWith('image/'));
            if (files.length) {
                this.uploadImages(files);
            }
        });
    }

    async uploadImages(files) {
        const form = new FormData();
        files.forEach(file => form.append('images', file));
        if (this.sessionId) {
            form.append('session_id', this.sessionId);
        }
        this.toggleLoading(true, 'Uploading images…');
        try {
            const response = await fetch(this.endpoints.uploadUrl, { method: 'POST', body: form });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Upload failed');
            this.sessionId = data.session_id;
            data.images.forEach(img => {
                if (!this.images.find(existing => existing.id === img.id)) {
                    this.images.push(img);
                    this.totalBytes += img.size_bytes || 0;
                }
            });
            this.renderGrid();
        } catch (error) {
            alert(error.message);
        } finally {
            this.toggleLoading(false);
        }
    }

    renderGrid() {
        this.imageGrid.innerHTML = '';
        this.fileCountEl.textContent = `${this.images.length} images`;
        this.totalSizeEl.textContent = `${(this.totalBytes / (1024 * 1024)).toFixed(2)} MB`;
        this.images.forEach((img, index) => {
            const card = document.createElement('article');
            card.className = 'image-card';
            card.dataset.imageId = img.id;
            if (this.selected.has(img.id)) {
                card.classList.add('selected');
            }
            card.innerHTML = `
                <img src="${img.thumbnail_url}" alt="${img.filename}">
                <footer>
                    <strong>${img.filename}</strong>
                    <div>${img.dimensions}</div>
                    <div>${img.size}</div>
                </footer>
            `;
            card.addEventListener('click', () => this.toggleSelection(img.id));
            card.addEventListener('dblclick', () => {
                this.currentPreviewIndex = index;
                this.showPreview(index);
            });
            this.imageGrid.appendChild(card);
        });
    }

    toggleSelection(imageId) {
        if (this.selected.has(imageId)) {
            this.selected.delete(imageId);
        } else {
            this.selected.add(imageId);
        }
        this.renderGrid();
    }

    resetConfig() {
        document.querySelectorAll('[data-op]').forEach(input => { input.checked = false; });
        document.querySelectorAll('[data-param]').forEach(input => {
            if (input.type === 'range') {
                input.value = input.defaultValue || input.value;
            } else {
                input.value = '';
            }
        });
    }

    gatherConfig() {
        const config = {};
        document.querySelectorAll('[data-op]').forEach(input => {
            const op = input.dataset.op;
            config[op] = { enabled: input.checked };
        });
        document.querySelectorAll('[data-param]').forEach(input => {
            const [op, param] = input.dataset.param.split('.');
            config[op] = config[op] || {};
            if (input.type === 'range' || input.type === 'number') {
                config[op][param] = input.value !== '' ? Number(input.value) : null;
            } else {
                config[op][param] = input.value;
            }
        });
        return config;
    }

    async processSelection(applyAll = false) {
        if (!this.sessionId) {
            alert('Upload images first.');
            return;
        }
        const targets = applyAll ? this.images.map(img => img.id) : Array.from(this.selected);
        if (!targets.length) {
            targets = this.images.map(img => img.id);
        }
        this.toggleLoading(true, 'Processing images…');
        try {
            const response = await fetch(this.endpoints.processUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    image_ids: targets,
                    operations: this.gatherConfig(),
                }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Processing failed');
            data.results.forEach(result => {
                this.processedMap.set(result.id, result);
            });
            if (data.results.length) {
                this.currentPreviewIndex = this.images.findIndex(img => img.id === data.results[0].id);
                this.showPreview(this.currentPreviewIndex);
            }
        } catch (error) {
            alert(error.message);
        } finally {
            this.toggleLoading(false);
        }
    }

    showPreview(index) {
        if (!this.images.length) return;
        if (index < 0) index = this.images.length - 1;
        if (index >= this.images.length) index = 0;
        this.currentPreviewIndex = index;
        const image = this.images[index];
        const processed = this.processedMap.get(image.id);
        this.originalPreview.src = processed?.original_url || image.original_url;
        this.processedPreview.src = processed?.processed_url || image.original_url;
        this.metricPsnr.textContent = processed?.metrics?.psnr ?? '—';
        this.metricSsim.textContent = processed?.metrics?.ssim ?? '—';
        this.previewIndex.textContent = `${index + 1} / ${this.images.length}`;
    }

    async downloadZip() {
        if (!this.sessionId || !this.processedMap.size) {
            alert('Process some images first.');
            return;
        }
        this.toggleLoading(true, 'Preparing ZIP…');
        try {
            const response = await fetch(this.endpoints.downloadUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    image_ids: Array.from(this.processedMap.keys()),
                }),
            });
            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.error || 'Download failed');
            }
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const anchor = document.createElement('a');
            anchor.href = url;
            anchor.download = `autoda_image_hub_${Date.now()}.zip`;
            anchor.click();
            URL.revokeObjectURL(url);
        } catch (error) {
            alert(error.message);
        } finally {
            this.toggleLoading(false);
        }
    }

    downloadSelected() {
        if (!this.selected.size) {
            alert('Select images to download.');
            return;
        }
        this.selected.forEach(id => {
            const processed = this.processedMap.get(id);
            if (processed) {
                const anchor = document.createElement('a');
                anchor.href = processed.processed_url;
                anchor.download = `${id}_processed.png`;
                anchor.click();
            }
        });
    }

    async runAutoMode() {
        if (!this.sessionId || !this.images.length) {
            alert('Upload images first.');
            return;
        }
        const desired = parseInt(this.autoInput.value, 10);
        if (!desired || desired <= 0) {
            alert('Enter the number of augmented images you need.');
            return;
        }
        const width = parseInt(document.getElementById('auto-width').value, 10) || 512;
        const height = parseInt(document.getElementById('auto-height').value, 10) || 512;
        const preset = document.getElementById('auto-preset').value;
        this.toggleLoading(true, 'Running Auto Mode…');
        try {
            const response = await fetch(this.endpoints.autoUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    count: desired,
                    width,
                    height,
                    preset,
                }),
            });
            const raw = await response.text();
            let data;
            try {
                data = JSON.parse(raw);
            } catch (parseErr) {
                throw new Error(raw.slice(0, 180) || 'Auto Mode failed');
            }
            if (!response.ok) throw new Error(data.error || 'Auto Mode failed');
            if (this.autoResult) {
                this.autoResult.hidden = false;
                this.autoResultText.textContent = `Generated ${data.count} augmented images.`;
                this.autoResultLink.href = data.zip_url;
                this.autoResultLink.click();
            }
            alert(`Auto Mode generated ${data.count} images. Download started.`);
        } catch (error) {
            alert(error.message);
        } finally {
            this.toggleLoading(false);
        }
    }

    clearImages() {
        this.images = [];
        this.selected.clear();
        this.processedMap.clear();
        this.totalBytes = 0;
        this.imageGrid.innerHTML = '';
        this.fileCountEl.textContent = '0 images';
        this.totalSizeEl.textContent = '0 MB';
        this.originalPreview.src = '';
        this.processedPreview.src = '';
        this.metricPsnr.textContent = '—';
        this.metricSsim.textContent = '—';
        this.previewIndex.textContent = '0 / 0';
    }

    toggleLoading(show, message = 'Working…') {
        if (!this.overlay) return;
        this.overlay.hidden = !show;
        if (show) this.overlayMessage.textContent = message;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    if (window.IMAGE_HUB_CONFIG) {
        new ImageHub(window.IMAGE_HUB_CONFIG);
    }
});
