class InterfaceManager {
    constructor() {
        // Initialize core components
        this.propertyViewer = new PropertyViewer();
        this.floorManager = new FloorManager();
        this.situationPlan = new SituationPlan();
        this.propertyAnalyzer = new PropertyAnalyzer();
        this.chatAssistant = new ChatAssistant();

        // Initialize UI state
        this.currentView = '3d';
        this.isAnalyzing = false;
        this.currentFloor = '1';
        this.activeTools = new Set();

        // Initialize event listeners
        this.initializeEventListeners();
    }

    async initialize() {
        try {
            // Initialize 3D viewer
            await this.propertyViewer.init('3d-container');
            
            // Initialize situation plan
            await this.situationPlan.init('situation-container');
            
            // Initialize floor manager
            await this.floorManager.initialize();
            
            // Setup UI controls
            this.setupControls();
            
            // Setup drag and drop
            this.setupDragAndDrop();
            
        } catch (error) {
            console.error('Error initializing interface:', error);
            this.showError('Det oppstod en feil under initialisering av grensesnittet.');
        }
    }

    initializeEventListeners() {
        // Address search
        document.getElementById('searchAddress').addEventListener('click', () => {
            this.handleAddressSearch();
        });

        // URL analysis
        document.getElementById('analyzeUrl').addEventListener('click', () => {
            this.handleUrlAnalysis();
        });

        // View controls
        document.querySelectorAll('#viewButtons button').forEach(button => {
            button.addEventListener('click', () => {
                this.switchView(button.dataset.view);
            });
        });

        // Layer controls
        document.querySelectorAll('.layer-controls input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                this.toggleLayer(checkbox.id, checkbox.checked);
            });
        });

        // Save project button
        document.getElementById('saveProject').addEventListener('click', () => {
            this.saveProject();
        });

        // Start analysis button
        document.getElementById('startAnalysis').addEventListener('click', () => {
            this.startAnalysis();
        });

        // Window resize event
        window.addEventListener('resize', () => {
            this.handleResize();
        });
    }

    setupControls() {
        // Setup measurement tools
        const measurementTools = document.createElement('div');
        measurementTools.className = 'measurement-tools';
        measurementTools.innerHTML = `
            <button class="tool-button" data-tool="distance" title="Mål avstand">
                <i class="fas fa-ruler"></i>
            </button>
            <button class="tool-button" data-tool="area" title="Mål areal">
                <i class="fas fa-vector-square"></i>
            </button>
            <button class="tool-button" data-tool="height" title="Mål høyde">
                <i class="fas fa-arrows-alt-v"></i>
            </button>
        `;
        document.getElementById('3d-container').appendChild(measurementTools);

        // Add tool event listeners
        measurementTools.querySelectorAll('.tool-button').forEach(button => {
            button.addEventListener('click', () => {
                this.toggleTool(button.dataset.tool);
            });
        });
    }

    setupDragAndDrop() {
        const dropZone = document.getElementById('uploadArea');
        const fileInput = document.getElementById('floorplanUpload');

        // File input change event
        fileInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                this.handleFileUpload(file);
            }
        });

        // Drag and drop events
        dropZone.addEventListener('dragover', (event) => {
            event.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (event) => {
            event.preventDefault();
            dropZone.classList.remove('dragover');
            const file = event.dataTransfer.files[0];
            if (file) {
                this.handleFileUpload(file);
            }
        });

        // Click to upload
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });
    }

    async handleAddressSearch() {
        const addressInput = document.getElementById('addressInput');
        const address = addressInput.value.trim();

        if (!address) {
            this.showError('Vennligst skriv inn en adresse');
            return;
        }

        try {
            this.showLoading('Søker etter eiendom...');
            const propertyData = await this.propertyAnalyzer.analyzeProperty(address, 'address');
            await this.updatePropertyDisplay(propertyData);
        } catch (error) {
            console.error('Error searching address:', error);
            this.showError('Kunne ikke finne eiendommen. Vennligst sjekk adressen og prøv igjen.');
        } finally {
            this.hideLoading();
        }
    }

    async handleUrlAnalysis() {
        const urlInput = document.getElementById('urlInput');
        const url = urlInput.value.trim();

        if (!url) {
            this.showError('Vennligst skriv inn en URL');
            return;
        }

        try {
            this.showLoading('Analyserer nettside...');
            const propertyData = await this.propertyAnalyzer.analyzeProperty(url, 'url');
            await this.updatePropertyDisplay(propertyData);
        } catch (error) {
            console.error('Error analyzing URL:', error);
            this.showError('Kunne ikke analysere nettsiden. Vennligst sjekk URL-en og prøv igjen.');
        } finally {
            this.hideLoading();
        }
    }

    async handleFileUpload(file) {
        if (!file.type.startsWith('image/') && file.type !== 'application/pdf') {
            this.showError('Vennligst last opp en gyldig plantegning (bilde eller PDF)');
            return;
        }

        try {
            this.showLoading('Analyserer plantegning...');
            const propertyData = await this.propertyAnalyzer.analyzeProperty(file, 'floorplan');
            await this.updatePropertyDisplay(propertyData);
        } catch (error) {
            console.error('Error processing floorplan:', error);
            this.showError('Kunne ikke analysere plantegningen. Vennligst prøv igjen.');
        } finally {
            this.hideLoading();
        }
    }

    async updatePropertyDisplay(propertyData) {
        try {
            // Update 3D model
            await this.propertyViewer.loadProperty(propertyData);

            // Update floor plans
            await this.floorManager.loadFloorPlans(propertyData.floors);

            // Update situation plan
            await this.situationPlan.loadProperty(propertyData);

            // Update analysis results
            this.updateAnalysisResults(propertyData.analysis);

            // Notify chat assistant
            this.chatAssistant.updateContext({
                propertyData,
                analysisResults: propertyData.analysis
            });

            // Show success message
            this.showSuccess('Eiendom lastet inn vellykket');

        } catch (error) {
            console.error('Error updating property display:', error);
            throw error;
        }
    }

    switchView(view) {
        // Update button states
        document.querySelectorAll('#viewButtons button').forEach(button => {
            button.classList.toggle('active', button.dataset.view === view);
        });

        // Update view
        this.currentView = view;
        switch (view) {
            case '3d':
                this.propertyViewer.set3DView();
                break;
            case 'top':
                this.propertyViewer.setTopView();
                break;
            case 'front':
                this.propertyViewer.setFrontView();
                break;
        }
    }

    toggleLayer(layerId, visible) {
        switch (layerId) {
            case 'wallsLayer':
                this.propertyViewer.toggleWalls(visible);
                break;
            case 'furnitureLayer':
                this.propertyViewer.toggleFurniture(visible);
                break;
            case 'measurementsLayer':
                this.propertyViewer.toggleMeasurements(visible);
                break;
        }
    }

    toggleTool(tool) {
        const button = document.querySelector(`[data-tool="${tool}"]`);
        const isActive = this.activeTools.has(tool);

        if (isActive) {
            this.activeTools.delete(tool);
            button.classList.remove('active');
            this.propertyViewer.deactivateTool(tool);
        } else {
            this.activeTools.add(tool);
            button.classList.add('active');
            this.propertyViewer.activateTool(tool);
        }
    }

    async saveProject() {
        try {
            this.showLoading('Lagrer prosjekt...');

            const projectData = {
                propertyData: this.propertyViewer.getPropertyData(),
                floors: this.floorManager.exportFloorData(),
                situationPlan: this.situationPlan.exportSituationPlan(),
                analysis: this.propertyAnalyzer.getAnalysisResults()
            };

            const response = await fetch('/api/projects/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(projectData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.showSuccess('Prosjekt lagret vellykket');

        } catch (error) {
            console.error('Error saving project:', error);
            this.showError('Kunne ikke lagre prosjektet. Vennligst prøv igjen.');
        } finally {
            this.hideLoading();
        }
    }

    async startAnalysis() {
        if (this.isAnalyzing) return;

        try {
            this.isAnalyzing = true;
            this.showLoading('Utfører analyse...');

            const propertyData = this.propertyViewer.getPropertyData();
            const analysisResults = await this.propertyAnalyzer.performAnalysis(propertyData);

            // Update UI with results
            this.updateAnalysisResults(analysisResults);

            // Notify chat assistant
            this.chatAssistant.updateContext({
                analysisResults
            });

            this.showSuccess('Analyse fullført');

        } catch (error) {
            console.error('Error performing analysis:', error);
            this.showError('Kunne ikke fullføre analysen. Vennligst prøv igjen.');
        } finally {
            this.isAnalyzing = false;
            this.hideLoading();
        }
    }

    updateAnalysisResults(results) {
        const resultsContainer = document.getElementById('analysisResults');
        if (!resultsContainer) return;

        // Clear previous results
        resultsContainer.innerHTML = '';

        // Create results sections
        Object.entries(results).forEach(([category, data]) => {
            const section = this.createAnalysisSection(category, data);
            resultsContainer.appendChild(section);
        });

        // Animate sections
        resultsContainer.querySelectorAll('.analysis-section').forEach((section, index) => {
            setTimeout(() => {
                section.classList.add('visible');
            }, index * 100);
        });
    }

    createAnalysisSection(category, data) {
        const section = document.createElement('div');
        section.className = 'analysis-section';
        
        const title = document.createElement('h4');
        title.textContent = this.formatCategory(category);
        section.appendChild(title);

        const content = document.createElement('div');
        content.className = 'analysis-content';
        
        if (Array.isArray(data)) {
            content.appendChild(this.createList(data));
        } else if (typeof data === 'object') {
            content.appendChild(this.createDetailsList(data));
        } else {
            const p = document.createElement('p');
            p.textContent = data;
            content.appendChild(p);
        }

        section.appendChild(content);
        return section;
    }

    formatCategory(category) {
        return category
            .split(/(?=[A-Z])/)
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    createList(items) {
        const ul = document.createElement('ul');
        items.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            ul.appendChild(li);
        });
        return ul;
    }

    createDetailsList(data) {
        const dl = document.createElement('dl');
        Object.entries(data).forEach(([key, value]) => {
            const dt = document.createElement('dt');
            dt.textContent = this.formatCategory(key);
            
            const dd = document.createElement('dd');
            dd.textContent = value;
            
            dl.appendChild(dt);
            dl.appendChild(dd);
        });
        return dl;
    }

    handleResize() {
        // Update 3D viewer
        this.propertyViewer.onWindowResize();

        // Update situation plan
        this.situationPlan.onWindowResize();
    }

    showLoading(message) {
        const loadingOverlay = document.getElementById('loadingOverlay');
        const loadingText = loadingOverlay.querySelector('.loading-text');
        
        if (loadingText) {
            loadingText.textContent = message;
        }
        
        loadingOverlay.classList.add('active');
    }

    hideLoading() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        loadingOverlay.classList.remove('active');
    }

    showError(message) {
        // Implementation depends on your notification system
        console.error(message);
        // You could use a toast notification system here
    }

    showSuccess(message) {
        // Implementation depends on your notification system
        console.log(message);
        // You could use a toast notification system here
    }
}