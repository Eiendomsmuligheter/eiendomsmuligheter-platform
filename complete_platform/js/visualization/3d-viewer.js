class PropertyViewer {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.floors = new Map();
        this.currentFloor = '1';
        this.measurements = new Map();
        this.isInitialized = false;
        this.loadingManager = new THREE.LoadingManager();
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        // Bind methods
        this.init = this.init.bind(this);
        this.animate = this.animate.bind(this);
        this.onWindowResize = this.onWindowResize.bind(this);
        this.handleMouseMove = this.handleMouseMove.bind(this);
        this.handleClick = this.handleClick.bind(this);
    }

    async init(containerId) {
        if (this.isInitialized) return;

        // Get container
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error('Container not found:', containerId);
            return;
        }

        // Setup scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x070b15);

        // Setup camera
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        this.camera.position.set(10, 10, 10);

        // Setup renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);

        // Setup controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        // Setup lighting
        this.setupLighting();

        // Setup grid
        this.setupGrid();

        // Add event listeners
        window.addEventListener('resize', this.onWindowResize);
        this.renderer.domElement.addEventListener('mousemove', this.handleMouseMove);
        this.renderer.domElement.addEventListener('click', this.handleClick);

        // Start animation loop
        this.animate();
        this.isInitialized = true;
    }

    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);

        // Directional light
        const dirLight = new THREE.DirectionalLight(0xffffff, 1);
        dirLight.position.set(10, 10, 10);
        dirLight.castShadow = true;
        dirLight.shadow.mapSize.width = 2048;
        dirLight.shadow.mapSize.height = 2048;
        this.scene.add(dirLight);

        // Add rim light for better depth perception
        const rimLight = new THREE.DirectionalLight(0x0ED3CF, 0.3);
        rimLight.position.set(-10, 5, -10);
        this.scene.add(rimLight);
    }

    setupGrid() {
        const gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
        this.scene.add(gridHelper);
    }

    async loadFloorModel(floorLevel, modelPath) {
        return new Promise((resolve, reject) => {
            const loader = new THREE.GLTFLoader(this.loadingManager);
            
            loader.load(modelPath,
                (gltf) => {
                    const model = gltf.scene;
                    model.traverse((child) => {
                        if (child.isMesh) {
                            child.castShadow = true;
                            child.receiveShadow = true;
                        }
                    });
                    
                    this.floors.set(floorLevel, model);
                    resolve(model);
                },
                undefined,
                (error) => {
                    console.error('Error loading model:', error);
                    reject(error);
                }
            );
        });
    }

    switchFloor(floorLevel) {
        // Hide all floors
        this.floors.forEach(floor => {
            floor.visible = false;
        });

        // Show selected floor
        const selectedFloor = this.floors.get(floorLevel);
        if (selectedFloor) {
            selectedFloor.visible = true;
            this.currentFloor = floorLevel;
            this.updateFloorIndicator();
        }
    }

    updateFloorIndicator() {
        const indicator = document.querySelector('.floor-indicator');
        if (indicator) {
            indicator.textContent = `Etasje: ${this.currentFloor}`;
        }
    }

    addMeasurement(start, end) {
        const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
        const material = new THREE.LineBasicMaterial({ color: 0x0ED3CF });
        const line = new THREE.Line(geometry, material);
        
        // Calculate distance
        const distance = start.distanceTo(end);
        
        // Create measurement label
        const midPoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
        const label = this.createMeasurementLabel(distance.toFixed(2) + 'm');
        label.position.copy(midPoint);
        
        // Store measurement
        const measurementId = Date.now().toString();
        this.measurements.set(measurementId, { line, label });
        
        // Add to scene
        this.scene.add(line);
        this.scene.add(label);
        
        return measurementId;
    }

    createMeasurementLabel(text) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        context.font = 'Bold 20px Arial';
        context.fillStyle = 'white';
        context.fillText(text, 0, 20);
        
        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ map: texture });
        return new THREE.Sprite(material);
    }

    removeMeasurement(measurementId) {
        const measurement = this.measurements.get(measurementId);
        if (measurement) {
            this.scene.remove(measurement.line);
            this.scene.remove(measurement.label);
            this.measurements.delete(measurementId);
        }
    }

    handleMouseMove(event) {
        // Calculate mouse position in normalized device coordinates
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        // Update raycaster
        this.raycaster.setFromCamera(this.mouse, this.camera);

        // Check for intersections with objects
        const intersects = this.raycaster.intersectObjects(this.scene.children, true);
        
        // Update hover effects
        this.updateHoverEffects(intersects);
    }

    handleClick(event) {
        // Handle measurements and object selection
        const intersects = this.raycaster.intersectObjects(this.scene.children, true);
        if (intersects.length > 0) {
            const selected = intersects[0].object;
            this.selectObject(selected);
        }
    }

    selectObject(object) {
        // Remove previous selection
        this.scene.traverse((child) => {
            if (child.isMesh) {
                child.material.emissive = new THREE.Color(0x000000);
            }
        });

        // Highlight selected object
        if (object.isMesh) {
            object.material.emissive = new THREE.Color(0x0ED3CF);
            this.showObjectInfo(object);
        }
    }

    showObjectInfo(object) {
        // Display object information in the measurement overlay
        const overlay = document.getElementById('measurementInfo');
        if (overlay) {
            overlay.textContent = `${object.name}: ${object.userData.type || 'Unknown'}`;
            overlay.classList.add('active');
        }
    }

    updateHoverEffects(intersects) {
        // Update cursor and hover information
        if (intersects.length > 0) {
            this.renderer.domElement.style.cursor = 'pointer';
            const hovered = intersects[0].object;
            this.showHoverInfo(hovered, intersects[0].point);
        } else {
            this.renderer.domElement.style.cursor = 'default';
            this.hideHoverInfo();
        }
    }

    showHoverInfo(object, position) {
        const hoverInfo = document.querySelector('.hover-info');
        if (hoverInfo && object.userData.info) {
            hoverInfo.textContent = object.userData.info;
            hoverInfo.style.display = 'block';
            
            // Convert 3D position to screen coordinates
            const screenPosition = this.getScreenPosition(position);
            hoverInfo.style.left = screenPosition.x + 'px';
            hoverInfo.style.top = screenPosition.y + 'px';
        }
    }

    hideHoverInfo() {
        const hoverInfo = document.querySelector('.hover-info');
        if (hoverInfo) {
            hoverInfo.style.display = 'none';
        }
    }

    getScreenPosition(position) {
        const vector = position.clone();
        vector.project(this.camera);

        const rect = this.renderer.domElement.getBoundingClientRect();
        const x = (vector.x + 1) * rect.width / 2 + rect.left;
        const y = (-vector.y + 1) * rect.height / 2 + rect.top;

        return { x, y };
    }

    onWindowResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    animate() {
        requestAnimationFrame(this.animate);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    dispose() {
        // Clean up resources
        window.removeEventListener('resize', this.onWindowResize);
        this.renderer.domElement.removeEventListener('mousemove', this.handleMouseMove);
        this.renderer.domElement.removeEventListener('click', this.handleClick);
        
        this.scene.traverse((object) => {
            if (object.geometry) object.geometry.dispose();
            if (object.material) {
                if (Array.isArray(object.material)) {
                    object.material.forEach(material => material.dispose());
                } else {
                    object.material.dispose();
                }
            }
        });
        
        this.renderer.dispose();
        this.isInitialized = false;
    }
}