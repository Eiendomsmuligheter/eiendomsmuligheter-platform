class SituationPlan {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.layers = new Map();
        this.propertyBoundary = null;
        this.buildingFootprint = null;
        this.isInitialized = false;
        
        // Bind methods
        this.init = this.init.bind(this);
        this.animate = this.animate.bind(this);
        this.onWindowResize = this.onWindowResize.bind(this);
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
        this.camera = new THREE.OrthographicCamera(
            width / -2, width / 2,
            height / 2, height / -2,
            1, 1000
        );
        this.camera.position.set(0, 100, 0);
        this.camera.lookAt(0, 0, 0);

        // Setup renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.shadowMap.enabled = true;
        this.container.appendChild(this.renderer.domElement);

        // Setup controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.maxPolarAngle = Math.PI / 2;

        // Setup lighting
        this.setupLighting();

        // Setup grid
        this.setupGrid();

        // Initialize layers
        this.initializeLayers();

        // Add event listeners
        window.addEventListener('resize', this.onWindowResize);

        // Start animation loop
        this.animate();
        this.isInitialized = true;
    }

    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        // Directional light for shadows
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(50, 100, 50);
        dirLight.castShadow = true;
        dirLight.shadow.camera.left = -100;
        dirLight.shadow.camera.right = 100;
        dirLight.shadow.camera.top = 100;
        dirLight.shadow.camera.bottom = -100;
        dirLight.shadow.mapSize.width = 2048;
        dirLight.shadow.mapSize.height = 2048;
        this.scene.add(dirLight);
    }

    setupGrid() {
        const gridHelper = new THREE.GridHelper(200, 20, 0x444444, 0x222222);
        this.scene.add(gridHelper);
    }

    initializeLayers() {
        // Initialize layer groups
        this.layers.set('boundaries', new THREE.Group());
        this.layers.set('parking', new THREE.Group());
        this.layers.set('vegetation', new THREE.Group());
        this.layers.set('buildings', new THREE.Group());
        this.layers.set('terrain', new THREE.Group());

        // Add layers to scene
        this.layers.forEach(layer => {
            this.scene.add(layer);
        });

        // Setup layer controls
        this.setupLayerControls();
    }

    setupLayerControls() {
        const controls = document.querySelector('.situation-controls');
        if (!controls) return;

        controls.querySelectorAll('button').forEach(button => {
            const layerName = button.dataset.layer;
            button.addEventListener('click', () => {
                const layer = this.layers.get(layerName);
                if (layer) {
                    layer.visible = !layer.visible;
                    button.classList.toggle('active');
                }
            });
        });
    }

    setPropertyBoundary(coordinates) {
        // Remove existing boundary
        if (this.propertyBoundary) {
            this.layers.get('boundaries').remove(this.propertyBoundary);
        }

        // Create boundary line
        const points = coordinates.map(coord => new THREE.Vector3(coord.x, 0, coord.z));
        points.push(points[0]); // Close the loop

        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
            color: 0x0ED3CF,
            linewidth: 2
        });

        this.propertyBoundary = new THREE.Line(geometry, material);
        this.layers.get('boundaries').add(this.propertyBoundary);

        // Create boundary area visualization
        const shape = new THREE.Shape();
        shape.moveTo(coordinates[0].x, coordinates[0].z);
        for (let i = 1; i < coordinates.length; i++) {
            shape.lineTo(coordinates[i].x, coordinates[i].z);
        }

        const areaGeometry = new THREE.ShapeGeometry(shape);
        const areaMaterial = new THREE.MeshBasicMaterial({
            color: 0x0ED3CF,
            transparent: true,
            opacity: 0.1,
            side: THREE.DoubleSide
        });

        const areaMesh = new THREE.Mesh(areaGeometry, areaMaterial);
        areaMesh.rotation.x = -Math.PI / 2;
        this.layers.get('boundaries').add(areaMesh);
    }

    setBuildingFootprint(coordinates) {
        // Remove existing footprint
        if (this.buildingFootprint) {
            this.layers.get('buildings').remove(this.buildingFootprint);
        }

        // Create building footprint
        const shape = new THREE.Shape();
        shape.moveTo(coordinates[0].x, coordinates[0].z);
        for (let i = 1; i < coordinates.length; i++) {
            shape.lineTo(coordinates[i].x, coordinates[i].z);
        }

        const geometry = new THREE.ExtrudeGeometry(shape, {
            depth: 0.2,
            bevelEnabled: false
        });

        const material = new THREE.MeshPhongMaterial({
            color: 0x2D46B9,
            transparent: true,
            opacity: 0.8
        });

        this.buildingFootprint = new THREE.Mesh(geometry, material);
        this.buildingFootprint.rotation.x = -Math.PI / 2;
        this.layers.get('buildings').add(this.buildingFootprint);
    }

    addParkingArea(coordinates, type = 'surface') {
        const shape = new THREE.Shape();
        shape.moveTo(coordinates[0].x, coordinates[0].z);
        for (let i = 1; i < coordinates.length; i++) {
            shape.lineTo(coordinates[i].x, coordinates[i].z);
        }

        const geometry = new THREE.ShapeGeometry(shape);
        const material = new THREE.MeshPhongMaterial({
            color: type === 'surface' ? 0x666666 : 0x444444,
            transparent: true,
            opacity: 0.8
        });

        const parkingArea = new THREE.Mesh(geometry, material);
        parkingArea.rotation.x = -Math.PI / 2;
        parkingArea.position.y = type === 'surface' ? 0.1 : -0.1;
        
        this.layers.get('parking').add(parkingArea);

        // Add parking symbols
        this.addParkingSymbols(coordinates, type);
    }

    addParkingSymbols(coordinates, type) {
        // Calculate parking spots based on area
        const area = this.calculateArea(coordinates);
        const spotsCount = Math.floor(area / 12.5); // Standard parking spot is about 12.5m²

        // Add parking spot markers
        for (let i = 0; i < spotsCount; i++) {
            const symbol = this.createParkingSymbol();
            // Position symbols evenly within the parking area
            // This is a simplified placement - would need more complex logic for real applications
            symbol.position.set(
                coordinates[0].x + (i % 3) * 2.5,
                type === 'surface' ? 0.15 : -0.05,
                coordinates[0].z + Math.floor(i / 3) * 5
            );
            this.layers.get('parking').add(symbol);
        }
    }

    createParkingSymbol() {
        const geometry = new THREE.PlaneGeometry(2, 4);
        const material = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            side: THREE.DoubleSide
        });
        const symbol = new THREE.Mesh(geometry, material);
        symbol.rotation.x = -Math.PI / 2;
        return symbol;
    }

    addVegetation(position, type = 'tree') {
        let vegetation;
        
        if (type === 'tree') {
            vegetation = this.createTree();
        } else if (type === 'bush') {
            vegetation = this.createBush();
        } else {
            vegetation = this.createGrassArea(position);
        }

        vegetation.position.copy(position);
        this.layers.get('vegetation').add(vegetation);
    }

    createTree() {
        const group = new THREE.Group();

        // Tree trunk
        const trunkGeometry = new THREE.CylinderGeometry(0.2, 0.3, 2, 8);
        const trunkMaterial = new THREE.MeshPhongMaterial({ color: 0x8B4513 });
        const trunk = new THREE.Mesh(trunkGeometry, trunkMaterial);
        trunk.position.y = 1;
        group.add(trunk);

        // Tree crown
        const crownGeometry = new THREE.SphereGeometry(1.5, 8, 8);
        const crownMaterial = new THREE.MeshPhongMaterial({ color: 0x228B22 });
        const crown = new THREE.Mesh(crownGeometry, crownMaterial);
        crown.position.y = 2.5;
        group.add(crown);

        return group;
    }

    createBush() {
        const geometry = new THREE.SphereGeometry(0.7, 8, 8);
        const material = new THREE.MeshPhongMaterial({ color: 0x228B22 });
        const bush = new THREE.Mesh(geometry, material);
        bush.position.y = 0.7;
        return bush;
    }

    createGrassArea(position) {
        // Create a grass patch using a textured plane
        const geometry = new THREE.PlaneGeometry(5, 5);
        const material = new THREE.MeshPhongMaterial({
            color: 0x90EE90,
            side: THREE.DoubleSide
        });
        const grass = new THREE.Mesh(geometry, material);
        grass.rotation.x = -Math.PI / 2;
        grass.position.y = 0.01; // Slightly above ground to prevent z-fighting
        return grass;
    }

    calculateArea(coordinates) {
        let area = 0;
        for (let i = 0; i < coordinates.length; i++) {
            const j = (i + 1) % coordinates.length;
            area += coordinates[i].x * coordinates[j].z;
            area -= coordinates[j].x * coordinates[i].z;
        }
        return Math.abs(area) / 2;
    }

    onWindowResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.left = width / -2;
        this.camera.right = width / 2;
        this.camera.top = height / 2;
        this.camera.bottom = height / -2;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height);
    }

    animate() {
        requestAnimationFrame(this.animate);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    dispose() {
        window.removeEventListener('resize', this.onWindowResize);
        
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

    // Export situation plan data
    exportSituationPlan() {
        const exportData = {
            propertyBoundary: this.propertyBoundary ? {
                type: 'boundary',
                coordinates: this.extractCoordinates(this.propertyBoundary)
            } : null,
            buildingFootprint: this.buildingFootprint ? {
                type: 'building',
                coordinates: this.extractCoordinates(this.buildingFootprint)
            } : null,
            parking: Array.from(this.layers.get('parking').children).map(parking => ({
                type: 'parking',
                coordinates: this.extractCoordinates(parking),
                parkingType: parking.userData.type
            })),
            vegetation: Array.from(this.layers.get('vegetation').children).map(veg => ({
                type: 'vegetation',
                position: {
                    x: veg.position.x,
                    y: veg.position.y,
                    z: veg.position.z
                },
                vegetationType: veg.userData.type
            }))
        };

        return exportData;
    }

    extractCoordinates(object) {
        if (!object) return null;
        
        // Extract vertices from geometry
        const positions = object.geometry.attributes.position.array;
        const coordinates = [];
        
        for (let i = 0; i < positions.length; i += 3) {
            coordinates.push({
                x: positions[i],
                y: positions[i + 1],
                z: positions[i + 2]
            });
        }
        
        return coordinates;
    }
}