class FloorManager {
    constructor() {
        this.floors = new Map();
        this.currentFloor = null;
        this.floorHeights = new Map();
        this.floorPlans = new Map();
        this.floorTypes = ['basement', '1', '2', '3', '4', 'roof'];
    }

    async initialize() {
        // Set default floor heights (in meters)
        this.floorHeights.set('basement', -2.8);
        this.floorHeights.set('1', 0);
        this.floorHeights.set('2', 2.8);
        this.floorHeights.set('3', 5.6);
        this.floorHeights.set('4', 8.4);
        this.floorHeights.set('roof', 11.2);

        // Initialize floor buttons
        this.initializeFloorControls();
    }

    initializeFloorControls() {
        const floorButtons = document.getElementById('floorButtons');
        if (!floorButtons) return;

        floorButtons.innerHTML = '';
        this.floorTypes.reverse().forEach(floor => {
            const button = document.createElement('button');
            button.className = 'btn btn-outline-light';
            button.dataset.floor = floor;
            button.textContent = this.getFloorDisplayName(floor);
            
            button.addEventListener('click', () => this.switchToFloor(floor));
            floorButtons.appendChild(button);
        });
    }

    getFloorDisplayName(floor) {
        const displayNames = {
            'basement': 'Kjeller',
            'roof': 'Tak'
        };
        return displayNames[floor] || `${floor}. Etasje`;
    }

    async loadFloorPlan(floor, planData) {
        try {
            // Create floor container
            const floorContainer = new THREE.Group();
            floorContainer.name = `floor-${floor}`;

            // Set floor height
            const height = this.floorHeights.get(floor) || 0;
            floorContainer.position.y = height;

            // Process floor plan data
            await this.processFloorPlanData(floorContainer, planData);

            // Store floor plan
            this.floors.set(floor, floorContainer);
            this.floorPlans.set(floor, planData);

            return floorContainer;
        } catch (error) {
            console.error(`Error loading floor plan for floor ${floor}:`, error);
            throw error;
        }
    }

    async processFloorPlanData(container, planData) {
        // Create walls
        if (planData.walls) {
            planData.walls.forEach(wall => {
                const wallMesh = this.createWall(wall);
                container.add(wallMesh);
            });
        }

        // Create windows
        if (planData.windows) {
            planData.windows.forEach(window => {
                const windowMesh = this.createWindow(window);
                container.add(windowMesh);
            });
        }

        // Create doors
        if (planData.doors) {
            planData.doors.forEach(door => {
                const doorMesh = this.createDoor(door);
                container.add(doorMesh);
            });
        }

        // Create rooms
        if (planData.rooms) {
            planData.rooms.forEach(room => {
                const roomMesh = this.createRoom(room);
                container.add(roomMesh);
            });
        }
    }

    createWall(wallData) {
        const wallGeometry = new THREE.BoxGeometry(
            wallData.length,
            wallData.height || 2.8,
            wallData.thickness || 0.2
        );
        const wallMaterial = new THREE.MeshPhongMaterial({
            color: 0xcccccc,
            transparent: true,
            opacity: 0.9
        });
        const wall = new THREE.Mesh(wallGeometry, wallMaterial);
        
        // Position and rotate wall
        wall.position.set(wallData.x, wallData.height / 2, wallData.z);
        wall.rotation.y = wallData.rotation || 0;
        
        // Add metadata
        wall.userData = {
            type: 'wall',
            info: `Vegg - Lengde: ${wallData.length}m, Høyde: ${wallData.height}m`
        };

        return wall;
    }

    createWindow(windowData) {
        const windowGeometry = new THREE.BoxGeometry(
            windowData.width,
            windowData.height,
            0.1
        );
        const windowMaterial = new THREE.MeshPhongMaterial({
            color: 0x88ccff,
            transparent: true,
            opacity: 0.4
        });
        const window = new THREE.Mesh(windowGeometry, windowMaterial);
        
        window.position.set(windowData.x, windowData.y, windowData.z);
        window.rotation.y = windowData.rotation || 0;
        
        window.userData = {
            type: 'window',
            info: `Vindu - Bredde: ${windowData.width}m, Høyde: ${windowData.height}m`
        };

        return window;
    }

    createDoor(doorData) {
        const doorGeometry = new THREE.BoxGeometry(
            doorData.width,
            doorData.height,
            0.1
        );
        const doorMaterial = new THREE.MeshPhongMaterial({
            color: 0x8b4513
        });
        const door = new THREE.Mesh(doorGeometry, doorMaterial);
        
        door.position.set(doorData.x, doorData.y, doorData.z);
        door.rotation.y = doorData.rotation || 0;
        
        door.userData = {
            type: 'door',
            info: `Dør - Bredde: ${doorData.width}m, Høyde: ${doorData.height}m`
        };

        return door;
    }

    createRoom(roomData) {
        // Create room floor
        const shape = new THREE.Shape();
        shape.moveTo(roomData.points[0].x, roomData.points[0].y);
        for (let i = 1; i < roomData.points.length; i++) {
            shape.lineTo(roomData.points[i].x, roomData.points[i].y);
        }
        shape.lineTo(roomData.points[0].x, roomData.points[0].y);

        const geometry = new THREE.ShapeGeometry(shape);
        const material = new THREE.MeshPhongMaterial({
            color: 0xeeeeee,
            side: THREE.DoubleSide
        });
        const floor = new THREE.Mesh(geometry, material);
        
        floor.rotation.x = -Math.PI / 2;
        floor.position.set(roomData.x || 0, 0, roomData.z || 0);
        
        floor.userData = {
            type: 'room',
            name: roomData.name,
            area: roomData.area,
            info: `${roomData.name} - Areal: ${roomData.area}m²`
        };

        return floor;
    }

    switchToFloor(floor) {
        if (this.currentFloor === floor) return;

        // Update button states
        const buttons = document.querySelectorAll('#floorButtons button');
        buttons.forEach(button => {
            button.classList.toggle('active', button.dataset.floor === floor);
        });

        // Hide all floors
        this.floors.forEach(floorObj => {
            floorObj.visible = false;
        });

        // Show selected floor
        const selectedFloor = this.floors.get(floor);
        if (selectedFloor) {
            selectedFloor.visible = true;
            this.currentFloor = floor;
            this.updateFloorIndicator();
        }

        // Dispatch event for other components
        const event = new CustomEvent('floorChanged', { detail: { floor } });
        window.dispatchEvent(event);
    }

    updateFloorIndicator() {
        const indicator = document.querySelector('.floor-indicator');
        if (indicator) {
            indicator.textContent = this.getFloorDisplayName(this.currentFloor);
        }
    }

    getCurrentFloorHeight() {
        return this.floorHeights.get(this.currentFloor) || 0;
    }

    getFloorData(floor) {
        return this.floorPlans.get(floor);
    }

    // Helper method to calculate total building height
    getTotalBuildingHeight() {
        let maxHeight = -Infinity;
        this.floorHeights.forEach(height => {
            if (height > maxHeight) maxHeight = height;
        });
        return maxHeight + 2.8; // Add typical floor height for roof
    }

    // Method to export floor data
    exportFloorData() {
        const exportData = {};
        this.floorPlans.forEach((planData, floor) => {
            exportData[floor] = {
                height: this.floorHeights.get(floor),
                data: planData
            };
        });
        return exportData;
    }
}