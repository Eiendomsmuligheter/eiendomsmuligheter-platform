import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import styles from '../styles/PropertyVisualizer.module.css';

const PropertyVisualizer = ({ propertyData, modifications }) => {
    const mountRef = useRef(null);
    const sceneRef = useRef(null);
    const cameraRef = useRef(null);
    const rendererRef = useRef(null);
    const controlsRef = useRef(null);

    useEffect(() => {
        // Scene setup
        const scene = new THREE.Scene();
        sceneRef.current = scene;
        scene.background = new THREE.Color(0xf0f0f0);

        // Camera setup
        const camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        cameraRef.current = camera;
        camera.position.set(10, 10, 10);

        // Renderer setup
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        rendererRef.current = renderer;
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.shadowMap.enabled = true;
        mountRef.current.appendChild(renderer.domElement);

        // Controls setup
        const controls = new OrbitControls(camera, renderer.domElement);
        controlsRef.current = controls;
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 10);
        directionalLight.castShadow = true;
        scene.add(directionalLight);

        // Load base property model
        loadPropertyModel(propertyData);

        // Add modifications if any
        if (modifications) {
            addModifications(modifications);
        }

        // Animation loop
        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        // Handle window resize
        const handleResize = () => {
            const width = window.innerWidth;
            const height = window.innerHeight;

            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        };
        window.addEventListener('resize', handleResize);

        // Cleanup
        return () => {
            window.removeEventListener('resize', handleResize);
            mountRef.current?.removeChild(renderer.domElement);
            scene.clear();
        };
    }, [propertyData]);

    // Load the base property model
    const loadPropertyModel = (propertyData) => {
        const loader = new GLTFLoader();

        // Load appropriate model based on property type
        const modelPath = getModelPath(propertyData.propertyType);
        
        loader.load(modelPath, (gltf) => {
            const model = gltf.scene;
            model.scale.set(1, 1, 1);
            model.position.set(0, 0, 0);
            model.castShadow = true;
            model.receiveShadow = true;
            sceneRef.current.add(model);

            // Add measurements and annotations
            addMeasurements(model, propertyData);
            addAnnotations(propertyData);
        });
    };

    // Add proposed modifications to the model
    const addModifications = (modifications) => {
        modifications.forEach(mod => {
            switch (mod.type) {
                case 'extension':
                    addExtension(mod);
                    break;
                case 'basement':
                    addBasement(mod);
                    break;
                case 'roof':
                    modifyRoof(mod);
                    break;
                case 'garage':
                    addGarage(mod);
                    break;
                default:
                    console.warn('Unknown modification type:', mod.type);
            }
        });
    };

    // Helper functions for modifications
    const addExtension = (modification) => {
        const geometry = new THREE.BoxGeometry(
            modification.dimensions.width,
            modification.dimensions.height,
            modification.dimensions.depth
        );
        const material = new THREE.MeshStandardMaterial({
            color: 0x88cc88,
            transparent: true,
            opacity: 0.7
        });
        const extension = new THREE.Mesh(geometry, material);
        extension.position.set(
            modification.position.x,
            modification.position.y,
            modification.position.z
        );
        sceneRef.current.add(extension);
    };

    const addBasement = (modification) => {
        // Implementer basement-visualisering
    };

    const modifyRoof = (modification) => {
        // Implementer tak-modifikasjon
    };

    const addGarage = (modification) => {
        // Implementer garasje-tillegg
    };

    // Helper functions for measurements and annotations
    const addMeasurements = (model, propertyData) => {
        // Legg til målelinjer og dimensjoner
    };

    const addAnnotations = (propertyData) => {
        // Legg til informative markører og tekst
    };

    // Helper function to get appropriate 3D model path
    const getModelPath = (propertyType) => {
        switch (propertyType.toLowerCase()) {
            case 'enebolig':
                return '/models/single_family_house.glb';
            case 'leilighet':
                return '/models/apartment.glb';
            case 'rekkehus':
                return '/models/townhouse.glb';
            default:
                return '/models/default_house.glb';
        }
    };

    return (
        <div className={styles.visualizerContainer} ref={mountRef}>
            <div className={styles.controls}>
                <button onClick={() => resetCamera()}>Reset Visning</button>
                <button onClick={() => toggleWireframe()}>Wireframe</button>
                <button onClick={() => toggleMeasurements()}>Mål</button>
                <button onClick={() => toggleAnnotations()}>Anmerkninger</button>
            </div>
        </div>
    );
};

export default PropertyVisualizer;