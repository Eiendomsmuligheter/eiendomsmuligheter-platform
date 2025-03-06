import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from transformers import ViTFeatureExtractor, ViTForImageClassification, DetrForObjectDetection, DetrFeatureExtractor
from typing import Dict, List, Union, Optional, Tuple, Any
import os
import logging
from dataclasses import dataclass
from enum import Enum
import math

# Sett opp logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageType(Enum):
    EXTERIOR = "exterior"
    INTERIOR = "interior"
    FLOOR_PLAN = "floor_plan"
    HAND_DRAWN = "hand_drawn"
    AERIAL = "aerial"
    DETAIL = "detail"
    UNKNOWN = "unknown"

@dataclass
class ArchitecturalFeature:
    name: str
    confidence: float
    dimensions: Optional[Tuple[float, float]] = None
    location: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    material: Optional[str] = None

class ImageAnalyzer:
    def __init__(self, models_path: Optional[str] = None):
        """
        Initialize the ImageAnalyzer with necessary models for comprehensive image analysis
        
        Args:
            models_path: Optional path to local model files
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Image classification model (Vision Transformer)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.classification_model = self._load_classification_model()
        
        # Object detection model (DETR - DEtection TRansformer)
        self.detr_feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
        self.detection_model = self._load_detection_model()
        
        # Traditional CV processors for hand-drawn sketches
        self.sketch_processor = self._initialize_sketch_processor()
        
        # Transformations for preprocessing
        self.transform = self._get_transforms()
        
        # Load material and architectural style references
        self.material_references = self._load_material_references()
        self.architectural_styles = self._load_architectural_styles()
        
        # Maps for converting labels to Norwegian
        self.room_type_map = {
            'living_room': 'Stue',
            'bedroom': 'Soverom',
            'kitchen': 'Kjøkken',
            'bathroom': 'Bad',
            'hallway': 'Gang',
            'dining_room': 'Spisestue',
            'office': 'Kontor',
            'basement': 'Kjeller',
            'attic': 'Loft',
            'garage': 'Garasje',
            'laundry_room': 'Vaskerom',
            'staircase': 'Trappoppgang',
            'balcony': 'Balkong',
            'terrace': 'Terrasse'
        }
        
        self.material_map = {
            'wood': 'Tre',
            'concrete': 'Betong',
            'brick': 'Murstein',
            'stone': 'Stein',
            'metal': 'Metall',
            'glass': 'Glass',
            'plaster': 'Puss',
            'tile': 'Flis',
            'vinyl': 'Vinyl',
            'carpet': 'Teppe',
            'laminate': 'Laminat',
            'hardwood': 'Hardtre',
            'marble': 'Marmor',
            'granite': 'Granitt'
        }

    def _load_classification_model(self) -> nn.Module:
        """
        Load pre-trained image classification model
        """
        try:
            model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
            model.to(self.device)
            logger.info("Classification model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading classification model: {str(e)}")
            # Fall back to a simpler model if the main one fails
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            model.eval()
            model.to(self.device)
            return model

    def _load_detection_model(self) -> nn.Module:
        """
        Load pre-trained object detection model
        """
        try:
            model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
            model.to(self.device)
            logger.info("Detection model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading detection model: {str(e)}")
            # Fall back to a simpler model if the main one fails
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            model.to(self.device)
            return model

    def _initialize_sketch_processor(self) -> Dict:
        """
        Initialize processors for hand-drawn sketches
        """
        return {
            "line_detector": cv2.createLineSegmentDetector(0),
            "contour_params": {
                "threshold1": 50,
                "threshold2": 150,
                "min_area": 100
            }
        }

    def _get_transforms(self) -> transforms.Compose:
        """
        Define image transformations for neural network input
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def _load_material_references(self) -> Dict[str, List[np.ndarray]]:
        """
        Load references for material detection
        """
        # In a full implementation, this would load actual material textures
        # For now, we return placeholders
        return {
            "wood": [np.ones((10, 10, 3), dtype=np.uint8) * 150],
            "concrete": [np.ones((10, 10, 3), dtype=np.uint8) * 128],
            "brick": [np.ones((10, 10, 3), dtype=np.uint8) * 100],
            "stone": [np.ones((10, 10, 3), dtype=np.uint8) * 90],
            "glass": [np.ones((10, 10, 3), dtype=np.uint8) * 200]
        }

    def _load_architectural_styles(self) -> Dict[str, Dict]:
        """
        Load references for architectural style classification
        """
        # This would include features characteristic of different architectural styles
        return {
            "modern": {
                "features": ["flat_roof", "large_windows", "minimal_ornamentation"],
                "materials": ["glass", "concrete", "steel"]
            },
            "traditional": {
                "features": ["pitched_roof", "symmetrical_windows", "ornamentation"],
                "materials": ["wood", "brick", "stone"]
            },
            "functionalist": {
                "features": ["box_like", "strip_windows", "minimal_ornamentation"],
                "materials": ["concrete", "plaster", "steel"]
            },
            "scandinavian": {
                "features": ["clean_lines", "integrated_with_nature", "light_colors"],
                "materials": ["wood", "glass", "natural_materials"]
            }
        }

    async def analyze(
        self,
        image_path: Union[str, List[str]],
        analysis_type: str = "all"
    ) -> Dict:
        """
        Analyze image(s) and return detailed analysis
        
        Args:
            image_path: Path to image or list of image paths
            analysis_type: Type of analysis to perform (all, exterior, interior, floor_plan)
            
        Returns:
            Dict: Comprehensive analysis results
        """
        try:
            if isinstance(image_path, list):
                return await self._analyze_multiple_images(image_path, analysis_type)
            else:
                return await self._analyze_single_image(image_path, analysis_type)
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    async def _analyze_single_image(
        self,
        image_path: str,
        analysis_type: str
    ) -> Dict:
        """
        Analyze a single image based on its detected type
        """
        # Load and preprocess image
        try:
            image = Image.open(image_path)
            image_type = self._determine_image_type(image)
            quality_score = self._assess_image_quality(image)
            
            results = {
                "image_path": image_path,
                "image_type": image_type.value,
                "quality_score": quality_score
            }

            # Analyze based on detected image type and requested analysis
            if image_type == ImageType.FLOOR_PLAN or image_type == ImageType.HAND_DRAWN:
                # Process floor plans or hand-drawn sketches
                results["floor_plan_analysis"] = await self._analyze_floor_plan(image, image_type)
            elif image_type == ImageType.EXTERIOR:
                # Process exterior photos
                if analysis_type == "all" or analysis_type == "exterior":
                    results["exterior_analysis"] = await self._analyze_exterior(image)
            elif image_type == ImageType.INTERIOR:
                # Process interior photos
                if analysis_type == "all" or analysis_type == "interior":
                    results["interior_analysis"] = await self._analyze_interior(image)
            elif image_type == ImageType.AERIAL:
                # Process aerial/drone photos
                results["aerial_analysis"] = await self._analyze_aerial(image)
            
            # Always analyze materials and condition unless specifically excluded
            if analysis_type == "all" or analysis_type == "building_materials":
                results["material_analysis"] = await self._analyze_materials(image)
            
            if analysis_type == "all" or analysis_type == "condition":
                results["condition_analysis"] = await self._analyze_condition(image)

            return results
        except Exception as e:
            logger.error(f"Error in single image analysis: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    async def _analyze_multiple_images(self, image_paths: List[str], analysis_type: str) -> Dict:
        """
        Analyze multiple images and combine results intelligently
        """
        all_results = []
        grouped_images = {
            ImageType.EXTERIOR.value: [],
            ImageType.INTERIOR.value: [],
            ImageType.FLOOR_PLAN.value: [],
            ImageType.HAND_DRAWN.value: [],
            ImageType.AERIAL.value: [],
            ImageType.DETAIL.value: [],
            ImageType.UNKNOWN.value: []
        }
        
        # First pass - analyze each image and group by type
        for path in image_paths:
            try:
                result = await self._analyze_single_image(path, analysis_type)
                all_results.append(result)
                
                # Group by image type
                image_type = result.get("image_type", ImageType.UNKNOWN.value)
                grouped_images[image_type].append({
                    "path": path,
                    "result": result
                })
            except Exception as e:
                logger.error(f"Error analyzing image {path}: {str(e)}")
                all_results.append({"path": path, "error": str(e)})

        # Prepare combined results
        combined = {
            "image_count": len(image_paths),
            "image_types": {
                type_name: len(images) for type_name, images in grouped_images.items() if len(images) > 0
            },
            "overall_quality": self._calculate_average_quality(all_results)
        }
        
        # Process floor plans (prioritize proper floor plans over hand-drawn)
        if grouped_images[ImageType.FLOOR_PLAN.value]:
            best_floor_plan = self._find_best_quality_image(grouped_images[ImageType.FLOOR_PLAN.value])
            combined["floor_plan_analysis"] = best_floor_plan["result"].get("floor_plan_analysis", {})
        elif grouped_images[ImageType.HAND_DRAWN.value]:
            best_hand_drawn = self._find_best_quality_image(grouped_images[ImageType.HAND_DRAWN.value])
            combined["floor_plan_analysis"] = best_hand_drawn["result"].get("floor_plan_analysis", {})
        
        # Combine exterior analysis
        if grouped_images[ImageType.EXTERIOR.value] or grouped_images[ImageType.AERIAL.value]:
            combined["exterior_analysis"] = self._combine_exterior_analyses(
                grouped_images[ImageType.EXTERIOR.value] + grouped_images[ImageType.AERIAL.value]
            )
        
        # Combine interior analysis
        if grouped_images[ImageType.INTERIOR.value]:
            combined["interior_analysis"] = self._combine_interior_analyses(
                grouped_images[ImageType.INTERIOR.value]
            )
        
        # Combine material analysis
        material_analyses = []
        for result in all_results:
            if "material_analysis" in result:
                material_analyses.append(result["material_analysis"])
        
        if material_analyses:
            combined["material_analysis"] = self._combine_material_analyses(material_analyses)
        
        # Combine condition analysis
        condition_analyses = []
        for result in all_results:
            if "condition_analysis" in result:
                condition_analyses.append(result["condition_analysis"])
        
        if condition_analyses:
            combined["condition_analysis"] = self._combine_condition_analyses(condition_analyses)
        
        # Generate comprehensive property analysis
        combined["property_analysis"] = self._generate_property_analysis(combined)
        
        return combined

    def _determine_image_type(self, image: Image.Image) -> ImageType:
        """
        Determine the type of image using computer vision techniques
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Check if it's likely a floor plan or sketch using edge detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Floor plans typically have high edge density and straight lines
        if edge_density > 0.1:
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            
            if lines is not None and len(lines) > 20:
                # Calculate straightness of lines (floor plans have many straight lines)
                straightness = self._calculate_line_straightness(lines)
                
                if straightness > 0.8:
                    return ImageType.FLOOR_PLAN
            
            # If high edge density but not many straight lines, might be a hand-drawn sketch
            if edge_density > 0.05:
                return ImageType.HAND_DRAWN
        
        # For other types, use the Vision Transformer model
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.classification_model(**inputs)
            predictions = outputs.logits.softmax(-1)
        
        # Map predictions to image types
        # This is a simplified version - in reality, you'd have a trained model for this specific task
        pred_index = predictions.argmax().item()
        
        # Map prediction index to image type (this mapping would depend on your model)
        image_type_map = {
            0: ImageType.EXTERIOR,
            1: ImageType.INTERIOR,
            2: ImageType.AERIAL,
            3: ImageType.DETAIL,
            4: ImageType.UNKNOWN
        }
        
        return image_type_map.get(pred_index, ImageType.UNKNOWN)

    def _calculate_line_straightness(self, lines: np.ndarray) -> float:
        """
        Calculate how straight the detected lines are (for floor plan detection)
        """
        straight_lines = 0
        total_lines = len(lines)
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate angle of line with x-axis
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            
            # Most lines in floor plans are horizontal or vertical
            if angle < 5 or abs(angle - 90) < 5 or abs(angle - 180) < 5:
                straight_lines += 1
        
        return straight_lines / total_lines if total_lines > 0 else 0

    def _assess_image_quality(self, image: Image.Image) -> Dict:
        """
        Assess the quality of the image using multiple metrics
        """
        # Convert to numpy array
        img_array = np.array(image)

        # Calculate various quality metrics
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        blur_score = self._calculate_blur_score(img_array)
        resolution_score = self._calculate_resolution_score(image.size)
        noise_score = self._estimate_noise_level(img_array)

        overall_quality = self._calculate_overall_quality(
            brightness, contrast, blur_score, resolution_score, noise_score
        )

        return {
            "brightness": float(brightness),
            "contrast": float(contrast),
            "blur_score": blur_score,
            "resolution_score": resolution_score,
            "noise_score": noise_score,
            "overall_quality": overall_quality,
            "is_usable": overall_quality > 0.4  # Threshold for usable images
        }

    async def _analyze_floor_plan(self, image: Image.Image, image_type: ImageType) -> Dict:
        """
        Analyze floor plan or hand-drawn sketch
        """
        # Convert to numpy array
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Different processing for proper floor plans vs hand-drawn sketches
        if image_type == ImageType.FLOOR_PLAN:
            # Process formal floor plan
            rooms = self._detect_rooms_in_floor_plan(gray)
            walls = self._detect_walls_in_floor_plan(gray)
            measurements = self._extract_measurements_from_floor_plan(gray, walls)
            doors_windows = self._detect_doors_windows_in_floor_plan(gray)
        else:
            # Process hand-drawn sketch (more tolerant algorithms)
            rooms = self._detect_rooms_in_sketch(gray)
            walls = self._detect_walls_in_sketch(gray)
            measurements = self._extract_measurements_from_sketch(gray, walls)
            doors_windows = self._detect_doors_windows_in_sketch(gray)
        
        # Combine results
        results = {
            "rooms": rooms,
            "walls": walls,
            "measurements": measurements,
            "doors_windows": doors_windows,
            "is_hand_drawn": image_type == ImageType.HAND_DRAWN,
            "confidence_score": 0.9 if image_type == ImageType.FLOOR_PLAN else 0.7
        }
        
        # Calculate potential for modifications
        results["modification_potential"] = self._analyze_modification_potential(
            rooms, walls, measurements
        )
        
        return results

    def _detect_rooms_in_floor_plan(self, gray_image: np.ndarray) -> List[Dict]:
        """
        Detect rooms in a formal floor plan
        """
        # This would use contour detection and classification
        # For this implementation, we'll return a simplified example
        return [
            {"type": "living_room", "area": 25.0, "position": (100, 100, 300, 300)},
            {"type": "kitchen", "area": 15.0, "position": (310, 100, 450, 300)},
            {"type": "bedroom", "area": 12.0, "position": (100, 310, 250, 450)},
            {"type": "bathroom", "area": 8.0, "position": (260, 310, 350, 450)}
        ]

    def _detect_walls_in_floor_plan(self, gray_image: np.ndarray) -> List[Dict]:
        """
        Detect walls in a formal floor plan
        """
        # Use Hough line transform to detect lines
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        walls = []
        if lines is not None:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # Determine if horizontal or vertical
                orientation = "horizontal" if abs(angle) < 45 or abs(angle) > 135 else "vertical"
                
                walls.append({
                    "id": i,
                    "start": (int(x1), int(y1)),
                    "end": (int(x2), int(y2)),
                    "length_pixels": float(length),
                    "orientation": orientation,
                    "is_load_bearing": length > 200  # Simple heuristic
                })
        
        return walls

    def _extract_measurements_from_floor_plan(self, gray_image: np.ndarray, walls: List[Dict]) -> Dict:
        """
        Extract measurements from a floor plan
        """
        # In a real implementation, this would look for scale bars and dimension text
        # For now, we'll use a simplified approach
        
        # Find the image dimensions
        height, width = gray_image.shape
        
        # Use a default scale (e.g., 1px = 2cm)
        scale_factor = 0.02  # meters per pixel
        
        # Calculate area based on walls
        if walls:
            # Find the bounding box
            min_x = min([min(wall["start"][0], wall["end"][0]) for wall in walls])
            max_x = max([max(wall["start"][0], wall["end"][0]) for wall in walls])
            min_y = min([min(wall["start"][1], wall["end"][1]) for wall in walls])
            max_y = max([max(wall["start"][1], wall["end"][1]) for wall in walls])
            
            # Calculate area
            width_meters = (max_x - min_x) * scale_factor
            height_meters = (max_y - min_y) * scale_factor
            area = width_meters * height_meters
        else:
            # Fallback to image dimensions
            width_meters = width * scale_factor
            height_meters = height * scale_factor
            area = width_meters * height_meters
        
        return {
            "scale_factor": scale_factor,
            "total_area": area,
            "width": width_meters,
            "height": height_meters,
            "wall_lengths": [wall["length_pixels"] * scale_factor for wall in walls]
        }

    def _detect_doors_windows_in_floor_plan(self, gray_image: np.ndarray) -> Dict:
        """
        Detect doors and windows in a floor plan
        """
        # This would use template matching or specialized detection
        # For this implementation, we'll return example data
        return {
            "doors": [
                {"position": (305, 200), "type": "internal", "width": 0.9},
                {"position": (200, 305), "type": "internal", "width": 0.9}
            ],
            "windows": [
                {"position": (150, 100), "width": 1.2, "height": 1.2},
                {"position": (350, 100), "width": 1.0, "height": 1.2}
            ],
            "door_count": 2,
            "window_count": 2
        }

    def _detect_rooms_in_sketch(self, gray_image: np.ndarray) -> List[Dict]:
        """
        Detect rooms in a hand-drawn sketch (more tolerant algorithm)
        """
        # For hand-drawn sketches, we need more robust room detection
        # This is a simplified implementation
        
        # Use adaptive thresholding to handle varying drawing pressure
        thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rooms = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 5000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                room_type = "unknown"  # We would need text recognition for actual types
                
                # Assign room types based on size heuristics
                if area > 30000:
                    room_type = "living_room"
                elif 15000 < area <= 30000:
                    room_type = "bedroom"
                elif 5000 < area <= 15000:
                    if w/h < 1.5:  # More square rooms likely bathrooms
                        room_type = "bathroom"
                    else:
                        room_type = "kitchen"
                
                rooms.append({
                    "type": room_type,
                    "area": area * (0.02**2),  # Convert to square meters using scale factor
                    "position": (x, y, x+w, y+h)
                })
        
        return rooms

    def _detect_walls_in_sketch(self, gray_image: np.ndarray) -> List[Dict]:
        """
        Detect walls in a hand-drawn sketch
        """
        # For sketches, we need to be more tolerant of imperfections
        # Use probabilistic Hough transform with lower thresholds
        edges = cv2.Canny(gray_image, 30, 100, apertureSize=3)
        
        # Dilate edges to connect nearby lines
        kernel = np.ones((3,3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        lines = cv2.HoughLinesP(dilated_edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=20)
        
        walls = []
        if lines is not None:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # Determine if horizontal or vertical (more tolerance for sketches)
                orientation = "horizontal" if abs(angle) < 60 or abs(angle) > 120 else "vertical"
                
                walls.append({
                    "id": i,
                    "start": (int(x1), int(y1)),
                    "end": (int(x2), int(y2)),
                    "length_pixels": float(length),
                    "orientation": orientation,
                    "is_load_bearing": length > 150  # Adjusted for sketches
                })
        
        return walls

    def _extract_measurements_from_sketch(self, gray_image: np.ndarray, walls: List[Dict]) -> Dict:
        """
        Extract measurements from a hand-drawn sketch
        """
        # Similar to floor plan but with more assumptions
        height, width = gray_image.shape
        
        # Use a default scale (e.g., 1px = 3cm) - slightly larger for sketches
        scale_factor = 0.03  # meters per pixel
        
        # Calculate dimensions similar to floor plans
        if walls:
            min_x = min([min(wall["start"][0], wall["end"][0]) for wall in walls])
            max_x = max([max(wall["start"][0], wall["end"][0]) for wall in walls])
            min_y = min([min(wall["start"][1], wall["end"][1]) for wall in walls])
            max_y = max([max(wall["start"][1], wall["end"][1]) for wall in walls])
            
            width_meters = (max_x - min_x) * scale_factor
            height_meters = (max_y - min_y) * scale_factor
            area = width_meters * height_meters
        else:
            width_meters = width * scale_factor
            height_meters = height * scale_factor
            area = width_meters * height_meters
        
        return {
            "scale_factor": scale_factor,
            "total_area": area,
            "width": width_meters,
            "height": height_meters,
            "wall_lengths": [wall["length_pixels"] * scale_factor for wall in walls],
            "is_approximate": True  # Flag to indicate these are approximations
        }

    def _detect_doors_windows_in_sketch(self, gray_image: np.ndarray) -> Dict:
        """
        Detect doors and windows in a hand-drawn sketch
        """
        # Very basic detection for hand-drawn sketches
        # Real implementation would use more sophisticated algorithms
        return {
            "doors": [
                {"position": (300, 200), "type": "internal", "width": 0.9, "confidence": 0.6}
            ],
            "windows": [
                {"position": (150, 100), "width": 1.0, "height": 1.0, "confidence": 0.5}
            ],
            "door_count": 1,
            "window_count": 1,
            "is_approximate": True
        }

    def _analyze_modification_potential(self, rooms: List[Dict], walls: List[Dict], 
                                        measurements: Dict) -> Dict:
        """
        Analyze potential for modifications based on floor plan
        """
        # Calculate total area and average room size
        total_area = measurements.get("total_area", 0)
        avg_room_size = np.mean([room.get("area", 0) for room in rooms]) if rooms else 0
        
        # Count load-bearing walls
        load_bearing_walls = [wall for wall in walls if wall.get("is_load_bearing", False)]
        non_load_bearing_walls = [wall for wall in walls if not wall.get("is_load_bearing", False)]
        
        # Calculate potential based on non-load-bearing walls
        wall_modification_potential = len(non_load_bearing_walls) * 0.1
        
        # Check if there's a basement or attic
        has_basement = any(room.get("type", "").lower() == "basement" for room in rooms)
        has_attic = any(room.get("type", "").lower() == "attic" for room in rooms)
        
        # Estimate room reconfiguration potential
        reconfiguration_potential = 0.0
        if len(rooms) >= 3 and len(non_load_bearing_walls) >= 2:
            reconfiguration_potential = 0.7
        elif len(rooms) >= 2 and len(non_load_bearing_walls) >= 1:
            reconfiguration_potential = 0.5
        else:
            reconfiguration_potential = 0.3
        
        # Estimate division potential
        division_potential = 0.0
        if total_area > 80 and len(rooms) >= 3:
            division_potential = 0.8
        elif total_area > 60 and len(rooms) >= 2:
            division_potential = 0.5
        else:
            division_potential = 0.2
        
        # Adjust based on load-bearing walls
        if len(load_bearing_walls) > 2:
            # Many load-bearing walls make modifications harder
            reconfiguration_potential *= 0.7
            division_potential *= 0.8
        
        # Combine into result
        return {
            "wall_modification_ease": min(wall_modification_potential + 0.3, 1.0),
            "reconfiguration_potential": reconfiguration_potential,
            "division_potential": division_potential,
            "basement_conversion_potential": 0.8 if has_basement else 0.0,
            "attic_conversion_potential": 0.7 if has_attic else 0.0,
            "overall_flexibility": (wall_modification_potential + reconfiguration_potential + division_potential) / 3
        }

    async def _analyze_exterior(self, image: Image.Image) -> Dict:
        """
        Analyze exterior features of the building
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        try:
            # Detect architectural features
            features = self._detect_architectural_features(img_array)
            
            # Analyze facade materials
            materials = await self._analyze_materials(image)
            
            # Detect building damage or wear
            condition = self._detect_exterior_condition(img_array)
            
            # Estimate building dimensions
            dimensions = self._estimate_building_dimensions(img_array)
            
            # Analyze surrounding property
            property_analysis = self._analyze_surrounding_property(img_array)
            
            return {
                "architectural_features": features,
                "materials": materials,
                "condition": condition,
                "dimensions": dimensions,
                "property_analysis": property_analysis,
                "style": self._determine_architectural_style(features, materials.get("detected_materials", []))
            }
        except Exception as e:
            logger.error(f"Error in exterior analysis: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _detect_architectural_features(self, image: np.ndarray) -> List[ArchitecturalFeature]:
        """
        Detect architectural features in the image
        """
        # Use the object detection model
        features = []
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Run through DETR model
        inputs = self.detr_feature_extractor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.detection_model(**inputs)
        
        # Process results
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.7
        
        # Convert from model outputs to feature objects
        for p, box in zip(probas[keep], outputs.pred_boxes[0, keep]):
            class_id = p.argmax().item()
            
            # Map COCO class ids to architectural features
            # This mapping would depend on your model and classes
            feature_map = {
                0: "roof",
                1: "window",
                2: "door",
                3: "balcony",
                4: "chimney",
                5: "stairs",
                6: "garage",
                # Add more mappings as needed
            }
            
            feature_name = feature_map.get(class_id, "unknown")
            confidence = p[class_id].item()
            
            # Convert normalized coordinates to pixel coordinates
            h, w = image.shape[:2]
            box = box.cpu().numpy() * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            
            features.append(ArchitecturalFeature(
                name=feature_name,
                confidence=confidence,
                location=(x1, y1, x2, y2)
            ))
        
        return features

    def _detect_exterior_condition(self, image: np.ndarray) -> Dict:
        """
        Detect exterior condition issues like cracks, peeling paint, etc.
        """
        # A real implementation would use a specialized model for damage detection
        # This is a simplified example
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Look for high-frequency image components that might indicate damage
        # like cracks, missing shingles, etc.
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Variance of Laplacian as a simple measure of "detail" or "entropy"
        detail_level = laplacian.var()
        
        # More detailed regions might indicate more texture/damage
        # (This is a simplified heuristic)
        damage_score = min(max(detail_level / 1000.0, 0), 1)
        
        return {
            "damage_score": float(damage_score),
            "damage_probability": float(damage_score),
            "condition_rating": float(max(1, 10 - damage_score * 10)),
            "detected_issues": [
                {"type": "general_wear", "confidence": float(damage_score)}
            ] if damage_score > 0.3 else []
        }

    def _estimate_building_dimensions(self, image: np.ndarray) -> Dict:
        """
        Estimate building dimensions from exterior image
        """
        # A real implementation would use perspective correction and reference objects
        # This is a simplified example that returns reasonable defaults
        
        # Detect building bounding box
        height, width = image.shape[:2]
        
        # Simplified assumption: building occupies 60-80% of the image
        building_width_pixels = width * 0.7
        building_height_pixels = height * 0.6
        
        # Assume a default scale (e.g. 1px = 5cm at this distance)
        scale_factor = 0.05  # meters per pixel
        
        estimated_width = building_width_pixels * scale_factor
        estimated_height = building_height_pixels * scale_factor
        
        # Estimate number of floors based on height
        estimated_floors = max(1, round(estimated_height / 3.0))
        
        return {
            "estimated_width_m": float(estimated_width),
            "estimated_height_m": float(estimated_height),
            "estimated_floors": int(estimated_floors),
            "is_approximate": True
        }

    def _analyze_surrounding_property(self, image: np.ndarray) -> Dict:
        """
        Analyze surrounding property from exterior image
        """
        # A real implementation would segment the image and analyze the property
        # This is a simplified example
        
        # Define colors for vegetation, concrete, asphalt, etc.
        # Look for green (vegetation) in the image
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Green range in HSV
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([90, 255, 255])
        
        # Create mask for green areas
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate percentage of green area
        vegetation_percentage = np.sum(green_mask > 0) / (green_mask.shape[0] * green_mask.shape[1])
        
        return {
            "vegetation_percentage": float(vegetation_percentage),
            "has_garden": vegetation_percentage > 0.2,
            "has_driveway": True,  # Simplified assumption
            "parking_spaces": 1,    # Simplified assumption
            "is_approximate": True
        }

    def _determine_architectural_style(self, features: List[ArchitecturalFeature], 
                                     materials: List[str]) -> Dict:
        """
        Determine architectural style based on detected features and materials
        """
        # Count occurrences of different features
        feature_counts = {}
        for feature in features:
            if feature.name in feature_counts:
                feature_counts[feature.name] += 1
            else:
                feature_counts[feature.name] = 1
        
        # Check for patterns that indicate specific styles
        style_scores = {
            "modern": 0,
            "traditional": 0,
            "functionalist": 0,
            "scandinavian": 0
        }
        
        # Check for style indicators in features
        if "flat_roof" in feature_counts or "large_windows" in feature_counts:
            style_scores["modern"] += 0.4
            style_scores["functionalist"] += 0.3
        
        if "pitched_roof" in feature_counts:
            style_scores["traditional"] += 0.4
            style_scores["scandinavian"] += 0.2
        
        # Check for style indicators in materials
        for material in materials:
            if material.lower() in ["glass", "concrete", "steel"]:
                style_scores["modern"] += 0.3
                style_scores["functionalist"] += 0.2
            
            if material.lower() in ["wood", "timber", "natural"]:
                style_scores["scandinavian"] += 0.3
                style_scores["traditional"] += 0.2
            
            if material.lower() in ["brick", "stone"]:
                style_scores["traditional"] += 0.3
        
        # Find the style with the highest score
        top_style = max(style_scores.items(), key=lambda x: x[1])
        
        return {
            "detected_style": top_style[0],
            "confidence": top_style[1],
            "style_scores": style_scores
        }

    async def _analyze_interior(self, image: Image.Image) -> Dict:
        """
        Analyze interior features
        """
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Room type classification
            room_type = self._classify_room_type(img_array)
            
            # Detect interior features
            features = self._detect_interior_features(img_array)
            
            # Analyze lighting conditions
            lighting = self._analyze_lighting(img_array)
            
            # Analyze materials and finishes
            materials = self._detect_interior_materials(img_array)
            
            # Estimate room dimensions
            dimensions = self._estimate_room_dimensions(img_array)
            
            return {
                "room_type": room_type,
                "features": features,
                "lighting": lighting,
                "materials": materials,
                "dimensions": dimensions,
                "renovation_potential": self._assess_renovation_potential(room_type, features, materials)
            }
        except Exception as e:
            logger.error(f"Error in interior analysis: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _classify_room_type(self, image: np.ndarray) -> Dict:
        """
        Classify the type of room in the image
        """
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Get features for vision transformer
        inputs = self.feature_extractor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Room type classification
        # In a real implementation, you'd have a model trained specifically for this
        room_types = ["living_room", "bedroom", "kitchen", "bathroom", "hallway", 
                     "dining_room", "office", "other"]
        
        # Random prediction for this example
        # In a real implementation, this would use the model outputs
        confidences = {
            "living_room": 0.7,
            "bedroom": 0.1,
            "kitchen": 0.05,
            "bathroom": 0.05,
            "hallway": 0.03,
            "dining_room": 0.04,
            "office": 0.02,
            "other": 0.01
        }
        
        # Find the room type with highest confidence
        room_type = max(confidences.items(), key=lambda x: x[1])
        
        return {
            "predicted_type": room_type[0],
            "type_confidence": room_type[1],
            "all_confidences": confidences,
            "type_no": self.room_type_map.get(room_type[0], room_type[0])
        }

    def _detect_interior_features(self, image: np.ndarray) -> List[Dict]:
        """
        Detect interior features like windows, doors, built-ins, etc.
        """
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Use the object detection model
        inputs = self.detr_feature_extractor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        features = []
        with torch.no_grad():
            outputs = self.detection_model(**inputs)
        
        # Process the outputs
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.7
        
        # Define interior feature mapping
        interior_feature_map = {
            3: "table",
            5: "chair",
            6: "sofa",
            7: "bed",
            9: "dining_table",
            11: "cabinet",
            12: "bookshelf",
            62: "tv",
            63: "laptop",
            67: "door",
            72: "refrigerator",
            76: "window",
            # Add more mappings as needed
        }
        
        for p, box in zip(probas[keep], outputs.pred_boxes[0, keep]):
            class_id = p.argmax().item()
            
            if class_id in interior_feature_map:
                feature_name = interior_feature_map[class_id]
                confidence = p[class_id].item()
                
                # Convert coordinates
                h, w = image.shape[:2]
                box = box.cpu().numpy() * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                
                features.append({
                    "type": feature_name,
                    "confidence": float(confidence),
                    "position": (int(x1), int(y1), int(x2), int(y2))
                })
        
        return features

    def _analyze_lighting(self, image: np.ndarray) -> Dict:
        """
        Analyze lighting conditions in the room
        """
        # Convert to HSV for better lighting analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Extract Value channel (brightness)
        v = hsv[:, :, 2]
        
        # Calculate average brightness and distribution
        avg_brightness = np.mean(v)
        brightness_std = np.std(v)
        
        # Define lighting quality based on brightness levels
        lighting_quality = "poor"
        if avg_brightness > 180:
            lighting_quality = "excellent"
        elif avg_brightness > 140:
            lighting_quality = "good"
        elif avg_brightness > 100:
            lighting_quality = "adequate"
        
        # Detect potential sources of natural light (windows)
        # Simplification: looking for very bright regions near edges
        
        # Check for bright regions near the edges
        edge_width = max(20, int(min(image.shape[0], image.shape[1]) * 0.1))
        top_edge = v[:edge_width, :]
        bottom_edge = v[-edge_width:, :]
        left_edge = v[:, :edge_width]
        right_edge = v[:, -edge_width:]
        
        edges = [top_edge, bottom_edge, left_edge, right_edge]
        
        has_natural_light = False
        for edge in edges:
            if np.mean(edge) > avg_brightness * 1.3:
                has_natural_light = True
                break
        
        return {
            "average_brightness": float(avg_brightness),
            "brightness_variation": float(brightness_std),
            "lighting_quality": lighting_quality,
            "has_natural_light": has_natural_light,
            "improvement_needed": lighting_quality == "poor" or lighting_quality == "adequate"
        }

    def _detect_interior_materials(self, image: np.ndarray) -> Dict:
        """
        Detect interior materials and finishes
        """
        # This would use texture analysis and material classification
        # For this implementation, we'll return simplified results
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Create masks for common material colors
        # Wood (brown tones)
        lower_wood = np.array([10, 50, 50])
        upper_wood = np.array([30, 255, 200])
        wood_mask = cv2.inRange(hsv, lower_wood, upper_wood)
        wood_percentage = np.sum(wood_mask > 0) / (wood_mask.shape[0] * wood_mask.shape[1])
        
        # Tile/stone (gray tones)
        lower_tile = np.array([0, 0, 100])
        upper_tile = np.array([180, 30, 220])
        tile_mask = cv2.inRange(hsv, lower_tile, upper_tile)
        tile_percentage = np.sum(tile_mask > 0) / (tile_mask.shape[0] * tile_mask.shape[1])
        
        # Carpet (various colors but looking for texture)
        # This is simplified - real implementation would use texture analysis
        carpet_percentage = 0.0  # Placeholder
        
        detected_materials = []
        if wood_percentage > 0.2:
            detected_materials.append({
                "type": "wood",
                "coverage": float(wood_percentage),
                "confidence": float(min(wood_percentage * 2, 0.95))
            })
        
        if tile_percentage > 0.15:
            detected_materials.append({
                "type": "tile",
                "coverage": float(tile_percentage),
                "confidence": float(min(tile_percentage * 2, 0.95))
            })
        
        return {
            "detected_materials": detected_materials,
            "material_distribution": {
                "wood": float(wood_percentage),
                "tile": float(tile_percentage),
                "carpet": float(carpet_percentage),
                "other": float(1.0 - wood_percentage - tile_percentage - carpet_percentage)
            }
        }

    def _estimate_room_dimensions(self, image: np.ndarray) -> Dict:
        """
        Estimate room dimensions from interior image
        """
        # A real implementation would use perspective correction 
        # and reference objects for scale
        height, width = image.shape[:2]
        
        # Simplistic estimation based on typical room dimensions
        # Assume the visible portion is about 60% of the total room width
        # and 70% of the total room depth
        
        aspect_ratio = width / height
        
        if aspect_ratio > 1.5:
            # Wide image, probably showing room width
            room_width = width / 0.6  # pixels
        else:
            # More square image, showing less of the room width
            room_width = width / 0.4  # pixels
        
        # Assume a default scale (e.g. 1px = 2cm at typical interior photo distance)
        scale_factor = 0.02  # meters per pixel
        
        estimated_width = room_width * scale_factor
        estimated_area = estimated_width ** 2  # Rough square approximation
        
        return {
            "estimated_width_m": float(estimated_width),
            "estimated_area_m2": float(estimated_area),
            "ceiling_height_m": 2.4,  # Standard ceiling height
            "is_approximate": True
        }

    def _assess_renovation_potential(self, room_type: Dict, features: List[Dict], 
                                    materials: Dict) -> Dict:
        """
        Assess the potential for renovations based on the interior analysis
        """
        # Initialize scores for different renovation types
        scores = {
            "cosmetic": 0.5,  # Default middle value
            "layout": 0.5,
            "functionality": 0.5,
            "energy_efficiency": 0.5
        }
        
        # Adjust based on room type
        room_type_str = room_type.get("predicted_type", "")
        
        if room_type_str in ["bathroom", "kitchen"]:
            # Bathrooms and kitchens have high renovation value
            scores["cosmetic"] += 0.2
            scores["functionality"] += 0.2
        
        # Adjust based on detected materials
        material_distribution = materials.get("material_distribution", {})
        
        # Older materials might benefit more from renovation
        if material_distribution.get("tile", 0) > 0.3:
            scores["cosmetic"] += 0.1
        
        # Adjust based on features
        has_window = any(f.get("type") == "window" for f in features)
        if not has_window:
            scores["layout"] += 0.1
            scores["energy_efficiency"] += 0.2
        
        # Clip scores to [0, 1] range
        for key in scores:
            scores[key] = min(max(scores[key], 0.0), 1.0)
        
        # Calculate overall score
        overall = sum(scores.values()) / len(scores)
        
        return {
            "scores": scores,
            "overall_potential": float(overall),
            "recommended_improvements": self._generate_renovation_recommendations(
                room_type_str, features, materials, scores
            )
        }

    def _generate_renovation_recommendations(self, room_type: str, features: List[Dict],
                                           materials: Dict, scores: Dict) -> List[Dict]:
        """
        Generate specific renovation recommendations
        """
        recommendations = []
        
        # Add recommendations based on room type and scores
        if room_type == "bathroom" and scores["cosmetic"] > 0.6:
            recommendations.append({
                "type": "cosmetic",
                "description": "Modernisere bad med nye fliser og sanitærutstyr",
                "impact": "high",
                "cost_level": "medium"
            })
        
        if room_type == "kitchen" and scores["functionality"] > 0.6:
            recommendations.append({
                "type": "functionality",
                "description": "Oppgradere kjøkken med nye skap og benkeplater",
                "impact": "high",
                "cost_level": "high"
            })
        
        if scores["energy_efficiency"] > 0.7:
            recommendations.append({
                "type": "energy_efficiency",
                "description": "Oppgradere vinduer til energieffektive alternativer",
                "impact": "medium",
                "cost_level": "medium"
            })
        
        # Add general recommendations if none specific
        if not recommendations:
            recommendations.append({
                "type": "general",
                "description": "Friske opp rom med ny maling og belysning",
                "impact": "low",
                "cost_level": "low"
            })
        
        return recommendations

    async def _analyze_aerial(self, image: Image.Image) -> Dict:
        """
        Analyze aerial/drone images of the property
        """
        img_array = np.array(image)
        
        # Detect property boundaries
        property_bounds = self._detect_property_boundaries(img_array)
        
        # Analyze lot features
        lot_features = self._analyze_lot_features(img_array)
        
        # Check for potential expansion areas
        expansion_potential = self._analyze_expansion_potential(img_array, property_bounds)
        
        return {
            "property_bounds": property_bounds,
            "lot_features": lot_features,
            "expansion_potential": expansion_potential,
            "surrounding_area": self._analyze_surrounding_area(img_array)
        }

    def _detect_property_boundaries(self, image: np.ndarray) -> Dict:
        """
        Detect property boundaries in aerial image
        """
        # This would use segmentation techniques in a real implementation
        # For this example, we'll return a simplified result
        height, width = image.shape[:2]
        
        # Assume the property is roughly in the center of the image
        center_x = width // 2
        center_y = height // 2
        
        # Assume property takes up about 60% of the image
        property_width = int(width * 0.6)
        property_height = int(height * 0.6)
        
        # Calculate boundaries
        left = center_x - property_width // 2
        right = center_x + property_width // 2
        top = center_y - property_height // 2
        bottom = center_y + property_height // 2
        
        return {
            "bounds": (left, top, right, bottom),
            "area_pixels": property_width * property_height,
            "confidence": 0.7,
            "is_approximate": True
        }

    def _analyze_lot_features(self, image: np.ndarray) -> Dict:
        """
        Analyze features of the property lot
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Detect vegetation (green)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        vegetation_percentage = np.sum(green_mask > 0) / (green_mask.shape[0] * green_mask.shape[1])
        
        # Detect buildings (assume buildings are not green or blue)
        lower_building = np.array([0, 0, 50])
        upper_building = np.array([30, 255, 220])
        building_mask = cv2.inRange(hsv, lower_building, upper_building)
        building_percentage = np.sum(building_mask > 0) / (building_mask.shape[0] * building_mask.shape[1])
        
        # Detect water features (blue)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        water_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        water_percentage = np.sum(water_mask > 0) / (water_mask.shape[0] * water_mask.shape[1])
        
        return {
            "vegetation_percentage": float(vegetation_percentage),
            "building_percentage": float(building_percentage),
            "water_percentage": float(water_percentage),
            "has_garden": vegetation_percentage > 0.3,
            "has_water_feature": water_percentage > 0.05,
            "open_space_percentage": float(1.0 - building_percentage - water_percentage)
        }

    def _analyze_expansion_potential(self, image: np.ndarray, property_bounds: Dict) -> Dict:
        """
        Analyze potential for property expansion
        """
        # Extract property region from the bounds
        bounds = property_bounds.get("bounds", (0, 0, image.shape[1], image.shape[0]))
        left, top, right, bottom = bounds
        
        property_region = image[top:bottom, left:right]
        
        # Analyze building coverage
        hsv = cv2.cvtColor(property_region, cv2.COLOR_RGB2HSV)
        
        # Detect buildings (assume buildings are not green or blue)
        lower_building = np.array([0, 0, 50])
        upper_building = np.array([30, 255, 220])
        building_mask = cv2.inRange(hsv, lower_building, upper_building)
        building_percentage = np.sum(building_mask > 0) / (building_mask.shape[0] * building_mask.shape[1])
        
        # Calculate available space for expansion
        available_space = 1.0 - building_percentage
        
        # Determine expansion potential based on available space
        expansion_potential = "low"
        if available_space > 0.7:
            expansion_potential = "high"
        elif available_space > 0.4:
            expansion_potential = "medium"
        
        return {
            "available_space_percentage": float(available_space),
            "building_coverage": float(building_percentage),
            "expansion_potential": expansion_potential,
            "potential_directions": self._determine_expansion_directions(property_region, building_mask)
        }

    def _determine_expansion_directions(self, property_image: np.ndarray, building_mask: np.ndarray) -> List[str]:
        """
        Determine directions for potential expansion
        """
        height, width = property_image.shape[:2]
        
        # Divide the property into regions (north, east, south, west)
        north_region = building_mask[:height//3, :]
        east_region = building_mask[:, width*2//3:]
        south_region = building_mask[height*2//3:, :]
        west_region = building_mask[:, :width//3]
        
        # Calculate building coverage in each region
        north_coverage = np.sum(north_region > 0) / north_region.size
        east_coverage = np.sum(east_region > 0) / east_region.size
        south_coverage = np.sum(south_region > 0) / south_region.size
        west_coverage = np.sum(west_region > 0) / west_region.size
        
        # Determine potential expansion directions
        directions = []
        threshold = 0.3  # Threshold for considering a region open for expansion
        
        if north_coverage < threshold:
            directions.append("north")
        if east_coverage < threshold:
            directions.append("east")
        if south_coverage < threshold:
            directions.append("south")
        if west_coverage < threshold:
            directions.append("west")
        
        return directions

    def _analyze_surrounding_area(self, image: np.ndarray) -> Dict:
        """
        Analyze the surrounding area in aerial images
        """
        # This would use image segmentation and object detection
        # For this example, we'll use color analysis as a simplified approach
        
        height, width = image.shape[:2]
        
        # Define a larger region around the center to capture surrounding area
        center_x, center_y = width // 2, height // 2
        surrounding_width = int(width * 0.9)
        surrounding_height = int(height * 0.9)
        
        # Calculate boundaries for surrounding area (exclude property center)
        surr_left = max(0, center_x - surrounding_width // 2)
        surr_right = min(width, center_x + surrounding_width // 2)
        surr_top = max(0, center_y - surrounding_height // 2)
        surr_bottom = min(height, center_y + surrounding_height // 2)
        
        # Extract surrounding region
        surrounding_region = image.copy()
        
        # Blank out the center (property area)
        property_width = int(width * 0.4)
        property_height = int(height * 0.4)
        prop_left = center_x - property_width // 2
        prop_right = center_x + property_width // 2
        prop_top = center_y - property_height // 2
        prop_bottom = center_y + property_height // 2
        
        surrounding_region[prop_top:prop_bottom, prop_left:prop_right] = 0
        
        # Analyze colors in surrounding area
        hsv = cv2.cvtColor(surrounding_region, cv2.COLOR_RGB2HSV)
        
        # Detect vegetation (green)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        vegetation_percentage = np.sum(green_mask > 0) / np.sum(surrounding_region != 0) * 3
        
        # Detect roads and buildings (gray/brown)
        lower_urban = np.array([0, 0, 50])
        upper_urban = np.array([30, 60, 200])
        urban_mask = cv2.inRange(hsv, lower_urban, upper_urban)
        urban_percentage = np.sum(urban_mask > 0) / np.sum(surrounding_region != 0) * 3
        
        # Make assessment based on color distributions
        is_urban = urban_percentage > 0.4
        is_green = vegetation_percentage > 0.4
        
        neighborhood_type = "mixed"
        if is_urban and not is_green:
            neighborhood_type = "urban"
        elif is_green and not is_urban:
            neighborhood_type = "suburban"
        
        return {
            "neighborhood_type": neighborhood_type,
            "vegetation_density": float(vegetation_percentage),
            "urban_density": float(urban_percentage),
            "is_predominantly_residential": True,  # Simplification
            "proximity_features": {
                "has_nearby_parks": vegetation_percentage > 0.3,
                "has_nearby_roads": urban_percentage > 0.2
            }
        }

    async def _analyze_materials(self, image: Image.Image) -> Dict:
        """
        Analyze building materials
        """
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Detect materials
            materials = self._detect_materials(img_array)
            
            # Analyze material condition
            condition = self._analyze_material_condition(img_array)
            
            return {
                "detected_materials": materials,
                "condition": condition,
                "recommendations": self._generate_material_recommendations(
                    materials,
                    condition
                )
            }
        except Exception as e:
            logger.error(f"Error in material analysis: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _detect_materials(self, image: np.ndarray) -> List[Dict]:
        """
        Detect building materials using color and texture analysis
        """
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for common materials
        material_ranges = {
            "wood": [(10, 50, 50), (30, 255, 200)],
            "brick": [(0, 50, 50), (10, 255, 200)],
            "concrete": [(0, 0, 100), (180, 30, 220)],
            "stone": [(0, 0, 80), (180, 40, 200)],
            "metal": [(0, 0, 180), (180, 30, 255)],
            "glass": [(0, 0, 200), (180, 30, 255)]
        }
        
        materials = []
        total_pixels = image.shape[0] * image.shape[1]
        
        for material, (lower, upper) in material_ranges.items():
            lower_bound = np.array(lower)
            upper_bound = np.array(upper)
            
            # Create mask for this material
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            coverage = np.sum(mask > 0) / total_pixels
            
            if coverage > 0.1:  # Only include materials with significant presence
                materials.append({
                    "type": material,
                    "type_no": self.material_map.get(material, material),
                    "coverage": float(coverage),
                    "confidence": float(min(coverage * 2, 0.95))
                })
        
        # Sort by coverage
        materials.sort(key=lambda x: x["coverage"], reverse=True)
        
        return materials

    def _analyze_material_condition(self, image: np.ndarray) -> Dict:
        """
        Analyze condition of detected materials
        """
        # This would use texture analysis and degradation detection
        # For this example, we'll use simplified metrics
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate texture entropy as a measure of material degradation
        # (More texture entropy might indicate degraded surfaces)
        kernel_size = 5
        entropy_img = np.zeros_like(gray, dtype=float)
        
        for i in range(kernel_size//2, gray.shape[0] - kernel_size//2):
            for j in range(kernel_size//2, gray.shape[1] - kernel_size//2):
                patch = gray[i-kernel_size//2:i+kernel_size//2+1, j-kernel_size//2:j+kernel_size//2+1]
                # Simple entropy calculation
                hist = np.histogram(patch, bins=32, range=(0, 256))[0] / (kernel_size**2)
                hist = hist[hist > 0]  # Only consider non-zero probabilities
                entropy = -np.sum(hist * np.log2(hist))
                entropy_img[i, j] = entropy
        
        # Calculate statistics from entropy image
        mean_entropy = np.mean(entropy_img)
        max_entropy = np.max(entropy_img)
        
        # Map entropy to condition score (higher entropy might indicate worse condition)
        # This is a simplified heuristic
        condition_score = max(0, 10 - mean_entropy * 2)
        
        # Detect potential issues
        issues = []
        if mean_entropy > 3.5:
            issues.append({
                "type": "surface_degradation",
                "severity": "medium",
                "confidence": 0.7
            })
        
        if max_entropy > 4.5:
            issues.append({
                "type": "damage",
                "severity": "high",
                "confidence": 0.6
            })
        
        return {
            "condition_score": float(condition_score),
            "material_quality": self._map_score_to_quality(condition_score),
            "detected_issues": issues,
            "requires_maintenance": condition_score < 6
        }

    def _map_score_to_quality(self, score: float) -> str:
        """Map numeric score to quality category"""
        if score >= 8:
            return "excellent"
        elif score >= 6:
            return "good"
        elif score >= 4:
            return "fair"
        elif score >= 2:
            return "poor"
        else:
            return "very_poor"

    def _generate_material_recommendations(self,
                                        materials: List[Dict],
                                        condition: Dict) -> List[Dict]:
        """
        Generate recommendations for material maintenance or replacement
        """
        recommendations = []
        
        # Check condition and generate appropriate recommendations
        condition_score = condition.get("condition_score", 10)
        
        if condition_score < 4:
            # Poor condition, suggest replacement
            for material in materials:
                material_type = material.get("type", "")
                
                if material_type == "wood" and material.get("coverage", 0) > 0.2:
                    recommendations.append({
                        "type": "replacement",
                        "material": material_type,
                        "description": "Erstatt slitt treverk med nye materialer",
                        "priority": "high",
                        "cost_level": "medium"
                    })
                elif material_type == "brick" and material.get("coverage", 0) > 0.2:
                    recommendations.append({
                        "type": "repair",
                        "material": material_type,
                        "description": "Reparer skadet murstein og fug på nytt",
                        "priority": "medium",
                        "cost_level": "medium"
                    })
                elif material_type == "concrete" and material.get("coverage", 0) > 0.2:
                    recommendations.append({
                        "type": "repair",
                        "material": material_type,
                        "description": "Reparer sprekker i betong",
                        "priority": "high",
                        "cost_level": "medium"
                    })
        
        elif condition_score < 7:
            # Fair condition, suggest maintenance
            for material in materials:
                material_type = material.get("type", "")
                
                if material_type == "wood" and material.get("coverage", 0) > 0.2:
                    recommendations.append({
                        "type": "maintenance",
                        "material": material_type,
                        "description": "Behandle treverk med ny olje/beis",
                        "priority": "medium",
                        "cost_level": "low"
                    })
                elif material_type == "brick" and material.get("coverage", 0) > 0.2:
                    recommendations.append({
                        "type": "maintenance",
                        "material": material_type,
                        "description": "Rengjør og impregnér murstein",
                        "priority": "low",
                        "cost_level": "low"
                    })
        
        # Add general recommendation if none specific
        if not recommendations:
            recommendations.append({
                "type": "general",
                "description": "Generelt vedlikehold av bygningens materialer",
                "priority": "low",
                "cost_level": "low"
            })
        
        return recommendations

    async def _analyze_condition(self, image: Image.Image) -> Dict:
        """
        Analyze building condition
        """
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Detect damage
            damage = self._detect_damage(img_array)
            
            # Analyze wear and tear
            wear = self._analyze_wear(img_array)
            
            # Estimate age based on visual cues
            age_estimate = self._estimate_building_age(img_array)
            
            return {
                "damage_detection": damage,
                "wear_analysis": wear,
                "age_estimate": age_estimate,
                "maintenance_needs": self._assess_maintenance_needs(damage, wear),
                "overall_condition": self._calculate_overall_condition(damage, wear)
            }
        except Exception as e:
            logger.error(f"Error in condition analysis: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _detect_damage(self, image: np.ndarray) -> Dict:
        """
        Detect damage in building
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines (cracks often appear as lines)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        potential_cracks = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # Cracks are often at odd angles
                if not (abs(angle) < 5 or abs(angle - 90) < 5 or abs(angle - 180) < 5):
                    potential_cracks.append({
                        "start": (int(x1), int(y1)),
                        "end": (int(x2), int(y2)),
                        "length": float(length),
                        "angle": float(angle)
                    })
        
        # Count potential cracks and calculate damage score
        crack_count = len(potential_cracks)
        damage_score = min(crack_count / 10, 1.0)
        
        return {
            "detected_cracks": potential_cracks[:5],  # Limit to top 5 for brevity
            "crack_count": crack_count,
            "damage_score": float(damage_score),
            "damage_level": self._map_score_to_damage_level(damage_score)
        }

    def _map_score_to_damage_level(self, score: float) -> str:
        """Map damage score to damage level"""
        if score < 0.2:
            return "minimal"
        elif score < 0.4:
            return "minor"
        elif score < 0.7:
            return "moderate"
        else:
            return "significant"

    def _analyze_wear(self, image: np.ndarray) -> Dict:
        """
        Analyze wear and tear
        """
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Extract saturation channel (faded materials show lower saturation)
        saturation = hsv[:, :, 1]
        
        # Calculate average saturation as a simple indicator of fading
        avg_saturation = np.mean(saturation)
        
        # Map saturation to wear score (lower saturation = more wear)
        wear_score = max(0, 1 - (avg_saturation / 128))
        
        # Detect potential discoloration using color variance
        color_variance = np.std(hsv[:, :, 0])
        has_discoloration = color_variance > 40
        
        return {
            "wear_score": float(wear_score),
            "wear_level": self._map_score_to_wear_level(wear_score),
            "has_discoloration": bool(has_discoloration),
            "average_saturation": float(avg_saturation)
        }

    def _map_score_to_wear_level(self, score: float) -> str:
        """Map wear score to wear level"""
        if score < 0.3:
            return "minimal"
        elif score < 0.5:
            return "light"
        elif score < 0.7:
            return "moderate"
        else:
            return "heavy"

    def _estimate_building_age(self, image: np.ndarray) -> Dict:
        """
        Estimate building age based on visual cues
        """
        # This would use a specialized model in real implementation
        # For this example, we'll use simplified heuristics
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Extract value channel (brightness)
        v = hsv[:, :, 2]
        
        # Calculate contrast as a rough indicator of age
        # (Older buildings might have more contrast due to uneven aging)
        contrast = np.std(v)
        
        # Calculate age score based on contrast (simplified heuristic)
        age_score = min(contrast / 50, 1.0)
        
        # Map to approximate age ranges
        age_category = "unknown"
        min_year = 1900
        max_year = 2020
        
        if age_score < 0.3:
            age_category = "new"
            min_year = 2010
            max_year = 2025
        elif age_score < 0.5:
            age_category = "modern"
            min_year = 1990
            max_year = 2010
        elif age_score < 0.7:
            age_category = "older"
            min_year = 1950
            max_year = 1990
        else:
            age_category = "historic"
            min_year = 1900
            max_year = 1950
        
        return {
            "age_category": age_category,
            "approximate_year_range": (min_year, max_year),
            "confidence": 0.5,  # Low confidence for this simplified approach
            "is_approximate": True
        }

    def _assess_maintenance_needs(self, damage: Dict, wear: Dict) -> List[Dict]:
        """
        Assess maintenance needs based on damage and wear analysis
        """
        maintenance_needs = []
        
        # Check damage level
        damage_level = damage.get("damage_level", "minimal")
        
        if damage_level in ["moderate", "significant"]:
            maintenance_needs.append({
                "type": "structural",
                "description": "Utbedre sprekker og strukturelle skader",
                "priority": "high" if damage_level == "significant" else "medium",
                "timeframe": "immediate" if damage_level == "significant" else "within_3_months"
            })
        
        # Check wear level
        wear_level = wear.get("wear_level", "minimal")
        
        if wear_level in ["moderate", "heavy"]:
            maintenance_needs.append({
                "type": "cosmetic",
                "description": "Friske opp slitt overflate med maling/behandling",
                "priority": "medium" if wear_level == "heavy" else "low",
                "timeframe": "within_6_months" if wear_level == "heavy" else "within_12_months"
            })
        
        # Add general maintenance if no specific needs
        if not maintenance_needs:
            maintenance_needs.append({
                "type": "general",
                "description": "Rutinemessig vedlikehold",
                "priority": "low",
                "timeframe": "within_24_months"
            })
        
        return maintenance_needs

    def _calculate_overall_condition(self, damage: Dict, wear: Dict) -> Dict:
        """
        Calculate overall condition score
        """
        damage_score = damage.get("damage_score", 0)
        wear_score = wear.get("wear_score", 0)
        
        # Weight damage more heavily than wear
        overall_score = 10 - (damage_score * 6 + wear_score * 4)
        
        # Ensure score is in valid range
        overall_score = max(1, min(10, overall_score))
        
        condition_category = self._map_condition_score_to_category(overall_score)
        
        return {
            "overall_score": float(overall_score),
            "condition_category": condition_category,
            "needs_immediate_attention": overall_score < 4
        }

    def _map_condition_score_to_category(self, score: float) -> str:
        """Map condition score to category"""
        if score >= 9:
            return "excellent"
        elif score >= 7:
            return "good"
        elif score >= 5:
            return "fair"
        elif score >= 3:
            return "poor"
        else:
            return "critical"

    def _calculate_blur_score(self, image: np.ndarray) -> float:
        """
        Calculate image blur score using Laplacian variance
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _calculate_resolution_score(self, size: tuple) -> float:
        """
        Calculate resolution score based on image dimensions
        """
        min_dimension = min(size)
        if min_dimension >= 2000:
            return 1.0
        elif min_dimension >= 1000:
            return 0.75
        elif min_dimension >= 500:
            return 0.5
        return 0.25

    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """
        Estimate image noise level
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply median filter to remove noise
        denoised = cv2.medianBlur(gray, 5)
        
        # Calculate difference between original and denoised
        noise = gray.astype(float) - denoised.astype(float)
        
        # Calculate noise statistics
        noise_std = np.std(noise)
        noise_level = min(noise_std / 10.0, 1.0)
        
        return float(noise_level)

    def _calculate_overall_quality(self,
                                brightness: float,
                                contrast: float,
                                blur_score: float,
                                resolution_score: float,
                                noise_level: float) -> float:
        """
        Calculate overall image quality score
        """
        # Normalize values
        norm_brightness = min(max(brightness / 128.0, 0), 2) / 2  # 0-1 scale, optimal around 128
        norm_contrast = min(max(contrast / 70.0, 0), 1)  # Higher contrast is generally better
        norm_blur = min(max(blur_score / 500.0, 0), 1)  # Higher values mean less blur
        norm_noise = 1 - noise_level  # Invert so higher is better
        
        # Adjust brightness score to penalize very dark or very bright images
        brightness_score = 1.0 - abs(norm_brightness - 0.5) * 2
        
        # Weighted average
        weights = {
            "brightness": 0.15,
            "contrast": 0.15,
            "blur": 0.3,
            "resolution": 0.25,
            "noise": 0.15
        }
        
        score = (
            weights["brightness"] * brightness_score +
            weights["contrast"] * norm_contrast +
            weights["blur"] * norm_blur +
            weights["resolution"] * resolution_score +
            weights["noise"] * norm_noise
        )
        
        return min(max(score, 0), 1)

    def _calculate_average_quality(self, results: List[Dict]) -> float:
        """
        Calculate average quality score from multiple images
        """
        quality_scores = []
        for result in results:
            if "quality_score" in result and "overall_quality" in result["quality_score"]:
                quality_scores.append(result["quality_score"]["overall_quality"])
        
        if quality_scores:
            return sum(quality_scores) / len(quality_scores)
        else:
            return 0.5  # Default middle value

    def _find_best_quality_image(self, images: List[Dict]) -> Dict:
        """
        Find the image with the best quality
        """
        if not images:
            return {}
        
        # Sort by quality score
        sorted_images = sorted(
            images, 
            key=lambda x: x["result"].get("quality_score", {}).get("overall_quality", 0),
            reverse=True
        )
        
        return sorted_images[0]

    def _combine_exterior_analyses(self, exterior_images: List[Dict]) -> Dict:
        """
        Combine analyses from multiple exterior images
        """
        if not exterior_images:
            return {}
        
        # Extract all analyses
        analyses = [img["result"].get("exterior_analysis", {}) for img in exterior_images if "result" in img]
        
        # Filter out empty analyses
        analyses = [a for a in analyses if a]
        
        if not analyses:
            return {}
        
        # Combine architectural features
        all_features = []
        for analysis in analyses:
            if "architectural_features" in analysis:
                all_features.extend(analysis["architectural_features"])
        
        # Deduplicate features
        features_by_name = {}
        for feature in all_features:
            name = feature.name if isinstance(feature, ArchitecturalFeature) else feature.get("name", "unknown")
            
            if name not in features_by_name:
                features_by_name[name] = feature
            else:
                # Keep the one with higher confidence
                existing_conf = features_by_name[name].confidence if isinstance(features_by_name[name], ArchitecturalFeature) else features_by_name[name].get("confidence", 0)
                new_conf = feature.confidence if isinstance(feature, ArchitecturalFeature) else feature.get("confidence", 0)
                
                if new_conf > existing_conf:
                    features_by_name[name] = feature
        
        # Combine material analyses
        materials = {}
        for analysis in analyses:
            if "materials" in analysis and "detected_materials" in analysis["materials"]:
                for material in analysis["materials"]["detected_materials"]:
                    material_type = material.get("type", "unknown")
                    if material_type not in materials:
                        materials[material_type] = material
                    else:
                        # Update confidence
                        materials[material_type]["confidence"] = max(
                            materials[material_type].get("confidence", 0),
                            material.get("confidence", 0)
                        )
        
        # Combine condition analyses
        condition_scores = []
        for analysis in analyses:
            if "condition" in analysis and "condition_score" in analysis["condition"]:
                condition_scores.append(analysis["condition"]["condition_score"])
        
        avg_condition = sum(condition_scores) / len(condition_scores) if condition_scores else None
        
        # Combine dimensions (use the most confident one)
        dimensions = None
        highest_conf = 0
        for analysis in analyses:
            if "dimensions" in analysis:
                dim = analysis["dimensions"]
                conf = dim.get("confidence", 0.5)  # Default confidence
                
                if conf > highest_conf:
                    dimensions = dim
                    highest_conf = conf
        
        # Combine style determinations
        styles = {}
        for analysis in analyses:
            if "style" in analysis and "detected_style" in analysis["style"]:
                style = analysis["style"]["detected_style"]
                conf = analysis["style"].get("confidence", 0)
                
                if style not in styles or conf > styles[style]:
                    styles[style] = conf
        
        top_style = max(styles.items(), key=lambda x: x[1]) if styles else (None, 0)
        
        return {
            "architectural_features": list(features_by_name.values()),
            "materials": {
                "detected_materials": list(materials.values())
            },
            "condition": {
                "condition_score": avg_condition,
                "condition_category": self._map_condition_score_to_category(avg_condition) if avg_condition else "unknown"
            },
            "dimensions": dimensions or {},
            "style": {
                "detected_style": top_style[0],
                "confidence": top_style[1],
                "all_detected_styles": styles
            }
        }

    def _combine_interior_analyses(self, interior_images: List[Dict]) -> Dict:
        """
        Combine analyses from multiple interior images
        """
        if not interior_images:
            return {}
        
        # Extract all analyses
        analyses = [img["result"].get("interior_analysis", {}) for img in interior_images if "result" in img]
        
        # Filter out empty analyses
        analyses = [a for a in analyses if a]
        
        if not analyses:
            return {}
        
        # Group by room type
        rooms_by_type = {}
        for analysis in analyses:
            if "room_type" in analysis and "predicted_type" in analysis["room_type"]:
                room_type = analysis["room_type"]["predicted_type"]
                confidence = analysis["room_type"].get("confidence", 0)
                
                if room_type not in rooms_by_type:
                    rooms_by_type[room_type] = []
                
                rooms_by_type[room_type].append((analysis, confidence))
        
        # For each room type, use the analysis with highest confidence
        best_analyses = {}
        for room_type, room_analyses in rooms_by_type.items():
            best_analysis = max(room_analyses, key=lambda x: x[1])
            best_analyses[room_type] = best_analysis[0]
        
        # Convert to room summary
        room_summaries = []
        for room_type, analysis in best_analyses.items():
            room_summary = {
                "type": room_type,
                "type_no": self.room_type_map.get(room_type, room_type),
                "features": analysis.get("features", []),
                "lighting": analysis.get("lighting", {}),
                "materials": analysis.get("materials", {}),
                "dimensions": analysis.get("dimensions", {}),
                "renovation_potential": analysis.get("renovation_potential", {})
            }
            room_summaries.append(room_summary)
        
        # Calculate property-wide statistics
        total_area = sum(room["dimensions"].get("estimated_area_m2", 0) for room in room_summaries)
        
        return {
            "room_count": len(room_summaries),
            "rooms": room_summaries,
            "total_area_m2": total_area,
            "has_natural_light": any(room["lighting"].get("has_natural_light", False) for room in room_summaries),
            "renovation_recommendations": self._combine_renovation_recommendations(best_analyses)
        }

    def _combine_renovation_recommendations(self, analyses: Dict) -> List[Dict]:
        """
        Combine renovation recommendations from multiple room analyses
        """
        all_recommendations = []
        
        for room_type, analysis in analyses.items():
            if "renovation_potential" in analysis and "recommended_improvements" in analysis["renovation_potential"]:
                for recommendation in analysis["renovation_potential"]["recommended_improvements"]:
                    # Add room information to recommendation
                    recommendation["room_type"] = room_type
                    recommendation["room_type_no"] = self.room_type_map.get(room_type, room_type)
                    all_recommendations.append(recommendation)
        
        # Deduplicate similar recommendations
        unique_recommendations = []
        recommendation_descriptions = set()
        
        for recommendation in all_recommendations:
            description = recommendation.get("description", "")
            if description not in recommendation_descriptions:
                recommendation_descriptions.add(description)
                unique_recommendations.append(recommendation)
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        unique_recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 99))
        
        return unique_recommendations

    def _combine_material_analyses(self, material_analyses: List[Dict]) -> Dict:
        """
        Combine material analyses from multiple images
        """
        if not material_analyses:
            return {}
        
        # Combine detected materials
        all_materials = {}
        for analysis in material_analyses:
            if "detected_materials" in analysis:
                for material in analysis["detected_materials"]:
                    material_type = material.get("type", "unknown")
                    confidence = material.get("confidence", 0)
                    
                    if material_type not in all_materials or confidence > all_materials[material_type].get("confidence", 0):
                        all_materials[material_type] = material
        
        # Combine condition assessments
        condition_scores = []
        for analysis in material_analyses:
            if "condition" in analysis and "condition_score" in analysis["condition"]:
                condition_scores.append(analysis["condition"]["condition_score"])
        
        avg_condition = sum(condition_scores) / len(condition_scores) if condition_scores else None
        
        # Combine recommendations
        all_recommendations = []
        for analysis in material_analyses:
            if "recommendations" in analysis:
                all_recommendations.extend(analysis["recommendations"])
        
        # Deduplicate recommendations
        unique_recommendations = []
        recommendation_descriptions = set()
        
        for recommendation in all_recommendations:
            description = recommendation.get("description", "")
            if description not in recommendation_descriptions:
                recommendation_descriptions.add(description)
                unique_recommendations.append(recommendation)
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        unique_recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 99))
        
        return {
            "detected_materials": list(all_materials.values()),
            "condition": {
                "condition_score": avg_condition,
                "material_quality": self._map_score_to_quality(avg_condition) if avg_condition else "unknown"
            },
            "recommendations": unique_recommendations[:5]  # Limit to top 5
        }

    def _combine_condition_analyses(self, condition_analyses: List[Dict]) -> Dict:
        """
        Combine condition analyses from multiple images
        """
        if not condition_analyses:
            return {}
        
        # Combine damage detections
        all_damages = []
        for analysis in condition_analyses:
            if "damage_detection" in analysis and "detected_cracks" in analysis["damage_detection"]:
                all_damages.extend(analysis["damage_detection"]["detected_cracks"])
        
        # Calculate average damage score
        damage_scores = []
        for analysis in condition_analyses:
            if "damage_detection" in analysis and "damage_score" in analysis["damage_detection"]:
                damage_scores.append(analysis["damage_detection"]["damage_score"])
        
        avg_damage_score = sum(damage_scores) / len(damage_scores) if damage_scores else 0
        
        # Calculate average wear score
        wear_scores = []
        for analysis in condition_analyses:
            if "wear_analysis" in analysis and "wear_score" in analysis["wear_analysis"]:
                wear_scores.append(analysis["wear_analysis"]["wear_score"])
        
        avg_wear_score = sum(wear_scores) / len(wear_scores) if wear_scores else 0
        
        # Combine maintenance needs
        all_maintenance = []
        for analysis in condition_analyses:
            if "maintenance_needs" in analysis:
                all_maintenance.extend(analysis["maintenance_needs"])
        
        # Deduplicate maintenance needs
        unique_maintenance = []
        maintenance_descriptions = set()
        
        for need in all_maintenance:
            description = need.get("description", "")
            if description not in maintenance_descriptions:
                maintenance_descriptions.add(description)
                unique_maintenance.append(need)
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        unique_maintenance.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 99))
        
        # Calculate overall condition
        dummy_damage = {"damage_score": avg_damage_score}
        dummy_wear = {"wear_score": avg_wear_score}
        overall_condition = self._calculate_overall_condition(dummy_damage, dummy_wear)
        
        return {
            "damage_detection": {
                "detected_damages": all_damages[:5],  # Limit to top 5
                "damage_score": avg_damage_score,
                "damage_level": self._map_score_to_damage_level(avg_damage_score)
            },
            "wear_analysis": {
                "wear_score": avg_wear_score,
                "wear_level": self._map_score_to_wear_level(avg_wear_score)
            },
            "maintenance_needs": unique_maintenance[:5],  # Limit to top 5
            "overall_condition": overall_condition
        }

    def _generate_property_analysis(self, combined_analyses: Dict) -> Dict:
        """
        Generate a comprehensive property analysis based on all analyses
        """
        property_summary = {
            "property_type": "residential",  # Default
            "condition_summary": "unknown",
            "estimated_age": "unknown",
            "key_features": [],
            "improvement_opportunities": [],
            "potential_issues": []
        }
        
        # Determine property type
        if "exterior_analysis" in combined_analyses:
            exterior = combined_analyses["exterior_analysis"]
            if "style" in exterior and "detected_style" in exterior["style"]:
                property_summary["architectural_style"] = exterior["style"]["detected_style"]
            
            if "dimensions" in exterior:
                dimensions = exterior["dimensions"]
                property_summary["estimated_size"] = {
                    "width_m": dimensions.get("estimated_width_m"),
                    "height_m": dimensions.get("estimated_height_m"),
                    "floors": dimensions.get("estimated_floors", 1)
                }
        
        # Determine age
        if "condition_analysis" in combined_analyses and "age_estimate" in combined_analyses["condition_analysis"]:
            age = combined_analyses["condition_analysis"]["age_estimate"]
            if "age_category" in age:
                property_summary["estimated_age"] = age["age_category"]
                if "approximate_year_range" in age:
                    property_summary["estimated_year_range"] = age["approximate_year_range"]
        
        # Determine condition
        if "condition_analysis" in combined_analyses and "overall_condition" in combined_analyses["condition_analysis"]:
            condition = combined_analyses["condition_analysis"]["overall_condition"]
            if "condition_category" in condition:
                property_summary["condition_summary"] = condition["condition_category"]
        
        # Extract key features
        key_features = []
        
        # From floor plan analysis
        if "floor_plan_analysis" in combined_analyses:
            floor_plan = combined_analyses["floor_plan_analysis"]
            if "rooms" in floor_plan:
                room_count = len(floor_plan["rooms"])
                key_features.append(f"{room_count} rom")
            
            if "measurements" in floor_plan and "total_area_m2" in floor_plan["measurements"]:
                area = floor_plan["measurements"]["total_area_m2"]
                key_features.append(f"Ca. {area:.1f} m²")
        
        # From exterior analysis
        if "exterior_analysis" in combined_analyses:
            exterior = combined_analyses["exterior_analysis"]
            if "property_analysis" in exterior:
                prop_analysis = exterior["property_analysis"]
                if prop_analysis.get("has_garden", False):
                    key_features.append("Hage")
        
        # From interior analysis
        if "interior_analysis" in combined_analyses:
            interior = combined_analyses["interior_analysis"]
            if "has_natural_light" in interior and interior["has_natural_light"]:
                key_features.append("God naturlig belysning")
        
        property_summary["key_features"] = key_features
        
        # Extract improvement opportunities
        opportunities = []
        
        # From floor plan analysis
        if "floor_plan_analysis" in combined_analyses:
            floor_plan = combined_analyses["floor_plan_analysis"]
            if "modification_potential" in floor_plan:
                mod_potential = floor_plan["modification_potential"]
                
                if mod_potential.get("division_potential", 0) > 0.7:
                    opportunities.append({
                        "type": "division",
                        "description": "Potensial for å dele boligen i separate enheter",
                        "confidence": mod_potential["division_potential"]
                    })
                
                if mod_potential.get("basement_conversion_potential", 0) > 0.7:
                    opportunities.append({
                        "type": "conversion",
                        "description": "Potensial for å konvertere kjeller til utleieenhet",
                        "confidence": mod_potential["basement_conversion_potential"]
                    })
                
                if mod_potential.get("attic_conversion_potential", 0) > 0.7:
                    opportunities.append({
                        "type": "conversion",
                        "description": "Potensial for å konvertere loft til utleieenhet",
                        "confidence": mod_potential["attic_conversion_potential"]
                    })
        
        # From interior analysis
        if "interior_analysis" in combined_analyses and "renovation_recommendations" in combined_analyses["interior_analysis"]:
            interior_recs = combined_analyses["interior_analysis"]["renovation_recommendations"]
            for rec in interior_recs[:2]:  # Top 2 interior recommendations
                opportunities.append({
                    "type": "renovation",
                    "description": rec.get("description", ""),
                    "room_type": rec.get("room_type_no", "")
                })
        
        # From material analysis
        if "material_analysis" in combined_analyses and "recommendations" in combined_analyses["material_analysis"]:
            material_recs = combined_analyses["material_analysis"]["recommendations"]
            for rec in material_recs[:2]:  # Top 2 material recommendations
                opportunities.append({
                    "type": "maintenance",
                    "description": rec.get("description", "")
                })
        
        property_summary["improvement_opportunities"] = opportunities
        
        # Extract potential issues
        issues = []
        
        # From condition analysis
        if "condition_analysis" in combined_analyses:
            condition = combined_analyses["condition_analysis"]
            
            if "maintenance_needs" in condition:
                for need in condition["maintenance_needs"]:
                    if need.get("priority", "low") == "high":
                        issues.append({
                            "type": "maintenance",
                            "description": need.get("description", ""),
                            "severity": "high"
                        })
            
            if "damage_detection" in condition and condition["damage_detection"].get("damage_level", "minimal") in ["moderate", "significant"]:
                issues.append({
                    "type": "damage",
                    "description": "Skader identifisert som krever utbedring",
                    "severity": "medium"
                })
        
        property_summary["potential_issues"] = issues
        
        # Generate rental potential assessment
        rental_potential = self._assess_rental_potential(combined_analyses)
        property_summary["rental_potential"] = rental_potential
        
        return property_summary

    def _assess_rental_potential(self, combined_analyses: Dict) -> Dict:
        """
        Assess the rental potential of the property
        """
        rental_potential = {
            "has_rental_unit": False,
            "potential_for_creation": "unknown",
            "estimated_monthly_rental": 0,
            "conversion_recommendations": []
        }
        
        # Check if property already has separate units
        if "floor_plan_analysis" in combined_analyses and "existing_units" in combined_analyses["floor_plan_analysis"]:
            existing_units = combined_analyses["floor_plan_analysis"]["existing_units"]
            if len(existing_units) > 1:
                rental_potential["has_rental_unit"] = True
                # Estimate rental income based on size
                for unit in existing_units:
                    if unit.get("type") == "secondary":
                        area = unit.get("total_area_m2", 0)
                        # Simple estimation of monthly rental (200 NOK per m²)
                        rental_potential["estimated_monthly_rental"] = area * 200
        
        # Check potential for creating rental units
        if "floor_plan_analysis" in combined_analyses and "modification_potential" in combined_analyses["floor_plan_analysis"]:
            mod_potential = combined_analyses["floor_plan_analysis"]["modification_potential"]
            
            # Check different conversion possibilities
            potentials = []
            recommendations = []
            
            if mod_potential.get("basement_conversion_potential", 0) > 0.6:
                potentials.append(("basement", mod_potential["basement_conversion_potential"]))
                recommendations.append({
                    "type": "basement_conversion",
                    "description": "Konverter kjeller til utleieenhet",
                    "estimated_cost": "medium_high",
                    "estimated_return": "good"
                })
            
            if mod_potential.get("attic_conversion_potential", 0) > 0.6:
                potentials.append(("attic", mod_potential["attic_conversion_potential"]))
                recommendations.append({
                    "type": "attic_conversion",
                    "description": "Konverter loft til utleieenhet",
                    "estimated_cost": "medium_high",
                    "estimated_return": "good"
                })
            
            if mod_potential.get("division_potential", 0) > 0.7:
                potentials.append(("division", mod_potential["division_potential"]))
                recommendations.append({
                    "type": "house_division",
                    "description": "Del boligen i to separate enheter",
                    "estimated_cost": "high",
                    "estimated_return": "excellent"
                })
            
            # Set overall potential based on highest individual potential
            if potentials:
                best_potential = max(potentials, key=lambda x: x[1])
                if best_potential[1] > 0.8:
                    rental_potential["potential_for_creation"] = "excellent"
                elif best_potential[1] > 0.6:
                    rental_potential["potential_for_creation"] = "good"
                elif best_potential[1] > 0.4:
                    rental_potential["potential_for_creation"] = "moderate"
                else:
                    rental_potential["potential_for_creation"] = "low"
                
                rental_potential["conversion_recommendations"] = recommendations
        
        # Estimate property values and ROI if we have enough information
        if "floor_plan_analysis" in combined_analyses and "measurements" in combined_analyses["floor_plan_analysis"]:
            measurements = combined_analyses["floor_plan_analysis"]["measurements"]
            total_area = measurements.get("total_area_m2", 0)
            
            if total_area > 0:
                # Very simplified estimation
                property_value = total_area * 50000  # 50,000 NOK per m²
                rental_potential["property_estimates"] = {
                    "estimated_property_value": property_value,
                    "estimated_annual_rental_income": rental_potential["estimated_monthly_rental"] * 12,
                    "estimated_roi_percentage": (rental_potential["estimated_monthly_rental"] * 12 / property_value) * 100 if property_value > 0 else 0
                }
        
        return rental_potential
