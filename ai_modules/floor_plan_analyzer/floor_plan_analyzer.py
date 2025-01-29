import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from typing import Dict, List, Tuple

class FloorPlanAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.transform = self._get_transforms()

    def _load_model(self) -> nn.Module:
        """
        Load pre-trained floor plan analysis model
        """
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'models/floor_plan_detector.pt')
        model.to(self.device)
        return model

    def _get_transforms(self) -> transforms.Compose:
        """
        Define image transformations
        """
        return transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    async def analyze(self, image_path: str) -> Dict:
        """
        Analyze floor plan image and return detailed analysis
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Detect rooms and features
            with torch.no_grad():
                detections = self.model(image_tensor)

            # Process detections
            rooms = self._process_room_detections(detections)
            walls = self._detect_walls(image)
            measurements = self._calculate_measurements(rooms, walls)
            doors_windows = self._detect_doors_windows(image)

            return {
                "rooms": rooms,
                "walls": walls,
                "measurements": measurements,
                "doors_windows": doors_windows,
                "analysis": self._generate_analysis(rooms, measurements)
            }

        except Exception as e:
            print(f"Error analyzing floor plan: {str(e)}")
            return {}

    def _process_room_detections(self, detections) -> List[Dict]:
        """
        Process room detections from model output
        """
        rooms = []
        for det in detections.pred[0]:
            if det[4] > 0.5:  # Confidence threshold
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                room_type = self.model.names[int(cls)]
                
                room = {
                    "type": room_type,
                    "confidence": float(conf),
                    "coordinates": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    },
                    "area": self._calculate_room_area(x1, y1, x2, y2)
                }
                rooms.append(room)
        
        return rooms

    def _detect_walls(self, image: Image.Image) -> List[Dict]:
        """
        Detect walls in the floor plan
        """
        # Convert to numpy array
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        
        walls = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                walls.append({
                    "coordinates": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    },
                    "length": self._calculate_wall_length(x1, y1, x2, y2)
                })
        
        return walls

    def _calculate_measurements(
        self,
        rooms: List[Dict],
        walls: List[Dict]
    ) -> Dict:
        """
        Calculate measurements and dimensions
        """
        measurements = {
            "total_area": sum(room["area"] for room in rooms),
            "room_dimensions": self._calculate_room_dimensions(rooms),
            "wall_lengths": [wall["length"] for wall in walls],
            "total_wall_length": sum(wall["length"] for wall in walls)
        }
        
        return measurements

    def _detect_doors_windows(self, image: Image.Image) -> Dict:
        """
        Detect doors and windows in the floor plan
        """
        # Convert to numpy array
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Template matching for doors and windows
        door_template = cv2.imread('templates/door.png', 0)
        window_template = cv2.imread('templates/window.png', 0)
        
        doors = self._template_match(gray, door_template, threshold=0.8)
        windows = self._template_match(gray, window_template, threshold=0.8)
        
        return {
            "doors": doors,
            "windows": windows,
            "door_count": len(doors),
            "window_count": len(windows)
        }

    def _template_match(
        self,
        image: np.ndarray,
        template: np.ndarray,
        threshold: float
    ) -> List[Dict]:
        """
        Perform template matching for feature detection
        """
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        
        features = []
        for pt in zip(*locations[::-1]):
            features.append({
                "coordinates": {
                    "x": int(pt[0]),
                    "y": int(pt[1])
                },
                "width": template.shape[1],
                "height": template.shape[0]
            })
        
        return features

    def _calculate_room_area(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float
    ) -> float:
        """
        Calculate room area in square meters
        """
        # Assuming scale is 1 pixel = 5cm
        scale = 0.05
        width = abs(x2 - x1) * scale
        height = abs(y2 - y1) * scale
        return width * height

    def _calculate_wall_length(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float
    ) -> float:
        """
        Calculate wall length in meters
        """
        # Assuming scale is 1 pixel = 5cm
        scale = 0.05
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * scale

    def _calculate_room_dimensions(self, rooms: List[Dict]) -> List[Dict]:
        """
        Calculate dimensions for each room
        """
        dimensions = []
        for room in rooms:
            coords = room["coordinates"]
            width = abs(coords["x2"] - coords["x1"]) * 0.05  # Scale to meters
            height = abs(coords["y2"] - coords["y1"]) * 0.05  # Scale to meters
            
            dimensions.append({
                "room_type": room["type"],
                "width": width,
                "height": height,
                "area": width * height
            })
        
        return dimensions

    def _generate_analysis(
        self,
        rooms: List[Dict],
        measurements: Dict
    ) -> Dict:
        """
        Generate comprehensive analysis of the floor plan
        """
        return {
            "total_area": measurements["total_area"],
            "room_count": len(rooms),
            "room_types": self._count_room_types(rooms),
            "layout_efficiency": self._calculate_layout_efficiency(rooms, measurements),
            "recommendations": self._generate_recommendations(rooms, measurements)
        }

    def _count_room_types(self, rooms: List[Dict]) -> Dict:
        """
        Count number of each room type
        """
        room_counts = {}
        for room in rooms:
            room_type = room["type"]
            room_counts[room_type] = room_counts.get(room_type, 0) + 1
        return room_counts

    def _calculate_layout_efficiency(
        self,
        rooms: List[Dict],
        measurements: Dict
    ) -> float:
        """
        Calculate layout efficiency score
        """
        usable_area = sum(room["area"] for room in rooms)
        total_area = measurements["total_area"]
        
        return (usable_area / total_area) * 100 if total_area > 0 else 0

    def _generate_recommendations(
        self,
        rooms: List[Dict],
        measurements: Dict
    ) -> List[str]:
        """
        Generate recommendations for layout improvement
        """
        recommendations = []
        
        # Check room sizes
        for room in rooms:
            if room["area"] < self._get_minimum_room_size(room["type"]):
                recommendations.append(
                    f"{room['type']} er mindre enn anbefalt minimumsstørrelse"
                )

        # Check layout efficiency
        efficiency = self._calculate_layout_efficiency(rooms, measurements)
        if efficiency < 80:
            recommendations.append(
                "Planløsningen kan optimaliseres for bedre arealutnyttelse"
            )

        return recommendations

    def _get_minimum_room_size(self, room_type: str) -> float:
        """
        Get minimum recommended size for different room types
        """
        minimum_sizes = {
            "bedroom": 7.0,
            "living_room": 15.0,
            "kitchen": 6.0,
            "bathroom": 4.0,
            "hallway": 3.0
        }
        
        return minimum_sizes.get(room_type.lower(), 0.0)