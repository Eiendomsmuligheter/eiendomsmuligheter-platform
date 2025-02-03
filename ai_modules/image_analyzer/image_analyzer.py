import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from transformers import ViTFeatureExtractor, ViTForImageClassification
from typing import Dict, List, Union, Optional

class ImageAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.model = self._load_model()
        self.transform = self._get_transforms()

    def _load_model(self) -> nn.Module:
        """
        Load pre-trained image analysis model
        """
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        model.to(self.device)
        return model

    def _get_transforms(self) -> transforms.Compose:
        """
        Define image transformations
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    async def analyze(
        self,
        image_path: Union[str, List[str]],
        analysis_type: str = "all"
    ) -> Dict:
        """
        Analyze image(s) and return detailed analysis
        """
        try:
            if isinstance(image_path, list):
                return await self._analyze_multiple_images(image_path)
            else:
                return await self._analyze_single_image(image_path, analysis_type)
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            return {}

    async def _analyze_single_image(
        self,
        image_path: str,
        analysis_type: str
    ) -> Dict:
        """
        Analyze a single image
        """
        # Load and preprocess image
        image = Image.open(image_path)
        
        results = {
            "image_type": self._determine_image_type(image),
            "quality_score": self._assess_image_quality(image)
        }

        # Perform specific analysis based on image type and requested analysis
        if analysis_type == "all" or analysis_type == "exterior":
            results["exterior_analysis"] = await self._analyze_exterior(image)
        
        if analysis_type == "all" or analysis_type == "interior":
            results["interior_analysis"] = await self._analyze_interior(image)
        
        if analysis_type == "all" or analysis_type == "building_materials":
            results["material_analysis"] = await self._analyze_materials(image)
        
        if analysis_type == "all" or analysis_type == "condition":
            results["condition_analysis"] = await self._analyze_condition(image)

        return results

    async def _analyze_multiple_images(self, image_paths: List[str]) -> Dict:
        """
        Analyze multiple images and combine results
        """
        all_results = []
        for path in image_paths:
            result = await self._analyze_single_image(path, "all")
            all_results.append(result)

        return self._combine_analysis_results(all_results)

    def _determine_image_type(self, image: Image.Image) -> str:
        """
        Determine the type of image (exterior, interior, detail, etc.)
        """
        # Prepare image for model
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.softmax(-1)

        # Map predictions to image types
        image_types = ["exterior", "interior", "detail", "aerial"]
        predicted_type = image_types[predictions.argmax().item()]

        return predicted_type

    def _assess_image_quality(self, image: Image.Image) -> Dict:
        """
        Assess the quality of the image
        """
        # Convert to numpy array
        img_array = np.array(image)

        # Calculate various quality metrics
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        blur_score = self._calculate_blur_score(img_array)
        resolution_score = self._calculate_resolution_score(image.size)

        return {
            "brightness": float(brightness),
            "contrast": float(contrast),
            "blur_score": blur_score,
            "resolution_score": resolution_score,
            "overall_quality": self._calculate_overall_quality(
                brightness, contrast, blur_score, resolution_score
            )
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
            
            # Detect damage or wear
            condition = self._detect_exterior_condition(img_array)
            
            return {
                "architectural_features": features,
                "materials": materials,
                "condition": condition,
                "style": self._determine_architectural_style(features)
            }
        except Exception as e:
            print(f"Error in exterior analysis: {str(e)}")
            return {}

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
            
            return {
                "room_type": room_type,
                "features": features,
                "lighting": lighting,
                "measurements": self._estimate_room_measurements(img_array)
            }
        except Exception as e:
            print(f"Error in interior analysis: {str(e)}")
            return {}

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
            print(f"Error in material analysis: {str(e)}")
            return {}

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
            
            return {
                "damage_detection": damage,
                "wear_analysis": wear,
                "maintenance_needs": self._assess_maintenance_needs(damage, wear)
            }
        except Exception as e:
            print(f"Error in condition analysis: {str(e)}")
            return {}

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

    def _calculate_overall_quality(
        self,
        brightness: float,
        contrast: float,
        blur_score: float,
        resolution_score: float
    ) -> float:
        """
        Calculate overall image quality score
        """
        # Normalize values
        norm_brightness = min(max(brightness / 255.0, 0), 1)
        norm_contrast = min(max(contrast / 255.0, 0), 1)
        norm_blur = min(max(blur_score / 1000.0, 0), 1)
        
        # Weighted average
        weights = {
            "brightness": 0.2,
            "contrast": 0.2,
            "blur": 0.3,
            "resolution": 0.3
        }
        
        score = (
            weights["brightness"] * norm_brightness +
            weights["contrast"] * norm_contrast +
            weights["blur"] * norm_blur +
            weights["resolution"] * resolution_score
        )
        
        return min(max(score, 0), 1)

    def _combine_analysis_results(self, results: List[Dict]) -> Dict:
        """
        Combine results from multiple images into a comprehensive analysis
        """
        combined = {
            "exterior_analysis": {},
            "interior_analysis": {},
            "material_analysis": {},
            "condition_analysis": {},
            "overall_quality": 0.0
        }
        
        # Count number of each type of analysis
        counts = {
            "exterior": 0,
            "interior": 0,
            "material": 0,
            "condition": 0
        }
        
        # Combine results
        for result in results:
            if "exterior_analysis" in result:
                self._update_exterior_analysis(
                    combined["exterior_analysis"],
                    result["exterior_analysis"]
                )
                counts["exterior"] += 1
            
            if "interior_analysis" in result:
                self._update_interior_analysis(
                    combined["interior_analysis"],
                    result["interior_analysis"]
                )
                counts["interior"] += 1
            
            if "material_analysis" in result:
                self._update_material_analysis(
                    combined["material_analysis"],
                    result["material_analysis"]
                )
                counts["material"] += 1
            
            if "condition_analysis" in result:
                self._update_condition_analysis(
                    combined["condition_analysis"],
                    result["condition_analysis"]
                )
                counts["condition"] += 1
            
            combined["overall_quality"] += result.get("quality_score", {}).get(
                "overall_quality", 0
            )
        
        # Calculate averages
        if results:
            combined["overall_quality"] /= len(results)
        
        return combined