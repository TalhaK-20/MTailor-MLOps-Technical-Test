import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
from torchvision import transforms
from typing import Tuple, List, Optional
import os
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:

    def __init__(self):
        # ImageNet normalization values
        self.mean = [0.485, 0.456, 0.406]  # RGB means
        self.std = [0.229, 0.224, 0.225]  # RGB standard deviations
        self.target_size = (224, 224)

        # Setup preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.target_size),
            transforms.ToTensor(),  # Converts to [0,1] and changes to CHW format
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        logger.info("ImagePreprocessor initialized with ImageNet standards")

    def preprocess_image(self, image_input) -> np.ndarray:
        try:
            # Handle different input types
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                image = Image.open(image_input)
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input)
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")

            if image.mode != 'RGB':
                image = image.convert('RGB')
                logger.debug(f"Converted image from {image.mode} to RGB")

            # Apply preprocessing pipeline
            processed_tensor = self.transform(image)

            # Convert to numpy and add batch dimension
            processed_array = processed_tensor.numpy()
            processed_array = np.expand_dims(processed_array, axis=0)

            logger.debug(f"Image preprocessed to shape: {processed_array.shape}")
            return processed_array.astype(np.float32)

        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def preprocess_batch(self, image_list: List) -> np.ndarray:
        batch_arrays = []

        for image_input in image_list:
            processed = self.preprocess_image(image_input)
            batch_arrays.append(processed[0])  # Remove batch dimension for individual images

        batch_array = np.stack(batch_arrays, axis=0)
        logger.info(f"Batch preprocessed to shape: {batch_array.shape}")

        return batch_array.astype(np.float32)


class ONNXModelHandler:
    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.preprocessor = ImagePreprocessor()

        if providers is None:
            self.providers = ['CPUExecutionProvider']
        else:
            self.providers = providers

        # Load model
        self._load_model()

    def _load_model(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=self.providers
            )

            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            input_shape = self.session.get_inputs()[0].shape
            output_shape = self.session.get_outputs()[0].shape

            logger.info(f"ONNX model loaded successfully")
            logger.info(f"Input shape: {input_shape}, Output shape: {output_shape}")
            logger.info(f"Execution providers: {self.session.get_providers()}")

        except Exception as e:
            logger.error(f"Error loading ONNX model: {str(e)}")
            raise

    def predict(self, image_input, return_probabilities: bool = False) -> Tuple[int, float, Optional[np.ndarray]]:
        try:
            start_time = time.time()

            processed_image = self.preprocessor.preprocess_image(image_input)
            outputs = self.session.run([self.output_name], {self.input_name: processed_image})

            logits = outputs[0]

            probabilities = self._softmax(logits[0])

            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])

            inference_time = time.time() - start_time
            logger.debug(f"Inference completed in {inference_time:.3f} seconds")

            return predicted_class, confidence, probabilities if return_probabilities else None

        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise

    def predict_batch(self, image_list: List, return_probabilities: bool = False) -> List[
        Tuple[int, float, Optional[np.ndarray]]]:

        try:
            start_time = time.time()

            # Preprocess batch
            batch_array = self.preprocessor.preprocess_batch(image_list)

            # Run batch inference
            outputs = self.session.run([self.output_name], {self.input_name: batch_array})
            batch_logits = outputs[0]

            # Process results
            results = []
            for logits in batch_logits:
                probabilities = self._softmax(logits)
                predicted_class = int(np.argmax(probabilities))
                confidence = float(probabilities[predicted_class])

                results.append((
                    predicted_class,
                    confidence,
                    probabilities if return_probabilities else None
                ))

            inference_time = time.time() - start_time
            logger.info(f"Batch inference ({len(image_list)} images) completed in {inference_time:.3f} seconds")

            return results

        except Exception as e:
            logger.error(f"Error during batch inference: {str(e)}")
            raise

    def get_top_k_predictions(self, image_input, k: int = 5) -> List[Tuple[int, float]]:
        try:
            _, _, probabilities = self.predict(image_input, return_probabilities=True)

            top_k_indices = np.argsort(probabilities)[-k:][::-1]

            top_k_predictions = [
                (int(idx), float(probabilities[idx]))
                for idx in top_k_indices
            ]

            return top_k_predictions

        except Exception as e:
            logger.error(f"Error getting top-k predictions: {str(e)}")
            raise

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def get_model_info(self) -> dict:
        if self.session is None:
            return {"error": "Model not loaded"}

        return {
            "model_path": self.model_path,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "input_shape": self.session.get_inputs()[0].shape,
            "output_shape": self.session.get_outputs()[0].shape,
            "providers": self.session.get_providers(),
            "num_classes": self.session.get_outputs()[0].shape[1] if len(
                self.session.get_outputs()[0].shape) > 1 else None
        }


# Example usage and testing
if __name__ == "__main__":
    try:
        # Initialize model handler
        model_handler = ONNXModelHandler("model.onnx")

        # Print model info
        print("Model Information:")
        info = model_handler.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")

        test_image = "n01667114_mud_turtle.JPEG"

        if os.path.exists(test_image):
            print(f"\nTesting with {test_image}:")
            predicted_class, confidence, _ = model_handler.predict(test_image)
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {confidence:.4f}")

            top_5 = model_handler.get_top_k_predictions(test_image, k=5)
            print("\nTop-5 predictions:")
            for i, (class_id, conf) in enumerate(top_5):
                print(f"  {i + 1}. Class {class_id}: {conf:.4f}")

    except Exception as e:
        print(f"Error: {str(e)}")