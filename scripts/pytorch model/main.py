import os
import io
import base64
import logging
from typing import Dict, Any
from PIL import Image
from model import ONNXModelHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
model_handler = None


def initialize_model():
    global model_handler
    try:
        model_path = "../onnx/model.onnx"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model_handler = ONNXModelHandler(model_path)
        logger.info("Model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return False


def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    global model_handler

    if model_handler is None:
        if not initialize_model():
            return {
                "error": "Failed to initialize model",
                "status": "error"
            }

    try:
        if not isinstance(payload, dict):
            return {
                "error": "Payload must be a dictionary",
                "status": "error"
            }

        image_path = None
        temp_file_created = False

        if 'image_base64' in payload:
            image_path = handle_base64_image(payload['image_base64'])
            temp_file_created = True

        elif 'image_url' in payload:
            image_path = handle_image_url(payload['image_url'])
            temp_file_created = True

        elif 'image_path' in payload:
            image_path = payload['image_path']

        else:
            return {
                "error": "No image provided. Use 'image_base64', 'image_url', or 'image_path'",
                "status": "error"
            }

        if not image_path or not os.path.exists(image_path):
            return {
                "error": f"Image file not found or invalid: {image_path}",
                "status": "error"
            }

        predicted_class, confidence, probabilities = model_handler.predict(image_path)

        if temp_file_created and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass

        response = {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "status": "success"
        }

        if payload.get('include_top_predictions', False):
            top_n = min(payload.get('top_n', 5), len(probabilities))
            response["top_predictions"] = [
                {
                    "class": class_name,
                    "confidence": float(prob)
                }
                for class_name, prob in probabilities[:top_n]
            ]

        if payload.get('include_metadata', False):
            response["metadata"] = {
                "model_type": "ONNX ResNet-18",
                "input_shape": [224, 224, 3],
                "num_classes": len(probabilities)
            }

        logger.info(f"Prediction successful: {predicted_class} ({confidence:.4f})")
        return response

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {
            "error": str(e),
            "status": "error"
        }


def handle_base64_image(base64_string: str) -> str:
    try:
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]

        image_data = base64.b64decode(base64_string)

        image = Image.open(io.BytesIO(image_data))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        temp_path = "/tmp/temp_image.jpg"
        image.save(temp_path, 'JPEG')

        return temp_path

    except Exception as e:
        raise ValueError(f"Failed to process base64 image: {str(e)}")


def handle_image_url(image_url: str) -> str:
    try:
        import requests

        response = requests.get(image_url, timeout=30)
        response.raise_for_status()

        # Create PIL Image
        image = Image.open(io.BytesIO(response.content))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        temp_path = "/tmp/temp_image_url.jpg"
        image.save(temp_path, 'JPEG')

        return temp_path

    except Exception as e:
        raise ValueError(f"Failed to process image URL: {str(e)}")


def health_check() -> Dict[str, Any]:
    global model_handler

    try:
        if model_handler is None:
            return {
                "status": "unhealthy",
                "message": "Model not initialized",
                "model_loaded": False
            }

        dummy_image = Image.new('RGB', (224, 224), (128, 128, 128))
        temp_path = "/tmp/health_check.jpg"
        dummy_image.save(temp_path)

        _, confidence, _ = model_handler.predict(temp_path)

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            "status": "healthy",
            "message": "Model is working correctly",
            "model_loaded": True,
            "test_confidence": float(confidence)
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}",
            "model_loaded": model_handler is not None
        }


def get_model_info() -> Dict[str, Any]:
    global model_handler

    if model_handler is None:
        return {
            "error": "Model not initialized",
            "status": "error"
        }

    try:
        # Get model metadata
        input_shape = model_handler.session.get_inputs()[0].shape
        output_shape = model_handler.session.get_outputs()[0].shape

        return {
            "model_type": "ONNX ResNet-18",
            "model_path": "model.onnx",
            "input_shape": input_shape,
            "output_shape": output_shape,
            "input_name": model_handler.session.get_inputs()[0].name,
            "output_name": model_handler.session.get_outputs()[0].name,
            "providers": model_handler.session.get_providers(),
            "status": "success"
        }

    except Exception as e:
        return {
            "error": f"Failed to get model info: {str(e)}",
            "status": "error"
        }


__all__ = ['predict', 'health_check', 'get_model_info']

if __name__ != "__main__":
    initialize_model()
