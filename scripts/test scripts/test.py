import os
import sys
import time
import argparse
import numpy as np
from PIL import Image
import logging

from model import ONNXModelHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_images(output_dir: str = "test_images"):
    os.makedirs(output_dir, exist_ok=True)

    test_cases = [
        ("red_square.jpg", (255, 0, 0)),
        ("green_circle.jpg", (0, 255, 0)),
        ("blue_triangle.jpg", (0, 0, 255)),
        ("gradient.jpg", None)  # Special case for gradient
    ]

    for filename, color in test_cases:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            if color:
                # Create solid color image
                img = Image.new('RGB', (224, 224), color)
            else:
                # Create gradient image
                img_array = np.zeros((224, 224, 3), dtype=np.uint8)
                for i in range(224):
                    for j in range(224):
                        img_array[i, j] = [i % 256, j % 256, (i + j) % 256]
                img = Image.fromarray(img_array)

            img.save(filepath)
            logger.info(f"Created test image: {filepath}")

    return [os.path.join(output_dir, case[0]) for case in test_cases]


class LocalModelTester:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_handler = None
        self.test_results = []

    def setup_model(self):
        try:
            logger.info(f"Loading ONNX model from: {self.model_path}")
            self.model_handler = ONNXModelHandler(self.model_path)
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False

    def test_single_image(self, image_path: str):
        try:
            logger.info(f"Testing image: {image_path}")

            start_time = time.time()
            predicted_class, confidence, probabilities = self.model_handler.predict(image_path)
            inference_time = time.time() - start_time

            result = {
                'image_path': image_path,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'inference_time': inference_time,
                'top_5_probs': probabilities[:5] if len(probabilities) > 5 else probabilities,
                'status': 'success'
            }

            logger.info(f"✅ Prediction: {predicted_class} (confidence: {confidence:.4f}, time: {inference_time:.3f}s)")
            return result

        except Exception as e:
            logger.error(f"Failed to predict {image_path}: {str(e)}")
            return {
                'image_path': image_path,
                'error': str(e),
                'status': 'error'
            }

    def test_batch_images(self, image_paths: list):
        logger.info(f"Testing {len(image_paths)} images...")

        for image_path in image_paths:
            if os.path.exists(image_path):
                result = self.test_single_image(image_path)
                self.test_results.append(result)
            else:
                logger.warning(f"Image not found: {image_path}")

    def benchmark_performance(self, image_path: str, num_runs: int = 10):
        logger.info(f"Benchmarking performance with {num_runs} runs...")

        if not os.path.exists(image_path):
            logger.error(f"Benchmark image not found: {image_path}")
            return None

        times = []
        for i in range(num_runs):
            start_time = time.time()
            try:
                _, _, _ = self.model_handler.predict(image_path)
                inference_time = time.time() - start_time
                times.append(inference_time)
            except Exception as e:
                logger.error(f"Benchmark run {i + 1} failed: {str(e)}")
                continue

        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)

            benchmark_results = {
                'num_runs': len(times),
                'avg_time': avg_time,
                'std_time': std_time,
                'min_time': min_time,
                'max_time': max_time,
                'fps': 1.0 / avg_time
            }

            logger.info(f"Benchmark Results:")
            logger.info(f"   Average time: {avg_time:.4f}s ± {std_time:.4f}s")
            logger.info(f"   Min/Max time: {min_time:.4f}s / {max_time:.4f}s")
            logger.info(f"   Throughput: {1.0 / avg_time:.2f} FPS")

            return benchmark_results

        return None

    def validate_model_outputs(self):
        logger.info("🔍 Validating model outputs...")
        successful_tests = [r for r in self.test_results if r.get('status') == 'success']

        if not successful_tests:
            logger.error("No successful predictions to validate!")
            return False

        validation_passed = True

        for result in successful_tests:
            confidence = result.get('confidence', 0)
            predicted_class = result.get('predicted_class', '')

            if not (0 <= confidence <= 1):
                logger.error(f"Invalid confidence value: {confidence}")
                validation_passed = False

            if not predicted_class or not isinstance(predicted_class, str):
                logger.error(f"Invalid predicted class: {predicted_class}")
                validation_passed = False

            if 'top_5_probs' in result:
                prob_sum = sum(p[1] for p in result['top_5_probs'])
                if not (0.8 <= prob_sum <= 1.2):  # Allow some tolerance
                    logger.warning(f"Probabilities don't sum to 1: {prob_sum}")

        if validation_passed:
            logger.info("Model output validation passed!")
        else:
            logger.error("Model output validation failed!")

        return validation_passed

    def generate_test_report(self):
        logger.info("Generating test report...")

        successful_tests = [r for r in self.test_results if r.get('status') == 'success']
        failed_tests = [r for r in self.test_results if r.get('status') == 'error']

        report = {
            'model_path': self.model_path,
            'total_tests': len(self.test_results),
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'success_rate': len(successful_tests) / len(self.test_results) if self.test_results else 0,
            'average_inference_time': np.mean(
                [r['inference_time'] for r in successful_tests]) if successful_tests else 0,
            'test_results': self.test_results
        }

        print("\n" + "=" * 60)
        print("🧪 LOCAL MODEL TEST REPORT")
        print("=" * 60)
        print(f"Model Path: {report['model_path']}")
        print(f"Total Tests: {report['total_tests']}")
        print(f"Successful: {report['successful_tests']}")
        print(f"Failed: {report['failed_tests']}")
        print(f"Success Rate: {report['success_rate']:.2%}")
        print(f"Avg Inference Time: {report['average_inference_time']:.4f}s")

        if successful_tests:
            print(f"Estimated Throughput: {1 / report['average_inference_time']:.2f} FPS")

        print("\n Individual Test Results:")
        for i, result in enumerate(self.test_results, 1):
            status_icon = " " if result['status'] == 'success' else " "
            image_name = os.path.basename(result['image_path'])

            if result['status'] == 'success':
                print(
                    f"{i:2d}. {status_icon} {image_name:<20} -> {result['predicted_class']:<15} ({result['confidence']:.4f}) [{result['inference_time']:.3f}s]")
            else:
                print(f"{i:2d}. {status_icon} {image_name:<20} -> ERROR: {result['error']}")

        print("=" * 60)

        return report


def main():
    parser = argparse.ArgumentParser(description='Test ONNX model locally before deployment')
    parser.add_argument('--model', type=str, default='model.onnx', help='Path to ONNX model file')
    parser.add_argument('--test_images', type=str, help='Directory containing test images')
    parser.add_argument('--create_samples', action='store_true', help='Create sample test images')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--benchmark_runs', type=int, default=10, help='Number of benchmark runs')

    args = parser.parse_args()

    if not os.path.exists(args.model):
        logger.error(f" Model file not found: {args.model}")
        logger.info("💡 Make sure to run convert_to_onnx.py first to create the ONNX model")
        sys.exit(1)

    # Initialize tester
    tester = LocalModelTester(args.model)

    # Setup model
    if not tester.setup_model():
        logger.error(" Failed to setup model. Exiting.")
        sys.exit(1)

    # Create or find test images
    if args.create_samples or not args.test_images:
        test_images = create_test_images()
    else:
        # Use provided test images directory
        if os.path.isdir(args.test_images):
            test_images = [os.path.join(args.test_images, f) for f in os.listdir(args.test_images)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        else:
            logger.error(f" Test images directory not found: {args.test_images}")
            sys.exit(1)

    if not test_images:
        logger.error(" No test images found!")
        sys.exit(1)

    tester.test_batch_images(test_images)

    tester.validate_model_outputs()

    if args.benchmark and test_images:
        tester.benchmark_performance(test_images[0], args.benchmark_runs)

    report = tester.generate_test_report()

    if report['success_rate'] < 1.0:
        logger.warning("Some tests failed. Please review the results.")
        sys.exit(1)
    else:
        logger.info("All tests passed! Model is ready for deployment.")
        sys.exit(0)


if __name__ == "__main__":
    main()
