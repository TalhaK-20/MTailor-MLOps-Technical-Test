"""
Self-contained Remote Server Testing
Tests the deployed ONNX model on Cerebrium platform

This script is completely self-contained and includes:
- API endpoint and key configuration
- Test image generation
- Comprehensive testing suite
- Performance benchmarking
- Error handling and retry logic
"""

import os
import sys
import time
import base64
import json
import requests
import logging
from io import BytesIO
from typing import Dict, List, Any, Optional
import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RemoteModelTester:
    def __init__(self, api_endpoint: str, api_key: str):
        """
        Initialize remote model tester

        Args:
            api_endpoint: Cerebrium API endpoint URL
            api_key: Cerebrium API key
        """
        self.api_endpoint = api_endpoint.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
        self.test_results = []

    def create_test_image_base64(self, image_type: str = "gradient") -> str:
        """
        Create test images and return as base64 string

        Args:
            image_type: Type of test image to create

        Returns:
            Base64 encoded image string
        """
        try:
            if image_type == "red_square":
                img = Image.new('RGB', (224, 224), (255, 0, 0))
            elif image_type == "green_circle":
                img = Image.new('RGB', (224, 224), (0, 255, 0))
                # Draw a simple circle by modifying pixels
                img_array = np.array(img)
                center = 112
                radius = 80
                y, x = np.ogrid[:224, :224]
                mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
                img_array[mask] = [0, 255, 0]
                img = Image.fromarray(img_array)
            elif image_type == "blue_triangle":
                img = Image.new('RGB', (224, 224), (0, 0, 255))
            elif image_type == "gradient":
                img_array = np.zeros((224, 224, 3), dtype=np.uint8)
                for i in range(224):
                    for j in range(224):
                        img_array[i, j] = [i % 256, j % 256, (i + j) % 256]
                img = Image.fromarray(img_array)
            elif image_type == "random_noise":
                img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
            else:
                # Default: solid gray
                img = Image.new('RGB', (224, 224), (128, 128, 128))

            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format='JPEG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return img_base64

        except Exception as e:
            logger.error(f"Failed to create test image {image_type}: {str(e)}")
            return ""

    def test_api_connectivity(self) -> bool:
        """Test basic API connectivity"""
        try:
            logger.info("🔗 Testing API connectivity...")

            # Try health check first
            health_url = f"{self.api_endpoint}/health"
            response = self.session.get(health_url, timeout=30)

            if response.status_code == 200:
                logger.info("✅ Health check endpoint accessible")
                return True
            elif response.status_code == 404:
                # Health endpoint might not exist, try main endpoint
                logger.info("ℹ️ Health endpoint not found, testing main endpoint...")
                return self.test_main_endpoint()
            else:
                logger.warning(f"⚠️ Health check returned status {response.status_code}")
                return self.test_main_endpoint()

        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ Health check failed: {str(e)}")
            return self.test_main_endpoint()

    def test_main_endpoint(self) -> bool:
        """Test main prediction endpoint with minimal payload"""
        try:
            # Create a simple test image
            test_image = self.create_test_image_base64("gradient")
            if not test_image:
                return False

            payload = {
                "image_base64": test_image
            }

            response = self.session.post(self.api_endpoint, json=payload, timeout=60)

            if response.status_code == 200:
                logger.info("✅ Main endpoint accessible")
                return True
            else:
                logger.error(f"❌ Main endpoint returned status {response.status_code}: {response.text}")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to test main endpoint: {str(e)}")
            return False

    def make_prediction_request(self, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        """
        Make a prediction request with retry logic

        Args:
            payload: Request payload
            timeout: Request timeout in seconds

        Returns:
            Response dictionary
        """
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                logger.debug(f"Making prediction request (attempt {attempt + 1}/{max_retries})")

                response = self.session.post(
                    self.api_endpoint,
                    json=payload,
                    timeout=timeout
                )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    logger.warning(f"Rate limited, waiting {retry_delay * 2}s...")
                    time.sleep(retry_delay * 2)
                    continue
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    if attempt == max_retries - 1:
                        return {"error": error_msg, "status": "error"}
                    else:
                        logger.warning(f"Request failed: {error_msg}, retrying...")
                        time.sleep(retry_delay)
                        continue

            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    return {"error": "Request timeout", "status": "error"}
                logger.warning(f"Request timeout, retrying in {retry_delay}s...")
                time.sleep(retry_delay)

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    return {"error": f"Request failed: {str(e)}", "status": "error"}
                logger.warning(f"Request failed: {str(e)}, retrying...")
                time.sleep(retry_delay)

        return {"error": "Max retries exceeded", "status": "error"}

    def test_single_prediction(self, image_type: str, include_extras: bool = False) -> Dict[str, Any]:
        """
        Test a single prediction

        Args:
            image_type: Type of test image
            include_extras: Whether to include extra metadata

        Returns:
            Test result dictionary
        """
        logger.info(f"🧪 Testing prediction with {image_type} image...")

        try:
            # Create test image
            image_base64 = self.create_test_image_base64(image_type)
            if not image_base64:
                return {
                    "image_type": image_type,
                    "error": "Failed to create test image",
                    "status": "error"
                }

            # Prepare payload
            payload = {
                "image_base64": image_base64
            }

            if include_extras:
                payload.update({
                    "include_top_predictions": True,
                    "top_n": 5,
                    "include_metadata": True
                })

            # Make request and time it
            start_time = time.time()
            response = self.make_prediction_request(payload)
            response_time = time.time() - start_time

            # Process response
            if response.get("status") == "success":
                result = {
                    "image_type": image_type,
                    "predicted_class": response.get("predicted_class"),
                    "confidence": response.get("confidence"),
                    "response_time": response_time,
                    "status": "success"
                }

                if "top_predictions" in response:
                    result["top_predictions"] = response["top_predictions"]

                if "metadata" in response:
                    result["metadata"] = response["metadata"]

                logger.info(f"✅ {image_type}: {result['predicted_class']} ({result['confidence']:.4f}) [{response_time:.3f}s]")
                return result
            else:
                error_msg = response.get("error", "Unknown error")
                logger.error(f"❌ {image_type}: {error_msg}")
                return {
                    "image_type": image_type,
                    "error": error_msg,
                    "response_time": response_time,
                    "status": "error"
                }

        except Exception as e:
            logger.error(f"❌ Test failed for {image_type}: {str(e)}")
            return {
                "image_type": image_type,
                "error": str(e),
                "status": "error"
            }

    def test_batch_predictions(self) -> List[Dict[str, Any]]:
        """Test multiple predictions with different image types"""
        logger.info("🔄 Running batch prediction tests...")

        test_cases = [
            "gradient",
            "red_square",
            "green_circle",
            "blue_triangle",
            "random_noise"
        ]

        results = []
        for i, image_type in enumerate(test_cases):
            # Add delay between requests to avoid rate limiting
            if i > 0:
                time.sleep(1)

            # Test with extras on first and last
            include_extras = (i == 0 or i == len(test_cases) - 1)
            result = self.test_single_prediction(image_type, include_extras)
            results.append(result)
            self.test_results.append(result)

        return results

    def benchmark_performance(self, num_requests: int = 5) -> Dict[str, Any]:
        """
        Benchmark server performance

        Args:
            num_requests: Number of requests to make

        Returns:
            Benchmark results
        """
        logger.info(f"📊 Benchmarking performance with {num_requests} requests...")

        # Use a simple image for benchmarking
        test_image = self.create_test_image_base64("gradient")
        if not test_image:
            return {"error": "Failed to create benchmark image"}

        payload = {"image_base64": test_image}
        times = []
        successful_requests = 0

        for i in range(num_requests):
            logger.debug(f"Benchmark request {i+1}/{num_requests}")

            start_time = time.time()
            response = self.make_prediction_request(payload, timeout=30)
            response_time = time.time() - start_time

            if response.get("status") == "success":
                times.append(response_time)
                successful_requests += 1
            else:
                logger.warning(f"Benchmark request {i+1} failed: {response.get('error')}")

            # Small delay between requests
            if i < num_requests - 1:
                time.sleep(0.5)

        if not times:
            return {"error": "No successful benchmark requests"}

        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        benchmark_results = {
            "total_requests": num_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / num_requests,
            "avg_response_time": avg_time,
            "std_response_time": std_time,
            "min_response_time": min_time,
            "max_response_time": max_time,
            "throughput_rps": 1.0 / avg_time if avg_time > 0 else 0
        }

        logger.info(f"📈 Benchmark Results:")
        logger.info(f"   Success Rate: {benchmark_results['success_rate']:.2%}")
        logger.info(f"   Avg Response Time: {avg_time:.3f}s ± {std_time:.3f}s")
        logger.info(f"   Min/Max Time: {min_time:.3f}s / {max_time:.3f}s")
        logger.info(f"   Throughput: {benchmark_results['throughput_rps']:.2f} RPS")

        return benchmark_results

    def test_error_handling(self) -> Dict[str, Any]:
        """Test various error conditions"""
        logger.info("🚨 Testing error handling...")

        error_tests = []

        # Test 1: Empty payload
        try:
            response = self.make_prediction_request({})
            error_tests.append({
                "test": "empty_payload",
                "status": "success" if "error" in response else "failed",
                "response": response
            })
        except Exception as e:
            error_tests.append({"test": "empty_payload", "status": "exception", "error": str(e)})

        # Test 2: Invalid base64
        try:
            response = self.make_prediction_request({"image_base64": "invalid_base64"})
            error_tests.append({
                "test": "invalid_base64",
                "status": "success" if "error" in response else "failed",
                "response": response
            })
        except Exception as e:
            error_tests.append({"test": "invalid_base64", "status": "exception", "error": str(e)})

        # Test 3: Malformed payload
        try:
            response = self.make_prediction_request({"wrong_field": "value"})
            error_tests.append({
                "test": "malformed_payload",
                "status": "success" if "error" in response else "failed",
                "response": response
            })
        except Exception as e:
            error_tests.append({"test": "malformed_payload", "status": "exception", "error": str(e)})

        logger.info(f"✅ Error handling tests completed: {len(error_tests)} tests")
        return {"error_tests": error_tests, "total_tests": len(error_tests)}

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive test report"""
        logger.info("📋 Generating comprehensive test report...")

        successful_tests = [r for r in self.test_results if r.get('status') == 'success']
        failed_tests = [r for r in self.test_results if r.get('status') == 'error']

        report = {
            "endpoint": self.api_endpoint,
            "total_tests": len(self.test_results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "success_rate": len(successful_tests) / len(self.test_results) if self.test_results else 0,
            "average_response_time": np.mean([r.get('response_time', 0) for r in successful_tests]) if successful_tests else 0,
            "test_details": self.test_results
        }

        # Print comprehensive report
        print("\n" + "="*80)
        print("🚀 REMOTE SERVER TEST REPORT")
        print("="*80)
        print(f"API Endpoint: {report['endpoint']}")
        print(f"Total Tests: {report['total_tests']}")
        print(f"Successful: {report['successful_tests']}")
        print(f"Failed: {report['failed_tests']}")
        print(f"Success Rate: {report['success_rate']:.2%}")
        print(f"Avg Response Time: {report['average_response_time']:.3f}s")

        if successful_tests:
            print(f"Estimated Throughput: {1/report['average_response_time']:.2f} RPS")

        print("\n📊 Individual Test Results:")
        for i, result in enumerate(self.test_results, 1):
            status_icon = "✅" if result['status'] == 'success' else "❌"
            image_type = result.get('image_type', 'unknown')

            if result['status'] == 'success':
                pred_class = result.get('predicted_class', 'N/A')
                confidence = result.get('confidence', 0)
                resp_time = result.get('response_time', 0)
                print(f"{i:2d}. {status_icon} {image_type:<15} -> {pred_class:<20} ({confidence:.4f}) [{resp_time:.3f}s]")
            else:
                error = result.get('error', 'Unknown error')
                print(f"{i:2d}. {status_icon} {image_type:<15} -> ERROR: {error}")

        print("="*80)

        return report

def main():
    """Main testing function"""
    print("🧪 Remote ONNX Model Server Tester")
    print("="*50)

    # Configuration - REPLACE WITH YOUR ACTUAL VALUES
    API_ENDPOINT = "https://api.cortex.cerebrium.ai/v4/p-4068f4c5/my-first-project/run  "  # Replace with actual endpoint
    API_KEY = "eyJjdHkiOiJKV1QiLCJlbmMiOiJBMjU2R0NNIiwiYWxnIjoiUlNBLU9BRVAifQ.vaVSTACwJMpRIJv1YYBF6Iobda0GTj2QuSYsFblQBoyDshS40eqWrAxRdhvmGQBAPetlY9PfAI8nk3tPsaxAn4bZcGgQCUj0i9yUAzQyH45vp1ya8xwGFkxBjODsdO77axee2XMX0x6ZnegQ8PD_45y7GzhWaovjKFNFd-HmghZUwBHliprDeuCfvgOFDqnLs2rmITaYxSZdP0DmToUx0ePiinfPHWQFiZf8yXbd7s7V-PuQtBbOrZSXmt9-NPYzlNWkn5Zdi78oRkmCNALtB3TzkpCT0rjQZkBGs1d4uTshv0WzSfixepD8lKpqvl6RSEy4fkAhBtmpcdWOkIn5Tg.G4gmQF0dulyEM2pv.enow4lr2dq31v5RNezec-f2otLnagsWV5kXVy4O1xO79XoyC68zX8Pmmj-Sqt8576TKGdqkoeqcy_49aeJnW5yBzPqkQXk6m7MQxdOSd2Dn_ycCZxflSEzHM_neaLeDfQCog2-sXaHlPHDGOld7crwArB2eAhvPPaWZiZDPWYdWUK87FlgUqtW9AqcbSFuGeI3RpCh5RpVVMzyDVMT15oKCBL1bZ_SgVqyhhx7Nay9qTEUVpCYkeo67pERyh2jinRr9jegDJVlpS6lhu__DkQTFIY6xq84vp4sA31DUeFlnyVqr_lGeERNOGmdasqauEuz0OxIwUCl8fR3PxXZAkqYebW0QPv7bC6Brtz91yBkdniq3R5jRZ0dq4F0QTxSAMsw3Mw9c1sZrF1Id5XRk5sQsL662AIfBgSWqT7epAs3nL0WOwPvxO5CGyFCUFkLKgXH8Y6H5V_LARHXZ4PPVT483xBCQqpn6zgxN5LHLAPOnuRzPGLTM9K0AcxloUJP6ftQOKlY9ydN2n_ijsEno6B4aT8BSKjIhNAQBGyrFDtvg-bgrLLDT4kTORnGWaYtIPF_m64y4NJgGsNq4KNHc4nX8SpK0AqMd-CNh1YY35UJ5QYNLdEtyUnxwY9yqC_5DTVQf5LhE1qqQ1L9p0jkLx1Xkd7lTYqbngkKJQznTIViiRagc-uD7iN4_xWY_kotNsr6w3KFzuNLsJlNOdJ2mwnSkn98zBHqLefGrRFDlKmUlU9XXvf742Y6BXPkUu0Jo31UkDbzUddB9Uhob5a2IZOLCkw4hqkx4FWvhlQztcvz8BYln2R00zPMz9SNswzrqnv89U_dUg-8pS3bO8_4TGM2yBr81f2jwAd38M0nNe5vfjDIoc--BSZlYjURBGKwne2RXVEtxmO0vSDOfctGRwsFzTDXpsIaou2vXANlayzNtd01DcWDVEzNKXjtoGKtVN0myMYBOf_QJbUBgOa7H_Qu2FqFo25FqWbQV5gECRM9AuPohmbd_DDvG9JDOxEJFxZt2uoxRKWRDNWJT879l4nlHKIqQNr--p_cEeR1MuvAoWpJueiFEOvYUqG5V3BS-ZZ0kjTaLQb56lLXXOuecHqC9d83Vmgcxd6agnt3t_kAFSUoFhvYeu3xAIfKzvAzeHl1N6Vik5nJFq0sy18eBprc00oSggvwWfm5fy-tdL4BApqdAzVn4GMLFOQ1Yfjr0-DbaHXfEdKbIl0tN-7kIluJgu6qfDAO_xVjU58_iWBZGFNa2Z-UTHLj-PcI8evT18H9YU1lRPT4xOuGCM5h9QTyubuszUxL9NJ5h5YZe3vNluCslAwexPtgshxQ.N8i64dDyI7uWuK_lZ-G8QA"  # Replace with actual API key

    # Check if configuration is set
    if "your-deployment-name" in API_ENDPOINT or "your-cerebrium-api-key" in API_KEY:
        logger.error("❌ Please update API_ENDPOINT and API_KEY with your actual Cerebrium deployment details!")
        logger.error("   1. Replace API_ENDPOINT with your Cerebrium deployment URL")
        logger.error("   2. Replace API_KEY with your Cerebrium API key")
        sys.exit(1)

    # Initialize tester
    tester = RemoteModelTester(API_ENDPOINT, API_KEY)

    try:
        # Test API connectivity
        if not tester.test_api_connectivity():
            logger.error("❌ Failed to connect to API. Please check your endpoint and API key.")
            sys.exit(1)

        # Run batch predictions
        batch_results = tester.test_batch_predictions()

        # Run performance benchmark
        benchmark_results = tester.benchmark_performance(num_requests=3)

        # Test error handling
        error_results = tester.test_error_handling()

        # Generate comprehensive report
        final_report = tester.generate_comprehensive_report()

        # Summary
        if final_report['success_rate'] >= 0.8:
            logger.info("🎉 Server testing completed successfully!")
            logger.info(f"✅ {final_report['successful_tests']}/{final_report['total_tests']} tests passed")
        else:
            logger.warning("⚠️ Some tests failed. Please review the results.")
            logger.warning(f"❌ {final_report['failed_tests']}/{final_report['total_tests']} tests failed")

        # Save detailed results
        results_file = "remote_test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "final_report": final_report,
                "benchmark_results": benchmark_results,
                "error_test_results": error_results
            }, f, indent=2)

        logger.info(f"📄 Detailed results saved to: {results_file}")

    except KeyboardInterrupt:
        logger.info("⏹️ Testing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Testing failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()