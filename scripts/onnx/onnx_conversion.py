import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import argparse
import os
from pytorch_model import Classifier, BasicBlock


class ONNXConverter:

    def __init__(self, pytorch_model_path: str, onnx_output_path: str):
        self.pytorch_model_path = pytorch_model_path
        self.onnx_output_path = onnx_output_path
        self.model = None

    def load_pytorch_model(self):
        print("Loading PyTorch model...")

        # Initialize model architecture
        self.model = Classifier(BasicBlock, [2, 2, 2, 2], num_classes=1000)

        # Loading the trained weights
        if not os.path.exists(self.pytorch_model_path):
            raise FileNotFoundError(f"PyTorch model not found at: {self.pytorch_model_path}")

        self.model.load_state_dict(torch.load(self.pytorch_model_path, map_location='cpu'))
        self.model.eval()

        print("Model loaded ...")
        return self.model

    def convert_to_onnx(self, input_shape=(1, 3, 224, 224)):

        if self.model is None:
            self.load_pytorch_model()

        dummy_input = torch.randn(*input_shape)

        # Export to ONNX
        torch.onnx.export(
            self.model,  # PyTorch model
            dummy_input,  # Model input (or a tuple for multiple inputs)
            self.onnx_output_path,  # Output file path
            export_params=True,
            opset_version=11,  # This is basically ONNX version
            do_constant_folding=True,
            input_names=['input'],  # Model input
            output_names=['output'],  # Model output
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        print("Model converted, ALHAMDULILLAH")

    # I have used CLAUDE AI for the following function for validity !
    def validate_onnx_model(self, test_image_path=None):
        print("Validating ONNX model...")

        onnx_model = onnx.load(self.onnx_output_path)
        onnx.checker.check_model(onnx_model)

        ort_session = ort.InferenceSession(self.onnx_output_path)

        if test_image_path and os.path.exists(test_image_path):
            img = Image.open(test_image_path)
            test_input = self.model.preprocess_image(img).unsqueeze(0).numpy()
        else:
            test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

        with torch.no_grad():
            pytorch_input = torch.from_numpy(test_input)
            pytorch_output = self.model(pytorch_input).numpy()

        onnx_output = ort_session.run(None, {'input': test_input})[0]

        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        print(f"Maximum difference between PyTorch and ONNX outputs: {max_diff}")

        if max_diff < 1e-5:
            print("ONNX model validation successful!")
            return True
        else:
            print("ONNX model validation failed - outputs differ significantly")
            return False


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
    parser.add_argument('--pytorch_model', type=str, default='pytorch_model_weights.pth',
                        help='Path to PyTorch model weights')
    parser.add_argument('--onnx_output', type=str, default='model.onnx',
                        help='Output path for ONNX model')
    parser.add_argument('--test_image', type=str, default=None,
                        help='Path to test image for validation')
    parser.add_argument('--validate', action='store_true',
                        help='Validate converted model')
    parser.add_argument('--info', action='store_true',
                        help='Show model information')

    args = parser.parse_args()
    converter = ONNXConverter(args.pytorch_model, args.onnx_output)
    converter.convert_to_onnx()
    if args.validate:
        converter.validate_onnx_model(args.test_image)
        print("\nConversion completed successfully!")
        print(f"ONNX model saved at: {args.onnx_output}")


if __name__ == "__main__":
    main()
