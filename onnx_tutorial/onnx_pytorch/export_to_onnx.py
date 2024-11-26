import onnx
import torch

from .model import MyModel

ONNX_MODEL_PATH = "onnx_models/my_image_classifier.onnx"

# After "*.onnx" file is created, you can visualize it using Netron.
# You can run Netron directly from the browser: https://netron.app/


def export_to_onnx() -> None:
    # Store ONNX model in torch.onnx.ONNXProgram as a binary protobuf file
    torch_model = MyModel()
    torch_input = torch.randn(1, 1, 32, 32)
    onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
    # Save the ONNX model to disk
    onnx_program.save(ONNX_MODEL_PATH)


def export_to_onnx_check() -> None:
    # Load the ONNX model from disk
    onnx_model = onnx.load(ONNX_MODEL_PATH)
    # Check the ONNX model
    onnx.checker.check_model(onnx_model)
