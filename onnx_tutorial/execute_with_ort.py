import numpy as np
import onnxruntime as ort
import torch

from .model import MyModel

ONNX_MODEL_PATH = "./onnx_tutorial/my_image_classifier.onnx"


def execute_with_ort():
    # Setup the PyTorch model and input
    torch_model = MyModel()
    torch_input = torch.randn(1, 1, 32, 32)
    onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
    onnx_program.save(ONNX_MODEL_PATH)

    onnx_input = onnx_program.adapt_torch_inputs_to_onnx(torch_input)

    print(f"ONNX input length: {len(onnx_input)}")
    print(f"ONNX input sample: {onnx_input}")

    # Create an ONNX Runtime InferenceSession
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH)

    onnxruntime_input = {
        k.name: _to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)
    }  # type: ignore
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)

    # Compare the PyTorch results with the ones from the ONNX Runtime
    torch_outputs = torch_model(torch_input)
    torch_outputs = onnx_program.adapt_torch_outputs_to_onnx(torch_outputs)

    assert len(torch_outputs) == len(onnxruntime_outputs)
    for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
        torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

    print("PyTorch and ONNX Runtime output matched!")
    print(f"Output length: {len(onnxruntime_outputs)}")
    print(f"Output sample: {onnxruntime_outputs}")


def _to_numpy(tensor) -> np.ndarray:
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )
