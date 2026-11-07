"""Export to ONNX, then FP16 / INT8 (dynamic) via onnxruntime. STATUS: not executed.
After quantising, re-run the quality metrics -- latency alone is not sufficient."""
import torch
from models.learned_isp.unet import AuroraUNet
def export(ckpt="models/checkpoints/aurora_unet.pt", out="models/onnx/aurora_unet.onnx"):
    m = AuroraUNet(); m.load_state_dict(torch.load(ckpt, map_location="cpu")); m.eval()
    torch.onnx.export(m, (torch.zeros(1, 4, 128, 128), torch.zeros(1, 5)), out, opset_version=17,
                      input_names=["packed", "meta"], output_names=["rgb"],
                      dynamic_axes={"packed": {2: "h", 3: "w"}, "rgb": {2: "H", 3: "W"}})
def quantise_int8(src="models/onnx/aurora_unet.onnx", dst="models/quantized/aurora_unet_int8.onnx"):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
if __name__ == "__main__": export(); quantise_int8()
