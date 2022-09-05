from onnxsim import simplify
import onnx


def simplify_onnx(path):
    out_path = path.replace('.onnx','_s.onnx')
    onnx_model = onnx.load(path)  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, out_path)
    print('finished exporting onnx')