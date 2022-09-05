import os
import sys

import numpy as np
import pycuda.autoinit
import tensorrt as trt

# sys.path.insert(1, os.path.join(sys.path[0], ".."))
# import common

import argparse
import os

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

try:
    # Sometimes python does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
    return val * 1 << 30


def add_help(description):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args, _ = parser.parse_known_args()


def find_sample_data(description="Runs a TensorRT Python sample", subfolder="", find_files=[], err_msg=""):
    '''
    Parses sample arguments.

    Args:
        description (str): Description of the sample.
        subfolder (str): The subfolder containing data relevant to this sample
        find_files (str): A list of filenames to find. Each filename will be replaced with an absolute path.

    Returns:
        str: Path of data directory.
    '''

    # Standard command-line arguments for all samples.
    kDEFAULT_DATA_ROOT = os.path.join(os.sep, "usr", "src", "tensorrt", "data")
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--datadir", help="Location of the TensorRT sample data directory, and any additional data directories.", action="append", default=[kDEFAULT_DATA_ROOT])
    args, _ = parser.parse_known_args()

    def get_data_path(data_dir):
        # If the subfolder exists, append it to the path, otherwise use the provided path as-is.
        data_path = os.path.join(data_dir, subfolder)
        if not os.path.exists(data_path):
            if data_dir != kDEFAULT_DATA_ROOT:
                print("WARNING: " + data_path + " does not exist. Trying " + data_dir + " instead.")
            data_path = data_dir
        # Make sure data directory exists.
        if not (os.path.exists(data_path)) and data_dir != kDEFAULT_DATA_ROOT:
            print("WARNING: {:} does not exist. Please provide the correct data path with the -d option.".format(data_path))
        return data_path

    data_paths = [get_data_path(data_dir) for data_dir in args.datadir]
    return data_paths, locate_files(data_paths, find_files, err_msg)

def locate_files(data_paths, filenames, err_msg=""):
    """
    Locates the specified files in the specified data directories.
    If a file exists in multiple data directories, the first directory is used.

    Args:
        data_paths (List[str]): The data directories.
        filename (List[str]): The names of the files to find.

    Returns:
        List[str]: The absolute paths of the files.

    Raises:
        FileNotFoundError if a file could not be located.
    """
    found_files = [None] * len(filenames)
    for data_path in data_paths:
        # Find all requested files.
        for index, (found, filename) in enumerate(zip(found_files, filenames)):
            if not found:
                file_path = os.path.abspath(os.path.join(data_path, filename))
                if os.path.exists(file_path):
                    found_files[index] = file_path

    # Check that all files were found
    for f, filename in zip(found_files, filenames):
        if not f or not os.path.exists(f):
            raise FileNotFoundError("Could not find {:}. Searched in data paths: {:}\n{:}".format(filename, data_paths, err_msg))
    return found_files

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
TRT_LOGGER = trt.Logger()

def get_engine(onnx_file_path, engine_file_path):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(EXPLICIT_BATCH)
        parser = trt.OnnxParser(network, TRT_LOGGER)
        runtime = trt.Runtime(TRT_LOGGER)

        # Parse model file
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        print('Completed parsing of ONNX file')

        # Print input info
        print('Network inputs:')
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            print(tensor.name, trt.nptype(tensor.dtype), tensor.shape)

        network.get_input(0).shape = [1, 3,224,224]
        # network.get_input(1).shape = [10, 1, 1, 16]
        # network.get_input(2).shape = [6, 1]
        # network.get_input(3).shape = [6, 1, 1, 16]

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.REFIT)
        config.max_workspace_size = 1 << 28  # 256MiB

        print('Building an engine from file {}; this may take a while...'.format(
            onnx_file_path))
        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")

        with open(engine_file_path, "wb") as f:
            f.write(plan)
        return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def main():
    onnx_file_path = 'body_net_pare-ok.onnx'
    engine_file_path = "pare_old1.trt"

    onnx_file_path = 'pare_new.onnx'
    engine_file_path = "pare_new111.trt"
    # input
    # context = 'A quick brown fox jumps over the lazy dog.'
    # query = 'What color is the fox?'
    # cw_str, _ = preprocess(context)
    # get ravelled data
    # cw, cc, qw, qc = get_inputs(context, query)

    # Do inference with TensorRT
    # weights_names = ["Parameter576_B_0", "W_0"]
    # refit_weights_dict = {name : np.load("{}.npy".format(name)) for name in weights_names}
    # fake_weights_dict = {name : np.ones_like(weights) for name, weights in refit_weights_dict.items()}
    engine = get_engine(onnx_file_path, engine_file_path)
    # refitter = trt.Refitter(engine, TRT_LOGGER)

    # for weights_dict, answer_correct in [(fake_weights_dict, False), (refit_weights_dict, True)]:
    #     print("Refitting engine...")
    #     # To get a list of all refittable weights' names
    #     # in the network, use refitter.get_all_weights().

    #     # Refit named weights via set_named_weights
    #     for name in weights_names:
    #         refitter.set_named_weights(name, weights_dict[name])

    #     # Get missing weights names. This should return empty
    #     # lists in this case.
    #     missing_weights = refitter.get_missing_weights()
    #     assert len(
    #         missing_weights) == 0, "Refitter found missing weights. Call set_named_weights() or set_weights() for all missing weights"
    #     # Refit the engine with the new weights. This will return True if
    #     # the refit operation succeeded.
    #     assert refitter.refit_cuda_engine()

    #     inputs, outputs, bindings, stream = allocate_buffers(engine)
    #     print("Doing inference...")
    #     # Do inference
    #     # Set host input. The common.do_inference_v2 function will copy the input to the GPU before executing.
    #     inputs[0].host = cw
    #     inputs[1].host = cc
    #     inputs[2].host = qw
    #     inputs[3].host = qc
    #     execution_context = engine.create_execution_context()
    #     trt_outputs = do_inference_v2(execution_context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    #     start = np.asscalar(trt_outputs[0])
    #     end = np.asscalar(trt_outputs[1])
    #     answer = [w.encode() for w in cw_str[start:end + 1].reshape(-1)]
    #     assert answer_correct == (answer == [b'brown'])
    print("Passed")


if __name__ == '__main__':
    main()