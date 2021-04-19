from pathlib import Path

import pycuda.driver as cuda

import pycuda.autoinit

import numpy as np

import time


import tensorrt as trt

def build_engine(model_file, max_ws=(1 << 30)*2, fp16=True):
    print("building engine")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    builder.fp16_mode = fp16
    config = builder.create_builder_config()
    config.max_workspace_size = max_ws
    config.default_device_type = trt.DeviceType.DLA
    config.DLA_core = 0
    if fp16:
        config.flags |= 1 << int(trt.BuilderFlag.FP16)

    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_file, 'rb') as model:
            parsed = parser.parse(model.read())
            print("network.num_layers", network.num_layers)
            for layer in range(network.num_layers):
                print("layer ", layer, " is DLA ok?", config.can_run_on_DLA(network.get_layer(layer)))
            last_layer = network.get_layer(network.num_layers - 1)
            network.mark_output(last_layer.get_output(0))
            engine = builder.build_engine(network, config=config)
            return engine

def alloc_buf(engine):
    # host cpu mem
    h_in_size = trt.volume(engine.get_binding_shape(0))
    h_out_size = trt.volume(engine.get_binding_shape(1))
    h_in_dtype = trt.nptype(engine.get_binding_dtype(0))
    h_out_dtype = trt.nptype(engine.get_binding_dtype(1))
    in_cpu = cuda.pagelocked_empty(h_in_size, h_in_dtype)
    out_cpu = cuda.pagelocked_empty(h_out_size, h_out_dtype)
    # allocate gpu mem
    in_gpu = cuda.mem_alloc(in_cpu.nbytes)
    out_gpu = cuda.mem_alloc(out_cpu.nbytes)
    stream = cuda.Stream()
    return in_cpu, out_cpu, in_gpu, out_gpu, stream

def inference(engine, context, inputs, out_cpu, in_gpu, out_gpu, stream):
    # async version
    # with engine.create_execution_context() as context:  # cost time to initialize
    # cuda.memcpy_htod_async(in_gpu, inputs, stream)
    # context.execute_async(1, [int(in_gpu), int(out_gpu)], stream.handle, None)
    # cuda.memcpy_dtoh_async(out_cpu, out_gpu, stream)
    # stream.synchronize()

    # sync version
    cuda.memcpy_htod(in_gpu, inputs)
    context.execute(1, [int(in_gpu), int(out_gpu)])
    cuda.memcpy_dtoh(out_cpu, out_gpu)
    return out_cpu

model_path = Path('./models/FCN-ResNet18-MHP-512x320/fcn_resnet18.onnx')
engine = build_engine(model_path)

with open('./models/engine512x320.trt', 'wb') as f:
    f.write(bytearray(engine.serialize()))

# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# runtime = trt.Runtime(TRT_LOGGER)
# with open('./models/engine.trt', 'rb') as f:
#     engine_bytes = f.read()
#     engine = runtime.deserialize_cuda_engine(engine_bytes)

segContext = engine.create_execution_context()

inputs = np.random.random((512, 320, 1, 1)).astype(np.float32)

in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
res = inference(engine, segContext, inputs.reshape(-1), out_cpu, in_gpu, out_gpu, stream)

print(res)


#     import tensorrt as trt
# from pathlib import Path

# model_path = Path('./models/FCN-ResNet18-MHP-512x320/fcn_resnet18.onnx')
# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
# with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
#     with open(model_path, 'rb') as model:
#         if not parser.parse(model.read()):
#             for error in range(parser.num_errors):
#                 print(parser.get_error(error))




# import tensorrt as trt


# model_path = "model.onnx"
# input_size = 32

# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# def build_engine(model_path):
#     with trt.Builder(TRT_LOGGER) as builder, \
#         builder.create_network() as network, \
#         trt.OnnxParser(network, TRT_LOGGER) as parser: 
#         builder.max_workspace_size = 2 * 1<<30 # 2 GB
#         builder.max_batch_size = 1
#         with open(model_path, "rb") as f:
#             parser.parse(f.read())
#         engine = builder.build_cuda_engine(network)
#         return engine






# if __name__ == "__main__":
#     inputs = np.random.random((1, 3, input_size, input_size)).astype(np.float32)
#     engine = build_engine(model_path)
#     context = engine.create_execution_context()
#     for _ in range(10):
#         t1 = time.time()
#         in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
#         res = inference(engine, context, inputs.reshape(-1), out_cpu, in_gpu, out_gpu, stream)
#         print(res)
#         print("cost time: ", time.time()-t1)


# # tensorrt docker image: docker pull nvcr.io/nvidia/tensorrt:19.09-py3 (See: https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt/tags)
# # NOTE: cuda driver >= 418