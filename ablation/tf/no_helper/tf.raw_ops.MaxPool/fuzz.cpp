#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions for input tensor (4D: batch, height, width, channels)
        int32_t batch = (data[offset] % 4) + 1;
        offset++;
        int32_t height = (data[offset] % 32) + 1;
        offset++;
        int32_t width = (data[offset] % 32) + 1;
        offset++;
        int32_t channels = (data[offset] % 16) + 1;
        offset++;
        
        // Extract ksize parameters (4D)
        int32_t ksize_batch = 1;
        int32_t ksize_height = (data[offset] % 8) + 1;
        offset++;
        int32_t ksize_width = (data[offset] % 8) + 1;
        offset++;
        int32_t ksize_channels = 1;
        
        // Extract stride parameters (4D)
        int32_t stride_batch = 1;
        int32_t stride_height = (data[offset] % 4) + 1;
        offset++;
        int32_t stride_width = (data[offset] % 4) + 1;
        offset++;
        int32_t stride_channels = 1;
        
        // Extract padding type
        std::string padding;
        switch (data[offset] % 3) {
            case 0: padding = "VALID"; break;
            case 1: padding = "SAME"; break;
            case 2: padding = "EXPLICIT"; break;
        }
        offset++;
        
        // Extract data format
        std::string data_format;
        switch (data[offset] % 3) {
            case 0: data_format = "NHWC"; break;
            case 1: data_format = "NCHW"; break;
            case 2: data_format = "NCHW_VECT_C"; break;
        }
        offset++;
        
        // Extract data type
        tensorflow::DataType dtype;
        switch (data[offset] % 11) {
            case 0: dtype = tensorflow::DT_HALF; break;
            case 1: dtype = tensorflow::DT_BFLOAT16; break;
            case 2: dtype = tensorflow::DT_FLOAT; break;
            case 3: dtype = tensorflow::DT_DOUBLE; break;
            case 4: dtype = tensorflow::DT_INT32; break;
            case 5: dtype = tensorflow::DT_INT64; break;
            case 6: dtype = tensorflow::DT_UINT8; break;
            case 7: dtype = tensorflow::DT_INT16; break;
            case 8: dtype = tensorflow::DT_INT8; break;
            case 9: dtype = tensorflow::DT_UINT16; break;
            case 10: dtype = tensorflow::DT_QINT8; break;
        }
        offset++;
        
        // Adjust dimensions based on data format
        tensorflow::TensorShape input_shape;
        if (data_format == "NHWC") {
            input_shape = tensorflow::TensorShape({batch, height, width, channels});
        } else if (data_format == "NCHW") {
            input_shape = tensorflow::TensorShape({batch, channels, height, width});
        } else { // NCHW_VECT_C
            input_shape = tensorflow::TensorShape({batch, channels, height, width});
        }
        
        // Create input tensor
        tensorflow::Tensor input_tensor(dtype, input_shape);
        
        // Fill tensor with remaining data
        size_t tensor_size = input_tensor.NumElements();
        size_t element_size = tensorflow::DataTypeSize(dtype);
        
        if (dtype == tensorflow::DT_FLOAT) {
            auto flat = input_tensor.flat<float>();
            for (int i = 0; i < tensor_size && offset < size; ++i) {
                flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
                offset++;
            }
        } else if (dtype == tensorflow::DT_INT32) {
            auto flat = input_tensor.flat<int32_t>();
            for (int i = 0; i < tensor_size && offset + 3 < size; ++i) {
                flat(i) = *reinterpret_cast<const int32_t*>(&data[offset]);
                offset += 4;
            }
        } else if (dtype == tensorflow::DT_UINT8) {
            auto flat = input_tensor.flat<uint8_t>();
            for (int i = 0; i < tensor_size && offset < size; ++i) {
                flat(i) = data[offset];
                offset++;
            }
        }
        
        // Create TensorFlow session
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input placeholder
        auto input_ph = tensorflow::ops::Placeholder(root, dtype);
        
        // Set up ksize and strides based on data format
        std::vector<int> ksize_vec, strides_vec;
        if (data_format == "NHWC") {
            ksize_vec = {ksize_batch, ksize_height, ksize_width, ksize_channels};
            strides_vec = {stride_batch, stride_height, stride_width, stride_channels};
        } else {
            ksize_vec = {ksize_batch, ksize_channels, ksize_height, ksize_width};
            strides_vec = {stride_batch, stride_channels, stride_height, stride_width};
        }
        
        // Create MaxPool operation
        auto maxpool_attrs = tensorflow::ops::MaxPool::Attrs()
            .DataFormat(data_format)
            .Padding(padding);
            
        auto maxpool_op = tensorflow::ops::MaxPool(root, input_ph, ksize_vec, strides_vec, maxpool_attrs);
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({{input_ph, input_tensor}}, {maxpool_op}, &outputs);
        
        if (!status.ok()) {
            std::cout << "MaxPool operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}