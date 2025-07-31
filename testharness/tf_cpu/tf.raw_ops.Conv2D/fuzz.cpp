#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 4
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 5) {
        case 0:
            dtype = tensorflow::DT_HALF;
            break;
        case 1:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 2:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 3:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 4:
            dtype = tensorflow::DT_INT32;
            break;
    }
    return dtype;
}

uint8_t parseRank(uint8_t byte) {
    constexpr uint8_t range = MAX_RANK - MIN_RANK + 1;
    uint8_t rank = byte % range + MIN_RANK;
    return rank;
}

std::vector<int64_t> parseShape(const uint8_t* data, size_t& offset, size_t total_size, uint8_t rank) {
    if (rank == 0) {
        return {};
    }

    std::vector<int64_t> shape;
    shape.reserve(rank);
    const auto sizeof_dim = sizeof(int64_t);

    for (uint8_t i = 0; i < rank; ++i) {
        if (offset + sizeof_dim <= total_size) {
            int64_t dim_val;
            std::memcpy(&dim_val, data + offset, sizeof_dim);
            offset += sizeof_dim;
            
            dim_val = MIN_TENSOR_SHAPE_DIMS_TF +
                    static_cast<int64_t>((static_cast<uint64_t>(std::abs(dim_val)) %
                                        static_cast<uint64_t>(MAX_TENSOR_SHAPE_DIMS_TF - MIN_TENSOR_SHAPE_DIMS_TF + 1)));

            shape.push_back(dim_val);
        } else {
             shape.push_back(1);
        }
    }

    return shape;
}

template <typename T>
void fillTensorWithData(tensorflow::Tensor& tensor, const uint8_t* data,
                        size_t& offset, size_t total_size) {
    auto flat = tensor.flat<T>();
    const size_t num_elements = flat.size();
    const size_t element_size = sizeof(T);

    for (size_t i = 0; i < num_elements; ++i) {
        if (offset + element_size <= total_size) {
            T value;
            std::memcpy(&value, data + offset, element_size);
            offset += element_size;
            flat(i) = value;
        } else {
            flat(i) = T{};
        }
    }
}

void fillTensorWithDataByType(tensorflow::Tensor& tensor,
                              tensorflow::DataType dtype, const uint8_t* data,
                              size_t& offset, size_t total_size) {
    switch (dtype) {
        case tensorflow::DT_FLOAT:
            fillTensorWithData<float>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_DOUBLE:
            fillTensorWithData<double>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_BFLOAT16:
            fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
            break;
        default:
            break;
    }
}

std::string parsePadding(uint8_t selector) {
    switch (selector % 3) {
        case 0:
            return "SAME";
        case 1:
            return "VALID";
        case 2:
            return "EXPLICIT";
        default:
            return "SAME";
    }
}

std::string parseDataFormat(uint8_t selector) {
    switch (selector % 2) {
        case 0:
            return "NHWC";
        case 1:
            return "NCHW";
        default:
            return "NHWC";
    }
}

std::vector<int> parseStrides(const uint8_t* data, size_t& offset, size_t total_size) {
    std::vector<int> strides(4);
    strides[0] = 1;
    strides[3] = 1;
    
    if (offset + sizeof(int) <= total_size) {
        int stride_val;
        std::memcpy(&stride_val, data + offset, sizeof(int));
        offset += sizeof(int);
        stride_val = 1 + (std::abs(stride_val) % 3);
        strides[1] = stride_val;
        strides[2] = stride_val;
    } else {
        strides[1] = 1;
        strides[2] = 1;
    }
    
    return strides;
}

std::vector<int> parseDilations(const uint8_t* data, size_t& offset, size_t total_size) {
    std::vector<int> dilations(4);
    dilations[0] = 1;
    dilations[3] = 1;
    
    if (offset + sizeof(int) <= total_size) {
        int dilation_val;
        std::memcpy(&dilation_val, data + offset, sizeof(int));
        offset += sizeof(int);
        dilation_val = 1 + (std::abs(dilation_val) % 3);
        dilations[1] = dilation_val;
        dilations[2] = dilation_val;
    } else {
        dilations[1] = 1;
        dilations[2] = 1;
    }
    
    return dilations;
}

std::vector<int> parseExplicitPaddings(const uint8_t* data, size_t& offset, size_t total_size, const std::string& padding) {
    std::vector<int> explicit_paddings;
    
    if (padding == "EXPLICIT") {
        explicit_paddings.resize(8);
        for (int i = 0; i < 8; ++i) {
            if (offset + sizeof(int) <= total_size) {
                int pad_val;
                std::memcpy(&pad_val, data + offset, sizeof(int));
                offset += sizeof(int);
                explicit_paddings[i] = std::abs(pad_val) % 5;
            } else {
                explicit_paddings[i] = 0;
            }
        }
    }
    
    return explicit_paddings;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 100) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        std::vector<int64_t> input_shape = parseShape(data, offset, size, 4);
        std::vector<int64_t> filter_shape = parseShape(data, offset, size, 4);
        
        if (input_shape.size() != 4 || filter_shape.size() != 4) {
            return 0;
        }
        
        filter_shape[2] = input_shape[3];
        
        tensorflow::TensorShape input_tensor_shape(input_shape);
        tensorflow::TensorShape filter_tensor_shape(filter_shape);
        
        tensorflow::Tensor input_tensor(dtype, input_tensor_shape);
        tensorflow::Tensor filter_tensor(dtype, filter_tensor_shape);
        
        fillTensorWithDataByType(input_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(filter_tensor, dtype, data, offset, size);
        
        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto filter_op = tensorflow::ops::Const(root, filter_tensor);
        
        std::vector<int> strides = parseStrides(data, offset, size);
        std::string padding = parsePadding(data[offset++]);
        std::string data_format = parseDataFormat(data[offset++]);
        std::vector<int> dilations = parseDilations(data, offset, size);
        std::vector<int> explicit_paddings = parseExplicitPaddings(data, offset, size, padding);
        bool use_cudnn_on_gpu = (data[offset++] % 2) == 0;
        
        auto conv2d_op = tensorflow::ops::Conv2D(
            root, 
            input_op, 
            filter_op, 
            strides, 
            padding,
            tensorflow::ops::Conv2D::DataFormat(data_format)
                .Dilations(dilations)
                .UseCudnnOnGpu(use_cudnn_on_gpu)
        );
        
        if (padding == "EXPLICIT" && !explicit_paddings.empty()) {
            // If we need to use explicit paddings, we'll need to use a different approach
            // since the C++ API doesn't directly expose explicit_paddings attribute
            // This is a workaround - in real code you might need to use the raw ops API
        }
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({conv2d_op}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}