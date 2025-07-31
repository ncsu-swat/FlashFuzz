#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 2
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
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

std::vector<int> parseStrides(const uint8_t* data, size_t& offset, size_t total_size, int rank) {
    std::vector<int> strides;
    strides.push_back(1);
    
    for (int i = 0; i < rank; ++i) {
        if (offset < total_size) {
            int stride = (data[offset] % 3) + 1;
            strides.push_back(stride);
            offset++;
        } else {
            strides.push_back(1);
        }
    }
    
    strides.push_back(1);
    return strides;
}

std::string parsePadding(uint8_t byte) {
    switch (byte % 3) {
        case 0: return "SAME";
        case 1: return "VALID";
        case 2: return "EXPLICIT";
        default: return "SAME";
    }
}

std::string parseDataFormat(uint8_t byte) {
    return (byte % 2 == 0) ? "NHWC" : "NCHW";
}

std::vector<int> parseDilations(const uint8_t* data, size_t& offset, size_t total_size, int rank) {
    std::vector<int> dilations;
    dilations.push_back(1);
    
    for (int i = 0; i < rank; ++i) {
        if (offset < total_size) {
            int dilation = (data[offset] % 3) + 1;
            dilations.push_back(dilation);
            offset++;
        } else {
            dilations.push_back(1);
        }
    }
    
    dilations.push_back(1);
    return dilations;
}

std::vector<int> parseExplicitPaddings(const uint8_t* data, size_t& offset, size_t total_size, const std::string& padding, int rank) {
    std::vector<int> explicit_paddings;
    
    if (padding == "EXPLICIT") {
        for (int i = 0; i < (rank + 2) * 2; ++i) {
            if (offset < total_size) {
                int pad = data[offset] % 4;
                explicit_paddings.push_back(pad);
                offset++;
            } else {
                explicit_paddings.push_back(0);
            }
        }
    }
    
    return explicit_paddings;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        uint8_t input_rank = parseRank(data[offset++]);
        if (input_rank < 3 || input_rank > 4) input_rank = 4;
        
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        uint8_t filter_rank = input_rank;
        std::vector<int64_t> filter_shape = parseShape(data, offset, size, filter_rank);
        
        if (input_shape.empty() || filter_shape.empty()) return 0;
        
        int spatial_dims = input_rank - 2;
        
        std::string data_format = parseDataFormat(data[offset++]);
        
        int64_t in_channels, out_channels;
        if (data_format == "NHWC") {
            in_channels = input_shape[input_rank - 1];
            filter_shape[filter_rank - 2] = in_channels;
            out_channels = filter_shape[filter_rank - 1];
        } else {
            in_channels = input_shape[1];
            filter_shape[filter_rank - 2] = in_channels;
            out_channels = filter_shape[filter_rank - 1];
        }
        
        tensorflow::Tensor input_tensor(dtype, tensorflow::TensorShape(input_shape));
        tensorflow::Tensor filter_tensor(dtype, tensorflow::TensorShape(filter_shape));
        
        fillTensorWithDataByType(input_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(filter_tensor, dtype, data, offset, size);
        
        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto filter_op = tensorflow::ops::Const(root, filter_tensor);
        
        std::vector<int> strides = parseStrides(data, offset, size, spatial_dims);
        std::string padding = parsePadding(data[offset++]);
        std::vector<int> explicit_paddings = parseExplicitPaddings(data, offset, size, padding, spatial_dims);
        std::vector<int> dilations = parseDilations(data, offset, size, spatial_dims);
        
        auto conv_op = tensorflow::ops::Conv2D(root, input_op, filter_op, strides, padding);
        
        if (padding == "EXPLICIT" && !explicit_paddings.empty()) {
            conv_op = tensorflow::ops::Conv2D(root, input_op, filter_op, strides, "EXPLICIT", explicit_paddings);
        }
        
        if (!dilations.empty()) {
            conv_op = tensorflow::ops::Conv2D(root, input_op, filter_op, strides, padding, dilations);
        }
        
        if (padding == "EXPLICIT" && !explicit_paddings.empty() && !dilations.empty()) {
            conv_op = tensorflow::ops::Conv2D(root, input_op, filter_op, strides, "EXPLICIT", dilations, explicit_paddings);
        }
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({conv_op}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}