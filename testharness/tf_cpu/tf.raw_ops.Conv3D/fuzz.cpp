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

#define MAX_RANK 5
#define MIN_RANK 5
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
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

std::vector<int> parseStrides(const uint8_t* data, size_t& offset, size_t total_size) {
    std::vector<int> strides(5);
    strides[0] = 1;
    strides[4] = 1;
    
    for (int i = 1; i < 4; ++i) {
        if (offset < total_size) {
            uint8_t stride_val = data[offset++];
            strides[i] = 1 + (stride_val % 3);
        } else {
            strides[i] = 1;
        }
    }
    
    return strides;
}

std::string parsePadding(uint8_t byte) {
    return (byte % 2 == 0) ? "VALID" : "SAME";
}

std::string parseDataFormat(uint8_t byte) {
    return (byte % 2 == 0) ? "NDHWC" : "NCDHW";
}

std::vector<int> parseDilations(const uint8_t* data, size_t& offset, size_t total_size) {
    std::vector<int> dilations(5);
    dilations[0] = 1;
    dilations[4] = 1;
    
    for (int i = 1; i < 4; ++i) {
        if (offset < total_size) {
            uint8_t dilation_val = data[offset++];
            dilations[i] = 1 + (dilation_val % 3);
        } else {
            dilations[i] = 1;
        }
    }
    
    return dilations;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        std::vector<int64_t> input_shape = parseShape(data, offset, size, 5);
        std::vector<int64_t> filter_shape = parseShape(data, offset, size, 5);
        
        if (input_shape.size() != 5 || filter_shape.size() != 5) {
            return 0;
        }
        
        filter_shape[3] = input_shape[4];
        
        tensorflow::Tensor input_tensor(dtype, tensorflow::TensorShape(input_shape));
        tensorflow::Tensor filter_tensor(dtype, tensorflow::TensorShape(filter_shape));
        
        fillTensorWithDataByType(input_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(filter_tensor, dtype, data, offset, size);
        
        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto filter_op = tensorflow::ops::Const(root, filter_tensor);
        
        std::vector<int> strides = parseStrides(data, offset, size);
        std::string padding = parsePadding(data[offset++]);
        std::string data_format = parseDataFormat(data[offset++]);
        std::vector<int> dilations = parseDilations(data, offset, size);
        
        auto conv3d_op = tensorflow::ops::Conv3D(
            root,
            input_op,
            filter_op,
            strides,
            padding,
            tensorflow::ops::Conv3D::Attrs()
                .DataFormat(data_format)
                .Dilations(dilations)
        );
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({conv3d_op}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
