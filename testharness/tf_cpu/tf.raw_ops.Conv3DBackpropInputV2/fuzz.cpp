#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>

#define MAX_RANK 5
#define MIN_RANK 5
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
    void logError(const std::string& message, const uint8_t* data, size_t size) {
        std::cerr << "Error: " << message << std::endl;
    }
}

tensorflow::DataType parseFilterDataType(uint8_t selector) {
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

tensorflow::DataType parseInputSizesDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
            break;
    }
    return dtype;
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
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType input_sizes_dtype = parseInputSizesDataType(data[offset++]);
        tensorflow::DataType filter_dtype = parseFilterDataType(data[offset++]);
        
        std::vector<int64_t> input_sizes_shape = {5};
        tensorflow::Tensor input_sizes_tensor(input_sizes_dtype, tensorflow::TensorShape(input_sizes_shape));
        fillTensorWithDataByType(input_sizes_tensor, input_sizes_dtype, data, offset, size);
        
        std::vector<int64_t> filter_shape = parseShape(data, offset, size, 5);
        tensorflow::Tensor filter_tensor(filter_dtype, tensorflow::TensorShape(filter_shape));
        fillTensorWithDataByType(filter_tensor, filter_dtype, data, offset, size);
        
        std::vector<int64_t> out_backprop_shape = parseShape(data, offset, size, 5);
        tensorflow::Tensor out_backprop_tensor(filter_dtype, tensorflow::TensorShape(out_backprop_shape));
        fillTensorWithDataByType(out_backprop_tensor, filter_dtype, data, offset, size);
        
        std::vector<int> strides = {1, 1, 1, 1, 1};
        if (offset + 5 <= size) {
            strides[1] = std::max(1, static_cast<int>(data[offset++] % 5 + 1));
            strides[2] = std::max(1, static_cast<int>(data[offset++] % 5 + 1));
            strides[3] = std::max(1, static_cast<int>(data[offset++] % 5 + 1));
        }
        
        std::string padding = (offset < size && data[offset++] % 2 == 0) ? "SAME" : "VALID";
        
        std::string data_format = (offset < size && data[offset++] % 2 == 0) ? "NDHWC" : "NCDHW";
        
        std::vector<int> dilations = {1, 1, 1, 1, 1};
        
        auto input_sizes_op = tensorflow::ops::Const(root, input_sizes_tensor);
        auto filter_op = tensorflow::ops::Const(root, filter_tensor);
        auto out_backprop_op = tensorflow::ops::Const(root, out_backprop_tensor);
        
        auto conv3d_backprop = tensorflow::ops::Conv3DBackpropInputV2(
            root, input_sizes_op, filter_op, out_backprop_op, strides, padding,
            tensorflow::ops::Conv3DBackpropInputV2::DataFormat(data_format)
                .Dilations(dilations));
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({conv3d_backprop}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
