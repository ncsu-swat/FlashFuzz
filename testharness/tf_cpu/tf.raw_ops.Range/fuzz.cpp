#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include <cstring>
#include <iostream>
#include <vector>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 10) {
        case 0:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 1:
            dtype = tensorflow::DT_HALF;
            break;
        case 2:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 3:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 4:
            dtype = tensorflow::DT_INT8;
            break;
        case 5:
            dtype = tensorflow::DT_INT16;
            break;
        case 6:
            dtype = tensorflow::DT_INT32;
            break;
        case 7:
            dtype = tensorflow::DT_INT64;
            break;
        case 8:
            dtype = tensorflow::DT_UINT16;
            break;
        case 9:
            dtype = tensorflow::DT_UINT32;
            break;
    }
    return dtype;
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
        case tensorflow::DT_INT16:
            fillTensorWithData<int16_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT8:
            fillTensorWithData<int8_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT16:
            fillTensorWithData<uint16_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT32:
            fillTensorWithData<uint32_t>(tensor, data, offset, total_size);
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
    if (size < 4) return 0;
    
    size_t offset = 0;
    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        tensorflow::TensorShape scalar_shape({});
        
        tensorflow::Tensor start_tensor(dtype, scalar_shape);
        tensorflow::Tensor limit_tensor(dtype, scalar_shape);
        tensorflow::Tensor delta_tensor(dtype, scalar_shape);
        
        fillTensorWithDataByType(start_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(limit_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(delta_tensor, dtype, data, offset, size);
        
        auto start_op = tensorflow::ops::Const(root, start_tensor);
        auto limit_op = tensorflow::ops::Const(root, limit_tensor);
        auto delta_op = tensorflow::ops::Const(root, delta_tensor);
        
        auto range_op = tensorflow::ops::Range(root, start_op, limit_op, delta_op);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({range_op}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
