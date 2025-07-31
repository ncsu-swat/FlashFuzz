#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
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

tensorflow::DataType parseDataTypeForStartStop(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
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
    }
    return dtype;
}

tensorflow::DataType parseDataTypeForNum(uint8_t selector) {
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
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType start_stop_dtype = parseDataTypeForStartStop(data[offset++]);
        tensorflow::DataType num_dtype = parseDataTypeForNum(data[offset++]);

        tensorflow::TensorShape start_shape({});
        tensorflow::Tensor start_tensor(start_stop_dtype, start_shape);
        fillTensorWithDataByType(start_tensor, start_stop_dtype, data, offset, size);

        tensorflow::TensorShape stop_shape({});
        tensorflow::Tensor stop_tensor(start_stop_dtype, stop_shape);
        fillTensorWithDataByType(stop_tensor, start_stop_dtype, data, offset, size);

        tensorflow::TensorShape num_shape({});
        tensorflow::Tensor num_tensor(num_dtype, num_shape);
        fillTensorWithDataByType(num_tensor, num_dtype, data, offset, size);

        if (num_dtype == tensorflow::DT_INT32) {
            auto num_flat = num_tensor.flat<int32_t>();
            if (num_flat(0) < 0) {
                num_flat(0) = std::abs(num_flat(0)) % 100 + 1;
            }
            if (num_flat(0) == 0) {
                num_flat(0) = 1;
            }
        } else if (num_dtype == tensorflow::DT_INT64) {
            auto num_flat = num_tensor.flat<int64_t>();
            if (num_flat(0) < 0) {
                num_flat(0) = std::abs(num_flat(0)) % 100 + 1;
            }
            if (num_flat(0) == 0) {
                num_flat(0) = 1;
            }
        }

        auto start_input = tensorflow::ops::Const(root, start_tensor);
        auto stop_input = tensorflow::ops::Const(root, stop_tensor);
        auto num_input = tensorflow::ops::Const(root, num_tensor);

        // Use the correct API for LinSpace
        auto linspace_op = tensorflow::ops::LinSpace(root.WithOpName("LinSpace"), 
                                                    start_input, 
                                                    stop_input, 
                                                    num_input);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({linspace_op}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}