#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>
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
    switch (selector % 6) {
        case 0:
            dtype = tensorflow::DT_QINT8;
            break;
        case 1:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 2:
            dtype = tensorflow::DT_QINT32;
            break;
        case 3:
            dtype = tensorflow::DT_QINT16;
            break;
        case 4:
            dtype = tensorflow::DT_QUINT16;
            break;
        case 5:
            dtype = tensorflow::DT_UINT8;
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
        case tensorflow::DT_UINT8:
            fillTensorWithData<uint8_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT8:
            fillTensorWithData<tensorflow::qint8>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QUINT8:
            fillTensorWithData<tensorflow::quint8>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT32:
            fillTensorWithData<tensorflow::qint32>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT16:
            fillTensorWithData<tensorflow::qint16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QUINT16:
            fillTensorWithData<tensorflow::quint16>(tensor, data, offset, total_size);
            break;
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType tensor_dtype = parseDataType(data[offset++]);
        
        uint8_t tensor_rank = parseRank(data[offset++]);
        std::vector<int64_t> tensor_shape = parseShape(data, offset, size, tensor_rank);
        
        tensorflow::TensorShape tf_tensor_shape;
        for (int64_t dim : tensor_shape) {
            tf_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_tensor(tensor_dtype, tf_tensor_shape);
        fillTensorWithDataByType(input_tensor, tensor_dtype, data, offset, size);
        
        uint8_t shape_rank = parseRank(data[offset++]);
        std::vector<int64_t> new_shape_dims = parseShape(data, offset, size, shape_rank);
        
        tensorflow::TensorShape shape_tensor_shape;
        shape_tensor_shape.AddDim(new_shape_dims.size());
        tensorflow::Tensor shape_tensor(tensorflow::DT_INT32, shape_tensor_shape);
        auto shape_flat = shape_tensor.flat<int32_t>();
        for (size_t i = 0; i < new_shape_dims.size(); ++i) {
            shape_flat(i) = static_cast<int32_t>(new_shape_dims[i]);
        }
        
        float input_min_val = 0.0f;
        float input_max_val = 1.0f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&input_min_val, data + offset, sizeof(float));
            offset += sizeof(float);
        }
        if (offset + sizeof(float) <= size) {
            std::memcpy(&input_max_val, data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        if (input_min_val > input_max_val) {
            std::swap(input_min_val, input_max_val);
        }
        
        tensorflow::Tensor input_min_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        input_min_tensor.scalar<float>()() = input_min_val;
        
        tensorflow::Tensor input_max_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        input_max_tensor.scalar<float>()() = input_max_val;
        
        auto tensor_input = tensorflow::ops::Const(root, input_tensor);
        auto shape_input = tensorflow::ops::Const(root, shape_tensor);
        auto input_min_input = tensorflow::ops::Const(root, input_min_tensor);
        auto input_max_input = tensorflow::ops::Const(root, input_max_tensor);
        
        auto quantized_reshape = tensorflow::ops::QuantizedReshape(
            root, tensor_input, shape_input, input_min_input, input_max_input);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({quantized_reshape.output, 
                                                quantized_reshape.output_min, 
                                                quantized_reshape.output_max}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
