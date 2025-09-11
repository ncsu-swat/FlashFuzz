#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <cstring>
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

tensorflow::DataType parseQuantizedDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 5) {
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
    }
    return dtype;
}

tensorflow::DataType parseOutputDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 5) {
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

void fillQuantizedTensorWithData(tensorflow::Tensor& tensor, tensorflow::DataType dtype, 
                                const uint8_t* data, size_t& offset, size_t total_size) {
    switch (dtype) {
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
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType features_dtype = parseQuantizedDataType(data[offset++]);
        uint8_t features_rank = parseRank(data[offset++]);
        std::vector<int64_t> features_shape = parseShape(data, offset, size, features_rank);
        
        tensorflow::TensorShape features_tensor_shape;
        for (int64_t dim : features_shape) {
            features_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor features_tensor(features_dtype, features_tensor_shape);
        fillQuantizedTensorWithData(features_tensor, features_dtype, data, offset, size);
        
        float max_value_val = 6.0f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&max_value_val, data + offset, sizeof(float));
            offset += sizeof(float);
            max_value_val = std::abs(max_value_val);
            if (max_value_val > 100.0f) max_value_val = 6.0f;
        }
        
        float min_features_val = -1.0f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&min_features_val, data + offset, sizeof(float));
            offset += sizeof(float);
            if (min_features_val > 0.0f) min_features_val = -std::abs(min_features_val);
            if (min_features_val < -100.0f) min_features_val = -1.0f;
        }
        
        float max_features_val = 1.0f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&max_features_val, data + offset, sizeof(float));
            offset += sizeof(float);
            max_features_val = std::abs(max_features_val);
            if (max_features_val > 100.0f) max_features_val = 1.0f;
            if (max_features_val <= min_features_val) max_features_val = min_features_val + 1.0f;
        }
        
        tensorflow::DataType out_type = tensorflow::DT_QUINT8;
        if (offset < size) {
            out_type = parseOutputDataType(data[offset++]);
        }
        
        tensorflow::TensorShape scalar_shape;
        tensorflow::Tensor max_value_tensor(tensorflow::DT_FLOAT, scalar_shape);
        max_value_tensor.scalar<float>()() = max_value_val;
        
        tensorflow::Tensor min_features_tensor(tensorflow::DT_FLOAT, scalar_shape);
        min_features_tensor.scalar<float>()() = min_features_val;
        
        tensorflow::Tensor max_features_tensor(tensorflow::DT_FLOAT, scalar_shape);
        max_features_tensor.scalar<float>()() = max_features_val;
        
        auto features_input = tensorflow::ops::Const(root, features_tensor);
        auto max_value_input = tensorflow::ops::Const(root, max_value_tensor);
        auto min_features_input = tensorflow::ops::Const(root, min_features_tensor);
        auto max_features_input = tensorflow::ops::Const(root, max_features_tensor);
        
        auto quantized_relu_x = tensorflow::ops::QuantizedReluX(
            root, 
            features_input, 
            max_value_input, 
            min_features_input, 
            max_features_input,
            tensorflow::ops::QuantizedReluX::OutType(out_type)
        );
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({quantized_relu_x.activations, 
                                                quantized_relu_x.min_activations, 
                                                quantized_relu_x.max_activations}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
