#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/types.h"
#include <iostream>
#include <cstring>
#include <vector>
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

void fillTensorWithDataByType(tensorflow::Tensor& tensor,
                              tensorflow::DataType dtype, const uint8_t* data,
                              size_t& offset, size_t total_size) {
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

std::vector<int> parseKsizeStrides(const uint8_t* data, size_t& offset, size_t total_size) {
    std::vector<int> result(4);
    for (int i = 0; i < 4; ++i) {
        if (offset < total_size) {
            uint8_t val = data[offset++];
            result[i] = (val % 5) + 1;
        } else {
            result[i] = 1;
        }
    }
    return result;
}

std::string parsePadding(uint8_t byte) {
    return (byte % 2 == 0) ? "SAME" : "VALID";
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType input_dtype = parseDataType(data[offset++]);
        uint8_t rank = parseRank(data[offset++]);
        
        std::vector<int64_t> input_shape = parseShape(data, offset, size, rank);
        
        tensorflow::TensorShape tensor_shape;
        for (int64_t dim : input_shape) {
            tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_tensor(input_dtype, tensor_shape);
        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        
        float min_input_val = 0.0f;
        float max_input_val = 1.0f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&min_input_val, data + offset, sizeof(float));
            offset += sizeof(float);
        }
        if (offset + sizeof(float) <= size) {
            std::memcpy(&max_input_val, data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        if (min_input_val >= max_input_val) {
            max_input_val = min_input_val + 1.0f;
        }
        
        tensorflow::Tensor min_input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        min_input_tensor.scalar<float>()() = min_input_val;
        
        tensorflow::Tensor max_input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        max_input_tensor.scalar<float>()() = max_input_val;
        
        std::vector<int> ksize = parseKsizeStrides(data, offset, size);
        std::vector<int> strides = parseKsizeStrides(data, offset, size);
        std::string padding = parsePadding(data[offset % size]);
        
        auto input_placeholder = tensorflow::ops::Placeholder(root, input_dtype);
        auto min_input_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto max_input_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        auto quantized_max_pool = tensorflow::ops::QuantizedMaxPool(
            root,
            input_placeholder,
            min_input_placeholder,
            max_input_placeholder,
            ksize,
            strides,
            padding
        );
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(
            {{input_placeholder, input_tensor},
             {min_input_placeholder, min_input_tensor},
             {max_input_placeholder, max_input_tensor}},
            {quantized_max_pool.output, quantized_max_pool.min_output, quantized_max_pool.max_output},
            &outputs
        );
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
