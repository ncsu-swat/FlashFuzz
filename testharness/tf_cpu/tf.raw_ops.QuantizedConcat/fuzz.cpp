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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        if (offset >= size) return 0;
        int32_t concat_dim_val = static_cast<int32_t>(data[offset] % 4);
        offset++;

        if (offset >= size) return 0;
        uint8_t num_tensors = (data[offset] % 3) + 2;
        offset++;

        if (offset >= size) return 0;
        tensorflow::DataType dtype = parseDataType(data[offset]);
        offset++;

        if (offset >= size) return 0;
        uint8_t rank = parseRank(data[offset]);
        offset++;

        tensorflow::TensorShape concat_dim_shape({});
        tensorflow::Tensor concat_dim_tensor(tensorflow::DT_INT32, concat_dim_shape);
        concat_dim_tensor.scalar<int32_t>()() = concat_dim_val;

        std::vector<tensorflow::Output> values;
        std::vector<tensorflow::Output> input_mins;
        std::vector<tensorflow::Output> input_maxes;

        for (uint8_t i = 0; i < num_tensors; ++i) {
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }

            tensorflow::Tensor value_tensor(dtype, tensor_shape);
            fillTensorWithDataByType(value_tensor, dtype, data, offset, size);

            float min_val = -1.0f;
            float max_val = 1.0f;
            if (offset + sizeof(float) <= size) {
                std::memcpy(&min_val, data + offset, sizeof(float));
                offset += sizeof(float);
            }
            if (offset + sizeof(float) <= size) {
                std::memcpy(&max_val, data + offset, sizeof(float));
                offset += sizeof(float);
            }

            tensorflow::TensorShape scalar_shape({});
            tensorflow::Tensor min_tensor(tensorflow::DT_FLOAT, scalar_shape);
            tensorflow::Tensor max_tensor(tensorflow::DT_FLOAT, scalar_shape);
            min_tensor.scalar<float>()() = min_val;
            max_tensor.scalar<float>()() = max_val;

            auto value_const = tensorflow::ops::Const(root, value_tensor);
            auto min_const = tensorflow::ops::Const(root, min_tensor);
            auto max_const = tensorflow::ops::Const(root, max_tensor);

            values.push_back(value_const);
            input_mins.push_back(min_const);
            input_maxes.push_back(max_const);
        }

        auto concat_dim_const = tensorflow::ops::Const(root, concat_dim_tensor);

        auto quantized_concat = tensorflow::ops::QuantizedConcat(
            root, concat_dim_const, values, input_mins, input_maxes);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({quantized_concat.output, quantized_concat.output_min, quantized_concat.output_max}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
