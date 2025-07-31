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
    std::cerr << "Error: " << message << std::endl;
}
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

void fillTensorWithDataByType(tensorflow::Tensor& tensor,
                              tensorflow::DataType dtype, const uint8_t* data,
                              size_t& offset, size_t total_size) {
    switch (dtype) {
        case tensorflow::DT_FLOAT:
            fillTensorWithData<float>(tensor, data, offset, total_size);
            break;
        default:
            break;
    }
}

std::string parseMode(uint8_t selector) {
    switch (selector % 3) {
        case 0:
            return "MIN_COMBINED";
        case 1:
            return "MIN_FIRST";
        case 2:
            return "SCALED";
        default:
            return "MIN_COMBINED";
    }
}

std::string parseRoundMode(uint8_t selector) {
    switch (selector % 2) {
        case 0:
            return "HALF_AWAY_FROM_ZERO";
        case 1:
            return "HALF_TO_EVEN";
        default:
            return "HALF_AWAY_FROM_ZERO";
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t input_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        tensorflow::TensorShape input_tensor_shape;
        for (int64_t dim : input_shape) {
            input_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_tensor_shape);
        fillTensorWithDataByType(input_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        if (offset >= size) return 0;
        
        float min_range_val, max_range_val;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&min_range_val, data + offset, sizeof(float));
            offset += sizeof(float);
        } else {
            min_range_val = -1.0f;
        }
        
        if (offset + sizeof(float) <= size) {
            std::memcpy(&max_range_val, data + offset, sizeof(float));
            offset += sizeof(float);
        } else {
            max_range_val = 1.0f;
        }
        
        if (min_range_val >= max_range_val) {
            max_range_val = min_range_val + 1.0f;
        }
        
        tensorflow::Tensor min_range_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        min_range_tensor.scalar<float>()() = min_range_val;
        
        tensorflow::Tensor max_range_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        max_range_tensor.scalar<float>()() = max_range_val;
        
        if (offset >= size) return 0;
        
        tensorflow::DataType output_dtype = parseOutputDataType(data[offset++]);
        
        if (offset >= size) return 0;
        std::string mode = parseMode(data[offset++]);
        
        if (offset >= size) return 0;
        std::string round_mode = parseRoundMode(data[offset++]);
        
        if (offset >= size) return 0;
        bool narrow_range = (data[offset++] % 2) == 1;
        
        if (offset >= size) return 0;
        int32_t axis = static_cast<int32_t>(static_cast<int8_t>(data[offset++]));
        
        float ensure_minimum_range = 0.01f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&ensure_minimum_range, data + offset, sizeof(float));
            offset += sizeof(float);
            if (ensure_minimum_range < 0.0f) {
                ensure_minimum_range = 0.01f;
            }
        }
        
        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto min_range_op = tensorflow::ops::Const(root, min_range_tensor);
        auto max_range_op = tensorflow::ops::Const(root, max_range_tensor);
        
        auto quantize_attrs = tensorflow::ops::QuantizeV2::Attrs()
            .Mode(mode)
            .RoundMode(round_mode)
            .NarrowRange(narrow_range)
            .Axis(axis)
            .EnsureMinimumRange(ensure_minimum_range);
        
        auto quantize_op = tensorflow::ops::QuantizeV2(root, input_op, min_range_op, max_range_op, output_dtype, quantize_attrs);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({quantize_op.output, quantize_op.output_min, quantize_op.output_max}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}