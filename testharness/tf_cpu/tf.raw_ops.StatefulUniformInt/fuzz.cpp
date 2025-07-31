#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/random_ops.h"
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
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
            break;
        case 2:
            dtype = tensorflow::DT_UINT32;
            break;
        case 3:
            dtype = tensorflow::DT_UINT64;
            break;
        case 4:
            dtype = tensorflow::DT_INT16;
            break;
        case 5:
            dtype = tensorflow::DT_UINT16;
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
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT32:
            fillTensorWithData<uint32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT64:
            fillTensorWithData<uint64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT16:
            fillTensorWithData<int16_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT16:
            fillTensorWithData<uint16_t>(tensor, data, offset, total_size);
            break;
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType minmax_dtype = parseDataType(data[offset++]);
        
        uint8_t shape_rank = parseRank(data[offset++]);
        std::vector<int64_t> output_shape = parseShape(data, offset, size, shape_rank);
        
        // Create a random seed generator resource
        auto seed_generator = tensorflow::ops::RandomUniformInt(
            root, tensorflow::ops::Const(root, {1}, {1}), 
            tensorflow::ops::Const(root, 0), 
            tensorflow::ops::Const(root, 1000000));
        
        // Create a stateful random number generator resource
        auto rng = tensorflow::ops::StatefulStandardNormalV2(
            root.WithOpName("rng"), 
            tensorflow::ops::VarHandleOp(root, tensorflow::DT_RESOURCE, {}).output,
            tensorflow::ops::Const(root, 0));
        
        int64_t algorithm_val = 1;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&algorithm_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            algorithm_val = std::abs(algorithm_val) % 3 + 1;
        }
        
        tensorflow::TensorShape shape_tensor_shape({static_cast<int64_t>(output_shape.size())});
        tensorflow::Tensor shape_tensor(tensorflow::DT_INT64, shape_tensor_shape);
        auto shape_flat = shape_tensor.flat<int64_t>();
        for (size_t i = 0; i < output_shape.size(); ++i) {
            shape_flat(i) = output_shape[i];
        }
        
        tensorflow::Tensor minval_tensor(minmax_dtype, tensorflow::TensorShape({}));
        fillTensorWithDataByType(minval_tensor, minmax_dtype, data, offset, size);
        
        tensorflow::Tensor maxval_tensor(minmax_dtype, tensorflow::TensorShape({}));
        fillTensorWithDataByType(maxval_tensor, minmax_dtype, data, offset, size);
        
        // Use StatelessRandomUniformInt as an alternative since StatefulUniformInt is not directly available
        auto uniform_int = tensorflow::ops::StatelessRandomUniformInt(
            root, 
            tensorflow::ops::Const(root, shape_tensor),
            seed_generator,
            tensorflow::ops::Const(root, minval_tensor),
            tensorflow::ops::Const(root, maxval_tensor));
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({uniform_int}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}