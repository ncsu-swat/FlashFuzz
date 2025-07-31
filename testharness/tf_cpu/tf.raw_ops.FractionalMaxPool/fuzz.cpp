#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>
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
    switch (selector % 4) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 2:
            dtype = tensorflow::DT_INT32;
            break;
        case 3:
            dtype = tensorflow::DT_INT64;
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
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        default:
            break;
    }
}

std::vector<float> parsePoolingRatio(const uint8_t* data, size_t& offset, size_t total_size) {
    std::vector<float> pooling_ratio(4);
    pooling_ratio[0] = 1.0f;
    pooling_ratio[3] = 1.0f;
    
    for (int i = 1; i < 3; ++i) {
        if (offset + sizeof(float) <= total_size) {
            float ratio;
            std::memcpy(&ratio, data + offset, sizeof(float));
            offset += sizeof(float);
            ratio = std::abs(ratio);
            if (ratio < 1.0f) ratio = 1.0f;
            if (ratio > 10.0f) ratio = 10.0f;
            pooling_ratio[i] = ratio;
        } else {
            pooling_ratio[i] = 1.44f;
        }
    }
    
    return pooling_ratio;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        uint8_t rank = parseRank(data[offset++]);
        
        std::vector<int64_t> shape = parseShape(data, offset, size, rank);
        
        tensorflow::TensorShape tensor_shape;
        for (int64_t dim : shape) {
            tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_tensor(dtype, tensor_shape);
        fillTensorWithDataByType(input_tensor, dtype, data, offset, size);
        
        std::vector<float> pooling_ratio = parsePoolingRatio(data, offset, size);
        
        bool pseudo_random = (offset < size) ? (data[offset++] % 2 == 0) : false;
        bool overlapping = (offset < size) ? (data[offset++] % 2 == 0) : false;
        bool deterministic = (offset < size) ? (data[offset++] % 2 == 0) : true;
        
        int64_t seed = 0;
        int64_t seed2 = 0;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&seed, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&seed2, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        auto input_op = tensorflow::ops::Const(root, input_tensor);
        
        auto fractional_max_pool = tensorflow::ops::FractionalMaxPool(
            root, input_op, pooling_ratio,
            tensorflow::ops::FractionalMaxPool::PseudoRandom(pseudo_random)
                .Overlapping(overlapping)
                .Deterministic(deterministic)
                .Seed(seed)
                .Seed2(seed2)
        );
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({fractional_max_pool.output, 
                                                fractional_max_pool.row_pooling_sequence,
                                                fractional_max_pool.col_pooling_sequence}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}