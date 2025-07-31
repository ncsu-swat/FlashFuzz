#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/random_ops.h"
#include <iostream>
#include <cstring>
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

tensorflow::DataType parseDataType(uint8_t selector) {
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
        default:
            return;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t shape_rank = parseRank(data[offset++]);
        std::vector<int64_t> output_shape = parseShape(data, offset, size, shape_rank);
        
        tensorflow::DataType minval_maxval_dtype = parseDataType(data[offset++]);
        
        tensorflow::TensorShape shape_tensor_shape({static_cast<int64_t>(output_shape.size())});
        tensorflow::Tensor shape_tensor(tensorflow::DT_INT64, shape_tensor_shape);
        auto shape_flat = shape_tensor.flat<int64_t>();
        for (size_t i = 0; i < output_shape.size(); ++i) {
            shape_flat(i) = output_shape[i];
        }
        
        tensorflow::TensorShape seed_shape({2});
        tensorflow::Tensor seed_tensor(tensorflow::DT_INT64, seed_shape);
        fillTensorWithDataByType(seed_tensor, tensorflow::DT_INT64, data, offset, size);
        
        tensorflow::TensorShape scalar_shape({});
        tensorflow::Tensor minval_tensor(minval_maxval_dtype, scalar_shape);
        fillTensorWithDataByType(minval_tensor, minval_maxval_dtype, data, offset, size);
        
        tensorflow::Tensor maxval_tensor(minval_maxval_dtype, scalar_shape);
        fillTensorWithDataByType(maxval_tensor, minval_maxval_dtype, data, offset, size);
        
        if (minval_maxval_dtype == tensorflow::DT_INT32) {
            auto minval_scalar = minval_tensor.scalar<int32_t>();
            auto maxval_scalar = maxval_tensor.scalar<int32_t>();
            if (minval_scalar() >= maxval_scalar()) {
                maxval_scalar() = minval_scalar() + 1;
            }
        } else if (minval_maxval_dtype == tensorflow::DT_INT64) {
            auto minval_scalar = minval_tensor.scalar<int64_t>();
            auto maxval_scalar = maxval_tensor.scalar<int64_t>();
            if (minval_scalar() >= maxval_scalar()) {
                maxval_scalar() = minval_scalar() + 1;
            }
        }
        
        std::cout << "Shape tensor: ";
        for (int i = 0; i < shape_tensor.NumElements(); ++i) {
            std::cout << shape_flat(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Seed tensor: ";
        auto seed_flat = seed_tensor.flat<int64_t>();
        for (int i = 0; i < seed_tensor.NumElements(); ++i) {
            std::cout << seed_flat(i) << " ";
        }
        std::cout << std::endl;
        
        if (minval_maxval_dtype == tensorflow::DT_INT32) {
            std::cout << "Minval: " << minval_tensor.scalar<int32_t>()() << std::endl;
            std::cout << "Maxval: " << maxval_tensor.scalar<int32_t>()() << std::endl;
        } else {
            std::cout << "Minval: " << minval_tensor.scalar<int64_t>()() << std::endl;
            std::cout << "Maxval: " << maxval_tensor.scalar<int64_t>()() << std::endl;
        }
        
        auto shape_op = tensorflow::ops::Const(root, shape_tensor);
        auto seed_op = tensorflow::ops::Const(root, seed_tensor);
        auto minval_op = tensorflow::ops::Const(root, minval_tensor);
        auto maxval_op = tensorflow::ops::Const(root, maxval_tensor);
        
        // Use the raw API instead of the ops namespace
        auto result = tensorflow::ops::StatelessRandomUniformIntV2(root, shape_op, seed_op, minval_op, maxval_op);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({result}, &outputs);
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }
        
        if (!outputs.empty()) {
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].dims(); ++i) {
                std::cout << outputs[0].dim_size(i) << " ";
            }
            std::cout << std::endl;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}