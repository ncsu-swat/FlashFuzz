#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/random_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
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

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 3) {
        case 0:
            dtype = tensorflow::DT_RESOURCE;
            break;
        case 1:
            dtype = tensorflow::DT_INT32;
            break;
        case 2:
            dtype = tensorflow::DT_UINT64;
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
        case tensorflow::DT_UINT64:
            fillTensorWithData<uint64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_RESOURCE:
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
        tensorflow::SessionOptions session_options;
        tensorflow::ClientSession session(root, session_options);

        // Create a resource variable
        auto resource_var = tensorflow::ops::Variable(root.WithOpName("resource_var"), 
                                                     {2}, tensorflow::DT_INT64);
        
        tensorflow::Tensor init_value(tensorflow::DT_INT64, tensorflow::TensorShape({2}));
        auto init_flat = init_value.flat<int64_t>();
        init_flat(0) = 0;
        init_flat(1) = 0;
        
        auto init_op = tensorflow::ops::Assign(root.WithOpName("init"), resource_var, 
                                              tensorflow::ops::Const(root, init_value));
        
        tensorflow::Status init_status = session.Run({init_op}, {});
        if (!init_status.ok()) {
            return -1;
        }

        uint8_t alg_rank = parseRank(data[offset++]);
        if (offset >= size) return 0;
        
        std::vector<int64_t> alg_shape = parseShape(data, offset, size, alg_rank);
        if (offset >= size) return 0;
        
        tensorflow::Tensor alg_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(alg_shape));
        fillTensorWithDataByType(alg_tensor, tensorflow::DT_INT32, data, offset, size);
        
        uint8_t delta_rank = parseRank(data[offset++]);
        if (offset >= size) return 0;
        
        std::vector<int64_t> delta_shape = parseShape(data, offset, size, delta_rank);
        if (offset >= size) return 0;
        
        tensorflow::Tensor delta_tensor(tensorflow::DT_UINT64, tensorflow::TensorShape(delta_shape));
        fillTensorWithDataByType(delta_tensor, tensorflow::DT_UINT64, data, offset, size);

        auto alg_input = tensorflow::ops::Const(root, alg_tensor);
        auto delta_input = tensorflow::ops::Const(root, delta_tensor);

        // Use raw op instead of the missing ops namespace function
        auto rng_op = tensorflow::ops::Operation(
            root.WithOpName("RngReadAndSkip"),
            "RngReadAndSkip",
            {resource_var.output, alg_input, delta_input}
        );

        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({rng_op}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
