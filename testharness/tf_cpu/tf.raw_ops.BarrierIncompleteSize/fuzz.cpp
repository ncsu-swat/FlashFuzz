#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/data_flow_ops.h"
#include <cstring>
#include <iostream>

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
    switch (selector % 1) {  
        case 0:
            dtype = tensorflow::DT_STRING;
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

void fillStringTensor(tensorflow::Tensor& tensor, const uint8_t* data, size_t& offset, size_t total_size) {
    auto flat = tensor.flat<tensorflow::tstring>();
    const size_t num_elements = flat.size();
    
    for (size_t i = 0; i < num_elements; ++i) {
        if (offset < total_size) {
            size_t str_len = std::min(static_cast<size_t>(8), total_size - offset);
            std::string str(reinterpret_cast<const char*>(data + offset), str_len);
            flat(i) = tensorflow::tstring(str);
            offset += str_len;
        } else {
            flat(i) = tensorflow::tstring("");
        }
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType handle_dtype = parseDataType(data[offset++]);
        uint8_t handle_rank = parseRank(data[offset++]);
        std::vector<int64_t> handle_shape = parseShape(data, offset, size, handle_rank);
        
        tensorflow::TensorShape handle_tensor_shape;
        for (int64_t dim : handle_shape) {
            handle_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor handle_tensor(handle_dtype, handle_tensor_shape);
        
        if (handle_dtype == tensorflow::DT_STRING) {
            fillStringTensor(handle_tensor, data, offset, size);
        }
        
        auto handle_input = tensorflow::ops::Const(root, handle_tensor);
        
        std::cout << "Handle tensor shape: ";
        for (int i = 0; i < handle_tensor_shape.dims(); ++i) {
            std::cout << handle_tensor_shape.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        auto barrier_incomplete_size_op = tensorflow::ops::BarrierIncompleteSize(root, handle_input);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({barrier_incomplete_size_op}, &outputs);
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
