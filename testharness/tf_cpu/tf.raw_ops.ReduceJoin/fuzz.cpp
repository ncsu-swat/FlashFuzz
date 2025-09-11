#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/string_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include <iostream>
#include <vector>
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
            uint8_t str_len = data[offset] % 10 + 1;
            offset++;
            
            std::string str;
            for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                char c = static_cast<char>((data[offset] % 26) + 'a');
                str += c;
                offset++;
            }
            flat(i) = tensorflow::tstring(str);
        } else {
            flat(i) = tensorflow::tstring("a");
        }
    }
}

std::vector<int32_t> parseReductionIndices(const uint8_t* data, size_t& offset, size_t total_size, uint8_t input_rank) {
    std::vector<int32_t> indices;
    
    if (offset >= total_size) {
        return indices;
    }
    
    uint8_t num_indices = data[offset] % (input_rank + 1);
    offset++;
    
    for (uint8_t i = 0; i < num_indices && offset < total_size; ++i) {
        int32_t idx = static_cast<int32_t>(data[offset] % input_rank);
        if (data[offset] & 0x80) {
            idx = -idx - 1;
        }
        indices.push_back(idx);
        offset++;
    }
    
    return indices;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t input_rank = parseRank(data[offset]);
        offset++;
        
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        tensorflow::TensorShape tensor_shape;
        for (int64_t dim : input_shape) {
            tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_tensor(tensorflow::DT_STRING, tensor_shape);
        fillStringTensor(input_tensor, data, offset, size);
        
        std::vector<int32_t> reduction_indices = parseReductionIndices(data, offset, size, input_rank);
        
        bool keep_dims = false;
        if (offset < size) {
            keep_dims = (data[offset] & 1) == 1;
            offset++;
        }
        
        std::string separator = "";
        if (offset < size) {
            uint8_t sep_len = data[offset] % 5;
            offset++;
            for (uint8_t i = 0; i < sep_len && offset < size; ++i) {
                separator += static_cast<char>((data[offset] % 26) + 'a');
                offset++;
            }
        }
        
        auto input_op = tensorflow::ops::Const(root, input_tensor);
        
        tensorflow::Tensor reduction_indices_tensor(tensorflow::DT_INT32, 
                                                   tensorflow::TensorShape({static_cast<int64_t>(reduction_indices.size())}));
        auto flat_indices = reduction_indices_tensor.flat<int32_t>();
        for (size_t i = 0; i < reduction_indices.size(); ++i) {
            flat_indices(i) = reduction_indices[i];
        }
        auto reduction_indices_op = tensorflow::ops::Const(root, reduction_indices_tensor);
        
        auto reduce_join_op = tensorflow::ops::ReduceJoin(root, input_op, reduction_indices_op,
                                                         tensorflow::ops::ReduceJoin::KeepDims(keep_dims)
                                                         .Separator(separator));
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({reduce_join_op}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
