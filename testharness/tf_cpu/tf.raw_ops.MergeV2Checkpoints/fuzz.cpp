#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/io_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
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
            uint8_t str_len = data[offset] % 20 + 1;
            offset++;
            
            std::string str;
            for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                char c = static_cast<char>(data[offset] % 94 + 33);
                str += c;
                offset++;
            }
            flat(i) = tensorflow::tstring(str);
        } else {
            flat(i) = tensorflow::tstring("default");
        }
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype1 = parseDataType(data[offset++]);
        uint8_t rank1 = parseRank(data[offset++]);
        std::vector<int64_t> shape1 = parseShape(data, offset, size, rank1);
        
        tensorflow::TensorShape tensor_shape1(shape1);
        tensorflow::Tensor checkpoint_prefixes_tensor(dtype1, tensor_shape1);
        
        if (dtype1 == tensorflow::DT_STRING) {
            fillStringTensor(checkpoint_prefixes_tensor, data, offset, size);
        }
        
        tensorflow::DataType dtype2 = parseDataType(data[offset % size]);
        offset++;
        uint8_t rank2 = parseRank(data[offset % size]);
        offset++;
        std::vector<int64_t> shape2 = parseShape(data, offset, size, rank2);
        
        tensorflow::TensorShape tensor_shape2(shape2);
        tensorflow::Tensor destination_prefix_tensor(dtype2, tensor_shape2);
        
        if (dtype2 == tensorflow::DT_STRING) {
            fillStringTensor(destination_prefix_tensor, data, offset, size);
        }
        
        bool delete_old_dirs = (data[offset % size] % 2) == 1;
        offset++;
        bool allow_missing_files = (data[offset % size] % 2) == 1;
        offset++;

        auto checkpoint_prefixes = tensorflow::ops::Const(root, checkpoint_prefixes_tensor);
        auto destination_prefix = tensorflow::ops::Const(root, destination_prefix_tensor);

        auto merge_op = tensorflow::ops::MergeV2Checkpoints(
            root,
            checkpoint_prefixes,
            destination_prefix,
            tensorflow::ops::MergeV2Checkpoints::DeleteOldDirs(delete_old_dirs).AllowMissingFiles(allow_missing_files)
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({}, {}, {merge_op.operation}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
