#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
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
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_RESOURCE;
            break;
        case 1:
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
            size_t str_len = std::min(static_cast<size_t>(data[offset] % 32 + 1), total_size - offset - 1);
            offset++;
            
            if (offset + str_len <= total_size) {
                std::string str(reinterpret_cast<const char*>(data + offset), str_len);
                flat(i) = tensorflow::tstring(str);
                offset += str_len;
            } else {
                flat(i) = tensorflow::tstring("default");
                offset = total_size;
            }
        } else {
            flat(i) = tensorflow::tstring("default");
        }
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::Output writer_tensor = tensorflow::ops::Placeholder(root, tensorflow::DT_RESOURCE);
        
        uint8_t db_uri_rank = parseRank(data[offset++]);
        std::vector<int64_t> db_uri_shape = parseShape(data, offset, size, db_uri_rank);
        tensorflow::TensorShape db_uri_tensor_shape(db_uri_shape);
        tensorflow::Tensor db_uri_input(tensorflow::DT_STRING, db_uri_tensor_shape);
        fillStringTensor(db_uri_input, data, offset, size);
        tensorflow::Output db_uri_tensor = tensorflow::ops::Const(root, db_uri_input);
        
        uint8_t experiment_name_rank = parseRank(data[offset++]);
        std::vector<int64_t> experiment_name_shape = parseShape(data, offset, size, experiment_name_rank);
        tensorflow::TensorShape experiment_name_tensor_shape(experiment_name_shape);
        tensorflow::Tensor experiment_name_input(tensorflow::DT_STRING, experiment_name_tensor_shape);
        fillStringTensor(experiment_name_input, data, offset, size);
        tensorflow::Output experiment_name_tensor = tensorflow::ops::Const(root, experiment_name_input);
        
        uint8_t run_name_rank = parseRank(data[offset++]);
        std::vector<int64_t> run_name_shape = parseShape(data, offset, size, run_name_rank);
        tensorflow::TensorShape run_name_tensor_shape(run_name_shape);
        tensorflow::Tensor run_name_input(tensorflow::DT_STRING, run_name_tensor_shape);
        fillStringTensor(run_name_input, data, offset, size);
        tensorflow::Output run_name_tensor = tensorflow::ops::Const(root, run_name_input);
        
        uint8_t user_name_rank = parseRank(data[offset++]);
        std::vector<int64_t> user_name_shape = parseShape(data, offset, size, user_name_rank);
        tensorflow::TensorShape user_name_tensor_shape(user_name_shape);
        tensorflow::Tensor user_name_input(tensorflow::DT_STRING, user_name_tensor_shape);
        fillStringTensor(user_name_input, data, offset, size);
        tensorflow::Output user_name_tensor = tensorflow::ops::Const(root, user_name_input);

        // Use raw op directly since summary_ops.h is not available
        auto create_summary_db_writer = tensorflow::Operation(
            root.WithOpName("CreateSummaryDbWriter")
                .WithAttr("T", tensorflow::DT_RESOURCE)
                .WithInput(writer_tensor)
                .WithInput(db_uri_tensor)
                .WithInput(experiment_name_tensor)
                .WithInput(run_name_tensor)
                .WithInput(user_name_tensor)
        );

        tensorflow::ClientSession session(root);

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
