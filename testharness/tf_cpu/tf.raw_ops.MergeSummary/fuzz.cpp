#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/summary.pb.h"
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
    return tensorflow::DT_STRING;
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

std::string createValidSummary(const uint8_t* data, size_t& offset, size_t total_size, const std::string& tag_prefix) {
    tensorflow::Summary summary;
    
    if (offset < total_size) {
        auto* value = summary.add_value();
        value->set_tag(tag_prefix + std::to_string(offset));
        
        if (offset + sizeof(float) <= total_size) {
            float scalar_value;
            std::memcpy(&scalar_value, data + offset, sizeof(float));
            offset += sizeof(float);
            value->set_simple_value(scalar_value);
        } else {
            value->set_simple_value(1.0f);
        }
    }
    
    std::string serialized;
    summary.SerializeToString(&serialized);
    return serialized;
}

void fillStringTensor(tensorflow::Tensor& tensor, const uint8_t* data, size_t& offset, size_t total_size, int num_summaries) {
    auto flat = tensor.flat<tensorflow::tstring>();
    const size_t num_elements = flat.size();
    
    for (size_t i = 0; i < num_elements; ++i) {
        std::string tag_prefix = "summary_" + std::to_string(i) + "_";
        std::string summary_str = createValidSummary(data, offset, total_size, tag_prefix);
        flat(i) = tensorflow::tstring(summary_str);
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_inputs_byte = data[offset++];
        int num_inputs = (num_inputs_byte % 5) + 1;
        
        std::vector<tensorflow::Output> input_list;
        
        for (int i = 0; i < num_inputs; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor input_tensor(dtype, tensor_shape);
            
            fillStringTensor(input_tensor, data, offset, size, i + 1);
            
            auto input_node = tensorflow::ops::Const(root.WithOpName("input_" + std::to_string(i)), input_tensor);
            input_list.push_back(input_node);
        }
        
        if (input_list.empty()) {
            tensorflow::TensorShape scalar_shape;
            tensorflow::Tensor default_tensor(tensorflow::DT_STRING, scalar_shape);
            std::string default_summary = createValidSummary(data, offset, size, "default_");
            default_tensor.scalar<tensorflow::tstring>()() = tensorflow::tstring(default_summary);
            auto default_input = tensorflow::ops::Const(root.WithOpName("default_input"), default_tensor);
            input_list.push_back(default_input);
        }
        
        // Use raw_ops.MergeSummary instead of ops::MergeSummary
        auto merge_summary_op = tensorflow::ops::internal::MergeSummary(root.WithOpName("merge_summary"), input_list);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({merge_summary_op}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}