#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include <iostream>
#include <cstring>
#include <vector>

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
    switch (selector % 11) {
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
            dtype = tensorflow::DT_UINT8;
            break;
        case 4:
            dtype = tensorflow::DT_INT16;
            break;
        case 5:
            dtype = tensorflow::DT_INT8;
            break;
        case 6:
            dtype = tensorflow::DT_INT64;
            break;
        case 7:
            dtype = tensorflow::DT_BOOL;
            break;
        case 8:
            dtype = tensorflow::DT_UINT16;
            break;
        case 9:
            dtype = tensorflow::DT_UINT32;
            break;
        case 10:
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
        case tensorflow::DT_FLOAT:
            fillTensorWithData<float>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_DOUBLE:
            fillTensorWithData<double>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT8:
            fillTensorWithData<uint8_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT16:
            fillTensorWithData<int16_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT8:
            fillTensorWithData<int8_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_BOOL:
            fillTensorWithData<bool>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT16:
            fillTensorWithData<uint16_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT32:
            fillTensorWithData<uint32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT64:
            fillTensorWithData<uint64_t>(tensor, data, offset, total_size);
            break;
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_inputs = (data[offset++] % 3) + 1;
        
        std::vector<tensorflow::Output> args;
        std::vector<tensorflow::DataType> input_types;
        std::vector<tensorflow::DataType> output_types;
        
        for (uint8_t i = 0; i < num_inputs; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            input_types.push_back(dtype);
            
            if (offset >= size) break;
            uint8_t rank = parseRank(data[offset++]);
            
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor input_tensor(dtype, tensor_shape);
            fillTensorWithDataByType(input_tensor, dtype, data, offset, size);
            
            auto placeholder = tensorflow::ops::Placeholder(root, dtype);
            args.push_back(placeholder);
        }
        
        if (args.empty()) return 0;
        
        output_types = input_types;
        
        tensorflow::FunctionDefLibrary function_lib;
        tensorflow::FunctionDef* func_def = function_lib.add_function();
        func_def->mutable_signature()->set_name("test_function");
        
        for (size_t i = 0; i < input_types.size(); ++i) {
            auto* input_arg = func_def->mutable_signature()->add_input_arg();
            input_arg->set_name("input_" + std::to_string(i));
            input_arg->set_type(input_types[i]);
        }
        
        for (size_t i = 0; i < output_types.size(); ++i) {
            auto* output_arg = func_def->mutable_signature()->add_output_arg();
            output_arg->set_name("output_" + std::to_string(i));
            output_arg->set_type(output_types[i]);
        }
        
        for (size_t i = 0; i < input_types.size(); ++i) {
            auto* node_def = func_def->add_node_def();
            node_def->set_name("Identity_" + std::to_string(i));
            node_def->set_op("Identity");
            node_def->add_input("input_" + std::to_string(i));
            (*node_def->mutable_attr())["T"].set_type(input_types[i]);
        }
        
        for (size_t i = 0; i < output_types.size(); ++i) {
            func_def->mutable_ret()->insert({"output_" + std::to_string(i), "Identity_" + std::to_string(i) + ":output:0"});
        }
        
        tensorflow::NameAttrList func_attr;
        func_attr.set_name("test_function");
        
        // Create StatefulPartitionedCall node using NodeBuilder
        tensorflow::NodeBuilder node_builder("stateful_partitioned_call", "StatefulPartitionedCall");
        
        // Add inputs
        for (const auto& arg : args) {
            node_builder.Input(arg.node());
        }
        
        // Add attributes
        node_builder.Attr("Tin", input_types);
        node_builder.Attr("Tout", output_types);
        node_builder.Attr("f", func_attr);
        
        // Build the node
        tensorflow::Node* stateful_call_node;
        tensorflow::Status status = node_builder.Finalize(root.graph(), &stateful_call_node);
        
        if (!status.ok()) {
            return -1;
        }
        
        std::vector<tensorflow::Output> stateful_call_outputs;
        for (int i = 0; i < stateful_call_node->num_outputs(); ++i) {
            stateful_call_outputs.push_back(tensorflow::Output(stateful_call_node, i));
        }
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict;
        
        for (size_t i = 0; i < args.size(); ++i) {
            tensorflow::DataType dtype = input_types[i];
            tensorflow::TensorShape tensor_shape;
            tensor_shape.AddDim(1);
            
            tensorflow::Tensor input_tensor(dtype, tensor_shape);
            fillTensorWithDataByType(input_tensor, dtype, data, offset, size);
            
            feed_dict.push_back({args[i].node()->name() + ":0", input_tensor});
        }
        
        std::vector<tensorflow::Output> fetch_outputs;
        for (const auto& output : stateful_call_outputs) {
            fetch_outputs.push_back(output);
        }
        
        status = session.Run(feed_dict, fetch_outputs, {}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}