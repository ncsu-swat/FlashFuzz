#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/control_flow_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/types.h"
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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        int32_t start_val, limit_val, delta_val;
        std::memcpy(&start_val, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        std::memcpy(&limit_val, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        std::memcpy(&delta_val, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        
        if (delta_val == 0) delta_val = 1;
        if (std::abs(limit_val - start_val) > 100) {
            limit_val = start_val + (delta_val > 0 ? 10 : -10);
        }

        auto start = tensorflow::ops::Const(root, start_val);
        auto limit = tensorflow::ops::Const(root, limit_val);
        auto delta = tensorflow::ops::Const(root, delta_val);

        uint8_t num_inputs = (data[offset++] % 3) + 1;
        std::vector<tensorflow::Output> input_tensors;

        for (uint8_t i = 0; i < num_inputs; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor tensor(dtype, tensor_shape);
            fillTensorWithDataByType(tensor, dtype, data, offset, size);
            
            auto const_op = tensorflow::ops::Const(root, tensor);
            input_tensors.push_back(const_op);
        }

        if (input_tensors.empty()) {
            tensorflow::Tensor default_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({2, 2}));
            auto flat = default_tensor.flat<float>();
            for (int i = 0; i < flat.size(); ++i) {
                flat(i) = 1.0f;
            }
            auto const_op = tensorflow::ops::Const(root, default_tensor);
            input_tensors.push_back(const_op);
        }

        tensorflow::FunctionDefLibrary fdef_lib;
        tensorflow::FunctionDef* fdef = fdef_lib.add_function();
        fdef->mutable_signature()->set_name("simple_body");
        
        auto* arg0 = fdef->mutable_signature()->add_input_arg();
        arg0->set_name("i");
        arg0->set_type(tensorflow::DT_INT32);
        
        for (size_t j = 0; j < input_tensors.size(); ++j) {
            auto* arg = fdef->mutable_signature()->add_input_arg();
            arg->set_name("input_" + std::to_string(j));
            arg->set_type(input_tensors[j].type());
        }
        
        for (size_t j = 0; j < input_tensors.size(); ++j) {
            auto* ret = fdef->mutable_signature()->add_output_arg();
            ret->set_name("output_" + std::to_string(j));
            ret->set_type(input_tensors[j].type());
        }

        auto* identity_node = fdef->add_node_def();
        identity_node->set_name("Identity");
        identity_node->set_op("Identity");
        identity_node->add_input("input_0");
        
        fdef->mutable_ret()->insert({"output_0", "Identity:output:0"});

        tensorflow::ClientSession session(root);
        tensorflow::Status status = root.graph()->AddFunctionLibrary(fdef_lib);
        if (!status.ok()) {
            return -1;
        }

        tensorflow::NameAttrList body_func;
        body_func.set_name("simple_body");

        // Create a NodeDef for the For operation
        tensorflow::NodeDef node_def;
        node_def.set_name("ForOp");
        node_def.set_op("For");
        
        // Add inputs to the NodeDef
        tensorflow::NodeDefBuilder builder("ForOp", "For");
        builder.Input(start.node()->name(), 0, tensorflow::DT_INT32);
        builder.Input(limit.node()->name(), 0, tensorflow::DT_INT32);
        builder.Input(delta.node()->name(), 0, tensorflow::DT_INT32);
        
        // Add the input tensors
        std::vector<tensorflow::NodeDefBuilder::NodeOut> input_nodes;
        for (const auto& input : input_tensors) {
            input_nodes.push_back({input.node()->name(), 0, input.type()});
        }
        builder.Input(input_nodes);
        
        // Add the body function attribute
        builder.Attr("body", body_func);
        
        // Build the NodeDef
        status = builder.Finalize(&node_def);
        if (!status.ok()) {
            return -1;
        }
        
        // Add the node to the graph
        tensorflow::Node* for_node;
        status = root.graph()->AddNode(node_def, &for_node);
        if (!status.ok()) {
            return -1;
        }
        
        // Connect the inputs
        root.graph()->AddEdge(start.node(), 0, for_node, 0);
        root.graph()->AddEdge(limit.node(), 0, for_node, 1);
        root.graph()->AddEdge(delta.node(), 0, for_node, 2);
        
        for (size_t i = 0; i < input_tensors.size(); ++i) {
            root.graph()->AddEdge(input_tensors[i].node(), 0, for_node, 3 + i);
        }
        
        // Create outputs
        std::vector<tensorflow::Output> outputs;
        for (size_t i = 0; i < input_tensors.size(); ++i) {
            outputs.push_back(tensorflow::Output(for_node, i));
        }

        std::vector<tensorflow::Tensor> output_tensors;
        status = session.Run({outputs[0]}, &output_tensors);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
