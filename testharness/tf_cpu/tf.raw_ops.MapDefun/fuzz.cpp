#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
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
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_arguments = (data[offset++] % 3) + 1;
        uint8_t num_captured = (data[offset++] % 3) + 1;
        uint8_t num_outputs = (data[offset++] % 3) + 1;

        std::vector<tensorflow::Output> arguments;
        std::vector<tensorflow::DataType> arg_types;
        
        for (uint8_t i = 0; i < num_arguments; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            arg_types.push_back(dtype);
            
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor tensor(dtype, tensor_shape);
            fillTensorWithDataByType(tensor, dtype, data, offset, size);
            
            auto placeholder = tensorflow::ops::Placeholder(root, dtype, 
                tensorflow::ops::Placeholder::Shape(tensor_shape));
            arguments.push_back(placeholder);
        }

        std::vector<tensorflow::Output> captured_inputs;
        std::vector<tensorflow::DataType> captured_types;
        
        for (uint8_t i = 0; i < num_captured; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            captured_types.push_back(dtype);
            
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor tensor(dtype, tensor_shape);
            fillTensorWithDataByType(tensor, dtype, data, offset, size);
            
            auto placeholder = tensorflow::ops::Placeholder(root, dtype,
                tensorflow::ops::Placeholder::Shape(tensor_shape));
            captured_inputs.push_back(placeholder);
        }

        std::vector<tensorflow::DataType> output_types;
        std::vector<tensorflow::TensorShape> output_shapes;
        
        for (uint8_t i = 0; i < num_outputs; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            output_types.push_back(dtype);
            
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            output_shapes.push_back(tensor_shape);
        }

        if (arguments.empty() || output_types.empty()) {
            return 0;
        }

        // Create a MapDefun node using NodeDef
        tensorflow::NodeDef node_def;
        node_def.set_name("test_map_defun");
        node_def.set_op("MapDefun");
        
        // Add inputs to the node
        for (size_t i = 0; i < arguments.size(); ++i) {
            node_def.add_input(arguments[i].node()->name());
        }
        
        for (size_t i = 0; i < captured_inputs.size(); ++i) {
            node_def.add_input(captured_inputs[i].node()->name());
        }
        
        // Set attributes
        auto* attr_map = node_def.mutable_attr();
        
        // Targuments
        auto* targuments = (*attr_map)["Targuments"].mutable_list();
        for (auto dtype : arg_types) {
            targuments->add_type(dtype);
        }
        
        // Tcaptured
        auto* tcaptured = (*attr_map)["Tcaptured"].mutable_list();
        for (auto dtype : captured_types) {
            tcaptured->add_type(dtype);
        }
        
        // output_types
        auto* output_types_attr = (*attr_map)["output_types"].mutable_list();
        for (auto dtype : output_types) {
            output_types_attr->add_type(dtype);
        }
        
        // output_shapes
        auto* output_shapes_attr = (*attr_map)["output_shapes"].mutable_list();
        for (const auto& shape : output_shapes) {
            auto* shape_proto = output_shapes_attr->add_shape();
            for (int i = 0; i < shape.dims(); ++i) {
                shape_proto->add_dim()->set_size(shape.dim_size(i));
            }
        }

        // Function attribute
        tensorflow::NameAttrList f_attr;
        f_attr.set_name("identity_func");
        (*attr_map)["f"].mutable_func()->CopyFrom(f_attr);
        
        // Other attributes
        (*attr_map)["max_intra_op_parallelism"].set_i(1);

        // Add the node to the graph
        tensorflow::Graph graph(tensorflow::OpRegistry::Global());
        tensorflow::Status status;
        tensorflow::Node* node;
        status = tensorflow::NodeBuilder(node_def.name(), node_def.op())
                    .Finalize(&graph, &node);

        if (!status.ok()) {
            std::cerr << "Failed to create node: " << status.error_message() << std::endl;
            return 0;
        }

        std::cout << "Created MapDefun node with " << arguments.size() << " arguments, " 
                  << captured_inputs.size() << " captured inputs, and " 
                  << output_types.size() << " outputs" << std::endl;

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return 0;
    } 

    return 0;
}