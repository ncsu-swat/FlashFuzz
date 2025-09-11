#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_inputs = (data[offset++] % 3) + 1;
        
        std::vector<tensorflow::Output> input_tensors;
        std::vector<tensorflow::DataType> input_types;
        
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
            
            auto input_const = tensorflow::ops::Const(root, input_tensor);
            input_tensors.push_back(input_const);
        }
        
        if (input_tensors.empty()) {
            return 0;
        }

        // Create condition function
        tensorflow::FunctionDef cond_func;
        tensorflow::OpDef* cond_sig = cond_func.mutable_signature();
        cond_sig->set_name("simple_cond");
        
        for (size_t i = 0; i < input_types.size(); ++i) {
            auto arg = cond_sig->add_input_arg();
            arg->set_type(input_types[i]);
            arg->set_name("cond_input_" + std::to_string(i));
        }
        
        auto ret_arg = cond_sig->add_output_arg();
        ret_arg->set_type(tensorflow::DT_BOOL);
        ret_arg->set_name("cond_output");
        
        // Create body function
        tensorflow::FunctionDef body_func;
        tensorflow::OpDef* body_sig = body_func.mutable_signature();
        body_sig->set_name("simple_body");
        
        for (size_t i = 0; i < input_types.size(); ++i) {
            auto arg = body_sig->add_input_arg();
            arg->set_type(input_types[i]);
            arg->set_name("body_input_" + std::to_string(i));
            
            auto ret_arg = body_sig->add_output_arg();
            ret_arg->set_type(input_types[i]);
            ret_arg->set_name("body_output_" + std::to_string(i));
        }
        
        // Register functions
        tensorflow::FunctionDefLibrary lib;
        *lib.add_function() = cond_func;
        *lib.add_function() = body_func;
        
        // Create StatelessWhile op
        tensorflow::NodeDef node_def;
        node_def.set_name("stateless_while");
        node_def.set_op("StatelessWhile");
        
        for (size_t i = 0; i < input_tensors.size(); ++i) {
            node_def.add_input(input_tensors[i].name());
        }
        
        auto attr_cond = node_def.mutable_attr();
        (*attr_cond)["cond"].mutable_func()->set_name("simple_cond");
        (*attr_cond)["body"].mutable_func()->set_name("simple_body");
        
        for (auto dtype : input_types) {
            (*attr_cond)["T"].mutable_list()->add_type(dtype);
        }
        
        // Add node to graph
        tensorflow::GraphDef graph_def;
        *graph_def.add_node() = node_def;
        *graph_def.mutable_library() = lib;
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        std::cout << "Created StatelessWhile operation with " << input_tensors.size() << " inputs" << std::endl;

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
