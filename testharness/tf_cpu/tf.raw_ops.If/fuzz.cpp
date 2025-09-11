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
#include <iostream>
#include <vector>
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
    switch (selector % 21) {  
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
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 9:
            dtype = tensorflow::DT_UINT16;
            break;
        case 10:
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 11:
            dtype = tensorflow::DT_HALF;
            break;
        case 12:
            dtype = tensorflow::DT_UINT32;
            break;
        case 13:
            dtype = tensorflow::DT_UINT64;
            break;
        case 14:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 15:
            dtype = tensorflow::DT_QINT8;
            break;
        case 16:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 17:
            dtype = tensorflow::DT_QINT32;
            break;
        case 18:
            dtype = tensorflow::DT_QINT16;
            break;
        case 19:
            dtype = tensorflow::DT_QUINT16;
            break;
        case 20:
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
        case tensorflow::DT_BFLOAT16:
            fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_COMPLEX64:
            fillTensorWithData<tensorflow::complex64>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_COMPLEX128:
            fillTensorWithData<tensorflow::complex128>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_STRING: {
            auto flat = tensor.flat<tensorflow::tstring>();
            const size_t num_elements = flat.size();
            for (size_t i = 0; i < num_elements; ++i) {
                if (offset < total_size) {
                    uint8_t str_len = data[offset] % 10;
                    offset++;
                    std::string str;
                    for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                        str += static_cast<char>(data[offset]);
                        offset++;
                    }
                    flat(i) = str;
                } else {
                    flat(i) = "";
                }
            }
            break;
        }
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
        uint8_t cond_rank = parseRank(data[offset++]);
        std::vector<int64_t> cond_shape = parseShape(data, offset, size, cond_rank);
        
        tensorflow::Tensor cond_tensor(tensorflow::DT_BOOL, tensorflow::TensorShape(cond_shape));
        fillTensorWithDataByType(cond_tensor, tensorflow::DT_BOOL, data, offset, size);
        
        auto cond_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_BOOL);
        
        if (offset >= size) return 0;
        uint8_t num_inputs = (data[offset++] % 3) + 1;
        
        std::vector<tensorflow::Output> input_placeholders;
        std::vector<tensorflow::Tensor> input_tensors;
        std::vector<tensorflow::DataType> input_types;
        
        for (uint8_t i = 0; i < num_inputs; ++i) {
            if (offset >= size) return 0;
            
            tensorflow::DataType input_dtype = parseDataType(data[offset++]);
            if (offset >= size) return 0;
            uint8_t input_rank = parseRank(data[offset++]);
            std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
            
            tensorflow::Tensor input_tensor(input_dtype, tensorflow::TensorShape(input_shape));
            fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
            
            auto input_placeholder = tensorflow::ops::Placeholder(root, input_dtype);
            input_placeholders.push_back(input_placeholder);
            input_tensors.push_back(input_tensor);
            input_types.push_back(input_dtype);
        }
        
        tensorflow::NameAttrList then_branch_attr;
        then_branch_attr.set_name("then_branch_func");
        
        tensorflow::NameAttrList else_branch_attr;
        else_branch_attr.set_name("else_branch_func");
        
        // Create a NodeDef for the If operation
        tensorflow::NodeDef if_node_def;
        if_node_def.set_name("if_op");
        if_node_def.set_op("If");
        
        // Add inputs to the NodeDef
        *if_node_def.add_input() = cond_placeholder.node()->name();
        for (const auto& input : input_placeholders) {
            *if_node_def.add_input() = input.node()->name();
        }
        
        // Set attributes
        tensorflow::AttrValue then_branch_value;
        *then_branch_value.mutable_func() = then_branch_attr;
        if_node_def.mutable_attr()->insert({"then_branch", then_branch_value});
        
        tensorflow::AttrValue else_branch_value;
        *else_branch_value.mutable_func() = else_branch_attr;
        if_node_def.mutable_attr()->insert({"else_branch", else_branch_value});
        
        tensorflow::AttrValue Tin_value;
        for (const auto& dtype : input_types) {
            Tin_value.mutable_list()->add_type(dtype);
        }
        if_node_def.mutable_attr()->insert({"Tin", Tin_value});
        
        tensorflow::AttrValue Tout_value;
        for (const auto& dtype : input_types) {
            Tout_value.mutable_list()->add_type(dtype);
        }
        if_node_def.mutable_attr()->insert({"Tout", Tout_value});
        
        // Add the node to the graph
        tensorflow::Status status;
        auto if_op = root.graph()->AddNode(if_node_def, &status);
        
        if (!status.ok()) {
            tf_fuzzer_utils::logError("Failed to add If node: " + status.ToString(), data, size);
            return 0;
        }
        
        tensorflow::ClientSession session(root);
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict;
        feed_dict.push_back({cond_placeholder.node()->name(), cond_tensor});
        
        for (size_t i = 0; i < input_placeholders.size(); ++i) {
            feed_dict.push_back({input_placeholders[i].node()->name(), input_tensors[i]});
        }
        
        std::vector<tensorflow::Tensor> outputs;
        
    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
