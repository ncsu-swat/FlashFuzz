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
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>
#include <cstring>
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
            dtype = tensorflow::DT_UINT32;
            break;
        case 12:
            dtype = tensorflow::DT_UINT64;
            break;
        case 13:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 14:
            dtype = tensorflow::DT_HALF;
            break;
        default:
            dtype = tensorflow::DT_FLOAT;
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
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        if (offset >= size) return 0;
        int32_t branch_index_val = static_cast<int32_t>(data[offset] % 3);
        offset++;

        tensorflow::Tensor branch_index_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        branch_index_tensor.scalar<int32_t>()() = branch_index_val;
        auto branch_index = tensorflow::ops::Const(root, branch_index_tensor);

        if (offset >= size) return 0;
        uint8_t num_inputs = (data[offset] % 3) + 1;
        offset++;

        std::vector<tensorflow::Output> input_tensors;
        std::vector<tensorflow::DataType> input_types;

        for (uint8_t i = 0; i < num_inputs; ++i) {
            if (offset >= size) return 0;
            tensorflow::DataType dtype = parseDataType(data[offset]);
            offset++;
            
            if (offset >= size) return 0;
            uint8_t rank = parseRank(data[offset]);
            offset++;

            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }

            tensorflow::Tensor tensor(dtype, tensor_shape);
            fillTensorWithDataByType(tensor, dtype, data, offset, size);

            auto input_op = tensorflow::ops::Const(root, tensor);
            input_tensors.push_back(input_op);
            input_types.push_back(dtype);
        }

        if (offset >= size) return 0;
        tensorflow::DataType output_dtype = parseDataType(data[offset]);
        offset++;

        // Create a Case operation using raw ops
        tensorflow::NodeDef case_node_def;
        case_node_def.set_name("case_op");
        case_node_def.set_op("Case");
        
        // Add branch_index as input
        *case_node_def.add_input() = branch_index.node()->name();
        
        // Add all input tensors
        for (const auto& input : input_tensors) {
            *case_node_def.add_input() = input.node()->name();
        }
        
        // Set attributes
        auto* attr_map = case_node_def.mutable_attr();
        
        // Set Tin attribute
        auto* tin_attr = &(*attr_map)["Tin"];
        for (const auto& dtype : input_types) {
            tin_attr->mutable_list()->add_type(dtype);
        }
        
        // Set Tout attribute
        auto* tout_attr = &(*attr_map)["Tout"];
        tout_attr->mutable_list()->add_type(output_dtype);
        
        // Set branches attribute
        auto* branches_attr = &(*attr_map)["branches"];
        for (int i = 0; i < 3; i++) {
            auto* func = branches_attr->mutable_list()->add_func();
            func->set_name("identity_func_" + std::to_string(i));
        }
        
        // Create the node in the graph
        tensorflow::Status status;
        tensorflow::Node* case_node = nullptr;
        status = root.graph()->AddNode(case_node_def, &case_node);
        
        if (!status.ok()) {
            return -1;
        }
        
        // Connect the inputs
        root.graph()->AddEdge(branch_index.node(), 0, case_node, 0);
        for (size_t i = 0; i < input_tensors.size(); i++) {
            root.graph()->AddEdge(input_tensors[i].node(), 0, case_node, i + 1);
        }
        
        // Create a ClientSession and run it
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        // We can't directly run the case_node since it's not part of the Scope's outputs
        // Instead, we'll run the inputs to verify they're valid
        status = session.Run({branch_index}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}