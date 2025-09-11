#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
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
    std::cerr << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 6) {
        case 0:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 1:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 2:
            dtype = tensorflow::DT_HALF;
            break;
        case 3:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 4:
            dtype = tensorflow::DT_INT32;
            break;
        case 5:
            dtype = tensorflow::DT_INT64;
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
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_BFLOAT16:
            fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
            break;
        default:
            break;
    }
}

std::string parseMergeOp(uint8_t selector) {
    switch (selector % 4) {
        case 0: return "Min";
        case 1: return "Max";
        case 2: return "Mul";
        case 3: return "Add";
    }
    return "Add";
}

std::string parseFinalOp(uint8_t selector) {
    switch (selector % 2) {
        case 0: return "Id";
        case 1: return "Div";
    }
    return "Id";
}

std::vector<int> parseSubdivOffsets(const uint8_t* data, size_t& offset, size_t total_size) {
    std::vector<int> subdiv_offsets;
    if (offset < total_size) {
        uint8_t num_offsets = data[offset] % 5;
        offset++;
        
        for (uint8_t i = 0; i < num_offsets && offset + sizeof(int) <= total_size; ++i) {
            int val;
            std::memcpy(&val, data + offset, sizeof(int));
            offset += sizeof(int);
            subdiv_offsets.push_back(std::abs(val) % 100);
        }
    }
    return subdiv_offsets;
}

std::vector<int> parseWaitFor(const uint8_t* data, size_t& offset, size_t total_size) {
    std::vector<int> wait_for;
    if (offset < total_size) {
        uint8_t num_waits = data[offset] % 3;
        offset++;
        
        for (uint8_t i = 0; i < num_waits && offset + sizeof(int) <= total_size; ++i) {
            int val;
            std::memcpy(&val, data + offset, sizeof(int));
            offset += sizeof(int);
            wait_for.push_back(std::abs(val) % 1000);
        }
    }
    return wait_for;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        uint8_t rank = parseRank(data[offset++]);
        std::vector<int64_t> shape = parseShape(data, offset, size, rank);
        
        tensorflow::TensorShape tensor_shape;
        for (int64_t dim : shape) {
            tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_tensor(dtype, tensor_shape);
        fillTensorWithDataByType(input_tensor, dtype, data, offset, size);
        
        if (offset >= size) return 0;
        
        int group_size = 1;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&group_size, data + offset, sizeof(int));
            offset += sizeof(int);
            group_size = std::abs(group_size) % 10 + 1;
        }
        
        int group_key = 0;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&group_key, data + offset, sizeof(int));
            offset += sizeof(int);
            group_key = std::abs(group_key) % 1000;
        }
        
        int instance_key = 0;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&instance_key, data + offset, sizeof(int));
            offset += sizeof(int);
            instance_key = std::abs(instance_key) % 1000;
        }
        
        std::string merge_op = parseMergeOp(data[offset++]);
        std::string final_op = parseFinalOp(data[offset++]);
        
        std::vector<int> subdiv_offsets = parseSubdivOffsets(data, offset, size);
        std::vector<int> wait_for = parseWaitFor(data, offset, size);
        
        std::string communication_hint = "auto";
        float timeout_seconds = 0.0f;
        
        auto input_op = tensorflow::ops::Const(root, input_tensor);
        
        // Use raw_ops.CollectiveReduce
        auto collective_reduce_attrs = tensorflow::ops::Const::Attrs();
        
        tensorflow::NodeDef node_def;
        node_def.set_op("CollectiveReduce");
        node_def.set_name("collective_reduce");
        
        auto* input_attr = node_def.add_input();
        *input_attr = input_op.node()->name();
        
        tensorflow::AttrValue group_size_attr;
        group_size_attr.set_i(group_size);
        (*node_def.mutable_attr())["group_size"] = group_size_attr;
        
        tensorflow::AttrValue group_key_attr;
        group_key_attr.set_i(group_key);
        (*node_def.mutable_attr())["group_key"] = group_key_attr;
        
        tensorflow::AttrValue instance_key_attr;
        instance_key_attr.set_i(instance_key);
        (*node_def.mutable_attr())["instance_key"] = instance_key_attr;
        
        tensorflow::AttrValue merge_op_attr;
        merge_op_attr.set_s(merge_op);
        (*node_def.mutable_attr())["merge_op"] = merge_op_attr;
        
        tensorflow::AttrValue final_op_attr;
        final_op_attr.set_s(final_op);
        (*node_def.mutable_attr())["final_op"] = final_op_attr;
        
        tensorflow::AttrValue subdiv_offsets_attr;
        for (int offset : subdiv_offsets) {
            subdiv_offsets_attr.mutable_list()->add_i(offset);
        }
        (*node_def.mutable_attr())["subdiv_offsets"] = subdiv_offsets_attr;
        
        tensorflow::AttrValue wait_for_attr;
        for (int wait : wait_for) {
            wait_for_attr.mutable_list()->add_i(wait);
        }
        (*node_def.mutable_attr())["wait_for"] = wait_for_attr;
        
        tensorflow::AttrValue communication_hint_attr;
        communication_hint_attr.set_s(communication_hint);
        (*node_def.mutable_attr())["communication_hint"] = communication_hint_attr;
        
        tensorflow::AttrValue timeout_seconds_attr;
        timeout_seconds_attr.set_f(timeout_seconds);
        (*node_def.mutable_attr())["timeout_seconds"] = timeout_seconds_attr;
        
        tensorflow::AttrValue t_attr;
        t_attr.set_type(dtype);
        (*node_def.mutable_attr())["T"] = t_attr;
        
        tensorflow::Status status;
        auto op = root.AddNode(node_def, &status);
        
        if (!status.ok()) {
            tf_fuzzer_utils::logError("Failed to create CollectiveReduce op: " + status.ToString(), data, size);
            return -1;
        }
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        std::cout << "Input tensor shape: ";
        for (int i = 0; i < input_tensor.shape().dims(); ++i) {
            std::cout << input_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Group size: " << group_size << std::endl;
        std::cout << "Group key: " << group_key << std::endl;
        std::cout << "Instance key: " << instance_key << std::endl;
        std::cout << "Merge op: " << merge_op << std::endl;
        std::cout << "Final op: " << final_op << std::endl;

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
