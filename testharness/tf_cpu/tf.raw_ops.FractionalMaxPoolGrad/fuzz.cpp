#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cstring>
#include <iostream>
#include <vector>

#define MAX_RANK 4
#define MIN_RANK 4
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
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
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        uint8_t rank = parseRank(data[offset++]);
        std::vector<int64_t> orig_input_shape = parseShape(data, offset, size, rank);
        
        std::vector<int64_t> orig_output_shape = parseShape(data, offset, size, rank);
        
        std::vector<int64_t> out_backprop_shape = parseShape(data, offset, size, rank);
        
        if (offset >= size) return 0;
        
        uint8_t row_seq_len = (data[offset++] % 10) + 2;
        uint8_t col_seq_len = (data[offset++] % 10) + 2;
        
        bool overlapping = (data[offset++] % 2) == 1;
        
        tensorflow::Tensor orig_input_tensor(dtype, tensorflow::TensorShape(orig_input_shape));
        fillTensorWithDataByType(orig_input_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor orig_output_tensor(dtype, tensorflow::TensorShape(orig_output_shape));
        fillTensorWithDataByType(orig_output_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor out_backprop_tensor(dtype, tensorflow::TensorShape(out_backprop_shape));
        fillTensorWithDataByType(out_backprop_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor row_pooling_sequence_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({row_seq_len}));
        auto row_flat = row_pooling_sequence_tensor.flat<int64_t>();
        for (int i = 0; i < row_seq_len; ++i) {
            if (offset + sizeof(int64_t) <= size) {
                int64_t val;
                std::memcpy(&val, data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                row_flat(i) = std::abs(val) % 100;
            } else {
                row_flat(i) = i;
            }
        }
        
        tensorflow::Tensor col_pooling_sequence_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({col_seq_len}));
        auto col_flat = col_pooling_sequence_tensor.flat<int64_t>();
        for (int i = 0; i < col_seq_len; ++i) {
            if (offset + sizeof(int64_t) <= size) {
                int64_t val;
                std::memcpy(&val, data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                col_flat(i) = std::abs(val) % 100;
            } else {
                col_flat(i) = i;
            }
        }
        
        auto orig_input = tensorflow::ops::Const(root, orig_input_tensor);
        auto orig_output = tensorflow::ops::Const(root, orig_output_tensor);
        auto out_backprop = tensorflow::ops::Const(root, out_backprop_tensor);
        auto row_pooling_sequence = tensorflow::ops::Const(root, row_pooling_sequence_tensor);
        auto col_pooling_sequence = tensorflow::ops::Const(root, col_pooling_sequence_tensor);
        
        // Use raw_ops for FractionalMaxPoolGrad since it's not in the C++ ops namespace
        tensorflow::NodeDef node_def;
        node_def.set_op("FractionalMaxPoolGrad");
        node_def.set_name("fractional_max_pool_grad");
        
        tensorflow::AttrValue overlapping_attr;
        overlapping_attr.set_b(overlapping);
        (*node_def.mutable_attr())["overlapping"] = overlapping_attr;
        
        tensorflow::Status status;
        auto op = root.AddNode(node_def, &status);
        
        if (!status.ok()) {
            return -1;
        }
        
        std::vector<tensorflow::Output> inputs = {
            orig_input, orig_output, out_backprop, row_pooling_sequence, col_pooling_sequence
        };
        
        root.graph()->AddEdge(inputs[0].node(), inputs[0].index(), op.node(), 0);
        root.graph()->AddEdge(inputs[1].node(), inputs[1].index(), op.node(), 1);
        root.graph()->AddEdge(inputs[2].node(), inputs[2].index(), op.node(), 2);
        root.graph()->AddEdge(inputs[3].node(), inputs[3].index(), op.node(), 3);
        root.graph()->AddEdge(inputs[4].node(), inputs[4].index(), op.node(), 4);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({tensorflow::Output(op.node(), 0)}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
