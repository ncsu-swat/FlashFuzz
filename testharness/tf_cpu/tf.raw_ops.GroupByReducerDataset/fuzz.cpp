#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/dataset_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
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
    switch (selector % 10) {
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
        case 4:
            dtype = tensorflow::DT_BOOL;
            break;
        case 5:
            dtype = tensorflow::DT_UINT8;
            break;
        case 6:
            dtype = tensorflow::DT_INT16;
            break;
        case 7:
            dtype = tensorflow::DT_UINT16;
            break;
        case 8:
            dtype = tensorflow::DT_UINT32;
            break;
        case 9:
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
        tensorflow::DataType input_dtype = parseDataType(data[offset++]);
        uint8_t input_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        tensorflow::TensorShape input_tensor_shape;
        for (int64_t dim : input_shape) {
            input_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_tensor(tensorflow::DT_VARIANT, tensorflow::TensorShape({}));
        
        std::vector<tensorflow::Tensor> key_func_other_arguments;
        std::vector<tensorflow::Tensor> init_func_other_arguments;
        std::vector<tensorflow::Tensor> reduce_func_other_arguments;
        std::vector<tensorflow::Tensor> finalize_func_other_arguments;
        
        if (offset < size) {
            tensorflow::DataType arg_dtype = parseDataType(data[offset++]);
            tensorflow::Tensor arg_tensor(arg_dtype, tensorflow::TensorShape({1}));
            fillTensorWithDataByType(arg_tensor, arg_dtype, data, offset, size);
            key_func_other_arguments.push_back(arg_tensor);
            init_func_other_arguments.push_back(arg_tensor);
            reduce_func_other_arguments.push_back(arg_tensor);
            finalize_func_other_arguments.push_back(arg_tensor);
        }
        
        tensorflow::FunctionDef key_func_def;
        key_func_def.mutable_signature()->set_name("key_func");
        key_func_def.mutable_signature()->add_input_arg()->set_name("input");
        key_func_def.mutable_signature()->add_input_arg()->set_type(tensorflow::DT_INT64);
        key_func_def.mutable_signature()->mutable_output_arg()->Add()->set_type(tensorflow::DT_INT64);
        
        tensorflow::FunctionDef init_func_def;
        init_func_def.mutable_signature()->set_name("init_func");
        init_func_def.mutable_signature()->add_input_arg()->set_type(tensorflow::DT_INT64);
        init_func_def.mutable_signature()->mutable_output_arg()->Add()->set_type(tensorflow::DT_FLOAT);
        
        tensorflow::FunctionDef reduce_func_def;
        reduce_func_def.mutable_signature()->set_name("reduce_func");
        reduce_func_def.mutable_signature()->add_input_arg()->set_type(tensorflow::DT_FLOAT);
        reduce_func_def.mutable_signature()->add_input_arg()->set_name("input");
        reduce_func_def.mutable_signature()->mutable_output_arg()->Add()->set_type(tensorflow::DT_FLOAT);
        
        tensorflow::FunctionDef finalize_func_def;
        finalize_func_def.mutable_signature()->set_name("finalize_func");
        finalize_func_def.mutable_signature()->add_input_arg()->set_type(tensorflow::DT_FLOAT);
        finalize_func_def.mutable_signature()->mutable_output_arg()->Add()->set_type(tensorflow::DT_FLOAT);
        
        std::vector<tensorflow::DataType> output_types = {tensorflow::DT_FLOAT};
        std::vector<tensorflow::PartialTensorShape> output_shapes = {tensorflow::PartialTensorShape({})};
        
        auto input_dataset_op = tensorflow::ops::Placeholder(root, tensorflow::DT_VARIANT);
        
        std::vector<tensorflow::Output> key_func_args;
        std::vector<tensorflow::Output> init_func_args;
        std::vector<tensorflow::Output> reduce_func_args;
        std::vector<tensorflow::Output> finalize_func_args;
        
        for (const auto& tensor : key_func_other_arguments) {
            auto placeholder = tensorflow::ops::Placeholder(root, tensor.dtype());
            key_func_args.push_back(placeholder);
        }
        
        for (const auto& tensor : init_func_other_arguments) {
            auto placeholder = tensorflow::ops::Placeholder(root, tensor.dtype());
            init_func_args.push_back(placeholder);
        }
        
        for (const auto& tensor : reduce_func_other_arguments) {
            auto placeholder = tensorflow::ops::Placeholder(root, tensor.dtype());
            reduce_func_args.push_back(placeholder);
        }
        
        for (const auto& tensor : finalize_func_other_arguments) {
            auto placeholder = tensorflow::ops::Placeholder(root, tensor.dtype());
            finalize_func_args.push_back(placeholder);
        }
        
        // Create a NodeDef for GroupByReducerDataset
        tensorflow::NodeDef node_def;
        node_def.set_name("GroupByReducerDataset");
        node_def.set_op("GroupByReducerDataset");
        
        // Add input dataset as input to the node
        tensorflow::NodeDefBuilder builder("GroupByReducerDataset", "GroupByReducerDataset");
        builder.Input(input_dataset_op.node()->name(), 0, tensorflow::DT_VARIANT);
        
        // Add other function arguments
        for (const auto& arg : key_func_args) {
            builder.Input(arg.node()->name(), 0, arg.type());
        }
        for (const auto& arg : init_func_args) {
            builder.Input(arg.node()->name(), 0, arg.type());
        }
        for (const auto& arg : reduce_func_args) {
            builder.Input(arg.node()->name(), 0, arg.type());
        }
        for (const auto& arg : finalize_func_args) {
            builder.Input(arg.node()->name(), 0, arg.type());
        }
        
        // Add attributes
        builder.Attr("key_func", key_func_def);
        builder.Attr("init_func", init_func_def);
        builder.Attr("reduce_func", reduce_func_def);
        builder.Attr("finalize_func", finalize_func_def);
        builder.Attr("Tkey_func_other_arguments", tensorflow::DataTypeVector{tensorflow::DT_INT64});
        builder.Attr("Tinit_func_other_arguments", tensorflow::DataTypeVector{tensorflow::DT_INT64});
        builder.Attr("Treduce_func_other_arguments", tensorflow::DataTypeVector{tensorflow::DT_INT64});
        builder.Attr("Tfinalize_func_other_arguments", tensorflow::DataTypeVector{tensorflow::DT_INT64});
        builder.Attr("output_types", output_types);
        builder.Attr("output_shapes", output_shapes);
        
        tensorflow::Status status = builder.Finalize(&node_def);
        if (!status.ok()) {
            return -1;
        }
        
        // Add the node to the graph
        tensorflow::Output group_by_reducer_dataset = root.AddNode(node_def);
        
        tensorflow::ClientSession session(root);
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict;
        feed_dict.push_back({input_dataset_op.node()->name(), input_tensor});
        
        for (size_t i = 0; i < key_func_other_arguments.size(); ++i) {
            feed_dict.push_back({key_func_args[i].node()->name(), key_func_other_arguments[i]});
        }
        
        for (size_t i = 0; i < init_func_other_arguments.size(); ++i) {
            feed_dict.push_back({init_func_args[i].node()->name(), init_func_other_arguments[i]});
        }
        
        for (size_t i = 0; i < reduce_func_other_arguments.size(); ++i) {
            feed_dict.push_back({reduce_func_args[i].node()->name(), reduce_func_other_arguments[i]});
        }
        
        for (size_t i = 0; i < finalize_func_other_arguments.size(); ++i) {
            feed_dict.push_back({finalize_func_args[i].node()->name(), finalize_func_other_arguments[i]});
        }
        
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run(feed_dict, {group_by_reducer_dataset}, {}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
