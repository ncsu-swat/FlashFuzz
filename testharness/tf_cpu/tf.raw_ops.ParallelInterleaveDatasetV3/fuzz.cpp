#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/dataset_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_shape.h"
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
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        auto input_dataset = tensorflow::ops::Placeholder(root, tensorflow::DT_VARIANT);
        
        std::vector<tensorflow::Output> other_arguments;
        
        if (offset < size) {
            tensorflow::DataType arg_dtype = parseDataType(data[offset++]);
            uint8_t arg_rank = parseRank(data[offset++]);
            std::vector<int64_t> arg_shape = parseShape(data, offset, size, arg_rank);
            
            tensorflow::TensorShape arg_tensor_shape;
            for (int64_t dim : arg_shape) {
                arg_tensor_shape.AddDim(dim);
            }
            
            auto arg_placeholder = tensorflow::ops::Placeholder(root, arg_dtype, 
                tensorflow::ops::Placeholder::Shape(arg_tensor_shape));
            other_arguments.push_back(arg_placeholder);
        }
        
        int64_t cycle_length_val = 2;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&cycle_length_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            cycle_length_val = std::abs(cycle_length_val) % 10 + 1;
        }
        auto cycle_length = tensorflow::ops::Const(root, cycle_length_val);
        
        int64_t block_length_val = 1;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&block_length_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            block_length_val = std::abs(block_length_val) % 10 + 1;
        }
        auto block_length = tensorflow::ops::Const(root, block_length_val);
        
        int64_t num_parallel_calls_val = 1;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&num_parallel_calls_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            num_parallel_calls_val = std::abs(num_parallel_calls_val) % 5 + 1;
        }
        auto num_parallel_calls = tensorflow::ops::Const(root, num_parallel_calls_val);
        
        tensorflow::NameAttrList f_attr;
        f_attr.set_name("identity_func");
        
        std::vector<tensorflow::DataType> output_types;
        std::vector<tensorflow::PartialTensorShape> output_shapes;
        
        if (offset < size) {
            tensorflow::DataType out_dtype = parseDataType(data[offset++]);
            output_types.push_back(out_dtype);
            
            uint8_t out_rank = parseRank(data[offset++]);
            std::vector<int64_t> out_shape = parseShape(data, offset, size, out_rank);
            
            tensorflow::PartialTensorShape out_tensor_shape(out_shape);
            output_shapes.push_back(out_tensor_shape);
        } else {
            output_types.push_back(tensorflow::DT_FLOAT);
            output_shapes.push_back(tensorflow::PartialTensorShape({1}));
        }
        
        std::string deterministic = "default";
        if (offset < size) {
            uint8_t det_choice = data[offset++] % 3;
            switch (det_choice) {
                case 0: deterministic = "true"; break;
                case 1: deterministic = "false"; break;
                case 2: deterministic = "default"; break;
            }
        }
        
        std::string metadata = "";
        
        // Use raw_ops approach instead of ops namespace
        tensorflow::NodeDef node_def;
        tensorflow::NodeDefBuilder builder("parallel_interleave", "ParallelInterleaveDatasetV3");
        
        builder.Input(tensorflow::NodeDefBuilder::NodeOut(input_dataset.node()->name(), 0, tensorflow::DT_VARIANT))
               .Input(tensorflow::NodeDefBuilder::NodeOut(cycle_length.node()->name(), 0, tensorflow::DT_INT64))
               .Input(tensorflow::NodeDefBuilder::NodeOut(block_length.node()->name(), 0, tensorflow::DT_INT64))
               .Input(tensorflow::NodeDefBuilder::NodeOut(num_parallel_calls.node()->name(), 0, tensorflow::DT_INT64));
        
        for (const auto& arg : other_arguments) {
            builder.Input(tensorflow::NodeDefBuilder::NodeOut(arg.node()->name(), 0, arg.type()));
        }
        
        builder.Attr("f", f_attr)
               .Attr("Targuments", tensorflow::DataTypeVector{tensorflow::DT_INT64})
               .Attr("output_types", output_types)
               .Attr("output_shapes", output_shapes)
               .Attr("deterministic", deterministic)
               .Attr("metadata", metadata);
        
        tensorflow::Status status = builder.Finalize(&node_def);
        if (!status.ok()) {
            tf_fuzzer_utils::logError("Failed to create node def: " + status.ToString(), data, size);
            return -1;
        }
        
        tensorflow::ClientSession session(root);
        
    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
