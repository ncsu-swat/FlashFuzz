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
#include "tensorflow/core/platform/types.h"
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
            dtype = tensorflow::DT_STRING;
            break;
        case 7:
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 8:
            dtype = tensorflow::DT_INT64;
            break;
        case 9:
            dtype = tensorflow::DT_BOOL;
            break;
        case 10:
            dtype = tensorflow::DT_QINT8;
            break;
        case 11:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 12:
            dtype = tensorflow::DT_QINT32;
            break;
        case 13:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 14:
            dtype = tensorflow::DT_QINT16;
            break;
        case 15:
            dtype = tensorflow::DT_QUINT16;
            break;
        case 16:
            dtype = tensorflow::DT_UINT16;
            break;
        case 17:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 18:
            dtype = tensorflow::DT_HALF;
            break;
        case 19:
            dtype = tensorflow::DT_UINT32;
            break;
        case 20:
            dtype = tensorflow::DT_UINT64;
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
        case tensorflow::DT_STRING:
            {
                auto flat = tensor.flat<tensorflow::tstring>();
                for (int i = 0; i < flat.size(); ++i) {
                    if (offset < total_size) {
                        uint8_t str_len = data[offset] % 10 + 1;
                        offset++;
                        std::string str;
                        for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                            str += static_cast<char>(data[offset] % 128);
                            offset++;
                        }
                        flat(i) = str;
                    } else {
                        flat(i) = "";
                    }
                }
            }
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
        tensorflow::Tensor input_dataset(tensorflow::DT_VARIANT, tensorflow::TensorShape({}));
        
        uint8_t num_key_args = data[offset++] % 3;
        uint8_t num_init_args = data[offset++] % 3;
        uint8_t num_reduce_args = data[offset++] % 3;
        uint8_t num_finalize_args = data[offset++] % 3;
        
        std::vector<tensorflow::Tensor> key_func_other_arguments;
        for (uint8_t i = 0; i < num_key_args; ++i) {
            if (offset >= size) break;
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            if (offset >= size) break;
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor tensor(dtype, tensor_shape);
            fillTensorWithDataByType(tensor, dtype, data, offset, size);
            key_func_other_arguments.push_back(tensor);
        }
        
        std::vector<tensorflow::Tensor> init_func_other_arguments;
        for (uint8_t i = 0; i < num_init_args; ++i) {
            if (offset >= size) break;
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            if (offset >= size) break;
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor tensor(dtype, tensor_shape);
            fillTensorWithDataByType(tensor, dtype, data, offset, size);
            init_func_other_arguments.push_back(tensor);
        }
        
        std::vector<tensorflow::Tensor> reduce_func_other_arguments;
        for (uint8_t i = 0; i < num_reduce_args; ++i) {
            if (offset >= size) break;
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            if (offset >= size) break;
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor tensor(dtype, tensor_shape);
            fillTensorWithDataByType(tensor, dtype, data, offset, size);
            reduce_func_other_arguments.push_back(tensor);
        }
        
        std::vector<tensorflow::Tensor> finalize_func_other_arguments;
        for (uint8_t i = 0; i < num_finalize_args; ++i) {
            if (offset >= size) break;
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            if (offset >= size) break;
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor tensor(dtype, tensor_shape);
            fillTensorWithDataByType(tensor, dtype, data, offset, size);
            finalize_func_other_arguments.push_back(tensor);
        }
        
        if (offset >= size) return 0;
        uint8_t num_output_types = data[offset++] % 5 + 1;
        std::vector<tensorflow::DataType> output_types;
        std::vector<tensorflow::PartialTensorShape> output_shapes;
        
        for (uint8_t i = 0; i < num_output_types; ++i) {
            if (offset >= size) break;
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            output_types.push_back(dtype);
            
            if (offset >= size) break;
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::PartialTensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            output_shapes.push_back(tensor_shape);
        }
        
        if (output_types.empty()) {
            output_types.push_back(tensorflow::DT_FLOAT);
            output_shapes.push_back(tensorflow::PartialTensorShape({}));
        }
        
        tensorflow::NameAttrList key_func;
        key_func.set_name("key_func");
        
        tensorflow::NameAttrList init_func;
        init_func.set_name("init_func");
        
        tensorflow::NameAttrList reduce_func;
        reduce_func.set_name("reduce_func");
        
        tensorflow::NameAttrList finalize_func;
        finalize_func.set_name("finalize_func");
        
        // Create a node def for ExperimentalGroupByReducerDataset
        tensorflow::NodeDef node_def;
        tensorflow::NodeDefBuilder builder("experimental_group_by_reducer_dataset", "ExperimentalGroupByReducerDataset");
        
        // Add inputs
        builder.Input("input_dataset", 0, tensorflow::DT_VARIANT);
        
        // Add key_func_other_arguments
        std::vector<tensorflow::NodeDefBuilder::NodeOut> key_func_args;
        for (size_t i = 0; i < key_func_other_arguments.size(); ++i) {
            key_func_args.emplace_back("key_func_arg_" + std::to_string(i), i, key_func_other_arguments[i].dtype());
        }
        builder.Input(key_func_args);
        
        // Add init_func_other_arguments
        std::vector<tensorflow::NodeDefBuilder::NodeOut> init_func_args;
        for (size_t i = 0; i < init_func_other_arguments.size(); ++i) {
            init_func_args.emplace_back("init_func_arg_" + std::to_string(i), i, init_func_other_arguments[i].dtype());
        }
        builder.Input(init_func_args);
        
        // Add reduce_func_other_arguments
        std::vector<tensorflow::NodeDefBuilder::NodeOut> reduce_func_args;
        for (size_t i = 0; i < reduce_func_other_arguments.size(); ++i) {
            reduce_func_args.emplace_back("reduce_func_arg_" + std::to_string(i), i, reduce_func_other_arguments[i].dtype());
        }
        builder.Input(reduce_func_args);
        
        // Add finalize_func_other_arguments
        std::vector<tensorflow::NodeDefBuilder::NodeOut> finalize_func_args;
        for (size_t i = 0; i < finalize_func_other_arguments.size(); ++i) {
            finalize_func_args.emplace_back("finalize_func_arg_" + std::to_string(i), i, finalize_func_other_arguments[i].dtype());
        }
        builder.Input(finalize_func_args);
        
        // Add attributes
        builder.Attr("key_func", key_func);
        builder.Attr("init_func", init_func);
        builder.Attr("reduce_func", reduce_func);
        builder.Attr("finalize_func", finalize_func);
        builder.Attr("Tkey_func_other_arguments", tensorflow::DataTypeVector{});
        builder.Attr("Tinit_func_other_arguments", tensorflow::DataTypeVector{});
        builder.Attr("Treduce_func_other_arguments", tensorflow::DataTypeVector{});
        builder.Attr("Tfinalize_func_other_arguments", tensorflow::DataTypeVector{});
        builder.Attr("output_types", output_types);
        builder.Attr("output_shapes", output_shapes);
        
        tensorflow::Status status = builder.Finalize(&node_def);
        if (!status.ok()) {
            tf_fuzzer_utils::logError("Failed to create NodeDef: " + status.ToString(), data, size);
            return 0;
        }
        
        // Create the operation using the node def
        tensorflow::Operation operation = root.graph()->CreateOperation(node_def);
        if (!operation.node()) {
            tf_fuzzer_utils::logError("Failed to create operation", data, size);
            return 0;
        }
        
        tensorflow::ClientSession session(root);
        
    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
