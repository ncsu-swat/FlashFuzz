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
            dtype = tensorflow::DT_QINT8;
            break;
        case 9:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 10:
            dtype = tensorflow::DT_QINT32;
            break;
        case 11:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 12:
            dtype = tensorflow::DT_QINT16;
            break;
        case 13:
            dtype = tensorflow::DT_QUINT16;
            break;
        case 14:
            dtype = tensorflow::DT_UINT16;
            break;
        case 15:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 16:
            dtype = tensorflow::DT_HALF;
            break;
        case 17:
            dtype = tensorflow::DT_UINT32;
            break;
        case 18:
            dtype = tensorflow::DT_UINT64;
            break;
        case 19:
            dtype = tensorflow::DT_COMPLEX64;
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
        tensorflow::DataType input_dtype = parseDataType(data[offset++]);
        uint8_t input_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        tensorflow::TensorShape tensor_shape;
        for (int64_t dim : input_shape) {
            tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_tensor(input_dtype, tensor_shape);
        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        
        auto tensor_slice = tensorflow::ops::Const(root, input_tensor);
        auto tensor_slice_dataset = tensorflow::ops::TensorSliceDataset(root, {tensor_slice}, {tensor_shape});
        
        uint8_t num_other_args = (offset < size) ? data[offset++] % 3 : 0;
        std::vector<tensorflow::Output> other_arguments;
        std::vector<tensorflow::DataType> other_arg_types;
        
        for (uint8_t i = 0; i < num_other_args && offset < size; ++i) {
            tensorflow::DataType arg_dtype = parseDataType(data[offset++]);
            uint8_t arg_rank = parseRank(data[offset++]);
            std::vector<int64_t> arg_shape = parseShape(data, offset, size, arg_rank);
            
            tensorflow::TensorShape arg_tensor_shape;
            for (int64_t dim : arg_shape) {
                arg_tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor arg_tensor(arg_dtype, arg_tensor_shape);
            fillTensorWithDataByType(arg_tensor, arg_dtype, data, offset, size);
            
            other_arguments.push_back(tensorflow::ops::Const(root, arg_tensor));
            other_arg_types.push_back(arg_dtype);
        }
        
        tensorflow::FunctionDef predicate_func;
        predicate_func.mutable_signature()->set_name("predicate_func");
        
        auto input_arg = predicate_func.mutable_signature()->add_input_arg();
        input_arg->set_name("input");
        input_arg->set_type(input_dtype);
        
        for (size_t i = 0; i < other_arg_types.size(); ++i) {
            auto other_arg = predicate_func.mutable_signature()->add_input_arg();
            other_arg->set_name("other_" + std::to_string(i));
            other_arg->set_type(other_arg_types[i]);
        }
        
        auto output_arg = predicate_func.mutable_signature()->add_output_arg();
        output_arg->set_name("output");
        output_arg->set_type(tensorflow::DT_BOOL);
        
        auto const_node = predicate_func.mutable_node_def()->Add();
        const_node->set_name("const_true");
        const_node->set_op("Const");
        (*const_node->mutable_attr())["dtype"].set_type(tensorflow::DT_BOOL);
        (*const_node->mutable_attr())["value"].mutable_tensor()->set_dtype(tensorflow::DT_BOOL);
        (*const_node->mutable_attr())["value"].mutable_tensor()->mutable_tensor_shape();
        (*const_node->mutable_attr())["value"].mutable_tensor()->add_bool_val(true);
        
        auto ret_node = predicate_func.mutable_ret();
        (*ret_node)["output"] = "const_true:output:0";
        
        std::vector<tensorflow::DataType> output_types = {input_dtype};
        std::vector<tensorflow::PartialTensorShape> output_shapes = {tensorflow::PartialTensorShape(input_shape)};
        
        tensorflow::NodeDef take_while_dataset_node;
        take_while_dataset_node.set_op("TakeWhileDataset");
        take_while_dataset_node.set_name("take_while_dataset");
        
        tensorflow::NodeDefBuilder builder("take_while_dataset", "TakeWhileDataset");
        builder.Input(tensorflow::NodeDefBuilder::NodeOut(tensor_slice_dataset.node()->name(), 0, tensorflow::DT_VARIANT));
        
        for (size_t i = 0; i < other_arguments.size(); ++i) {
            builder.Input(tensorflow::NodeDefBuilder::NodeOut(other_arguments[i].node()->name(), 0, other_arg_types[i]));
        }
        
        tensorflow::NameAttrList func_attr;
        func_attr.set_name("predicate_func");
        
        builder.Attr("predicate", func_attr);
        
        tensorflow::AttrValue output_types_attr;
        for (auto dtype : output_types) {
            output_types_attr.mutable_list()->add_type(dtype);
        }
        builder.Attr("output_types", output_types_attr);
        
        tensorflow::AttrValue output_shapes_attr;
        for (const auto& shape : output_shapes) {
            tensorflow::TensorShapeProto shape_proto;
            shape.AsProto(&shape_proto);
            *output_shapes_attr.mutable_list()->add_shape() = shape_proto;
        }
        builder.Attr("output_shapes", output_shapes_attr);
        
        tensorflow::Status status = builder.Finalize(&take_while_dataset_node);
        if (!status.ok()) {
            throw std::runtime_error("Failed to create TakeWhileDataset node: " + status.ToString());
        }
        
        root.graph()->AddNode(take_while_dataset_node);
        
        tensorflow::ClientSession session(root);
        
        std::cout << "Input tensor shape: ";
        for (int64_t dim : input_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Input dtype: " << tensorflow::DataTypeString(input_dtype) << std::endl;
        std::cout << "Number of other arguments: " << num_other_args << std::endl;
        
    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
