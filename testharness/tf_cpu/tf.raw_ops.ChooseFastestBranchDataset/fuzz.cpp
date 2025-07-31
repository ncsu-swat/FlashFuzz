#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/dataset_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"
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
                    flat(i) = "test_string";
                }
            }
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
        tensorflow::Tensor input_dataset(tensorflow::DT_VARIANT, tensorflow::TensorShape({}));
        
        if (offset + sizeof(int64_t) > size) return 0;
        int64_t ratio_num_val;
        std::memcpy(&ratio_num_val, data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        ratio_num_val = std::abs(ratio_num_val) % 100 + 1;
        
        if (offset + sizeof(int64_t) > size) return 0;
        int64_t ratio_den_val;
        std::memcpy(&ratio_den_val, data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        ratio_den_val = std::abs(ratio_den_val) % 100 + 1;
        
        tensorflow::Tensor ratio_numerator(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        ratio_numerator.scalar<int64_t>()() = ratio_num_val;
        
        tensorflow::Tensor ratio_denominator(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        ratio_denominator.scalar<int64_t>()() = ratio_den_val;
        
        if (offset >= size) return 0;
        uint8_t num_other_args = data[offset++] % 3;
        
        std::vector<tensorflow::Tensor> other_arguments;
        std::vector<tensorflow::DataType> other_arg_types;
        std::vector<int> other_arguments_lengths;
        
        for (int i = 0; i < num_other_args; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            if (offset >= size) break;
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (auto dim : shape) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor tensor(dtype, tensor_shape);
            fillTensorWithDataByType(tensor, dtype, data, offset, size);
            
            other_arguments.push_back(tensor);
            other_arg_types.push_back(dtype);
            other_arguments_lengths.push_back(1);
        }
        
        if (offset >= size) return 0;
        int num_elements_per_branch = (data[offset++] % 10) + 1;
        
        if (offset >= size) return 0;
        uint8_t num_branches = (data[offset++] % 3) + 1;
        
        std::vector<tensorflow::NameAttrList> branches;
        for (int i = 0; i < num_branches; ++i) {
            tensorflow::NameAttrList branch;
            branch.set_name("identity_func_" + std::to_string(i));
            branches.push_back(branch);
        }
        
        if (offset >= size) return 0;
        uint8_t num_output_types = (data[offset++] % 5) + 1;
        
        std::vector<tensorflow::DataType> output_types;
        std::vector<tensorflow::PartialTensorShape> output_shapes;
        
        for (int i = 0; i < num_output_types; ++i) {
            if (offset >= size) break;
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            output_types.push_back(dtype);
            
            if (offset >= size) break;
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::PartialTensorShape tensor_shape;
            if (!shape.empty()) {
                tensor_shape = tensorflow::PartialTensorShape(shape);
            }
            output_shapes.push_back(tensor_shape);
        }
        
        if (other_arguments_lengths.empty()) {
            other_arguments_lengths.push_back(0);
        }
        
        auto input_dataset_op = tensorflow::ops::Placeholder(root, tensorflow::DT_VARIANT);
        auto ratio_numerator_op = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
        auto ratio_denominator_op = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
        
        std::vector<tensorflow::ops::Placeholder> other_arg_ops;
        for (size_t i = 0; i < other_arguments.size(); ++i) {
            other_arg_ops.push_back(tensorflow::ops::Placeholder(root, other_arg_types[i]));
        }
        
        std::vector<tensorflow::Output> other_outputs;
        for (auto& op : other_arg_ops) {
            other_outputs.push_back(op);
        }
        
        // Create a raw op using the scope's internal operation builder
        tensorflow::Node* node;
        tensorflow::Status status;
        
        std::vector<tensorflow::NodeBuilder::NodeOut> other_args_node_out;
        for (const auto& output : other_outputs) {
            other_args_node_out.push_back(tensorflow::NodeBuilder::NodeOut(output.node()));
        }
        
        tensorflow::NodeBuilder node_builder = tensorflow::NodeBuilder("choose_fastest_branch_dataset", "ChooseFastestBranchDataset")
            .Input(input_dataset_op.node())
            .Input(ratio_numerator_op.node())
            .Input(ratio_denominator_op.node())
            .Input(other_args_node_out);
        
        // Add attributes
        node_builder.Attr("num_elements_per_branch", num_elements_per_branch);
        node_builder.Attr("branches", branches);
        node_builder.Attr("other_arguments_lengths", other_arguments_lengths);
        node_builder.Attr("output_types", output_types);
        node_builder.Attr("output_shapes", output_shapes);
        
        status = node_builder.Finalize(root.graph(), &node);
        
        if (!status.ok()) {
            return -1;
        }
        
        tensorflow::Output choose_fastest_op(node);

        tensorflow::ClientSession session(root);
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict;
        feed_dict.push_back({input_dataset_op.node()->name(), input_dataset});
        feed_dict.push_back({ratio_numerator_op.node()->name(), ratio_numerator});
        feed_dict.push_back({ratio_denominator_op.node()->name(), ratio_denominator});
        
        for (size_t i = 0; i < other_arguments.size(); ++i) {
            feed_dict.push_back({other_arg_ops[i].node()->name(), other_arguments[i]});
        }
        
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run(feed_dict, {choose_fastest_op}, {}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}