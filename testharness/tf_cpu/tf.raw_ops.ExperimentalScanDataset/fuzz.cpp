#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/dataset_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/types.h"
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
                        str += static_cast<char>(data[offset] % 26 + 'a');
                        offset++;
                    }
                    flat(i) = str;
                } else {
                    flat(i) = "default";
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
    if (size < 20) return 0;
    
    size_t offset = 0;
    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::Tensor input_dataset(tensorflow::DT_VARIANT, tensorflow::TensorShape({}));
        
        uint8_t num_initial_state = (data[offset++] % 3) + 1;
        std::vector<tensorflow::Output> initial_state_outputs;
        std::vector<tensorflow::DataType> state_types;
        
        for (uint8_t i = 0; i < num_initial_state; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            if (dtype == tensorflow::DT_STRING) dtype = tensorflow::DT_FLOAT;
            
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor tensor(dtype, tensor_shape);
            fillTensorWithDataByType(tensor, dtype, data, offset, size);
            
            auto const_op = tensorflow::ops::Const(root, tensor);
            initial_state_outputs.push_back(const_op);
            state_types.push_back(dtype);
        }
        
        uint8_t num_other_args = (data[offset++] % 2) + 1;
        std::vector<tensorflow::Output> other_arguments_outputs;
        
        for (uint8_t i = 0; i < num_other_args; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            if (dtype == tensorflow::DT_STRING) dtype = tensorflow::DT_INT32;
            
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor tensor(dtype, tensor_shape);
            fillTensorWithDataByType(tensor, dtype, data, offset, size);
            
            auto const_op = tensorflow::ops::Const(root, tensor);
            other_arguments_outputs.push_back(const_op);
        }
        
        std::vector<tensorflow::DataType> output_types = state_types;
        std::vector<tensorflow::PartialTensorShape> output_shapes;
        for (size_t i = 0; i < state_types.size(); ++i) {
            output_shapes.push_back(tensorflow::PartialTensorShape({1}));
        }
        std::vector<tensorflow::DataType> argument_types;
        argument_types.reserve(other_arguments_outputs.size());
        for (const auto& arg : other_arguments_outputs) {
            argument_types.push_back(arg.type());
        }
        
        bool preserve_cardinality = (data[offset % size] % 2) == 0;
        
        tensorflow::NameAttrList f_attr;
        f_attr.set_name("identity_func");
        
        // Build a simple RangeDataset to serve as input_dataset.
        auto start = tensorflow::ops::Const(root, static_cast<int64_t>(0));
        auto stop = tensorflow::ops::Const(root, static_cast<int64_t>(10));
        auto step = tensorflow::ops::Const(root, static_cast<int64_t>(1));
        std::vector<tensorflow::DataType> dataset_types = {tensorflow::DT_INT64};
        std::vector<tensorflow::PartialTensorShape> dataset_shapes = {tensorflow::PartialTensorShape({})};

        tensorflow::Node* range_dataset_node = nullptr;
        auto range_status = tensorflow::NodeBuilder("range_dataset", "RangeDataset")
                                .Input(start.node())
                                .Input(stop.node())
                                .Input(step.node())
                                .Attr("output_types", dataset_types)
                                .Attr("output_shapes", dataset_shapes)
                                .Finalize(root.graph(), &range_dataset_node);
        if (!range_status.ok()) {
            tf_fuzzer_utils::logError("Failed to build RangeDataset: " + range_status.ToString(), data, size);
            return -1;
        }

        std::vector<tensorflow::NodeBuilder::NodeOut> initial_state_nodes;
        for (const auto& state : initial_state_outputs) {
            initial_state_nodes.emplace_back(state.node());
        }

        std::vector<tensorflow::NodeBuilder::NodeOut> other_argument_nodes;
        for (const auto& arg : other_arguments_outputs) {
            other_argument_nodes.emplace_back(arg.node());
        }

        tensorflow::Node* scan_dataset_node = nullptr;
        auto scan_status = tensorflow::NodeBuilder("scan_dataset", "ExperimentalScanDataset")
                               .Input(range_dataset_node)
                               .Input(initial_state_nodes)
                               .Input(other_argument_nodes)
                               .Attr("f", f_attr)
                               .Attr("Tstate", state_types)
                               .Attr("Targuments", argument_types)
                               .Attr("output_types", output_types)
                               .Attr("output_shapes", output_shapes)
                               .Attr("preserve_cardinality", preserve_cardinality)
                               .Finalize(root.graph(), &scan_dataset_node);
        if (!scan_status.ok()) {
            tf_fuzzer_utils::logError("Failed to add node to graph: " + scan_status.ToString(), data, size);
            return -1;
        }
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
