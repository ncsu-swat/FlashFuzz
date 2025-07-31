#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/string_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>

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
            dtype = tensorflow::DT_STRING;
            break;
        case 7:
            dtype = tensorflow::DT_INT64;
            break;
        case 8:
            dtype = tensorflow::DT_BOOL;
            break;
        case 9:
            dtype = tensorflow::DT_UINT16;
            break;
        case 10:
            dtype = tensorflow::DT_UINT32;
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
                    flat(i) = "a";
                }
            }
            break;
        }
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType values_dtype = parseDataType(data[offset++]);
        
        uint8_t hyp_indices_rank = parseRank(data[offset++]);
        std::vector<int64_t> hyp_indices_shape = parseShape(data, offset, size, hyp_indices_rank);
        
        uint8_t hyp_values_rank = parseRank(data[offset++]);
        std::vector<int64_t> hyp_values_shape = parseShape(data, offset, size, hyp_values_rank);
        
        uint8_t hyp_shape_rank = parseRank(data[offset++]);
        std::vector<int64_t> hyp_shape_shape = parseShape(data, offset, size, hyp_shape_rank);
        
        uint8_t truth_indices_rank = parseRank(data[offset++]);
        std::vector<int64_t> truth_indices_shape = parseShape(data, offset, size, truth_indices_rank);
        
        uint8_t truth_values_rank = parseRank(data[offset++]);
        std::vector<int64_t> truth_values_shape = parseShape(data, offset, size, truth_values_rank);
        
        uint8_t truth_shape_rank = parseRank(data[offset++]);
        std::vector<int64_t> truth_shape_shape = parseShape(data, offset, size, truth_shape_rank);
        
        bool normalize = (data[offset++] % 2) == 1;

        tensorflow::Tensor hyp_indices_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(hyp_indices_shape));
        fillTensorWithDataByType(hyp_indices_tensor, tensorflow::DT_INT64, data, offset, size);
        
        tensorflow::Tensor hyp_values_tensor(values_dtype, tensorflow::TensorShape(hyp_values_shape));
        fillTensorWithDataByType(hyp_values_tensor, values_dtype, data, offset, size);
        
        tensorflow::Tensor hyp_shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(hyp_shape_shape));
        fillTensorWithDataByType(hyp_shape_tensor, tensorflow::DT_INT64, data, offset, size);
        
        tensorflow::Tensor truth_indices_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(truth_indices_shape));
        fillTensorWithDataByType(truth_indices_tensor, tensorflow::DT_INT64, data, offset, size);
        
        tensorflow::Tensor truth_values_tensor(values_dtype, tensorflow::TensorShape(truth_values_shape));
        fillTensorWithDataByType(truth_values_tensor, values_dtype, data, offset, size);
        
        tensorflow::Tensor truth_shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(truth_shape_shape));
        fillTensorWithDataByType(truth_shape_tensor, tensorflow::DT_INT64, data, offset, size);

        auto hyp_indices_op = tensorflow::ops::Const(root, hyp_indices_tensor);
        auto hyp_values_op = tensorflow::ops::Const(root, hyp_values_tensor);
        auto hyp_shape_op = tensorflow::ops::Const(root, hyp_shape_tensor);
        auto truth_indices_op = tensorflow::ops::Const(root, truth_indices_tensor);
        auto truth_values_op = tensorflow::ops::Const(root, truth_values_tensor);
        auto truth_shape_op = tensorflow::ops::Const(root, truth_shape_tensor);

        tensorflow::Node* edit_distance_node;
        tensorflow::NodeBuilder builder("edit_distance", "EditDistance");
        builder.Input(hyp_indices_op.node())
               .Input(hyp_values_op.node())
               .Input(hyp_shape_op.node())
               .Input(truth_indices_op.node())
               .Input(truth_values_op.node())
               .Input(truth_shape_op.node())
               .Attr("normalize", normalize);
        
        tensorflow::Status build_status = builder.Finalize(root.graph(), &edit_distance_node);
        if (!build_status.ok()) {
            return -1;
        }

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        // Create an Output object from the node
        tensorflow::Output edit_distance_output(edit_distance_node, 0);
        std::vector<tensorflow::Output> fetch_outputs = {edit_distance_output};
        
        tensorflow::Status status = session.Run(fetch_outputs, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}