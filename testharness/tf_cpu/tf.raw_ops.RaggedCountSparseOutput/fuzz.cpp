#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>
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

tensorflow::DataType parseValuesDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
            break;
    }
    return dtype;
}

tensorflow::DataType parseWeightsDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
        case 0:
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
            break;
        case 2:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 3:
            dtype = tensorflow::DT_DOUBLE;
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
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t splits_rank = parseRank(data[offset++]);
        std::vector<int64_t> splits_shape = parseShape(data, offset, size, splits_rank);
        
        tensorflow::Tensor splits_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(splits_shape));
        fillTensorWithDataByType(splits_tensor, tensorflow::DT_INT64, data, offset, size);
        
        uint8_t values_rank = parseRank(data[offset++]);
        std::vector<int64_t> values_shape = parseShape(data, offset, size, values_rank);
        tensorflow::DataType values_dtype = parseValuesDataType(data[offset++]);
        
        tensorflow::Tensor values_tensor(values_dtype, tensorflow::TensorShape(values_shape));
        fillTensorWithDataByType(values_tensor, values_dtype, data, offset, size);
        
        uint8_t weights_rank = parseRank(data[offset++]);
        std::vector<int64_t> weights_shape = parseShape(data, offset, size, weights_rank);
        tensorflow::DataType weights_dtype = parseWeightsDataType(data[offset++]);
        
        tensorflow::Tensor weights_tensor(weights_dtype, tensorflow::TensorShape(weights_shape));
        fillTensorWithDataByType(weights_tensor, weights_dtype, data, offset, size);
        
        bool binary_output = (data[offset++] % 2) == 1;
        
        int minlength = -1;
        int maxlength = -1;
        if (offset < size) {
            minlength = static_cast<int>(data[offset++] % 100) - 1;
        }
        if (offset < size) {
            maxlength = static_cast<int>(data[offset++] % 100) - 1;
        }

        auto splits_input = tensorflow::ops::Const(root, splits_tensor);
        auto values_input = tensorflow::ops::Const(root, values_tensor);
        auto weights_input = tensorflow::ops::Const(root, weights_tensor);

        // Use raw_ops namespace for RaggedCountSparseOutput
        tensorflow::OutputList outputs;
        tensorflow::Status status;
        
        // Create attributes dictionary
        std::vector<std::pair<std::string, tensorflow::AttrValue>> attrs;
        
        tensorflow::AttrValue binary_output_attr;
        binary_output_attr.set_b(binary_output);
        attrs.push_back({"binary_output", binary_output_attr});
        
        if (minlength >= 0) {
            tensorflow::AttrValue minlength_attr;
            minlength_attr.set_i(minlength);
            attrs.push_back({"minlength", minlength_attr});
        }
        
        if (maxlength >= 0) {
            tensorflow::AttrValue maxlength_attr;
            maxlength_attr.set_i(maxlength);
            attrs.push_back({"maxlength", maxlength_attr});
        }
        
        tensorflow::AttrValue Tvalues_attr;
        Tvalues_attr.set_type(values_dtype);
        attrs.push_back({"Tvalues", Tvalues_attr});
        
        tensorflow::AttrValue Tweights_attr;
        Tweights_attr.set_type(weights_dtype);
        attrs.push_back({"Tweights", Tweights_attr});
        
        // Call the raw op
        status = root.graph()->AddOperation(
            tensorflow::Operation::MakeOperation(
                root.WithOpName("RaggedCountSparseOutput"),
                "RaggedCountSparseOutput",
                {splits_input.node(), values_input.node(), weights_input.node()},
                attrs,
                3,
                &outputs));
        
        if (!status.ok()) {
            return -1;
        }
        
        auto output_indices = tensorflow::Output(outputs[0]);
        auto output_values = tensorflow::Output(outputs[1]);
        auto output_dense_shape = tensorflow::Output(outputs[2]);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> output_tensors;
        
        status = session.Run({output_indices, output_values, output_dense_shape}, &output_tensors);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}