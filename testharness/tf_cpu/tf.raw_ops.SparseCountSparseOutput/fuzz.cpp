#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/sparse_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
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
    std::cerr << "Error: " << message << std::endl;
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
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType values_dtype = parseValuesDataType(data[offset++]);
        tensorflow::DataType weights_dtype = parseWeightsDataType(data[offset++]);
        
        uint8_t indices_rank = parseRank(data[offset++]);
        if (indices_rank == 0) indices_rank = 2;
        
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        if (indices_shape.size() < 2) {
            indices_shape = {3, 2};
        }
        
        int64_t num_sparse_elements = indices_shape[0];
        int64_t sparse_dims = indices_shape[1];
        
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(indices_shape));
        fillTensorWithData<int64_t>(indices_tensor, data, offset, size);
        
        std::vector<int64_t> values_shape = {num_sparse_elements};
        tensorflow::Tensor values_tensor(values_dtype, tensorflow::TensorShape(values_shape));
        fillTensorWithDataByType(values_tensor, values_dtype, data, offset, size);
        
        std::vector<int64_t> dense_shape_dims = {sparse_dims};
        tensorflow::Tensor dense_shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(dense_shape_dims));
        fillTensorWithData<int64_t>(dense_shape_tensor, data, offset, size);
        
        bool use_weights = (offset < size) ? (data[offset++] % 2 == 1) : false;
        tensorflow::Tensor weights_tensor;
        
        if (use_weights) {
            weights_tensor = tensorflow::Tensor(weights_dtype, tensorflow::TensorShape(values_shape));
            fillTensorWithDataByType(weights_tensor, weights_dtype, data, offset, size);
        } else {
            weights_tensor = tensorflow::Tensor(weights_dtype, tensorflow::TensorShape({0}));
        }
        
        bool binary_output = (offset < size) ? (data[offset++] % 2 == 1) : false;
        
        int minlength = -1;
        int maxlength = -1;
        
        if (offset + sizeof(int32_t) <= size) {
            int32_t min_val;
            std::memcpy(&min_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            minlength = std::abs(min_val) % 100;
        }
        
        if (offset + sizeof(int32_t) <= size) {
            int32_t max_val;
            std::memcpy(&max_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            maxlength = minlength + (std::abs(max_val) % 100) + 1;
        }

        auto indices_input = tensorflow::ops::Const(root, indices_tensor);
        auto values_input = tensorflow::ops::Const(root, values_tensor);
        auto dense_shape_input = tensorflow::ops::Const(root, dense_shape_tensor);
        auto weights_input = tensorflow::ops::Const(root, weights_tensor);

        // Use raw_ops namespace for SparseCountSparseOutput
        tensorflow::OutputList outputs;
        tensorflow::Status status;
        
        if (use_weights) {
            auto op_attrs = tensorflow::ops::Raw::SparseCountSparseOutput::Attrs()
                .BinaryOutput(binary_output)
                .Minlength(minlength)
                .Maxlength(maxlength);
                
            outputs = tensorflow::ops::Raw::SparseCountSparseOutput(
                root, 
                indices_input, 
                values_input, 
                dense_shape_input, 
                weights_input, 
                op_attrs
            );
        } else {
            // Create empty weights tensor with correct dtype
            auto empty_weights = tensorflow::ops::Const(root, tensorflow::Tensor(weights_dtype, tensorflow::TensorShape({0})));
            
            auto op_attrs = tensorflow::ops::Raw::SparseCountSparseOutput::Attrs()
                .BinaryOutput(binary_output)
                .Minlength(minlength)
                .Maxlength(maxlength);
                
            outputs = tensorflow::ops::Raw::SparseCountSparseOutput(
                root, 
                indices_input, 
                values_input, 
                dense_shape_input, 
                empty_weights, 
                op_attrs
            );
        }

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> output_tensors;
        
        status = session.Run({outputs[0], outputs[1], outputs[2]}, &output_tensors);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}