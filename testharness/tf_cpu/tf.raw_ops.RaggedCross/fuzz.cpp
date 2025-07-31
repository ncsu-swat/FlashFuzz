#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>
#include <vector>
#include <cstring>
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

tensorflow::DataType parseDataTypeForValues(uint8_t selector) {
    switch (selector % 2) {
        case 0:
            return tensorflow::DT_INT64;
        case 1:
            return tensorflow::DT_STRING;
        default:
            return tensorflow::DT_INT64;
    }
}

tensorflow::DataType parseDataTypeForRowSplits(uint8_t selector) {
    switch (selector % 2) {
        case 0:
            return tensorflow::DT_INT32;
        case 1:
            return tensorflow::DT_INT64;
        default:
            return tensorflow::DT_INT32;
    }
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

void fillStringTensor(tensorflow::Tensor& tensor, const uint8_t* data,
                      size_t& offset, size_t total_size) {
    auto flat = tensor.flat<tensorflow::tstring>();
    const size_t num_elements = flat.size();

    for (size_t i = 0; i < num_elements; ++i) {
        if (offset < total_size) {
            uint8_t str_len = data[offset] % 10 + 1;
            offset++;
            
            std::string str;
            for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                str += static_cast<char>('a' + (data[offset] % 26));
                offset++;
            }
            flat(i) = tensorflow::tstring(str);
        } else {
            flat(i) = tensorflow::tstring("a");
        }
    }
}

void fillTensorWithDataByType(tensorflow::Tensor& tensor,
                              tensorflow::DataType dtype, const uint8_t* data,
                              size_t& offset, size_t total_size) {
    switch (dtype) {
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_STRING:
            fillStringTensor(tensor, data, offset, total_size);
            break;
        default:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_ragged = (data[offset++] % 3) + 1;
        uint8_t num_sparse = (data[offset++] % 3) + 1;
        uint8_t num_dense = (data[offset++] % 3) + 1;
        
        std::string input_order;
        for (int i = 0; i < num_ragged; ++i) input_order += "R";
        for (int i = 0; i < num_dense; ++i) input_order += "D";
        for (int i = 0; i < num_sparse; ++i) input_order += "S";
        
        bool hashed_output = (data[offset++] % 2) == 1;
        int64_t num_buckets = hashed_output ? ((data[offset++] % 100) + 1) : 0;
        int64_t hash_key = data[offset++];
        
        tensorflow::DataType out_values_type = parseDataTypeForValues(data[offset++]);
        tensorflow::DataType out_row_splits_type = parseDataTypeForRowSplits(data[offset++]);

        std::vector<tensorflow::Input> ragged_values;
        std::vector<tensorflow::Input> ragged_row_splits;
        
        for (int i = 0; i < num_ragged; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType values_dtype = parseDataTypeForValues(data[offset++]);
            uint8_t values_rank = parseRank(data[offset++]);
            std::vector<int64_t> values_shape = parseShape(data, offset, size, values_rank);
            
            tensorflow::Tensor values_tensor(values_dtype, tensorflow::TensorShape(values_shape));
            fillTensorWithDataByType(values_tensor, values_dtype, data, offset, size);
            
            auto values_input = tensorflow::ops::Const(root, values_tensor);
            ragged_values.push_back(values_input);
            
            tensorflow::DataType row_splits_dtype = parseDataTypeForRowSplits(data[offset++]);
            uint8_t row_splits_rank = 1;
            int64_t row_splits_size = values_shape.empty() ? 2 : values_shape[0] + 1;
            std::vector<int64_t> row_splits_shape = {row_splits_size};
            
            tensorflow::Tensor row_splits_tensor(row_splits_dtype, tensorflow::TensorShape(row_splits_shape));
            
            if (row_splits_dtype == tensorflow::DT_INT32) {
                auto flat = row_splits_tensor.flat<int32_t>();
                for (int j = 0; j < row_splits_size; ++j) {
                    flat(j) = j;
                }
            } else {
                auto flat = row_splits_tensor.flat<int64_t>();
                for (int j = 0; j < row_splits_size; ++j) {
                    flat(j) = j;
                }
            }
            
            auto row_splits_input = tensorflow::ops::Const(root, row_splits_tensor);
            ragged_row_splits.push_back(row_splits_input);
        }

        std::vector<tensorflow::Input> sparse_indices;
        std::vector<tensorflow::Input> sparse_values;
        std::vector<tensorflow::Input> sparse_shape;
        
        for (int i = 0; i < num_sparse; ++i) {
            if (offset >= size) break;
            
            tensorflow::Tensor indices_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({2, 2}));
            auto indices_flat = indices_tensor.flat<int64_t>();
            indices_flat(0) = 0; indices_flat(1) = 0;
            indices_flat(2) = 1; indices_flat(3) = 1;
            sparse_indices.push_back(tensorflow::ops::Const(root, indices_tensor));
            
            tensorflow::DataType values_dtype = parseDataTypeForValues(data[offset++]);
            tensorflow::Tensor values_tensor(values_dtype, tensorflow::TensorShape({2}));
            fillTensorWithDataByType(values_tensor, values_dtype, data, offset, size);
            sparse_values.push_back(tensorflow::ops::Const(root, values_tensor));
            
            tensorflow::Tensor shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({2}));
            auto shape_flat = shape_tensor.flat<int64_t>();
            shape_flat(0) = 2; shape_flat(1) = 2;
            sparse_shape.push_back(tensorflow::ops::Const(root, shape_tensor));
        }

        std::vector<tensorflow::Input> dense_inputs;
        for (int i = 0; i < num_dense; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType dense_dtype = parseDataTypeForValues(data[offset++]);
            uint8_t dense_rank = parseRank(data[offset++]);
            std::vector<int64_t> dense_shape = parseShape(data, offset, size, dense_rank);
            
            tensorflow::Tensor dense_tensor(dense_dtype, tensorflow::TensorShape(dense_shape));
            fillTensorWithDataByType(dense_tensor, dense_dtype, data, offset, size);
            
            auto dense_input = tensorflow::ops::Const(root, dense_tensor);
            dense_inputs.push_back(dense_input);
        }

        // Use raw_ops namespace for RaggedCross
        tensorflow::OutputList output_values;
        tensorflow::OutputList output_row_splits;
        
        tensorflow::Status status = tensorflow::ops::internal::RaggedCross(
            root.WithOpName("RaggedCross"),
            ragged_values,
            ragged_row_splits,
            sparse_indices,
            sparse_values,
            sparse_shape,
            dense_inputs,
            &output_values,
            &output_row_splits,
            tensorflow::ops::internal::RaggedCross::Attrs()
                .InputOrder(input_order)
                .HashedOutput(hashed_output)
                .NumBuckets(num_buckets)
                .HashKey(hash_key)
                .OutValuesType(out_values_type)
                .OutRowSplitsType(out_row_splits_type)
        );

        if (!status.ok()) {
            return -1;
        }

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        // Run the session with the first outputs from each list
        if (!output_values.empty() && !output_row_splits.empty()) {
            status = session.Run({output_values[0], output_row_splits[0]}, &outputs);
            
            if (!status.ok()) {
                return -1;
            }
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}