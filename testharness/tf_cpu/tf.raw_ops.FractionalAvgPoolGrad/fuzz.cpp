#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
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

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
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
        tensorflow::DataType out_backprop_dtype = parseDataType(data[offset++]);
        
        std::vector<int64_t> orig_input_shape = {4};
        if (offset + 4 * sizeof(int64_t) <= size) {
            for (int i = 0; i < 4; ++i) {
                int64_t dim;
                std::memcpy(&dim, data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                orig_input_shape[i] = MIN_TENSOR_SHAPE_DIMS_TF + 
                    (std::abs(dim) % (MAX_TENSOR_SHAPE_DIMS_TF - MIN_TENSOR_SHAPE_DIMS_TF + 1));
            }
        } else {
            orig_input_shape = {2, 4, 4, 2};
        }

        std::vector<int64_t> out_backprop_shape = {2, 2, 2, 2};
        if (offset + 4 * sizeof(int64_t) <= size) {
            for (int i = 0; i < 4; ++i) {
                int64_t dim;
                std::memcpy(&dim, data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                out_backprop_shape[i] = MIN_TENSOR_SHAPE_DIMS_TF + 
                    (std::abs(dim) % (MAX_TENSOR_SHAPE_DIMS_TF - MIN_TENSOR_SHAPE_DIMS_TF + 1));
            }
        }

        int64_t row_seq_len = 3;
        int64_t col_seq_len = 3;
        if (offset + 2 * sizeof(int64_t) <= size) {
            std::memcpy(&row_seq_len, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            std::memcpy(&col_seq_len, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            row_seq_len = 2 + (std::abs(row_seq_len) % 4);
            col_seq_len = 2 + (std::abs(col_seq_len) % 4);
        }

        bool overlapping = false;
        if (offset < size) {
            overlapping = (data[offset++] % 2) == 1;
        }

        tensorflow::Tensor orig_input_tensor_shape(tensorflow::DT_INT64, tensorflow::TensorShape({4}));
        auto orig_shape_flat = orig_input_tensor_shape.flat<int64_t>();
        for (int i = 0; i < 4; ++i) {
            orig_shape_flat(i) = orig_input_shape[i];
        }

        tensorflow::Tensor out_backprop(out_backprop_dtype, tensorflow::TensorShape(out_backprop_shape));
        fillTensorWithDataByType(out_backprop, out_backprop_dtype, data, offset, size);

        tensorflow::Tensor row_pooling_sequence(tensorflow::DT_INT64, tensorflow::TensorShape({row_seq_len}));
        auto row_seq_flat = row_pooling_sequence.flat<int64_t>();
        for (int64_t i = 0; i < row_seq_len; ++i) {
            row_seq_flat(i) = i * orig_input_shape[1] / (row_seq_len - 1);
        }

        tensorflow::Tensor col_pooling_sequence(tensorflow::DT_INT64, tensorflow::TensorShape({col_seq_len}));
        auto col_seq_flat = col_pooling_sequence.flat<int64_t>();
        for (int64_t i = 0; i < col_seq_len; ++i) {
            col_seq_flat(i) = i * orig_input_shape[2] / (col_seq_len - 1);
        }

        auto orig_input_shape_op = tensorflow::ops::Const(root, orig_input_tensor_shape);
        auto out_backprop_op = tensorflow::ops::Const(root, out_backprop);
        auto row_pooling_sequence_op = tensorflow::ops::Const(root, row_pooling_sequence);
        auto col_pooling_sequence_op = tensorflow::ops::Const(root, col_pooling_sequence);

        // Use raw_ops for FractionalAvgPoolGrad since it's not in the C++ ops namespace
        auto attrs = tensorflow::ops::FractionalAvgPool::Overlapping(overlapping);
        tensorflow::OutputList outputs;
        tensorflow::Status status = tensorflow::ops::internal::FractionalAvgPoolGrad(
            root.WithOpName("FractionalAvgPoolGrad"),
            orig_input_shape_op,
            out_backprop_op,
            row_pooling_sequence_op,
            col_pooling_sequence_op,
            attrs,
            &outputs
        );

        if (!status.ok()) {
            return -1;
        }

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> output_tensors;
        status = session.Run({outputs[0]}, &output_tensors);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}