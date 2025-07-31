#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include <cstring>
#include <iostream>
#include <vector>

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
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
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
        tensorflow::DataType inputs_dtype = parseDataType(data[offset++]);
        
        uint8_t inputs_rank = 3;
        std::vector<int64_t> inputs_shape = {3, 2, 4};
        
        tensorflow::Tensor inputs_tensor(inputs_dtype, tensorflow::TensorShape(inputs_shape));
        fillTensorWithDataByType(inputs_tensor, inputs_dtype, data, offset, size);
        
        int64_t num_labels = 2;
        tensorflow::Tensor labels_indices_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({num_labels, 2}));
        auto labels_indices_flat = labels_indices_tensor.flat<int64_t>();
        labels_indices_flat(0) = 0; labels_indices_flat(1) = 0;
        labels_indices_flat(2) = 0; labels_indices_flat(3) = 1;
        
        tensorflow::Tensor labels_values_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({num_labels}));
        auto labels_values_flat = labels_values_tensor.flat<int32_t>();
        labels_values_flat(0) = 1;
        labels_values_flat(1) = 2;
        
        tensorflow::Tensor sequence_length_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({2}));
        auto sequence_length_flat = sequence_length_tensor.flat<int32_t>();
        sequence_length_flat(0) = 3;
        sequence_length_flat(1) = 3;
        
        bool preprocess_collapse_repeated = (data[offset % size] % 2) == 0;
        bool ctc_merge_repeated = (data[(offset + 1) % size] % 2) == 0;
        bool ignore_longer_outputs_than_inputs = (data[(offset + 2) % size] % 2) == 0;
        
        auto inputs_op = tensorflow::ops::Const(root, inputs_tensor);
        auto labels_indices_op = tensorflow::ops::Const(root, labels_indices_tensor);
        auto labels_values_op = tensorflow::ops::Const(root, labels_values_tensor);
        auto sequence_length_op = tensorflow::ops::Const(root, sequence_length_tensor);
        
        // Create a sparse tensor for labels
        tensorflow::Tensor labels_shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({2}));
        auto labels_shape_flat = labels_shape_tensor.flat<int64_t>();
        labels_shape_flat(0) = 1;  // batch size
        labels_shape_flat(1) = 2;  // max time
        auto labels_shape_op = tensorflow::ops::Const(root, labels_shape_tensor);
        
        // Use raw_ops.CTCLoss directly
        std::vector<tensorflow::Output> ctc_loss_outputs;
        tensorflow::Status status = tensorflow::ops::CTCLoss(
            root.WithOpName("CTCLoss"),
            inputs_op,
            labels_indices_op,
            labels_values_op,
            labels_shape_op,
            sequence_length_op,
            preprocess_collapse_repeated,
            ctc_merge_repeated,
            ignore_longer_outputs_than_inputs,
            &ctc_loss_outputs
        );
        
        if (!status.ok()) {
            return -1;
        }

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({ctc_loss_outputs[0], ctc_loss_outputs[1]}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}