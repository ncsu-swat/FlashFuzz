#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 1
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
    void logError(const std::string& message, const uint8_t* data, size_t size) {
        std::cerr << "Error: " << message << std::endl;
    }
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_HALF;
            break;
        case 1:
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
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
            break;
        default:
            return;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        uint8_t x_rank = parseRank(data[offset++]);
        std::vector<int64_t> x_shape = parseShape(data, offset, size, x_rank);
        if (x_shape.size() != 2) {
            x_shape = {2, 4};
        }
        int64_t batch_size = x_shape[0];
        int64_t num_inputs = x_shape[1];
        
        uint8_t cs_rank = parseRank(data[offset++]);
        std::vector<int64_t> cs_shape = parseShape(data, offset, size, cs_rank);
        if (cs_shape.size() != 2) {
            cs_shape = {batch_size, 3};
        }
        int64_t num_units = cs_shape[1];
        
        std::vector<int64_t> h_shape = {batch_size, num_units};
        std::vector<int64_t> w_shape = {num_inputs + num_units, 4 * num_units};
        std::vector<int64_t> wci_shape = {num_units};
        std::vector<int64_t> wcf_shape = {num_units};
        std::vector<int64_t> wco_shape = {num_units};
        std::vector<int64_t> b_shape = {4 * num_units};

        tensorflow::Tensor x_tensor(dtype, tensorflow::TensorShape(x_shape));
        tensorflow::Tensor cs_prev_tensor(dtype, tensorflow::TensorShape(cs_shape));
        tensorflow::Tensor h_prev_tensor(dtype, tensorflow::TensorShape(h_shape));
        tensorflow::Tensor w_tensor(dtype, tensorflow::TensorShape(w_shape));
        tensorflow::Tensor wci_tensor(dtype, tensorflow::TensorShape(wci_shape));
        tensorflow::Tensor wcf_tensor(dtype, tensorflow::TensorShape(wcf_shape));
        tensorflow::Tensor wco_tensor(dtype, tensorflow::TensorShape(wco_shape));
        tensorflow::Tensor b_tensor(dtype, tensorflow::TensorShape(b_shape));

        fillTensorWithDataByType(x_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(cs_prev_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(h_prev_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(w_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(wci_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(wcf_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(wco_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(b_tensor, dtype, data, offset, size);

        auto x_op = tensorflow::ops::Const(root, x_tensor);
        auto cs_prev_op = tensorflow::ops::Const(root, cs_prev_tensor);
        auto h_prev_op = tensorflow::ops::Const(root, h_prev_tensor);
        auto w_op = tensorflow::ops::Const(root, w_tensor);
        auto wci_op = tensorflow::ops::Const(root, wci_tensor);
        auto wcf_op = tensorflow::ops::Const(root, wcf_tensor);
        auto wco_op = tensorflow::ops::Const(root, wco_tensor);
        auto b_op = tensorflow::ops::Const(root, b_tensor);

        float forget_bias = 1.0f;
        float cell_clip = 3.0f;
        bool use_peephole = false;
        
        if (offset < size) {
            uint8_t bias_byte = data[offset++];
            forget_bias = static_cast<float>(bias_byte) / 255.0f * 2.0f;
        }
        
        if (offset < size) {
            uint8_t clip_byte = data[offset++];
            cell_clip = static_cast<float>(clip_byte) / 255.0f * 10.0f;
        }
        
        if (offset < size) {
            use_peephole = (data[offset++] % 2) == 1;
        }

        // Use raw_ops directly instead of ops::LSTMBlockCell
        auto lstm_result = tensorflow::ops::internal::LSTMBlockCell(
            root, x_op, cs_prev_op, h_prev_op, w_op, wci_op, wcf_op, wco_op, b_op,
            forget_bias, cell_clip, use_peephole
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        // Adjust the output names to match the raw op outputs
        tensorflow::Status status = session.Run({
            lstm_result.i, lstm_result.cs, lstm_result.f, 
            lstm_result.o, lstm_result.ci, lstm_result.co, 
            lstm_result.h
        }, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
