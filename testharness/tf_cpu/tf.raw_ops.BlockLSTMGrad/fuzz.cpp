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
    if (size < 100) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        if (offset >= size) return 0;
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        if (offset >= size) return 0;
        uint8_t seq_len_max_rank = parseRank(data[offset++]);
        std::vector<int64_t> seq_len_max_shape = parseShape(data, offset, size, seq_len_max_rank);
        tensorflow::Tensor seq_len_max_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(seq_len_max_shape));
        fillTensorWithData<int64_t>(seq_len_max_tensor, data, offset, size);
        auto seq_len_max = tensorflow::ops::Const(root, seq_len_max_tensor);

        if (offset >= size) return 0;
        uint8_t x_rank = parseRank(data[offset++]);
        std::vector<int64_t> x_shape = parseShape(data, offset, size, x_rank);
        tensorflow::Tensor x_tensor(dtype, tensorflow::TensorShape(x_shape));
        fillTensorWithDataByType(x_tensor, dtype, data, offset, size);
        auto x = tensorflow::ops::Const(root, x_tensor);

        if (offset >= size) return 0;
        uint8_t cs_prev_rank = parseRank(data[offset++]);
        std::vector<int64_t> cs_prev_shape = parseShape(data, offset, size, cs_prev_rank);
        tensorflow::Tensor cs_prev_tensor(dtype, tensorflow::TensorShape(cs_prev_shape));
        fillTensorWithDataByType(cs_prev_tensor, dtype, data, offset, size);
        auto cs_prev = tensorflow::ops::Const(root, cs_prev_tensor);

        if (offset >= size) return 0;
        uint8_t h_prev_rank = parseRank(data[offset++]);
        std::vector<int64_t> h_prev_shape = parseShape(data, offset, size, h_prev_rank);
        tensorflow::Tensor h_prev_tensor(dtype, tensorflow::TensorShape(h_prev_shape));
        fillTensorWithDataByType(h_prev_tensor, dtype, data, offset, size);
        auto h_prev = tensorflow::ops::Const(root, h_prev_tensor);

        if (offset >= size) return 0;
        uint8_t w_rank = parseRank(data[offset++]);
        std::vector<int64_t> w_shape = parseShape(data, offset, size, w_rank);
        tensorflow::Tensor w_tensor(dtype, tensorflow::TensorShape(w_shape));
        fillTensorWithDataByType(w_tensor, dtype, data, offset, size);
        auto w = tensorflow::ops::Const(root, w_tensor);

        if (offset >= size) return 0;
        uint8_t wci_rank = parseRank(data[offset++]);
        std::vector<int64_t> wci_shape = parseShape(data, offset, size, wci_rank);
        tensorflow::Tensor wci_tensor(dtype, tensorflow::TensorShape(wci_shape));
        fillTensorWithDataByType(wci_tensor, dtype, data, offset, size);
        auto wci = tensorflow::ops::Const(root, wci_tensor);

        if (offset >= size) return 0;
        uint8_t wcf_rank = parseRank(data[offset++]);
        std::vector<int64_t> wcf_shape = parseShape(data, offset, size, wcf_rank);
        tensorflow::Tensor wcf_tensor(dtype, tensorflow::TensorShape(wcf_shape));
        fillTensorWithDataByType(wcf_tensor, dtype, data, offset, size);
        auto wcf = tensorflow::ops::Const(root, wcf_tensor);

        if (offset >= size) return 0;
        uint8_t wco_rank = parseRank(data[offset++]);
        std::vector<int64_t> wco_shape = parseShape(data, offset, size, wco_rank);
        tensorflow::Tensor wco_tensor(dtype, tensorflow::TensorShape(wco_shape));
        fillTensorWithDataByType(wco_tensor, dtype, data, offset, size);
        auto wco = tensorflow::ops::Const(root, wco_tensor);

        if (offset >= size) return 0;
        uint8_t b_rank = parseRank(data[offset++]);
        std::vector<int64_t> b_shape = parseShape(data, offset, size, b_rank);
        tensorflow::Tensor b_tensor(dtype, tensorflow::TensorShape(b_shape));
        fillTensorWithDataByType(b_tensor, dtype, data, offset, size);
        auto b = tensorflow::ops::Const(root, b_tensor);

        if (offset >= size) return 0;
        uint8_t i_rank = parseRank(data[offset++]);
        std::vector<int64_t> i_shape = parseShape(data, offset, size, i_rank);
        tensorflow::Tensor i_tensor(dtype, tensorflow::TensorShape(i_shape));
        fillTensorWithDataByType(i_tensor, dtype, data, offset, size);
        auto i = tensorflow::ops::Const(root, i_tensor);

        if (offset >= size) return 0;
        uint8_t cs_rank = parseRank(data[offset++]);
        std::vector<int64_t> cs_shape = parseShape(data, offset, size, cs_rank);
        tensorflow::Tensor cs_tensor(dtype, tensorflow::TensorShape(cs_shape));
        fillTensorWithDataByType(cs_tensor, dtype, data, offset, size);
        auto cs = tensorflow::ops::Const(root, cs_tensor);

        if (offset >= size) return 0;
        uint8_t f_rank = parseRank(data[offset++]);
        std::vector<int64_t> f_shape = parseShape(data, offset, size, f_rank);
        tensorflow::Tensor f_tensor(dtype, tensorflow::TensorShape(f_shape));
        fillTensorWithDataByType(f_tensor, dtype, data, offset, size);
        auto f = tensorflow::ops::Const(root, f_tensor);

        if (offset >= size) return 0;
        uint8_t o_rank = parseRank(data[offset++]);
        std::vector<int64_t> o_shape = parseShape(data, offset, size, o_rank);
        tensorflow::Tensor o_tensor(dtype, tensorflow::TensorShape(o_shape));
        fillTensorWithDataByType(o_tensor, dtype, data, offset, size);
        auto o = tensorflow::ops::Const(root, o_tensor);

        if (offset >= size) return 0;
        uint8_t ci_rank = parseRank(data[offset++]);
        std::vector<int64_t> ci_shape = parseShape(data, offset, size, ci_rank);
        tensorflow::Tensor ci_tensor(dtype, tensorflow::TensorShape(ci_shape));
        fillTensorWithDataByType(ci_tensor, dtype, data, offset, size);
        auto ci = tensorflow::ops::Const(root, ci_tensor);

        if (offset >= size) return 0;
        uint8_t co_rank = parseRank(data[offset++]);
        std::vector<int64_t> co_shape = parseShape(data, offset, size, co_rank);
        tensorflow::Tensor co_tensor(dtype, tensorflow::TensorShape(co_shape));
        fillTensorWithDataByType(co_tensor, dtype, data, offset, size);
        auto co = tensorflow::ops::Const(root, co_tensor);

        if (offset >= size) return 0;
        uint8_t h_rank = parseRank(data[offset++]);
        std::vector<int64_t> h_shape = parseShape(data, offset, size, h_rank);
        tensorflow::Tensor h_tensor(dtype, tensorflow::TensorShape(h_shape));
        fillTensorWithDataByType(h_tensor, dtype, data, offset, size);
        auto h = tensorflow::ops::Const(root, h_tensor);

        if (offset >= size) return 0;
        uint8_t cs_grad_rank = parseRank(data[offset++]);
        std::vector<int64_t> cs_grad_shape = parseShape(data, offset, size, cs_grad_rank);
        tensorflow::Tensor cs_grad_tensor(dtype, tensorflow::TensorShape(cs_grad_shape));
        fillTensorWithDataByType(cs_grad_tensor, dtype, data, offset, size);
        auto cs_grad = tensorflow::ops::Const(root, cs_grad_tensor);

        if (offset >= size) return 0;
        uint8_t h_grad_rank = parseRank(data[offset++]);
        std::vector<int64_t> h_grad_shape = parseShape(data, offset, size, h_grad_rank);
        tensorflow::Tensor h_grad_tensor(dtype, tensorflow::TensorShape(h_grad_shape));
        fillTensorWithDataByType(h_grad_tensor, dtype, data, offset, size);
        auto h_grad = tensorflow::ops::Const(root, h_grad_tensor);

        if (offset >= size) return 0;
        bool use_peephole = (data[offset++] % 2) == 1;

        tensorflow::ops::BlockLSTMGrad::Attrs attrs;
        if (use_peephole) {
            attrs = attrs.UsePeephole(true);
        }

        auto block_lstm_grad = tensorflow::ops::BlockLSTMGrad(
            root, seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b,
            i, cs, f, o, ci, co, h, cs_grad, h_grad, attrs);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({block_lstm_grad.x_grad, block_lstm_grad.cs_prev_grad,
                                                 block_lstm_grad.h_prev_grad, block_lstm_grad.w_grad,
                                                 block_lstm_grad.wci_grad, block_lstm_grad.wcf_grad,
                                                 block_lstm_grad.wco_grad, block_lstm_grad.b_grad}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}