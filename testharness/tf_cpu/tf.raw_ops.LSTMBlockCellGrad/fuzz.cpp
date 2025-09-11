#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <cstring>
#include <vector>
#include <iostream>

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
    if (size < 100) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        bool use_peephole = (data[offset++] % 2) == 1;
        
        uint8_t batch_size = (data[offset++] % 5) + 1;
        uint8_t num_inputs = (data[offset++] % 5) + 1;
        uint8_t num_units = (data[offset++] % 5) + 1;
        
        tensorflow::TensorShape x_shape({batch_size, num_inputs});
        tensorflow::TensorShape cs_prev_shape({batch_size, num_units});
        tensorflow::TensorShape h_prev_shape({batch_size, num_units});
        tensorflow::TensorShape w_shape({num_inputs + num_units, 4 * num_units});
        tensorflow::TensorShape wci_shape({num_units});
        tensorflow::TensorShape wcf_shape({num_units});
        tensorflow::TensorShape wco_shape({num_units});
        tensorflow::TensorShape b_shape({4 * num_units});
        tensorflow::TensorShape i_shape({batch_size, num_units});
        tensorflow::TensorShape cs_shape({batch_size, num_units});
        tensorflow::TensorShape f_shape({batch_size, num_units});
        tensorflow::TensorShape o_shape({batch_size, num_units});
        tensorflow::TensorShape ci_shape({batch_size, num_units});
        tensorflow::TensorShape co_shape({batch_size, num_units});
        tensorflow::TensorShape cs_grad_shape({batch_size, num_units});
        tensorflow::TensorShape h_grad_shape({batch_size, num_units});
        
        tensorflow::Tensor x_tensor(dtype, x_shape);
        tensorflow::Tensor cs_prev_tensor(dtype, cs_prev_shape);
        tensorflow::Tensor h_prev_tensor(dtype, h_prev_shape);
        tensorflow::Tensor w_tensor(dtype, w_shape);
        tensorflow::Tensor wci_tensor(dtype, wci_shape);
        tensorflow::Tensor wcf_tensor(dtype, wcf_shape);
        tensorflow::Tensor wco_tensor(dtype, wco_shape);
        tensorflow::Tensor b_tensor(dtype, b_shape);
        tensorflow::Tensor i_tensor(dtype, i_shape);
        tensorflow::Tensor cs_tensor(dtype, cs_shape);
        tensorflow::Tensor f_tensor(dtype, f_shape);
        tensorflow::Tensor o_tensor(dtype, o_shape);
        tensorflow::Tensor ci_tensor(dtype, ci_shape);
        tensorflow::Tensor co_tensor(dtype, co_shape);
        tensorflow::Tensor cs_grad_tensor(dtype, cs_grad_shape);
        tensorflow::Tensor h_grad_tensor(dtype, h_grad_shape);
        
        fillTensorWithDataByType(x_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(cs_prev_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(h_prev_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(w_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(wci_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(wcf_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(wco_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(b_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(i_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(cs_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(f_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(o_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(ci_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(co_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(cs_grad_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(h_grad_tensor, dtype, data, offset, size);
        
        auto x_input = tensorflow::ops::Const(root, x_tensor);
        auto cs_prev_input = tensorflow::ops::Const(root, cs_prev_tensor);
        auto h_prev_input = tensorflow::ops::Const(root, h_prev_tensor);
        auto w_input = tensorflow::ops::Const(root, w_tensor);
        auto wci_input = tensorflow::ops::Const(root, wci_tensor);
        auto wcf_input = tensorflow::ops::Const(root, wcf_tensor);
        auto wco_input = tensorflow::ops::Const(root, wco_tensor);
        auto b_input = tensorflow::ops::Const(root, b_tensor);
        auto i_input = tensorflow::ops::Const(root, i_tensor);
        auto cs_input = tensorflow::ops::Const(root, cs_tensor);
        auto f_input = tensorflow::ops::Const(root, f_tensor);
        auto o_input = tensorflow::ops::Const(root, o_tensor);
        auto ci_input = tensorflow::ops::Const(root, ci_tensor);
        auto co_input = tensorflow::ops::Const(root, co_tensor);
        auto cs_grad_input = tensorflow::ops::Const(root, cs_grad_tensor);
        auto h_grad_input = tensorflow::ops::Const(root, h_grad_tensor);
        
        // Use raw_ops namespace for LSTMBlockCellGrad
        auto lstm_grad = tensorflow::ops::LSTMBlockCellGrad(
            root,
            x_input,
            cs_prev_input,
            h_prev_input,
            w_input,
            wci_input,
            wcf_input,
            wco_input,
            b_input,
            i_input,
            cs_input,
            f_input,
            o_input,
            ci_input,
            co_input,
            cs_grad_input,
            h_grad_input,
            tensorflow::ops::LSTMBlockCellGrad::UsePeephole(use_peephole)
        );
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({lstm_grad.cs_prev_grad, lstm_grad.dicfo, lstm_grad.wci_grad, lstm_grad.wcf_grad, lstm_grad.wco_grad}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
