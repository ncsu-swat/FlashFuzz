#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/types.h"
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
    return tensorflow::DT_FLOAT;
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
    default:
      fillTensorWithData<float>(tensor, data, offset, total_size);
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 100) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        uint8_t batch_size = (data[offset++] % 5) + 1;
        uint8_t input_size = (data[offset++] % 5) + 1;
        uint8_t hidden_size = (data[offset++] % 5) + 1;
        
        std::vector<int64_t> x_shape = {batch_size, input_size};
        std::vector<int64_t> h_prev_shape = {batch_size, hidden_size};
        std::vector<int64_t> w_ru_shape = {input_size + hidden_size, 2 * hidden_size};
        std::vector<int64_t> w_c_shape = {input_size + hidden_size, hidden_size};
        std::vector<int64_t> b_ru_shape = {2 * hidden_size};
        std::vector<int64_t> b_c_shape = {hidden_size};
        std::vector<int64_t> r_shape = {batch_size, hidden_size};
        std::vector<int64_t> u_shape = {batch_size, hidden_size};
        std::vector<int64_t> c_shape = {batch_size, hidden_size};
        std::vector<int64_t> d_h_shape = {batch_size, hidden_size};

        tensorflow::Tensor x_tensor(dtype, tensorflow::TensorShape(x_shape));
        tensorflow::Tensor h_prev_tensor(dtype, tensorflow::TensorShape(h_prev_shape));
        tensorflow::Tensor w_ru_tensor(dtype, tensorflow::TensorShape(w_ru_shape));
        tensorflow::Tensor w_c_tensor(dtype, tensorflow::TensorShape(w_c_shape));
        tensorflow::Tensor b_ru_tensor(dtype, tensorflow::TensorShape(b_ru_shape));
        tensorflow::Tensor b_c_tensor(dtype, tensorflow::TensorShape(b_c_shape));
        tensorflow::Tensor r_tensor(dtype, tensorflow::TensorShape(r_shape));
        tensorflow::Tensor u_tensor(dtype, tensorflow::TensorShape(u_shape));
        tensorflow::Tensor c_tensor(dtype, tensorflow::TensorShape(c_shape));
        tensorflow::Tensor d_h_tensor(dtype, tensorflow::TensorShape(d_h_shape));

        fillTensorWithDataByType(x_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(h_prev_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(w_ru_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(w_c_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(b_ru_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(b_c_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(r_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(u_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(c_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(d_h_tensor, dtype, data, offset, size);

        auto x_input = tensorflow::ops::Const(root, x_tensor);
        auto h_prev_input = tensorflow::ops::Const(root, h_prev_tensor);
        auto w_ru_input = tensorflow::ops::Const(root, w_ru_tensor);
        auto w_c_input = tensorflow::ops::Const(root, w_c_tensor);
        auto b_ru_input = tensorflow::ops::Const(root, b_ru_tensor);
        auto b_c_input = tensorflow::ops::Const(root, b_c_tensor);
        auto r_input = tensorflow::ops::Const(root, r_tensor);
        auto u_input = tensorflow::ops::Const(root, u_tensor);
        auto c_input = tensorflow::ops::Const(root, c_tensor);
        auto d_h_input = tensorflow::ops::Const(root, d_h_tensor);

        tensorflow::Node* gru_grad_node;
        tensorflow::NodeBuilder builder("gru_block_cell_grad", "GRUBlockCellGrad");
        builder.Input(x_input.node())
               .Input(h_prev_input.node())
               .Input(w_ru_input.node())
               .Input(w_c_input.node())
               .Input(b_ru_input.node())
               .Input(b_c_input.node())
               .Input(r_input.node())
               .Input(u_input.node())
               .Input(c_input.node())
               .Input(d_h_input.node())
               .Attr("T", dtype);
        
        tensorflow::Status build_status = builder.Finalize(root.graph(), &gru_grad_node);
        if (!build_status.ok()) {
            return -1;
        }

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({tensorflow::Output(gru_grad_node, 0),
                                                 tensorflow::Output(gru_grad_node, 1),
                                                 tensorflow::Output(gru_grad_node, 2),
                                                 tensorflow::Output(gru_grad_node, 3)}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
