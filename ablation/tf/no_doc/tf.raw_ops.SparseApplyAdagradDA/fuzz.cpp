#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/training_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/core/framework/graph.pb.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 3) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 2:
            dtype = tensorflow::DT_HALF;
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
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    default:
      return;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 20) return 0;
        
        tensorflow::DataType var_dtype = parseDataType(data[offset++]);
        
        uint8_t var_rank = parseRank(data[offset++]);
        std::vector<int64_t> var_shape = parseShape(data, offset, size, var_rank);
        
        uint8_t accum_rank = parseRank(data[offset++]);
        std::vector<int64_t> accum_shape = parseShape(data, offset, size, accum_rank);
        
        uint8_t squared_accum_rank = parseRank(data[offset++]);
        std::vector<int64_t> squared_accum_shape = parseShape(data, offset, size, squared_accum_rank);
        
        uint8_t grad_rank = parseRank(data[offset++]);
        std::vector<int64_t> grad_shape = parseShape(data, offset, size, grad_rank);
        
        uint8_t indices_rank = parseRank(data[offset++]);
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        
        if (offset >= size) return 0;
        
        tensorflow::Tensor var_tensor(var_dtype, tensorflow::TensorShape(var_shape));
        tensorflow::Tensor accum_tensor(var_dtype, tensorflow::TensorShape(accum_shape));
        tensorflow::Tensor squared_accum_tensor(var_dtype, tensorflow::TensorShape(squared_accum_shape));
        tensorflow::Tensor grad_tensor(var_dtype, tensorflow::TensorShape(grad_shape));
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(indices_shape));
        
        fillTensorWithDataByType(var_tensor, var_dtype, data, offset, size);
        fillTensorWithDataByType(accum_tensor, var_dtype, data, offset, size);
        fillTensorWithDataByType(squared_accum_tensor, var_dtype, data, offset, size);
        fillTensorWithDataByType(grad_tensor, var_dtype, data, offset, size);
        fillTensorWithData<int32_t>(indices_tensor, data, offset, size);
        
        float lr = 0.01f;
        float l1 = 0.0f;
        float l2 = 0.0f;
        int64_t global_step = 1;
        
        if (offset + sizeof(float) <= size) {
            std::memcpy(&lr, data + offset, sizeof(float));
            offset += sizeof(float);
            lr = std::abs(lr);
            if (lr > 1.0f) lr = 1.0f;
        }
        
        if (offset + sizeof(float) <= size) {
            std::memcpy(&l1, data + offset, sizeof(float));
            offset += sizeof(float);
            l1 = std::abs(l1);
            if (l1 > 1.0f) l1 = 1.0f;
        }
        
        if (offset + sizeof(float) <= size) {
            std::memcpy(&l2, data + offset, sizeof(float));
            offset += sizeof(float);
            l2 = std::abs(l2);
            if (l2 > 1.0f) l2 = 1.0f;
        }
        
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&global_step, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            global_step = std::abs(global_step);
            if (global_step < 1) global_step = 1;
            if (global_step > 1000) global_step = 1000;
        }
        
        tensorflow::Tensor lr_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        lr_tensor.scalar<float>()() = lr;
        
        tensorflow::Tensor l1_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        l1_tensor.scalar<float>()() = l1;
        
        tensorflow::Tensor l2_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        l2_tensor.scalar<float>()() = l2;
        
        tensorflow::Tensor global_step_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        global_step_tensor.scalar<int64_t>()() = global_step;
        
        std::cout << "var shape: ";
        for (auto dim : var_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "accum shape: ";
        for (auto dim : accum_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "squared_accum shape: ";
        for (auto dim : squared_accum_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "grad shape: ";
        for (auto dim : grad_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "indices shape: ";
        for (auto dim : indices_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "lr: " << lr << std::endl;
        std::cout << "l1: " << l1 << std::endl;
        std::cout << "l2: " << l2 << std::endl;
        std::cout << "global_step: " << global_step << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto var_op = tensorflow::ops::Const(root, var_tensor);
        auto accum_op = tensorflow::ops::Const(root, accum_tensor);
        auto squared_accum_op = tensorflow::ops::Const(root, squared_accum_tensor);
        auto grad_op = tensorflow::ops::Const(root, grad_tensor);
        auto indices_op = tensorflow::ops::Const(root, indices_tensor);
        auto lr_op = tensorflow::ops::Const(root, lr_tensor);
        auto l1_op = tensorflow::ops::Const(root, l1_tensor);
        auto l2_op = tensorflow::ops::Const(root, l2_tensor);
        auto global_step_op = tensorflow::ops::Const(root, global_step_tensor);
        
        auto sparse_apply_adagrad_da = tensorflow::ops::SparseApplyAdagradDA(
            root, var_op, accum_op, squared_accum_op, grad_op, indices_op,
            lr_op, l1_op, l2_op, global_step_op);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({sparse_apply_adagrad_da.var}, &outputs);
        
        if (!status.ok()) {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
        } else {
            std::cout << "Operation succeeded" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}