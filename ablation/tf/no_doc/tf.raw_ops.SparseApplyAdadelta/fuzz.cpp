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
        
        if (size < 10) return 0;
        
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        uint8_t var_rank = parseRank(data[offset++]);
        std::vector<int64_t> var_shape = parseShape(data, offset, size, var_rank);
        
        uint8_t accum_rank = parseRank(data[offset++]);
        std::vector<int64_t> accum_shape = parseShape(data, offset, size, accum_rank);
        
        uint8_t accum_update_rank = parseRank(data[offset++]);
        std::vector<int64_t> accum_update_shape = parseShape(data, offset, size, accum_update_rank);
        
        uint8_t grad_rank = parseRank(data[offset++]);
        std::vector<int64_t> grad_shape = parseShape(data, offset, size, grad_rank);
        
        uint8_t indices_rank = parseRank(data[offset++]);
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        
        if (offset + 3 * sizeof(float) > size) return 0;
        
        float lr, rho, epsilon;
        std::memcpy(&lr, data + offset, sizeof(float));
        offset += sizeof(float);
        std::memcpy(&rho, data + offset, sizeof(float));
        offset += sizeof(float);
        std::memcpy(&epsilon, data + offset, sizeof(float));
        offset += sizeof(float);
        
        lr = std::abs(lr);
        rho = std::abs(rho);
        epsilon = std::abs(epsilon);
        if (lr > 1.0f) lr = 1.0f;
        if (rho > 1.0f) rho = 1.0f;
        if (epsilon < 1e-8f) epsilon = 1e-8f;
        
        tensorflow::TensorShape var_tensor_shape(var_shape);
        tensorflow::TensorShape accum_tensor_shape(accum_shape);
        tensorflow::TensorShape accum_update_tensor_shape(accum_update_shape);
        tensorflow::TensorShape grad_tensor_shape(grad_shape);
        tensorflow::TensorShape indices_tensor_shape(indices_shape);
        
        tensorflow::Tensor var_tensor(dtype, var_tensor_shape);
        tensorflow::Tensor accum_tensor(dtype, accum_tensor_shape);
        tensorflow::Tensor accum_update_tensor(dtype, accum_update_tensor_shape);
        tensorflow::Tensor grad_tensor(dtype, grad_tensor_shape);
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, indices_tensor_shape);
        tensorflow::Tensor lr_tensor(dtype, tensorflow::TensorShape({}));
        tensorflow::Tensor rho_tensor(dtype, tensorflow::TensorShape({}));
        tensorflow::Tensor epsilon_tensor(dtype, tensorflow::TensorShape({}));
        
        fillTensorWithDataByType(var_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(accum_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(accum_update_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(grad_tensor, dtype, data, offset, size);
        fillTensorWithData<int32_t>(indices_tensor, data, offset, size);
        
        switch (dtype) {
            case tensorflow::DT_FLOAT:
                lr_tensor.scalar<float>()() = lr;
                rho_tensor.scalar<float>()() = rho;
                epsilon_tensor.scalar<float>()() = epsilon;
                break;
            case tensorflow::DT_DOUBLE:
                lr_tensor.scalar<double>()() = static_cast<double>(lr);
                rho_tensor.scalar<double>()() = static_cast<double>(rho);
                epsilon_tensor.scalar<double>()() = static_cast<double>(epsilon);
                break;
            case tensorflow::DT_HALF:
                lr_tensor.scalar<Eigen::half>()() = Eigen::half(lr);
                rho_tensor.scalar<Eigen::half>()() = Eigen::half(rho);
                epsilon_tensor.scalar<Eigen::half>()() = Eigen::half(epsilon);
                break;
        }
        
        std::cout << "var shape: ";
        for (auto dim : var_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "accum shape: ";
        for (auto dim : accum_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "grad shape: ";
        for (auto dim : grad_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "indices shape: ";
        for (auto dim : indices_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "lr: " << lr << ", rho: " << rho << ", epsilon: " << epsilon << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto var_op = tensorflow::ops::Const(root, var_tensor);
        auto accum_op = tensorflow::ops::Const(root, accum_tensor);
        auto accum_update_op = tensorflow::ops::Const(root, accum_update_tensor);
        auto grad_op = tensorflow::ops::Const(root, grad_tensor);
        auto indices_op = tensorflow::ops::Const(root, indices_tensor);
        auto lr_op = tensorflow::ops::Const(root, lr_tensor);
        auto rho_op = tensorflow::ops::Const(root, rho_tensor);
        auto epsilon_op = tensorflow::ops::Const(root, epsilon_tensor);
        
        auto sparse_apply_adadelta = tensorflow::ops::SparseApplyAdadelta(
            root, var_op, accum_op, accum_update_op, lr_op, rho_op, epsilon_op, grad_op, indices_op);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({sparse_apply_adadelta}, &outputs);
        
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