#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
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
    case tensorflow::DT_UINT8:
      fillTensorWithData<uint8_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT16:
      fillTensorWithData<int16_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT8:
      fillTensorWithData<int8_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT64:
      fillTensorWithData<int64_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_BOOL:
      fillTensorWithData<bool>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_UINT16:
      fillTensorWithData<uint16_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_UINT32:
      fillTensorWithData<uint32_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_UINT64:
      fillTensorWithData<uint64_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_BFLOAT16:
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset,
                                               total_size);
      break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_COMPLEX64:
      fillTensorWithData<tensorflow::complex64>(tensor, data, offset,
                                                total_size);
      break;
    case tensorflow::DT_COMPLEX128:
      fillTensorWithData<tensorflow::complex128>(tensor, data, offset,
                                                 total_size);
      break;
    default:
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t indices_rank = parseRank(data[offset++]);
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(indices_shape));
        fillTensorWithDataByType(indices_tensor, tensorflow::DT_INT32, data, offset, size);
        auto indices = tensorflow::ops::Const(root, indices_tensor);

        uint8_t gradient_rank = parseRank(data[offset++]);
        std::vector<int64_t> gradient_shape = parseShape(data, offset, size, gradient_rank);
        tensorflow::Tensor gradient_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(gradient_shape));
        fillTensorWithDataByType(gradient_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto gradient = tensorflow::ops::Const(root, gradient_tensor);

        tensorflow::Tensor learning_rate_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        fillTensorWithDataByType(learning_rate_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto learning_rate = tensorflow::ops::Const(root, learning_rate_tensor);

        tensorflow::Tensor beta_1_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        fillTensorWithDataByType(beta_1_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto beta_1 = tensorflow::ops::Const(root, beta_1_tensor);

        tensorflow::Tensor epsilon_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        fillTensorWithDataByType(epsilon_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto epsilon = tensorflow::ops::Const(root, epsilon_tensor);

        uint8_t accumulator_rank = parseRank(data[offset++]);
        std::vector<int64_t> accumulator_shape = parseShape(data, offset, size, accumulator_rank);
        tensorflow::Tensor accumulator_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(accumulator_shape));
        fillTensorWithDataByType(accumulator_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto accumulator = tensorflow::ops::Const(root, accumulator_tensor);

        uint8_t momentum_rank = parseRank(data[offset++]);
        std::vector<int64_t> momentum_shape = parseShape(data, offset, size, momentum_rank);
        tensorflow::Tensor momentum_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(momentum_shape));
        fillTensorWithDataByType(momentum_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto momentum = tensorflow::ops::Const(root, momentum_tensor);

        uint8_t embedding_table_rank = parseRank(data[offset++]);
        std::vector<int64_t> embedding_table_shape = parseShape(data, offset, size, embedding_table_rank);
        tensorflow::Tensor embedding_table_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(embedding_table_shape));
        fillTensorWithDataByType(embedding_table_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto embedding_table = tensorflow::ops::Const(root, embedding_table_tensor);

        int feature_width = 1;
        if (offset < size) {
            feature_width = static_cast<int>(data[offset++]) % 100 + 1;
        }

        bool use_nesterov = false;
        if (offset < size) {
            use_nesterov = (data[offset++] % 2) == 1;
        }

        float beta_2 = 0.999f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&beta_2, data + offset, sizeof(float));
            offset += sizeof(float);
        }

        float exponent = 0.5f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&exponent, data + offset, sizeof(float));
            offset += sizeof(float);
        }

        std::cout << "indices shape: ";
        for (auto dim : indices_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "gradient shape: ";
        for (auto dim : gradient_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "feature_width: " << feature_width << std::endl;
        std::cout << "use_nesterov: " << use_nesterov << std::endl;
        std::cout << "beta_2: " << beta_2 << std::endl;
        std::cout << "exponent: " << exponent << std::endl;

        // Create operation using raw_ops
        tensorflow::OutputList outputs;
        tensorflow::Status status = tensorflow::ops::XlaSparseCoreAdagradMomentum(
            root.WithOpName("XlaSparseCoreAdagradMomentum"),
            indices,
            gradient,
            learning_rate,
            beta_1,
            epsilon,
            accumulator,
            momentum,
            embedding_table,
            feature_width,
            use_nesterov,
            beta_2,
            exponent,
            &outputs
        );

        if (!status.ok()) {
            std::cout << "Error creating operation: " << status.ToString() << std::endl;
            return -1;
        }

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> output_tensors;
        status = session.Run({outputs[0], outputs[1], outputs[2]}, &output_tensors);
        
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}