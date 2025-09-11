#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/random_ops.h"
#include "tensorflow/core/framework/types.h"
#include <cstring>
#include <iostream>
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t points_rank = parseRank(data[offset++]);
        if (points_rank < 2) points_rank = 2;
        
        std::vector<int64_t> points_shape = parseShape(data, offset, size, points_rank);
        
        tensorflow::TensorShape points_tensor_shape;
        for (int64_t dim : points_shape) {
            points_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor points_tensor(tensorflow::DT_FLOAT, points_tensor_shape);
        fillTensorWithDataByType(points_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        int64_t num_to_sample_val = 1;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&num_to_sample_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            num_to_sample_val = std::abs(num_to_sample_val) % 100 + 1;
        }
        
        int64_t seed_val = 42;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&seed_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        int64_t num_retries_val = 5;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&num_retries_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            num_retries_val = std::abs(num_retries_val) % 20;
        }
        
        tensorflow::Tensor num_to_sample_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        num_to_sample_tensor.scalar<int64_t>()() = num_to_sample_val;
        
        tensorflow::Tensor seed_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        seed_tensor.scalar<int64_t>()() = seed_val;
        
        tensorflow::Tensor num_retries_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        num_retries_tensor.scalar<int64_t>()() = num_retries_val;
        
        auto points_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto num_to_sample_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
        auto seed_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
        auto num_retries_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
        
        // Use raw operation instead of ops namespace
        auto kmeans_op = tensorflow::Operation(
            root.WithOpName("KmeansPlusPlusInitialization"),
            "KmeansPlusPlusInitialization",
            {points_placeholder, num_to_sample_placeholder, seed_placeholder, num_retries_placeholder}
        );
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(
            {{points_placeholder, points_tensor},
             {num_to_sample_placeholder, num_to_sample_tensor},
             {seed_placeholder, seed_tensor},
             {num_retries_placeholder, num_retries_tensor}},
            {tensorflow::Output(kmeans_op, 0)},
            &outputs
        );
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
