#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/random_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>
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
    if (size < 3) {
        return 0;
    }

    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t seed_rank = parseRank(data[offset++]);
        std::vector<int64_t> seed_shape = parseShape(data, offset, size, seed_rank);
        
        uint8_t seed2_rank = parseRank(data[offset++]);
        std::vector<int64_t> seed2_shape = parseShape(data, offset, size, seed2_rank);
        
        uint8_t reshuffle_rank = parseRank(data[offset++]);
        std::vector<int64_t> reshuffle_shape = parseShape(data, offset, size, reshuffle_rank);

        tensorflow::TensorShape seed_tensor_shape;
        for (auto dim : seed_shape) {
            seed_tensor_shape.AddDim(dim);
        }
        
        tensorflow::TensorShape seed2_tensor_shape;
        for (auto dim : seed2_shape) {
            seed2_tensor_shape.AddDim(dim);
        }
        
        tensorflow::TensorShape reshuffle_tensor_shape;
        for (auto dim : reshuffle_shape) {
            reshuffle_tensor_shape.AddDim(dim);
        }

        tensorflow::Tensor seed_tensor(tensorflow::DT_INT64, seed_tensor_shape);
        tensorflow::Tensor seed2_tensor(tensorflow::DT_INT64, seed2_tensor_shape);
        tensorflow::Tensor reshuffle_tensor(tensorflow::DT_BOOL, reshuffle_tensor_shape);

        fillTensorWithDataByType(seed_tensor, tensorflow::DT_INT64, data, offset, size);
        fillTensorWithDataByType(seed2_tensor, tensorflow::DT_INT64, data, offset, size);
        fillTensorWithDataByType(reshuffle_tensor, tensorflow::DT_BOOL, data, offset, size);

        std::cout << "Seed tensor shape: ";
        for (int i = 0; i < seed_tensor.shape().dims(); ++i) {
            std::cout << seed_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Seed2 tensor shape: ";
        for (int i = 0; i < seed2_tensor.shape().dims(); ++i) {
            std::cout << seed2_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Reshuffle tensor shape: ";
        for (int i = 0; i < reshuffle_tensor.shape().dims(); ++i) {
            std::cout << reshuffle_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

        auto seed_input = tensorflow::ops::Const(root, seed_tensor);
        auto seed2_input = tensorflow::ops::Const(root, seed2_tensor);
        auto reshuffle_input = tensorflow::ops::Const(root, reshuffle_tensor);

        // Use raw_ops directly since AnonymousSeedGenerator is not in tensorflow::ops namespace
        auto anonymous_seed_gen = tensorflow::ops::internal::AnonymousSeedGenerator(
            root.WithOpName("AnonymousSeedGenerator"), 
            seed_input, seed2_input, reshuffle_input);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({anonymous_seed_gen.handle, anonymous_seed_gen.deleter}, &outputs);
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }

        std::cout << "AnonymousSeedGenerator executed successfully" << std::endl;
        std::cout << "Output handle type: " << outputs[0].dtype() << std::endl;
        std::cout << "Output deleter type: " << outputs[1].dtype() << std::endl;

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}