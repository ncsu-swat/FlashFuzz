#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/image_ops.h"
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

        uint8_t input_rank = parseRank(data[offset++]);
        if (input_rank != 4) {
            input_rank = 4;
        }
        
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        if (input_shape.size() != 4) {
            input_shape = {2, 10, 10, 3};
        }
        
        tensorflow::TensorShape input_tensor_shape;
        for (auto dim : input_shape) {
            input_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_tensor_shape);
        fillTensorWithDataByType(input_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        std::vector<int64_t> size_shape = {2};
        tensorflow::TensorShape size_tensor_shape;
        for (auto dim : size_shape) {
            size_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor size_tensor(tensorflow::DT_INT32, size_tensor_shape);
        auto size_flat = size_tensor.flat<int32_t>();
        if (offset + sizeof(int32_t) <= size) {
            int32_t height;
            std::memcpy(&height, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            height = std::abs(height) % 5 + 1;
            size_flat(0) = height;
        } else {
            size_flat(0) = 3;
        }
        
        if (offset + sizeof(int32_t) <= size) {
            int32_t width;
            std::memcpy(&width, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            width = std::abs(width) % 5 + 1;
            size_flat(1) = width;
        } else {
            size_flat(1) = 3;
        }
        
        std::vector<int64_t> offsets_shape = {input_shape[0], 2};
        tensorflow::TensorShape offsets_tensor_shape;
        for (auto dim : offsets_shape) {
            offsets_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor offsets_tensor(tensorflow::DT_FLOAT, offsets_tensor_shape);
        fillTensorWithDataByType(offsets_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        bool centered = true;
        bool normalized = true;
        bool uniform_noise = true;
        std::string noise = "uniform";
        
        if (offset < size) {
            centered = (data[offset++] % 2) == 1;
        }
        if (offset < size) {
            normalized = (data[offset++] % 2) == 1;
        }
        if (offset < size) {
            uniform_noise = (data[offset++] % 2) == 1;
        }
        if (offset < size) {
            uint8_t noise_type = data[offset++] % 3;
            switch (noise_type) {
                case 0: noise = "uniform"; break;
                case 1: noise = "gaussian"; break;
                case 2: noise = "zero"; break;
            }
        }

        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto size_op = tensorflow::ops::Const(root, size_tensor);
        auto offsets_op = tensorflow::ops::Const(root, offsets_tensor);

        auto extract_glimpse = tensorflow::ops::ExtractGlimpse(
            root, input_op, size_op, offsets_op,
            tensorflow::ops::ExtractGlimpse::Attrs()
                .Centered(centered)
                .Normalized(normalized)
                .UniformNoise(uniform_noise)
                .Noise(noise)
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({extract_glimpse}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
