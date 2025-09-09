#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/random_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/platform/env.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

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

tensorflow::DataType parseShapeDataType(uint8_t selector) {
  switch (selector % 2) {
    case 0:
      return tensorflow::DT_INT32;
    case 1:
      return tensorflow::DT_INT64;
    default:
      return tensorflow::DT_INT32;
  }
}

tensorflow::DataType parseRateDataType(uint8_t selector) {
  switch (selector % 5) {
    case 0:
      return tensorflow::DT_HALF;
    case 1:
      return tensorflow::DT_FLOAT;
    case 2:
      return tensorflow::DT_DOUBLE;
    case 3:
      return tensorflow::DT_INT32;
    case 4:
      return tensorflow::DT_INT64;
    default:
      return tensorflow::DT_FLOAT;
  }
}

tensorflow::DataType parseOutputDataType(uint8_t selector) {
  switch (selector % 5) {
    case 0:
      return tensorflow::DT_HALF;
    case 1:
      return tensorflow::DT_FLOAT;
    case 2:
      return tensorflow::DT_DOUBLE;
    case 3:
      return tensorflow::DT_INT32;
    case 4:
      return tensorflow::DT_INT64;
    default:
      return tensorflow::DT_INT64;
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
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType shape_dtype = parseShapeDataType(data[offset++]);
        tensorflow::DataType rate_dtype = parseRateDataType(data[offset++]);
        tensorflow::DataType output_dtype = parseOutputDataType(data[offset++]);
        
        uint8_t shape_rank = parseRank(data[offset++]);
        uint8_t rate_rank = parseRank(data[offset++]);
        
        int seed = 0;
        int seed2 = 0;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&seed, data + offset, sizeof(int));
            offset += sizeof(int);
        }
        if (offset + sizeof(int) <= size) {
            std::memcpy(&seed2, data + offset, sizeof(int));
            offset += sizeof(int);
        }

        std::vector<int64_t> shape_dims = parseShape(data, offset, size, shape_rank);
        std::vector<int64_t> rate_dims = parseShape(data, offset, size, rate_rank);

        tensorflow::TensorShape shape_tensor_shape(shape_dims);
        tensorflow::TensorShape rate_tensor_shape(rate_dims);

        tensorflow::Tensor shape_tensor(shape_dtype, shape_tensor_shape);
        tensorflow::Tensor rate_tensor(rate_dtype, rate_tensor_shape);

        fillTensorWithDataByType(shape_tensor, shape_dtype, data, offset, size);
        fillTensorWithDataByType(rate_tensor, rate_dtype, data, offset, size);

        std::cout << "Shape tensor: " << shape_tensor.DebugString() << std::endl;
        std::cout << "Rate tensor: " << rate_tensor.DebugString() << std::endl;
        std::cout << "Seed: " << seed << ", Seed2: " << seed2 << std::endl;
        std::cout << "Output dtype: " << tensorflow::DataTypeString(output_dtype) << std::endl;

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto shape_op = tensorflow::ops::Const(root, shape_tensor);
        auto rate_op = tensorflow::ops::Const(root, rate_tensor);
        
        auto random_poisson = tensorflow::ops::RandomPoissonV2(
            root, shape_op, rate_op,
            tensorflow::ops::RandomPoissonV2::Seed(seed)
                .Seed2(seed2)
                .Dtype(output_dtype));

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({random_poisson}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "Output tensor: " << outputs[0].DebugString() << std::endl;
        } else {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}