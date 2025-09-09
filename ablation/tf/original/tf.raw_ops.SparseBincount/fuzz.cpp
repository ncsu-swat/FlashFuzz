#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/sparse_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

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
    case tensorflow::DT_INT64:
      fillTensorWithData<int64_t>(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
}

tensorflow::DataType parseValuesDataType(uint8_t selector) {
  switch (selector % 2) {
    case 0:
      return tensorflow::DT_INT32;
    case 1:
      return tensorflow::DT_INT64;
    default:
      return tensorflow::DT_INT32;
  }
}

tensorflow::DataType parseWeightsDataType(uint8_t selector) {
  switch (selector % 4) {
    case 0:
      return tensorflow::DT_INT32;
    case 1:
      return tensorflow::DT_INT64;
    case 2:
      return tensorflow::DT_FLOAT;
    case 3:
      return tensorflow::DT_DOUBLE;
    default:
      return tensorflow::DT_INT32;
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

        tensorflow::DataType values_dtype = parseValuesDataType(data[offset++]);
        tensorflow::DataType weights_dtype = parseWeightsDataType(data[offset++]);
        
        uint8_t indices_rank = parseRank(data[offset++]);
        uint8_t values_rank = parseRank(data[offset++]);
        uint8_t dense_shape_rank = parseRank(data[offset++]);
        uint8_t size_rank = parseRank(data[offset++]);
        uint8_t weights_rank = parseRank(data[offset++]);
        
        bool binary_output = (data[offset++] % 2) == 1;

        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        std::vector<int64_t> values_shape = parseShape(data, offset, size, values_rank);
        std::vector<int64_t> dense_shape_shape = parseShape(data, offset, size, dense_shape_rank);
        std::vector<int64_t> size_shape = parseShape(data, offset, size, size_rank);
        std::vector<int64_t> weights_shape = parseShape(data, offset, size, weights_rank);

        if (indices_shape.size() != 2) {
            indices_shape = {2, 2};
        }
        if (values_shape.size() != 1) {
            values_shape = {2};
        }
        if (dense_shape_shape.size() != 1) {
            dense_shape_shape = {1};
        }
        if (size_shape.size() != 0) {
            size_shape = {};
        }

        tensorflow::Tensor indices_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(indices_shape));
        tensorflow::Tensor values_tensor(values_dtype, tensorflow::TensorShape(values_shape));
        tensorflow::Tensor dense_shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(dense_shape_shape));
        tensorflow::Tensor size_tensor(values_dtype, tensorflow::TensorShape(size_shape));
        tensorflow::Tensor weights_tensor(weights_dtype, tensorflow::TensorShape(weights_shape));

        fillTensorWithDataByType(indices_tensor, tensorflow::DT_INT64, data, offset, size);
        fillTensorWithDataByType(values_tensor, values_dtype, data, offset, size);
        fillTensorWithDataByType(dense_shape_tensor, tensorflow::DT_INT64, data, offset, size);
        fillTensorWithDataByType(size_tensor, values_dtype, data, offset, size);
        fillTensorWithDataByType(weights_tensor, weights_dtype, data, offset, size);

        auto indices_flat = indices_tensor.flat<int64_t>();
        for (int i = 0; i < indices_flat.size(); ++i) {
            indices_flat(i) = std::abs(indices_flat(i)) % 10;
        }

        if (values_dtype == tensorflow::DT_INT32) {
            auto values_flat = values_tensor.flat<int32_t>();
            for (int i = 0; i < values_flat.size(); ++i) {
                values_flat(i) = std::abs(values_flat(i)) % 100;
            }
        } else {
            auto values_flat = values_tensor.flat<int64_t>();
            for (int i = 0; i < values_flat.size(); ++i) {
                values_flat(i) = std::abs(values_flat(i)) % 100;
            }
        }

        auto dense_shape_flat = dense_shape_tensor.flat<int64_t>();
        for (int i = 0; i < dense_shape_flat.size(); ++i) {
            dense_shape_flat(i) = std::abs(dense_shape_flat(i)) % 10 + 1;
        }

        if (values_dtype == tensorflow::DT_INT32) {
            auto size_flat = size_tensor.flat<int32_t>();
            for (int i = 0; i < size_flat.size(); ++i) {
                size_flat(i) = std::abs(size_flat(i)) % 100 + 1;
            }
        } else {
            auto size_flat = size_tensor.flat<int64_t>();
            for (int i = 0; i < size_flat.size(); ++i) {
                size_flat(i) = std::abs(size_flat(i)) % 100 + 1;
            }
        }

        std::cout << "indices_tensor shape: ";
        for (int i = 0; i < indices_tensor.shape().dims(); ++i) {
            std::cout << indices_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "values_tensor shape: ";
        for (int i = 0; i < values_tensor.shape().dims(); ++i) {
            std::cout << values_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "dense_shape_tensor shape: ";
        for (int i = 0; i < dense_shape_tensor.shape().dims(); ++i) {
            std::cout << dense_shape_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "size_tensor shape: ";
        for (int i = 0; i < size_tensor.shape().dims(); ++i) {
            std::cout << size_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "weights_tensor shape: ";
        for (int i = 0; i < weights_tensor.shape().dims(); ++i) {
            std::cout << weights_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "binary_output: " << binary_output << std::endl;

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto indices_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
        auto values_placeholder = tensorflow::ops::Placeholder(root, values_dtype);
        auto dense_shape_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
        auto size_placeholder = tensorflow::ops::Placeholder(root, values_dtype);
        auto weights_placeholder = tensorflow::ops::Placeholder(root, weights_dtype);

        auto sparse_bincount = tensorflow::ops::SparseBincount(
            root, 
            indices_placeholder, 
            values_placeholder, 
            dense_shape_placeholder, 
            size_placeholder, 
            weights_placeholder,
            tensorflow::ops::SparseBincount::BinaryOutput(binary_output)
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({
            {indices_placeholder, indices_tensor},
            {values_placeholder, values_tensor},
            {dense_shape_placeholder, dense_shape_tensor},
            {size_placeholder, size_tensor},
            {weights_placeholder, weights_tensor}
        }, {sparse_bincount}, &outputs);

        if (status.ok() && !outputs.empty()) {
            std::cout << "SparseBincount operation completed successfully" << std::endl;
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].shape().dims(); ++i) {
                std::cout << outputs[0].shape().dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "SparseBincount operation failed: " << status.ToString() << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}