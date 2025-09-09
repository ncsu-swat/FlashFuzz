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
        
        tensorflow::DataType grad_dtype = parseDataType(data[offset++]);
        uint8_t grad_rank = parseRank(data[offset++]);
        std::vector<int64_t> grad_shape = parseShape(data, offset, size, grad_rank);
        
        uint8_t indices_rank = parseRank(data[offset++]);
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        
        uint8_t segment_ids_rank = parseRank(data[offset++]);
        std::vector<int64_t> segment_ids_shape = parseShape(data, offset, size, segment_ids_rank);
        
        uint8_t output_dim0_rank = 0;
        std::vector<int64_t> output_dim0_shape = {1};
        
        if (offset >= size) return 0;
        
        tensorflow::TensorShape grad_tensor_shape(grad_shape);
        tensorflow::TensorShape indices_tensor_shape(indices_shape);
        tensorflow::TensorShape segment_ids_tensor_shape(segment_ids_shape);
        tensorflow::TensorShape output_dim0_tensor_shape(output_dim0_shape);
        
        tensorflow::Tensor grad_tensor(grad_dtype, grad_tensor_shape);
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, indices_tensor_shape);
        tensorflow::Tensor segment_ids_tensor(tensorflow::DT_INT32, segment_ids_tensor_shape);
        tensorflow::Tensor output_dim0_tensor(tensorflow::DT_INT32, output_dim0_tensor_shape);
        
        fillTensorWithDataByType(grad_tensor, grad_dtype, data, offset, size);
        fillTensorWithData<int32_t>(indices_tensor, data, offset, size);
        fillTensorWithData<int32_t>(segment_ids_tensor, data, offset, size);
        fillTensorWithData<int32_t>(output_dim0_tensor, data, offset, size);
        
        std::cout << "grad_tensor shape: ";
        for (int i = 0; i < grad_tensor.dims(); ++i) {
            std::cout << grad_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "indices_tensor shape: ";
        for (int i = 0; i < indices_tensor.dims(); ++i) {
            std::cout << indices_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "segment_ids_tensor shape: ";
        for (int i = 0; i < segment_ids_tensor.dims(); ++i) {
            std::cout << segment_ids_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "output_dim0_tensor shape: ";
        for (int i = 0; i < output_dim0_tensor.dims(); ++i) {
            std::cout << output_dim0_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto grad_placeholder = tensorflow::ops::Placeholder(root, grad_dtype);
        auto indices_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto segment_ids_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto output_dim0_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        
        auto sparse_segment_sqrt_n_grad_v2 = tensorflow::ops::SparseSegmentSqrtNGradV2(
            root, grad_placeholder, indices_placeholder, segment_ids_placeholder, output_dim0_placeholder);
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({
            {grad_placeholder, grad_tensor},
            {indices_placeholder, indices_tensor},
            {segment_ids_placeholder, segment_ids_tensor},
            {output_dim0_placeholder, output_dim0_tensor}
        }, {sparse_segment_sqrt_n_grad_v2}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "Operation executed successfully" << std::endl;
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].dims(); ++i) {
                std::cout << outputs[0].dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}