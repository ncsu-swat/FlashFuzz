#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/sparse_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>

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

tensorflow::DataType parseDataType(uint8_t selector) {
  tensorflow::DataType dtype; 
  switch (selector % 15) {  
    case 0:
      dtype = tensorflow::DT_FLOAT;
      break;
    case 1:
      dtype = tensorflow::DT_DOUBLE;
      break;
    case 2:
      dtype = tensorflow::DT_INT32;
      break;
    case 3:
      dtype = tensorflow::DT_UINT8;
      break;
    case 4:
      dtype = tensorflow::DT_INT16;
      break;
    case 5:
      dtype = tensorflow::DT_INT8;
      break;
    case 6:
      dtype = tensorflow::DT_INT64;
      break;
    case 7:
      dtype = tensorflow::DT_BOOL;
      break;
    case 8:
      dtype = tensorflow::DT_BFLOAT16;
      break;
    case 9:
      dtype = tensorflow::DT_UINT16;
      break;
    case 10:
      dtype = tensorflow::DT_COMPLEX64;
      break;
    case 11:
      dtype = tensorflow::DT_COMPLEX128;
      break;
    case 12:
      dtype = tensorflow::DT_HALF;
      break;
    case 13:
      dtype = tensorflow::DT_UINT32;
      break;
    case 14:
      dtype = tensorflow::DT_UINT64;
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType value_dtype = parseDataType(data[offset++]);
        
        uint8_t indices_rank = parseRank(data[offset++]);
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        
        uint8_t output_shape_rank = parseRank(data[offset++]);
        std::vector<int64_t> output_shape_shape = parseShape(data, offset, size, output_shape_rank);
        
        uint8_t values_rank = parseRank(data[offset++]);
        std::vector<int64_t> values_shape = parseShape(data, offset, size, values_rank);
        
        if (offset >= size) {
            return 0;
        }

        tensorflow::TensorShape indices_tensor_shape;
        for (int64_t dim : indices_shape) {
            indices_tensor_shape.AddDim(dim);
        }
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT64, indices_tensor_shape);
        fillTensorWithData<int64_t>(indices_tensor, data, offset, size);

        tensorflow::TensorShape output_shape_tensor_shape;
        for (int64_t dim : output_shape_shape) {
            output_shape_tensor_shape.AddDim(dim);
        }
        tensorflow::Tensor output_shape_tensor(tensorflow::DT_INT64, output_shape_tensor_shape);
        fillTensorWithData<int64_t>(output_shape_tensor, data, offset, size);

        tensorflow::TensorShape values_tensor_shape;
        for (int64_t dim : values_shape) {
            values_tensor_shape.AddDim(dim);
        }
        tensorflow::Tensor values_tensor(value_dtype, values_tensor_shape);
        fillTensorWithDataByType(values_tensor, value_dtype, data, offset, size);

        tensorflow::Tensor default_value_tensor(value_dtype, tensorflow::TensorShape({}));
        fillTensorWithDataByType(default_value_tensor, value_dtype, data, offset, size);

        std::cout << "indices_tensor shape: ";
        for (int i = 0; i < indices_tensor.shape().dims(); ++i) {
            std::cout << indices_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "output_shape_tensor shape: ";
        for (int i = 0; i < output_shape_tensor.shape().dims(); ++i) {
            std::cout << output_shape_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "values_tensor shape: ";
        for (int i = 0; i < values_tensor.shape().dims(); ++i) {
            std::cout << values_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "default_value_tensor shape: ";
        for (int i = 0; i < default_value_tensor.shape().dims(); ++i) {
            std::cout << default_value_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto indices_op = tensorflow::ops::Const(root, indices_tensor);
        auto output_shape_op = tensorflow::ops::Const(root, output_shape_tensor);
        auto values_op = tensorflow::ops::Const(root, values_tensor);
        auto default_value_op = tensorflow::ops::Const(root, default_value_tensor);

        auto sparse_to_dense_op = tensorflow::ops::SparseToDense(
            root, indices_op, output_shape_op, values_op, default_value_op);

        tensorflow::GraphDef graph;
        TF_CHECK_OK(root.ToGraphDef(&graph));

        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        TF_CHECK_OK(session->Create(graph));

        std::vector<tensorflow::Tensor> outputs;
        TF_CHECK_OK(session->Run({}, {sparse_to_dense_op.node()->name() + ":0"}, {}, &outputs));

        if (!outputs.empty()) {
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].shape().dims(); ++i) {
                std::cout << outputs[0].shape().dim_size(i) << " ";
            }
            std::cout << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}