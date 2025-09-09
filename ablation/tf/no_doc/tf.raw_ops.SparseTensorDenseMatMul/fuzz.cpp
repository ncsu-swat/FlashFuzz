#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/sparse_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/public/session.h>
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

tensorflow::DataType parseDataType(uint8_t selector) {
  tensorflow::DataType dtype; 
  switch (selector % 10) {  
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
      dtype = tensorflow::DT_INT64;
      break;
    case 4:
      dtype = tensorflow::DT_BFLOAT16;
      break;
    case 5:
      dtype = tensorflow::DT_HALF;
      break;
    case 6:
      dtype = tensorflow::DT_COMPLEX64;
      break;
    case 7:
      dtype = tensorflow::DT_COMPLEX128;
      break;
    case 8:
      dtype = tensorflow::DT_UINT8;
      break;
    case 9:
      dtype = tensorflow::DT_INT8;
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

        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        uint8_t indices_rank = parseRank(data[offset++]);
        if (indices_rank < 2) indices_rank = 2;
        
        uint8_t values_rank = parseRank(data[offset++]);
        if (values_rank != 1) values_rank = 1;
        
        uint8_t shape_rank = parseRank(data[offset++]);
        if (shape_rank != 1) shape_rank = 1;
        
        uint8_t dense_rank = parseRank(data[offset++]);
        if (dense_rank < 2) dense_rank = 2;
        
        bool adjoint_a = (data[offset++] % 2) == 1;
        bool adjoint_b = (data[offset++] % 2) == 1;
        
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        std::vector<int64_t> values_shape = parseShape(data, offset, size, values_rank);
        std::vector<int64_t> sparse_shape_shape = parseShape(data, offset, size, shape_rank);
        std::vector<int64_t> dense_shape = parseShape(data, offset, size, dense_rank);
        
        if (indices_shape.size() < 2) {
            indices_shape = {2, 2};
        }
        if (values_shape.size() < 1) {
            values_shape = {2};
        }
        if (sparse_shape_shape.size() < 1) {
            sparse_shape_shape = {2};
        }
        if (dense_shape.size() < 2) {
            dense_shape = {2, 2};
        }
        
        indices_shape[1] = sparse_shape_shape[0];
        values_shape[0] = indices_shape[0];
        
        if (!adjoint_a) {
            dense_shape[0] = sparse_shape_shape[0];
        } else {
            dense_shape[1] = sparse_shape_shape[0];
        }
        
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(indices_shape));
        tensorflow::Tensor values_tensor(dtype, tensorflow::TensorShape(values_shape));
        tensorflow::Tensor sparse_shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(sparse_shape_shape));
        tensorflow::Tensor dense_tensor(dtype, tensorflow::TensorShape(dense_shape));
        
        fillTensorWithData<int64_t>(indices_tensor, data, offset, size);
        fillTensorWithDataByType(values_tensor, dtype, data, offset, size);
        fillTensorWithData<int64_t>(sparse_shape_tensor, data, offset, size);
        fillTensorWithDataByType(dense_tensor, dtype, data, offset, size);
        
        auto indices_flat = indices_tensor.flat<int64_t>();
        for (int i = 0; i < indices_flat.size(); ++i) {
            indices_flat(i) = std::abs(indices_flat(i)) % 10;
        }
        
        auto sparse_shape_flat = sparse_shape_tensor.flat<int64_t>();
        for (int i = 0; i < sparse_shape_flat.size(); ++i) {
            sparse_shape_flat(i) = std::max(static_cast<int64_t>(1), std::abs(sparse_shape_flat(i)) % 20);
        }
        
        std::cout << "Indices shape: ";
        for (auto dim : indices_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "Values shape: ";
        for (auto dim : values_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "Sparse shape shape: ";
        for (auto dim : sparse_shape_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "Dense shape: ";
        for (auto dim : dense_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "Adjoint A: " << adjoint_a << ", Adjoint B: " << adjoint_b << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto indices_op = tensorflow::ops::Const(root, indices_tensor);
        auto values_op = tensorflow::ops::Const(root, values_tensor);
        auto sparse_shape_op = tensorflow::ops::Const(root, sparse_shape_tensor);
        auto dense_op = tensorflow::ops::Const(root, dense_tensor);
        
        auto sparse_dense_matmul = tensorflow::ops::SparseTensorDenseMatMul(
            root, indices_op, values_op, sparse_shape_op, dense_op,
            tensorflow::ops::SparseTensorDenseMatMul::AdjointA(adjoint_a)
                .AdjointB(adjoint_b));
        
        tensorflow::GraphDef graph;
        tensorflow::Status status = root.ToGraphDef(&graph);
        if (!status.ok()) {
            std::cout << "Graph creation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        status = session->Create(graph);
        if (!status.ok()) {
            std::cout << "Session creation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {sparse_dense_matmul.product.name()}, {}, &outputs);
        if (!status.ok()) {
            std::cout << "Session run failed: " << status.ToString() << std::endl;
        } else {
            std::cout << "Operation completed successfully" << std::endl;
            if (!outputs.empty()) {
                std::cout << "Output shape: " << outputs[0].shape().DebugString() << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}