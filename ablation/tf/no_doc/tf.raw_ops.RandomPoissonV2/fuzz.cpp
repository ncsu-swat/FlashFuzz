#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/random_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_ops.h>
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
  switch (selector % 6) {  
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
      dtype = tensorflow::DT_HALF;
      break;
    case 5:
      dtype = tensorflow::DT_BFLOAT16;
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

        uint8_t shape_rank = parseRank(data[offset++]);
        std::vector<int64_t> shape_dims = parseShape(data, offset, size, shape_rank);
        
        uint8_t lam_rank = parseRank(data[offset++]);
        std::vector<int64_t> lam_dims = parseShape(data, offset, size, lam_rank);
        
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        if (offset >= size) {
            return 0;
        }

        tensorflow::TensorShape shape_tensor_shape(shape_dims);
        tensorflow::Tensor shape_tensor(tensorflow::DT_INT32, shape_tensor_shape);
        fillTensorWithData<int32_t>(shape_tensor, data, offset, size);
        
        tensorflow::TensorShape lam_tensor_shape(lam_dims);
        tensorflow::Tensor lam_tensor(dtype, lam_tensor_shape);
        fillTensorWithDataByType(lam_tensor, dtype, data, offset, size);
        
        if (offset + sizeof(int64_t) <= size) {
            int64_t seed_val;
            std::memcpy(&seed_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            tensorflow::Tensor seed_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
            seed_tensor.scalar<int64_t>()() = seed_val;
            
            std::cout << "Shape tensor: " << shape_tensor.DebugString() << std::endl;
            std::cout << "Lam tensor: " << lam_tensor.DebugString() << std::endl;
            std::cout << "Seed tensor: " << seed_tensor.DebugString() << std::endl;
            std::cout << "Output dtype: " << tensorflow::DataTypeString(dtype) << std::endl;
            
            tensorflow::Scope root = tensorflow::Scope::NewRootScope();
            
            auto shape_const = tensorflow::ops::Const(root, shape_tensor);
            auto lam_const = tensorflow::ops::Const(root, lam_tensor);
            auto seed_const = tensorflow::ops::Const(root, seed_tensor);
            
            auto random_poisson = tensorflow::ops::RandomPoissonV2(root, shape_const, lam_const, seed_const, 
                                                                  tensorflow::ops::RandomPoissonV2::Dtype(dtype));
            
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
            status = session->Run({}, {random_poisson.node()->name() + ":0"}, {}, &outputs);
            if (!status.ok()) {
                std::cout << "Session run failed: " << status.ToString() << std::endl;
                return 0;
            }
            
            if (!outputs.empty()) {
                std::cout << "Output tensor: " << outputs[0].DebugString() << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}