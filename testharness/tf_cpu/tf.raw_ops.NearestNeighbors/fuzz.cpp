#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 1
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
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t points_rank = parseRank(data[offset++]);
        std::vector<int64_t> points_shape = parseShape(data, offset, size, points_rank);
        
        uint8_t centers_rank = parseRank(data[offset++]);
        std::vector<int64_t> centers_shape = parseShape(data, offset, size, centers_rank);
        
        if (points_shape.size() != 2 || centers_shape.size() != 2) {
            return 0;
        }
        
        if (points_shape[1] != centers_shape[1]) {
            centers_shape[1] = points_shape[1];
        }
        
        tensorflow::TensorShape points_tensor_shape(points_shape);
        tensorflow::TensorShape centers_tensor_shape(centers_shape);
        
        tensorflow::Tensor points_tensor(tensorflow::DT_FLOAT, points_tensor_shape);
        tensorflow::Tensor centers_tensor(tensorflow::DT_FLOAT, centers_tensor_shape);
        
        fillTensorWithDataByType(points_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(centers_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        int64_t k_value = 1;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&k_value, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            k_value = std::abs(k_value) % std::min(centers_shape[0], static_cast<int64_t>(10)) + 1;
        }
        
        tensorflow::Tensor k_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        k_tensor.scalar<int64_t>()() = k_value;
        
        auto points_input = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto centers_input = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto k_input = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
        
        // Use raw_ops.NearestNeighbors through the NodeBuilder API
        tensorflow::NodeBuilder nb = tensorflow::NodeBuilder("NearestNeighbors", "NearestNeighbors")
            .Input(points_input.node())
            .Input(centers_input.node())
            .Input(k_input.node());
        
        tensorflow::Node* nearest_neighbors_node;
        tensorflow::Status status = nb.Finalize(root.graph(), &nearest_neighbors_node);
        
        if (!status.ok()) {
            return -1;
        }
        
        tensorflow::Output nearest_center_indices(nearest_neighbors_node, 0);
        tensorflow::Output nearest_center_distances(nearest_neighbors_node, 1);
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({{points_input, points_tensor}, 
                              {centers_input, centers_tensor}, 
                              {k_input, k_tensor}}, 
                            {nearest_center_indices, nearest_center_distances}, 
                            &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}