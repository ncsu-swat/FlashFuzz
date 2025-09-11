#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_handle.h"
#include <cstring>
#include <iostream>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << message << std::endl;
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
        tensorflow::Tensor tree_ensemble_handle_tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape({}));
        
        uint8_t mean_gradients_rank = parseRank(data[offset++]);
        std::vector<int64_t> mean_gradients_shape = parseShape(data, offset, size, mean_gradients_rank);
        tensorflow::Tensor mean_gradients_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(mean_gradients_shape));
        fillTensorWithDataByType(mean_gradients_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        uint8_t mean_hessians_rank = parseRank(data[offset++]);
        std::vector<int64_t> mean_hessians_shape = parseShape(data, offset, size, mean_hessians_rank);
        tensorflow::Tensor mean_hessians_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(mean_hessians_shape));
        fillTensorWithDataByType(mean_hessians_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        tensorflow::Tensor l1_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        fillTensorWithDataByType(l1_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        tensorflow::Tensor l2_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        fillTensorWithDataByType(l2_tensor, tensorflow::DT_FLOAT, data, offset, size);

        auto tree_ensemble_handle = tensorflow::ops::Placeholder(root, tensorflow::DT_RESOURCE);
        auto mean_gradients = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto mean_hessians = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto l1 = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto l2 = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);

        // Use raw_ops directly instead of the missing boosted_trees_ops.h
        auto boosted_trees_center_bias = tensorflow::ops::Operation(
            root.WithOpName("BoostedTreesCenterBias"),
            "BoostedTreesCenterBias",
            {tree_ensemble_handle, mean_gradients, mean_hessians, l1, l2});

        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({
            {tree_ensemble_handle, tree_ensemble_handle_tensor},
            {mean_gradients, mean_gradients_tensor},
            {mean_hessians, mean_hessians_tensor},
            {l1, l1_tensor},
            {l2, l2_tensor}
        }, {boosted_trees_center_bias.output(0)}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
