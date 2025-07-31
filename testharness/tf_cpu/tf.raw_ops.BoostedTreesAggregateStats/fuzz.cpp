#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include <cstring>
#include <iostream>
#include <vector>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
    void logError(const std::string& message, const uint8_t* data, size_t size) {
        std::cerr << message << std::endl;
    }
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

uint8_t parseRank(uint8_t byte) {
    constexpr uint8_t range = MAX_RANK - MIN_RANK + 1;
    uint8_t rank = byte % range + MIN_RANK;
    return rank;
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
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_COMPLEX64:
      fillTensorWithData<tensorflow::complex64>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_COMPLEX128:
      fillTensorWithData<tensorflow::complex128>(tensor, data, offset, total_size);
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
        uint8_t node_ids_rank = parseRank(data[offset++]);
        if (node_ids_rank == 0) node_ids_rank = 1;
        std::vector<int64_t> node_ids_shape = parseShape(data, offset, size, node_ids_rank);
        
        uint8_t gradients_rank = parseRank(data[offset++]);
        if (gradients_rank == 0) gradients_rank = 2;
        std::vector<int64_t> gradients_shape = parseShape(data, offset, size, gradients_rank);
        
        uint8_t hessians_rank = parseRank(data[offset++]);
        if (hessians_rank == 0) hessians_rank = 2;
        std::vector<int64_t> hessians_shape = parseShape(data, offset, size, hessians_rank);
        
        uint8_t feature_rank = parseRank(data[offset++]);
        if (feature_rank == 0) feature_rank = 2;
        std::vector<int64_t> feature_shape = parseShape(data, offset, size, feature_rank);

        if (offset + 8 > size) return 0;
        
        int max_splits, num_buckets;
        std::memcpy(&max_splits, data + offset, sizeof(int));
        offset += sizeof(int);
        std::memcpy(&num_buckets, data + offset, sizeof(int));
        offset += sizeof(int);
        
        max_splits = std::abs(max_splits) % 100 + 1;
        num_buckets = std::abs(num_buckets) % 100 + 1;

        tensorflow::Tensor node_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(node_ids_shape));
        fillTensorWithDataByType(node_ids_tensor, tensorflow::DT_INT32, data, offset, size);

        tensorflow::Tensor gradients_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(gradients_shape));
        fillTensorWithDataByType(gradients_tensor, tensorflow::DT_FLOAT, data, offset, size);

        tensorflow::Tensor hessians_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(hessians_shape));
        fillTensorWithDataByType(hessians_tensor, tensorflow::DT_FLOAT, data, offset, size);

        tensorflow::Tensor feature_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(feature_shape));
        fillTensorWithDataByType(feature_tensor, tensorflow::DT_INT32, data, offset, size);

        auto node_ids_input = tensorflow::ops::Const(root, node_ids_tensor);
        auto gradients_input = tensorflow::ops::Const(root, gradients_tensor);
        auto hessians_input = tensorflow::ops::Const(root, hessians_tensor);
        auto feature_input = tensorflow::ops::Const(root, feature_tensor);

        // Use raw_ops directly instead of the missing header
        auto boosted_trees_aggregate_stats = tensorflow::ops::internal::BoostedTreesAggregateStats(
            root.WithOpName("BoostedTreesAggregateStats"),
            node_ids_input,
            gradients_input,
            hessians_input,
            feature_input,
            max_splits,
            num_buckets
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({boosted_trees_aggregate_stats.output_ref}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}