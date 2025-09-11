#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include <cstring>
#include <vector>
#include <iostream>

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
        uint8_t batch_size_byte = data[offset++];
        int batch_size = (batch_size_byte % 10) + 1;
        
        uint8_t logits_dim_byte = data[offset++];
        int logits_dimension = (logits_dim_byte % 5) + 1;
        
        uint8_t hessian_dim_byte = data[offset++];
        int hessian_dimension = (hessian_dim_byte % 5) + 1;
        
        uint8_t feature_dim_byte = data[offset++];
        int feature_dimension = (feature_dim_byte % 10) + 1;
        
        uint8_t sparse_entries_byte = data[offset++];
        int num_sparse_entries = (sparse_entries_byte % 20) + 1;
        
        uint8_t max_splits_byte = data[offset++];
        int max_splits = (max_splits_byte % 10) + 1;
        
        uint8_t num_buckets_byte = data[offset++];
        int num_buckets = (num_buckets_byte % 10) + 1;

        tensorflow::TensorShape node_ids_shape({batch_size});
        tensorflow::Tensor node_ids_tensor(tensorflow::DT_INT32, node_ids_shape);
        fillTensorWithDataByType(node_ids_tensor, tensorflow::DT_INT32, data, offset, size);

        tensorflow::TensorShape gradients_shape({batch_size, logits_dimension});
        tensorflow::Tensor gradients_tensor(tensorflow::DT_FLOAT, gradients_shape);
        fillTensorWithDataByType(gradients_tensor, tensorflow::DT_FLOAT, data, offset, size);

        tensorflow::TensorShape hessians_shape({batch_size, hessian_dimension});
        tensorflow::Tensor hessians_tensor(tensorflow::DT_FLOAT, hessians_shape);
        fillTensorWithDataByType(hessians_tensor, tensorflow::DT_FLOAT, data, offset, size);

        tensorflow::TensorShape feature_indices_shape({num_sparse_entries, 2});
        tensorflow::Tensor feature_indices_tensor(tensorflow::DT_INT32, feature_indices_shape);
        fillTensorWithDataByType(feature_indices_tensor, tensorflow::DT_INT32, data, offset, size);

        tensorflow::TensorShape feature_values_shape({num_sparse_entries});
        tensorflow::Tensor feature_values_tensor(tensorflow::DT_INT32, feature_values_shape);
        fillTensorWithDataByType(feature_values_tensor, tensorflow::DT_INT32, data, offset, size);

        tensorflow::TensorShape feature_shape_shape({2});
        tensorflow::Tensor feature_shape_tensor(tensorflow::DT_INT32, feature_shape_shape);
        auto feature_shape_flat = feature_shape_tensor.flat<int32_t>();
        feature_shape_flat(0) = batch_size;
        feature_shape_flat(1) = feature_dimension;

        auto node_ids_input = tensorflow::ops::Const(root, node_ids_tensor);
        auto gradients_input = tensorflow::ops::Const(root, gradients_tensor);
        auto hessians_input = tensorflow::ops::Const(root, hessians_tensor);
        auto feature_indices_input = tensorflow::ops::Const(root, feature_indices_tensor);
        auto feature_values_input = tensorflow::ops::Const(root, feature_values_tensor);
        auto feature_shape_input = tensorflow::ops::Const(root, feature_shape_tensor);

        // Create attributes for the operation
        tensorflow::ops::Scope scope = root.WithOpName("BoostedTreesSparseAggregateStats");
        auto attrs = tensorflow::ops::Scope::WithAttrs(scope);
        attrs = attrs.WithAttr("max_splits", max_splits);
        attrs = attrs.WithAttr("num_buckets", num_buckets);

        // Create the operation using raw API
        auto op = tensorflow::Operation(root.WithOpName("BoostedTreesSparseAggregateStats")
            .WithDevice("/cpu:0")
            .WithAttr("max_splits", max_splits)
            .WithAttr("num_buckets", num_buckets));

        std::vector<tensorflow::Output> outputs;
        tensorflow::Status status = tensorflow::ops::Internal::BoostedTreesSparseAggregateStats(
            attrs,
            node_ids_input,
            gradients_input,
            hessians_input,
            feature_indices_input,
            feature_values_input,
            feature_shape_input,
            &outputs);

        if (!status.ok()) {
            return -1;
        }

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> output_tensors;
        status = session.Run({outputs[0], outputs[1], outputs[2]}, &output_tensors);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
