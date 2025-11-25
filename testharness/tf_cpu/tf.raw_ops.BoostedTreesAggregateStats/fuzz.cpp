#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include <cstring>
#include <iostream>
#include <vector>

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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 12) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        // Derive simple, compatible shapes for the op requirements.
        const int batch_size = (data[offset++] % 8) + 1;
        const int logits_dim = (data[offset++] % 6) + 1;
        const int hess_dim = (data[offset++] % 6) + 1;
        const int feature_dim = (data[offset++] % 6) + 1;

        if (offset + 8 > size) return 0;
        
        int max_splits, num_buckets;
        std::memcpy(&max_splits, data + offset, sizeof(int));
        offset += sizeof(int);
        std::memcpy(&num_buckets, data + offset, sizeof(int));
        offset += sizeof(int);
        
        max_splits = std::abs(max_splits) % 100 + 1;
        num_buckets = std::abs(num_buckets) % 100 + 1;

        tensorflow::Tensor node_ids_tensor(tensorflow::DT_INT32,
                                           tensorflow::TensorShape({batch_size}));
        fillTensorWithDataByType(node_ids_tensor, tensorflow::DT_INT32, data, offset, size);

        tensorflow::Tensor gradients_tensor(tensorflow::DT_FLOAT,
                                            tensorflow::TensorShape({batch_size, logits_dim}));
        fillTensorWithDataByType(gradients_tensor, tensorflow::DT_FLOAT, data, offset, size);

        tensorflow::Tensor hessians_tensor(tensorflow::DT_FLOAT,
                                           tensorflow::TensorShape({batch_size, hess_dim}));
        fillTensorWithDataByType(hessians_tensor, tensorflow::DT_FLOAT, data, offset, size);

        tensorflow::Tensor feature_tensor(tensorflow::DT_INT32,
                                          tensorflow::TensorShape({batch_size, feature_dim}));
        fillTensorWithDataByType(feature_tensor, tensorflow::DT_INT32, data, offset, size);

        auto node_ids_input = tensorflow::ops::Const(root, node_ids_tensor);
        auto gradients_input = tensorflow::ops::Const(root, gradients_tensor);
        auto hessians_input = tensorflow::ops::Const(root, hessians_tensor);
        auto feature_input = tensorflow::ops::Const(root, feature_tensor);

        // Build the op via NodeBuilder since there is no generated C++ wrapper.
        tensorflow::Node* op_node = nullptr;
        auto builder = tensorflow::NodeBuilder(
                           root.GetUniqueNameForOp("BoostedTreesAggregateStats"),
                           "BoostedTreesAggregateStats")
                           .Input(tensorflow::ops::AsNodeOut(root, node_ids_input))
                           .Input(tensorflow::ops::AsNodeOut(root, gradients_input))
                           .Input(tensorflow::ops::AsNodeOut(root, hessians_input))
                           .Input(tensorflow::ops::AsNodeOut(root, feature_input))
                           .Attr("max_splits", max_splits)
                           .Attr("num_buckets", num_buckets);
        root.UpdateStatus(builder.Finalize(root.graph(), &op_node));
        if (!root.ok() || op_node == nullptr) {
            return -1;
        }

        tensorflow::Output stats_summary(op_node, 0);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({stats_summary}, &outputs);
        if (!status.ok()) {
            return -1;
        }
    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
