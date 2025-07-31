#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include <cstring>
#include <vector>
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
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t inputs_rank = parseRank(data[offset++]);
        if (inputs_rank != 3) inputs_rank = 3;
        std::vector<int64_t> inputs_shape = parseShape(data, offset, size, inputs_rank);
        
        tensorflow::Tensor inputs_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(inputs_shape));
        fillTensorWithDataByType(inputs_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto inputs = tensorflow::ops::Const(root, inputs_tensor);

        uint8_t labels_indices_rank = parseRank(data[offset++]);
        if (labels_indices_rank != 2) labels_indices_rank = 2;
        std::vector<int64_t> labels_indices_shape = parseShape(data, offset, size, labels_indices_rank);
        
        tensorflow::Tensor labels_indices_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(labels_indices_shape));
        fillTensorWithDataByType(labels_indices_tensor, tensorflow::DT_INT64, data, offset, size);
        auto labels_indices = tensorflow::ops::Const(root, labels_indices_tensor);

        uint8_t labels_values_rank = parseRank(data[offset++]);
        if (labels_values_rank != 1) labels_values_rank = 1;
        std::vector<int64_t> labels_values_shape = parseShape(data, offset, size, labels_values_rank);
        
        tensorflow::Tensor labels_values_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(labels_values_shape));
        fillTensorWithDataByType(labels_values_tensor, tensorflow::DT_INT32, data, offset, size);
        auto labels_values = tensorflow::ops::Const(root, labels_values_tensor);

        uint8_t sequence_length_rank = parseRank(data[offset++]);
        if (sequence_length_rank != 1) sequence_length_rank = 1;
        std::vector<int64_t> sequence_length_shape = parseShape(data, offset, size, sequence_length_rank);
        
        tensorflow::Tensor sequence_length_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(sequence_length_shape));
        fillTensorWithDataByType(sequence_length_tensor, tensorflow::DT_INT32, data, offset, size);
        auto sequence_length = tensorflow::ops::Const(root, sequence_length_tensor);

        bool preprocess_collapse_repeated = (offset < size) ? (data[offset++] % 2 == 1) : false;
        bool ctc_merge_repeated = (offset < size) ? (data[offset++] % 2 == 1) : true;
        bool ignore_longer_outputs_than_inputs = (offset < size) ? (data[offset++] % 2 == 1) : false;

        // Use raw_ops directly since ctc_ops.h is not available
        tensorflow::Output loss, gradient;
        tensorflow::ops::Scope scope = root.WithOpName("CTCLossV2");
        
        tensorflow::NodeBuilder builder = tensorflow::NodeBuilder("CTCLossV2", "CTCLossV2")
            .Input(inputs.node())
            .Input(labels_indices.node())
            .Input(labels_values.node())
            .Input(sequence_length.node())
            .Attr("preprocess_collapse_repeated", preprocess_collapse_repeated)
            .Attr("ctc_merge_repeated", ctc_merge_repeated)
            .Attr("ignore_longer_outputs_than_inputs", ignore_longer_outputs_than_inputs);
        
        tensorflow::Node* node;
        TF_CHECK_OK(builder.Finalize(scope.graph(), &node));
        
        loss = tensorflow::Output(node, 0);
        gradient = tensorflow::Output(node, 1);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({loss, gradient}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}