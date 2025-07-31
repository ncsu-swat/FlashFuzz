#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>
#include <cmath>

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
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_boxes_byte = data[offset++];
        int num_boxes = (num_boxes_byte % 10) + 1;
        
        tensorflow::TensorShape overlaps_shape({num_boxes, num_boxes});
        tensorflow::Tensor overlaps_tensor(tensorflow::DT_FLOAT, overlaps_shape);
        fillTensorWithDataByType(overlaps_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        tensorflow::TensorShape scores_shape({num_boxes});
        tensorflow::Tensor scores_tensor(tensorflow::DT_FLOAT, scores_shape);
        fillTensorWithDataByType(scores_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        tensorflow::TensorShape scalar_shape({});
        
        tensorflow::Tensor max_output_size_tensor(tensorflow::DT_INT32, scalar_shape);
        int32_t max_output_size_val = (offset < size) ? static_cast<int32_t>(data[offset++] % num_boxes + 1) : 1;
        max_output_size_tensor.scalar<int32_t>()() = max_output_size_val;
        
        tensorflow::Tensor overlap_threshold_tensor(tensorflow::DT_FLOAT, scalar_shape);
        float overlap_threshold_val = 0.5f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&overlap_threshold_val, data + offset, sizeof(float));
            offset += sizeof(float);
            overlap_threshold_val = std::abs(overlap_threshold_val);
            if (overlap_threshold_val > 1.0f) overlap_threshold_val = 1.0f;
        }
        overlap_threshold_tensor.scalar<float>()() = overlap_threshold_val;
        
        tensorflow::Tensor score_threshold_tensor(tensorflow::DT_FLOAT, scalar_shape);
        float score_threshold_val = 0.0f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&score_threshold_val, data + offset, sizeof(float));
            offset += sizeof(float);
        }
        score_threshold_tensor.scalar<float>()() = score_threshold_val;

        auto overlaps_input = tensorflow::ops::Const(root, overlaps_tensor);
        auto scores_input = tensorflow::ops::Const(root, scores_tensor);
        auto max_output_size_input = tensorflow::ops::Const(root, max_output_size_tensor);
        auto overlap_threshold_input = tensorflow::ops::Const(root, overlap_threshold_tensor);
        auto score_threshold_input = tensorflow::ops::Const(root, score_threshold_tensor);

        auto nms_op = tensorflow::ops::NonMaxSuppressionWithOverlaps(
            root,
            overlaps_input,
            scores_input,
            max_output_size_input,
            overlap_threshold_input,
            score_threshold_input
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({nms_op}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}