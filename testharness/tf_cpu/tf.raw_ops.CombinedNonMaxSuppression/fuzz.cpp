#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/core/framework/types.h"
#include <cstring>
#include <iostream>
#include <vector>
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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t boxes_rank = parseRank(data[offset++]);
        if (boxes_rank != 4) boxes_rank = 4;
        std::vector<int64_t> boxes_shape = parseShape(data, offset, size, boxes_rank);
        if (boxes_shape.size() != 4) {
            boxes_shape = {1, 2, 1, 4};
        }
        
        uint8_t scores_rank = parseRank(data[offset++]);
        if (scores_rank != 3) scores_rank = 3;
        std::vector<int64_t> scores_shape = parseShape(data, offset, size, scores_rank);
        if (scores_shape.size() != 3) {
            scores_shape = {1, 2, 1};
        }
        
        if (boxes_shape[0] != scores_shape[0] || boxes_shape[1] != scores_shape[1]) {
            boxes_shape[0] = scores_shape[0] = 1;
            boxes_shape[1] = scores_shape[1] = 2;
        }
        
        tensorflow::Tensor boxes_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(boxes_shape));
        fillTensorWithDataByType(boxes_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        tensorflow::Tensor scores_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(scores_shape));
        fillTensorWithDataByType(scores_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        int32_t max_output_size_per_class_val = 10;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&max_output_size_per_class_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            max_output_size_per_class_val = std::abs(max_output_size_per_class_val) % 100 + 1;
        }
        tensorflow::Tensor max_output_size_per_class_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        max_output_size_per_class_tensor.scalar<int32_t>()() = max_output_size_per_class_val;
        
        int32_t max_total_size_val = 20;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&max_total_size_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            max_total_size_val = std::abs(max_total_size_val) % 200 + 1;
        }
        tensorflow::Tensor max_total_size_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        max_total_size_tensor.scalar<int32_t>()() = max_total_size_val;
        
        float iou_threshold_val = 0.5f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&iou_threshold_val, data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isnan(iou_threshold_val) || std::isinf(iou_threshold_val)) {
                iou_threshold_val = 0.5f;
            }
            iou_threshold_val = std::max(0.0f, std::min(1.0f, std::abs(iou_threshold_val)));
        }
        tensorflow::Tensor iou_threshold_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        iou_threshold_tensor.scalar<float>()() = iou_threshold_val;
        
        float score_threshold_val = 0.1f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&score_threshold_val, data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isnan(score_threshold_val) || std::isinf(score_threshold_val)) {
                score_threshold_val = 0.1f;
            }
            score_threshold_val = std::max(0.0f, std::min(1.0f, std::abs(score_threshold_val)));
        }
        tensorflow::Tensor score_threshold_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        score_threshold_tensor.scalar<float>()() = score_threshold_val;
        
        bool pad_per_class = false;
        bool clip_boxes = true;
        if (offset < size) {
            pad_per_class = (data[offset] % 2) == 1;
            offset++;
        }
        if (offset < size) {
            clip_boxes = (data[offset] % 2) == 1;
            offset++;
        }
        
        auto boxes_input = tensorflow::ops::Const(root, boxes_tensor);
        auto scores_input = tensorflow::ops::Const(root, scores_tensor);
        auto max_output_size_per_class_input = tensorflow::ops::Const(root, max_output_size_per_class_tensor);
        auto max_total_size_input = tensorflow::ops::Const(root, max_total_size_tensor);
        auto iou_threshold_input = tensorflow::ops::Const(root, iou_threshold_tensor);
        auto score_threshold_input = tensorflow::ops::Const(root, score_threshold_tensor);
        
        auto combined_nms = tensorflow::ops::CombinedNonMaxSuppression(
            root,
            boxes_input,
            scores_input,
            max_output_size_per_class_input,
            max_total_size_input,
            iou_threshold_input,
            score_threshold_input,
            tensorflow::ops::CombinedNonMaxSuppression::Attrs()
                .PadPerClass(pad_per_class)
                .ClipBoxes(clip_boxes)
        );
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({combined_nms.nmsed_boxes, combined_nms.nmsed_scores, 
                                                combined_nms.nmsed_classes, combined_nms.valid_detections}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
