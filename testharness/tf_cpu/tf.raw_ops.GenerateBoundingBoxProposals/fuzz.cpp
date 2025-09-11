#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <cstring>
#include <vector>
#include <iostream>

#define MAX_RANK 4
#define MIN_RANK 1
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

REGISTER_OP("GenerateBoundingBoxProposals")
    .Input("scores: float")
    .Input("bbox_deltas: float")
    .Input("image_info: float")
    .Input("anchors: float")
    .Input("nms_threshold: float")
    .Input("pre_nms_topn: int32")
    .Input("min_size: float")
    .Attr("post_nms_topn: int = 300")
    .Output("rois: float")
    .Output("roi_probabilities: float");

namespace tensorflow {
namespace ops {

class GenerateBoundingBoxProposals {
 public:
  GenerateBoundingBoxProposals(const ::tensorflow::Scope& scope, 
                              ::tensorflow::Input scores, 
                              ::tensorflow::Input bbox_deltas,
                              ::tensorflow::Input image_info,
                              ::tensorflow::Input anchors,
                              ::tensorflow::Input nms_threshold,
                              ::tensorflow::Input pre_nms_topn,
                              ::tensorflow::Input min_size) {
    if (!scope.ok()) return;
    auto _scores = ::tensorflow::ops::AsNodeOut(scope, scores);
    if (!scope.ok()) return;
    auto _bbox_deltas = ::tensorflow::ops::AsNodeOut(scope, bbox_deltas);
    if (!scope.ok()) return;
    auto _image_info = ::tensorflow::ops::AsNodeOut(scope, image_info);
    if (!scope.ok()) return;
    auto _anchors = ::tensorflow::ops::AsNodeOut(scope, anchors);
    if (!scope.ok()) return;
    auto _nms_threshold = ::tensorflow::ops::AsNodeOut(scope, nms_threshold);
    if (!scope.ok()) return;
    auto _pre_nms_topn = ::tensorflow::ops::AsNodeOut(scope, pre_nms_topn);
    if (!scope.ok()) return;
    auto _min_size = ::tensorflow::ops::AsNodeOut(scope, min_size);
    if (!scope.ok()) return;
    
    ::tensorflow::Node* ret;
    const auto unique_name = scope.GetUniqueNameForOp("GenerateBoundingBoxProposals");
    auto builder = ::tensorflow::NodeBuilder(unique_name, "GenerateBoundingBoxProposals")
                     .Input(_scores)
                     .Input(_bbox_deltas)
                     .Input(_image_info)
                     .Input(_anchors)
                     .Input(_nms_threshold)
                     .Input(_pre_nms_topn)
                     .Input(_min_size)
                     .Attr("post_nms_topn", attrs_.post_nms_topn_);
    scope.UpdateBuilder(&builder);
    scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
    if (!scope.ok()) return;
    scope.UpdateStatus(scope.DoShapeInference(ret));
    this->operation = ::tensorflow::Operation(ret);
    this->rois = this->operation.output(0);
    this->roi_probabilities = this->operation.output(1);
  }
  
  class Attrs {
   public:
    Attrs() {}
    
    Attrs PostNmsTopn(int64_t x) {
      Attrs ret = *this;
      ret.post_nms_topn_ = x;
      return ret;
    }
    
    int64_t post_nms_topn_ = 300;
  };
  
  Attrs Attrs() {
    return Attrs();
  }
  
  ::tensorflow::Operation operation;
  ::tensorflow::Output rois;
  ::tensorflow::Output roi_probabilities;
 private:
  Attrs attrs_;
};

}  // namespace ops
}  // namespace tensorflow

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t scores_rank = parseRank(data[offset++]);
        if (scores_rank != 4) scores_rank = 4;
        std::vector<int64_t> scores_shape = parseShape(data, offset, size, scores_rank);
        if (scores_shape.size() != 4) {
            scores_shape = {1, 2, 2, 3};
        }
        
        tensorflow::Tensor scores_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(scores_shape));
        fillTensorWithDataByType(scores_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto scores = tensorflow::ops::Const(root, scores_tensor);

        uint8_t bbox_deltas_rank = parseRank(data[offset++]);
        if (bbox_deltas_rank != 4) bbox_deltas_rank = 4;
        std::vector<int64_t> bbox_deltas_shape = parseShape(data, offset, size, bbox_deltas_rank);
        if (bbox_deltas_shape.size() != 4) {
            bbox_deltas_shape = {1, 2, 2, 12};
        } else {
            bbox_deltas_shape[3] = scores_shape[3] * 4;
        }
        
        tensorflow::Tensor bbox_deltas_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(bbox_deltas_shape));
        fillTensorWithDataByType(bbox_deltas_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto bbox_deltas = tensorflow::ops::Const(root, bbox_deltas_tensor);

        std::vector<int64_t> image_info_shape = {scores_shape[0], 5};
        tensorflow::Tensor image_info_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(image_info_shape));
        fillTensorWithDataByType(image_info_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto image_info = tensorflow::ops::Const(root, image_info_tensor);

        std::vector<int64_t> anchors_shape = {scores_shape[3], 4};
        tensorflow::Tensor anchors_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(anchors_shape));
        fillTensorWithDataByType(anchors_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto anchors = tensorflow::ops::Const(root, anchors_tensor);

        tensorflow::Tensor nms_threshold_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        fillTensorWithDataByType(nms_threshold_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto nms_threshold = tensorflow::ops::Const(root, nms_threshold_tensor);

        tensorflow::Tensor pre_nms_topn_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        fillTensorWithDataByType(pre_nms_topn_tensor, tensorflow::DT_INT32, data, offset, size);
        auto pre_nms_topn = tensorflow::ops::Const(root, pre_nms_topn_tensor);

        tensorflow::Tensor min_size_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        fillTensorWithDataByType(min_size_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto min_size = tensorflow::ops::Const(root, min_size_tensor);

        int32_t post_nms_topn_val = 300;
        if (offset < size) {
            std::memcpy(&post_nms_topn_val, data + offset, std::min(sizeof(int32_t), size - offset));
            post_nms_topn_val = std::abs(post_nms_topn_val) % 1000 + 1;
        }

        auto generate_proposals = tensorflow::ops::GenerateBoundingBoxProposals(
            root, scores, bbox_deltas, image_info, anchors, nms_threshold, 
            pre_nms_topn, min_size);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({generate_proposals.rois, generate_proposals.roi_probabilities}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
