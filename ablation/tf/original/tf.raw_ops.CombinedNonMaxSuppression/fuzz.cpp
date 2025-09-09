#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/common_runtime/direct_session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/platform/logging.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/public/version.h>
#include <tensorflow/core/framework/node_def_util.h>
#include <tensorflow/core/graph/node_builder.h>
#include <tensorflow/core/graph/graph.h>
#include <tensorflow/core/common_runtime/executor.h>
#include <tensorflow/core/common_runtime/function.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 6;
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
        
        if (size < 20) {
            return 0;
        }

        uint8_t boxes_rank = parseRank(data[offset++]);
        if (boxes_rank != 4) {
            boxes_rank = 4;
        }
        std::vector<int64_t> boxes_shape = parseShape(data, offset, size, boxes_rank);
        if (boxes_shape.size() != 4) {
            boxes_shape = {1, 2, 1, 4};
        }
        
        uint8_t scores_rank = parseRank(data[offset++]);
        if (scores_rank != 3) {
            scores_rank = 3;
        }
        std::vector<int64_t> scores_shape = parseShape(data, offset, size, scores_rank);
        if (scores_shape.size() != 3) {
            scores_shape = {1, 2, 1};
        }

        tensorflow::Tensor boxes_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(boxes_shape));
        tensorflow::Tensor scores_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(scores_shape));
        
        tensorflow::Tensor max_output_size_per_class_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        tensorflow::Tensor max_total_size_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        tensorflow::Tensor iou_threshold_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor score_threshold_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));

        fillTensorWithDataByType(boxes_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(scores_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(max_output_size_per_class_tensor, tensorflow::DT_INT32, data, offset, size);
        fillTensorWithDataByType(max_total_size_tensor, tensorflow::DT_INT32, data, offset, size);
        fillTensorWithDataByType(iou_threshold_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(score_threshold_tensor, tensorflow::DT_FLOAT, data, offset, size);

        bool pad_per_class = (offset < size) ? (data[offset++] % 2 == 1) : false;
        bool clip_boxes = (offset < size) ? (data[offset++] % 2 == 1) : true;

        std::cout << "Boxes shape: ";
        for (auto dim : boxes_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        std::cout << "Scores shape: ";
        for (auto dim : scores_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        std::cout << "Max output size per class: " << max_output_size_per_class_tensor.scalar<int32_t>()() << std::endl;
        std::cout << "Max total size: " << max_total_size_tensor.scalar<int32_t>()() << std::endl;
        std::cout << "IOU threshold: " << iou_threshold_tensor.scalar<float>()() << std::endl;
        std::cout << "Score threshold: " << score_threshold_tensor.scalar<float>()() << std::endl;
        std::cout << "Pad per class: " << pad_per_class << std::endl;
        std::cout << "Clip boxes: " << clip_boxes << std::endl;

        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        tensorflow::GraphDef graph_def;
        tensorflow::Graph graph(tensorflow::OpRegistry::Global());
        
        tensorflow::Node* boxes_node;
        tensorflow::Node* scores_node;
        tensorflow::Node* max_output_size_per_class_node;
        tensorflow::Node* max_total_size_node;
        tensorflow::Node* iou_threshold_node;
        tensorflow::Node* score_threshold_node;
        
        tensorflow::NodeBuilder("boxes", "Placeholder")
            .Attr("dtype", tensorflow::DT_FLOAT)
            .Attr("shape", tensorflow::TensorShape(boxes_shape))
            .Finalize(&graph, &boxes_node);
            
        tensorflow::NodeBuilder("scores", "Placeholder")
            .Attr("dtype", tensorflow::DT_FLOAT)
            .Attr("shape", tensorflow::TensorShape(scores_shape))
            .Finalize(&graph, &scores_node);
            
        tensorflow::NodeBuilder("max_output_size_per_class", "Placeholder")
            .Attr("dtype", tensorflow::DT_INT32)
            .Attr("shape", tensorflow::TensorShape({}))
            .Finalize(&graph, &max_output_size_per_class_node);
            
        tensorflow::NodeBuilder("max_total_size", "Placeholder")
            .Attr("dtype", tensorflow::DT_INT32)
            .Attr("shape", tensorflow::TensorShape({}))
            .Finalize(&graph, &max_total_size_node);
            
        tensorflow::NodeBuilder("iou_threshold", "Placeholder")
            .Attr("dtype", tensorflow::DT_FLOAT)
            .Attr("shape", tensorflow::TensorShape({}))
            .Finalize(&graph, &iou_threshold_node);
            
        tensorflow::NodeBuilder("score_threshold", "Placeholder")
            .Attr("dtype", tensorflow::DT_FLOAT)
            .Attr("shape", tensorflow::TensorShape({}))
            .Finalize(&graph, &score_threshold_node);

        tensorflow::Node* combined_nms_node;
        tensorflow::NodeBuilder("combined_nms", "CombinedNonMaxSuppression")
            .Input(boxes_node)
            .Input(scores_node)
            .Input(max_output_size_per_class_node)
            .Input(max_total_size_node)
            .Input(iou_threshold_node)
            .Input(score_threshold_node)
            .Attr("pad_per_class", pad_per_class)
            .Attr("clip_boxes", clip_boxes)
            .Finalize(&graph, &combined_nms_node);

        graph.ToGraphDef(&graph_def);
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            std::cout << "Failed to create session: " << status.ToString() << std::endl;
            return 0;
        }

        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"boxes", boxes_tensor},
            {"scores", scores_tensor},
            {"max_output_size_per_class", max_output_size_per_class_tensor},
            {"max_total_size", max_total_size_tensor},
            {"iou_threshold", iou_threshold_tensor},
            {"score_threshold", score_threshold_tensor}
        };

        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {
            "combined_nms:0",
            "combined_nms:1", 
            "combined_nms:2",
            "combined_nms:3"
        };

        status = session->Run(inputs, output_names, {}, &outputs);
        if (!status.ok()) {
            std::cout << "Failed to run session: " << status.ToString() << std::endl;
            return 0;
        }

        std::cout << "CombinedNonMaxSuppression executed successfully" << std::endl;
        std::cout << "Output tensors count: " << outputs.size() << std::endl;
        
        for (size_t i = 0; i < outputs.size(); ++i) {
            std::cout << "Output " << i << " shape: ";
            for (int j = 0; j < outputs[i].dims(); ++j) {
                std::cout << outputs[i].dim_size(j) << " ";
            }
            std::cout << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}