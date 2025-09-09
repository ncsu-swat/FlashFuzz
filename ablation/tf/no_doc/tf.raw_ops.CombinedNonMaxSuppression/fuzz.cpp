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
#include <tensorflow/core/common_runtime/kernel_benchmark_testlib.h>
#include <tensorflow/core/kernels/ops_testutil.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_HALF;
            break;
    }
    return dtype;
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
      return;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 20) {
            return 0;
        }

        tensorflow::DataType boxes_dtype = parseDataType(data[offset++]);
        tensorflow::DataType scores_dtype = parseDataType(data[offset++]);
        
        uint8_t boxes_rank = parseRank(data[offset++]);
        uint8_t scores_rank = parseRank(data[offset++]);
        
        if (boxes_rank < 3) boxes_rank = 3;
        if (scores_rank < 3) scores_rank = 3;
        
        std::vector<int64_t> boxes_shape = parseShape(data, offset, size, boxes_rank);
        std::vector<int64_t> scores_shape = parseShape(data, offset, size, scores_rank);
        
        if (boxes_shape.size() >= 3) {
            boxes_shape[boxes_shape.size()-1] = 4;
        }
        
        int32_t max_output_size_per_class = 10;
        int32_t max_total_size = 50;
        float iou_threshold = 0.5f;
        float score_threshold = 0.1f;
        
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&max_output_size_per_class, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            max_output_size_per_class = std::abs(max_output_size_per_class) % 100 + 1;
        }
        
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&max_total_size, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            max_total_size = std::abs(max_total_size) % 200 + 1;
        }
        
        if (offset + sizeof(float) <= size) {
            std::memcpy(&iou_threshold, data + offset, sizeof(float));
            offset += sizeof(float);
            if (iou_threshold < 0.0f || iou_threshold > 1.0f || std::isnan(iou_threshold)) {
                iou_threshold = 0.5f;
            }
        }
        
        if (offset + sizeof(float) <= size) {
            std::memcpy(&score_threshold, data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isnan(score_threshold)) {
                score_threshold = 0.1f;
            }
        }

        tensorflow::TensorShape boxes_tensor_shape(boxes_shape);
        tensorflow::TensorShape scores_tensor_shape(scores_shape);
        
        tensorflow::Tensor boxes_tensor(boxes_dtype, boxes_tensor_shape);
        tensorflow::Tensor scores_tensor(scores_dtype, scores_tensor_shape);
        tensorflow::Tensor max_output_size_per_class_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        tensorflow::Tensor max_total_size_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        tensorflow::Tensor iou_threshold_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor score_threshold_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        
        fillTensorWithDataByType(boxes_tensor, boxes_dtype, data, offset, size);
        fillTensorWithDataByType(scores_tensor, scores_dtype, data, offset, size);
        
        max_output_size_per_class_tensor.scalar<int32_t>()() = max_output_size_per_class;
        max_total_size_tensor.scalar<int32_t>()() = max_total_size;
        iou_threshold_tensor.scalar<float>()() = iou_threshold;
        score_threshold_tensor.scalar<float>()() = score_threshold;

        std::cout << "Boxes tensor shape: ";
        for (int i = 0; i < boxes_tensor.dims(); ++i) {
            std::cout << boxes_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Scores tensor shape: ";
        for (int i = 0; i < scores_tensor.dims(); ++i) {
            std::cout << scores_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "max_output_size_per_class: " << max_output_size_per_class << std::endl;
        std::cout << "max_total_size: " << max_total_size << std::endl;
        std::cout << "iou_threshold: " << iou_threshold << std::endl;
        std::cout << "score_threshold: " << score_threshold << std::endl;

        tensorflow::OpKernelContext::Params params;
        tensorflow::DeviceBase device(tensorflow::Env::Default());
        params.device = &device;
        
        tensorflow::NodeDef node_def;
        node_def.set_name("combined_non_max_suppression");
        node_def.set_op("CombinedNonMaxSuppression");
        
        tensorflow::Status status;
        std::unique_ptr<tensorflow::OpKernel> kernel;
        
        tensorflow::OpKernelConstruction construction(
            tensorflow::DeviceType("CPU"), &device, tensorflow::AllocatorAttributes(),
            &node_def, tensorflow::OpDef(), &status);
            
        if (!status.ok()) {
            std::cout << "OpKernel construction failed: " << status.ToString() << std::endl;
            return 0;
        }

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}