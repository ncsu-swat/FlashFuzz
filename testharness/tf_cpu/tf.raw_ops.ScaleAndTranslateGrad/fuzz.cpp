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
#define MIN_RANK 2
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    return tensorflow::DT_FLOAT;
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
    default:
      fillTensorWithData<float>(tensor, data, offset, total_size);
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t grads_rank = parseRank(data[offset++]);
        std::vector<int64_t> grads_shape = parseShape(data, offset, size, grads_rank);
        tensorflow::DataType grads_dtype = parseDataType(data[offset++]);
        
        tensorflow::Tensor grads_tensor(grads_dtype, tensorflow::TensorShape(grads_shape));
        fillTensorWithDataByType(grads_tensor, grads_dtype, data, offset, size);
        auto grads = tensorflow::ops::Const(root, grads_tensor);

        uint8_t original_image_rank = parseRank(data[offset++]);
        std::vector<int64_t> original_image_shape = parseShape(data, offset, size, original_image_rank);
        tensorflow::DataType original_image_dtype = parseDataType(data[offset++]);
        
        tensorflow::Tensor original_image_tensor(original_image_dtype, tensorflow::TensorShape(original_image_shape));
        fillTensorWithDataByType(original_image_tensor, original_image_dtype, data, offset, size);
        auto original_image = tensorflow::ops::Const(root, original_image_tensor);

        std::vector<int64_t> scale_shape = {2};
        tensorflow::Tensor scale_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(scale_shape));
        fillTensorWithData<float>(scale_tensor, data, offset, size);
        auto scale = tensorflow::ops::Const(root, scale_tensor);

        std::vector<int64_t> translation_shape = {2};
        tensorflow::Tensor translation_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(translation_shape));
        fillTensorWithData<float>(translation_tensor, data, offset, size);
        auto translation = tensorflow::ops::Const(root, translation_tensor);

        std::string kernel_type = "lanczos3";
        if (offset < size) {
            uint8_t kernel_selector = data[offset++];
            switch (kernel_selector % 4) {
                case 0: kernel_type = "lanczos3"; break;
                case 1: kernel_type = "lanczos5"; break;
                case 2: kernel_type = "gaussian"; break;
                case 3: kernel_type = "box"; break;
            }
        }

        bool antialias = true;
        if (offset < size) {
            antialias = (data[offset++] % 2) == 0;
        }

        // Use raw_ops namespace for ScaleAndTranslateGrad
        auto scale_and_translate_grad = tensorflow::ops::internal::ScaleAndTranslateGrad(
            root, grads, original_image, scale, translation,
            tensorflow::ops::internal::ScaleAndTranslateGrad::Attrs()
                .KernelType(kernel_type)
                .Antialias(antialias));

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({scale_and_translate_grad}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}