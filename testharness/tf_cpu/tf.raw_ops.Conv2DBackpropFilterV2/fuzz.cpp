#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 4
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
        case 0:
            dtype = tensorflow::DT_HALF;
            break;
        case 1:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 2:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 3:
            dtype = tensorflow::DT_DOUBLE;
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
    case tensorflow::DT_BFLOAT16:
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset,
                                               total_size);
      break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {

        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        std::vector<int64_t> input_shape = {2, 4, 4, 3};
        std::vector<int64_t> filter_shape = {3, 3, 3, 2};
        std::vector<int64_t> out_backprop_shape = {2, 4, 4, 2};
        
        if (offset + 3 < size) {
            input_shape[0] = (data[offset++] % 3) + 1;
            input_shape[1] = (data[offset++] % 5) + 2;
            input_shape[2] = (data[offset++] % 5) + 2;
            input_shape[3] = (data[offset++] % 4) + 1;
            
            filter_shape[0] = (data[offset++] % 3) + 1;
            filter_shape[1] = (data[offset++] % 3) + 1;
            filter_shape[2] = input_shape[3];
            filter_shape[3] = (data[offset++] % 4) + 1;
            
            out_backprop_shape[0] = input_shape[0];
            out_backprop_shape[1] = input_shape[1];
            out_backprop_shape[2] = input_shape[2];
            out_backprop_shape[3] = filter_shape[3];
        }

        tensorflow::Tensor input_tensor(dtype, tensorflow::TensorShape(input_shape));
        tensorflow::Tensor filter_tensor(dtype, tensorflow::TensorShape(filter_shape));
        tensorflow::Tensor out_backprop_tensor(dtype, tensorflow::TensorShape(out_backprop_shape));

        fillTensorWithDataByType(input_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(filter_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(out_backprop_tensor, dtype, data, offset, size);

        auto input_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto filter_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto out_backprop_placeholder = tensorflow::ops::Placeholder(root, dtype);

        std::vector<int> strides = {1, 1, 1, 1};
        if (offset + 4 < size) {
            strides[1] = (data[offset++] % 2) + 1;
            strides[2] = (data[offset++] % 2) + 1;
        }

        std::string padding = "VALID";
        if (offset < size) {
            if (data[offset++] % 2 == 0) {
                padding = "SAME";
            }
        }

        bool use_cudnn_on_gpu = false;
        std::vector<int> explicit_paddings = {};
        std::string data_format = "NHWC";
        std::vector<int> dilations = {1, 1, 1, 1};

        auto conv2d_backprop_filter = tensorflow::ops::Conv2DBackpropFilter(
            root,
            input_placeholder,
            tensorflow::ops::Shape(root, filter_placeholder),
            out_backprop_placeholder,
            strides,
            padding,
            tensorflow::ops::Conv2DBackpropFilter::Attrs()
                .UseCudnnOnGpu(use_cudnn_on_gpu)
                .ExplicitPaddings(explicit_paddings)
                .DataFormat(data_format)
                .Dilations(dilations)
        );

        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(
            {{input_placeholder, input_tensor},
             {filter_placeholder, filter_tensor},
             {out_backprop_placeholder, out_backprop_tensor}},
            {conv2d_backprop_filter},
            &outputs
        );

        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
