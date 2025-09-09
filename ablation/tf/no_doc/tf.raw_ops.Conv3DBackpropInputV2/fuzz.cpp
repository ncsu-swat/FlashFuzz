#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

constexpr uint8_t MIN_RANK = 1;
constexpr uint8_t MAX_RANK = 5;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 3) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 2:
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
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    default:
      return;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 10) return 0;
        
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        std::vector<int64_t> input_shape = {1, 4, 4, 4, 2};
        std::vector<int64_t> filter_shape = {2, 2, 2, 2, 3};
        std::vector<int64_t> out_backprop_shape = {1, 3, 3, 3, 3};
        
        tensorflow::TensorShape input_tensor_shape(input_shape);
        tensorflow::TensorShape filter_tensor_shape(filter_shape);
        tensorflow::TensorShape out_backprop_tensor_shape(out_backprop_shape);
        
        tensorflow::Tensor input_sizes_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({5}));
        auto input_sizes_flat = input_sizes_tensor.flat<int32_t>();
        for (int i = 0; i < 5; ++i) {
            input_sizes_flat(i) = static_cast<int32_t>(input_shape[i]);
        }
        
        tensorflow::Tensor filter_tensor(dtype, filter_tensor_shape);
        tensorflow::Tensor out_backprop_tensor(dtype, out_backprop_tensor_shape);
        
        fillTensorWithDataByType(filter_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(out_backprop_tensor, dtype, data, offset, size);
        
        std::cout << "Input sizes shape: " << input_sizes_tensor.shape().DebugString() << std::endl;
        std::cout << "Filter shape: " << filter_tensor.shape().DebugString() << std::endl;
        std::cout << "Out backprop shape: " << out_backprop_tensor.shape().DebugString() << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto input_sizes_op = tensorflow::ops::Const(root, input_sizes_tensor);
        auto filter_op = tensorflow::ops::Const(root, filter_tensor);
        auto out_backprop_op = tensorflow::ops::Const(root, out_backprop_tensor);
        
        std::vector<int> strides = {1, 1, 1, 1, 1};
        std::string padding = "VALID";
        std::vector<int> dilations = {1, 1, 1, 1, 1};
        
        auto conv3d_backprop_input = tensorflow::ops::Conv3DBackpropInputV2(
            root, input_sizes_op, filter_op, out_backprop_op, strides, padding,
            tensorflow::ops::Conv3DBackpropInputV2::Attrs().Dilations(dilations));
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({conv3d_backprop_input}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "Output shape: " << outputs[0].shape().DebugString() << std::endl;
        } else {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}