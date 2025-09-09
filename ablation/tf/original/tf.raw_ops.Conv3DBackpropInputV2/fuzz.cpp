#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

constexpr uint8_t MIN_RANK = 1;
constexpr uint8_t MAX_RANK = 5;
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
    case tensorflow::DT_INT64:
      fillTensorWithData<int64_t>(tensor, data, offset, total_size);
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

tensorflow::DataType parseFilterDataType(uint8_t selector) {
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

tensorflow::DataType parseInputSizesDataType(uint8_t selector) {
  tensorflow::DataType dtype; 
  switch (selector % 2) {  
    case 0:
      dtype = tensorflow::DT_INT32;
      break;
    case 1:
      dtype = tensorflow::DT_INT64;
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 20) {
            return 0;
        }

        tensorflow::DataType input_sizes_dtype = parseInputSizesDataType(data[offset++]);
        tensorflow::DataType filter_dtype = parseFilterDataType(data[offset++]);
        
        std::string padding = (data[offset++] % 2 == 0) ? "SAME" : "VALID";
        std::string data_format = (data[offset++] % 2 == 0) ? "NDHWC" : "NCDHW";
        
        std::vector<int64_t> input_sizes_shape = {5};
        tensorflow::TensorShape input_sizes_tensor_shape(input_sizes_shape);
        tensorflow::Tensor input_sizes_tensor(input_sizes_dtype, input_sizes_tensor_shape);
        
        if (input_sizes_dtype == tensorflow::DT_INT32) {
            auto flat = input_sizes_tensor.flat<int32_t>();
            flat(0) = 2; flat(1) = 4; flat(2) = 4; flat(3) = 4; flat(4) = 3;
        } else {
            auto flat = input_sizes_tensor.flat<int64_t>();
            flat(0) = 2; flat(1) = 4; flat(2) = 4; flat(3) = 4; flat(4) = 3;
        }
        
        std::vector<int64_t> filter_shape = {3, 3, 3, 3, 2};
        tensorflow::TensorShape filter_tensor_shape(filter_shape);
        tensorflow::Tensor filter_tensor(filter_dtype, filter_tensor_shape);
        fillTensorWithDataByType(filter_tensor, filter_dtype, data, offset, size);
        
        std::vector<int64_t> out_backprop_shape = {2, 2, 2, 2, 2};
        tensorflow::TensorShape out_backprop_tensor_shape(out_backprop_shape);
        tensorflow::Tensor out_backprop_tensor(filter_dtype, out_backprop_tensor_shape);
        fillTensorWithDataByType(out_backprop_tensor, filter_dtype, data, offset, size);
        
        std::vector<int> strides = {1, 1, 1, 1, 1};
        std::vector<int> dilations = {1, 1, 1, 1, 1};
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto input_sizes_placeholder = tensorflow::ops::Placeholder(root, input_sizes_dtype);
        auto filter_placeholder = tensorflow::ops::Placeholder(root, filter_dtype);
        auto out_backprop_placeholder = tensorflow::ops::Placeholder(root, filter_dtype);
        
        auto conv3d_backprop_input = tensorflow::ops::Conv3DBackpropInputV2(
            root, input_sizes_placeholder, filter_placeholder, out_backprop_placeholder,
            strides, padding,
            tensorflow::ops::Conv3DBackpropInputV2::DataFormat(data_format)
                .Dilations(dilations));
        
        tensorflow::GraphDef graph;
        tensorflow::Status status = root.ToGraphDef(&graph);
        if (!status.ok()) {
            std::cout << "Failed to create graph: " << status.ToString() << std::endl;
            return 0;
        }
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        status = session->Create(graph);
        if (!status.ok()) {
            std::cout << "Failed to create session: " << status.ToString() << std::endl;
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {input_sizes_placeholder.node()->name(), input_sizes_tensor},
            {filter_placeholder.node()->name(), filter_tensor},
            {out_backprop_placeholder.node()->name(), out_backprop_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {conv3d_backprop_input.node()->name()}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "Conv3DBackpropInputV2 executed successfully" << std::endl;
            std::cout << "Output shape: " << outputs[0].shape().DebugString() << std::endl;
        } else {
            std::cout << "Conv3DBackpropInputV2 failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}