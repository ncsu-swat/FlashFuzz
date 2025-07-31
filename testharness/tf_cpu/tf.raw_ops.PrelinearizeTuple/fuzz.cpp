#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <cstring>
#include <vector>
#include <iostream>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10
#define MAX_NUM_INPUTS 5

tensorflow::DataType parseDataType(uint8_t selector) {
  tensorflow::DataType dtype; 
  switch (selector % 15) {  
    case 0:
      dtype = tensorflow::DT_FLOAT;
      break;
    case 1:
      dtype = tensorflow::DT_DOUBLE;
      break;
    case 2:
      dtype = tensorflow::DT_INT32;
      break;
    case 3:
      dtype = tensorflow::DT_UINT8;
      break;
    case 4:
      dtype = tensorflow::DT_INT16;
      break;
    case 5:
      dtype = tensorflow::DT_INT8;
      break;
    case 6:
      dtype = tensorflow::DT_INT64;
      break;
    case 7:
      dtype = tensorflow::DT_BOOL;
      break;
    case 8:
      dtype = tensorflow::DT_BFLOAT16;
      break;
    case 9:
      dtype = tensorflow::DT_UINT16;
      break;
    case 10:
      dtype = tensorflow::DT_HALF;
      break;
    case 11:
      dtype = tensorflow::DT_UINT32;
      break;
    case 12:
      dtype = tensorflow::DT_UINT64;
      break;
    case 13:
      dtype = tensorflow::DT_COMPLEX64;
      break;
    case 14:
      dtype = tensorflow::DT_COMPLEX128;
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
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_COMPLEX64:
      fillTensorWithData<tensorflow::complex64>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_COMPLEX128:
      fillTensorWithData<tensorflow::complex128>(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_inputs_byte = data[offset++];
        uint8_t num_inputs = (num_inputs_byte % MAX_NUM_INPUTS) + 1;
        
        std::vector<tensorflow::Output> inputs;
        std::vector<tensorflow::TensorShape> shapes;
        
        for (uint8_t i = 0; i < num_inputs; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            if (offset >= size) break;
            
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape_dims = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape_dims) {
                tensor_shape.AddDim(dim);
            }
            shapes.push_back(tensor_shape);
            
            tensorflow::Tensor tensor(dtype, tensor_shape);
            fillTensorWithDataByType(tensor, dtype, data, offset, size);
            
            auto const_op = tensorflow::ops::Const(root, tensor);
            inputs.push_back(const_op);
        }
        
        if (inputs.empty()) {
            return 0;
        }
        
        std::vector<int32_t> layouts;
        if (offset < size) {
            uint8_t num_layouts = data[offset++] % 10;
            for (uint8_t i = 0; i < num_layouts && offset < size; ++i) {
                int32_t layout_val = static_cast<int32_t>(data[offset++]) - 128;
                layouts.push_back(layout_val);
            }
        }
        
        // Create a vector of tensorflow::Input from tensorflow::Output
        std::vector<tensorflow::Input> input_tensors;
        for (const auto& output : inputs) {
            input_tensors.push_back(output);
        }
        
        // Convert shapes to a tensor
        std::vector<tensorflow::Tensor> shape_tensors;
        for (const auto& shape : shapes) {
            std::vector<int64_t> dims;
            for (int i = 0; i < shape.dims(); i++) {
                dims.push_back(shape.dim_size(i));
            }
            tensorflow::Tensor shape_tensor(tensorflow::DT_INT64, {static_cast<int64_t>(dims.size())});
            auto flat = shape_tensor.flat<int64_t>();
            for (size_t i = 0; i < dims.size(); i++) {
                flat(i) = dims[i];
            }
            shape_tensors.push_back(shape_tensor);
        }
        
        // Convert layouts to a tensor
        tensorflow::Tensor layouts_tensor(tensorflow::DT_INT32, {static_cast<int64_t>(layouts.size())});
        auto flat_layouts = layouts_tensor.flat<int32_t>();
        for (size_t i = 0; i < layouts.size(); i++) {
            flat_layouts(i) = layouts[i];
        }
        
        // Use raw_ops directly
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        // Create node for PrelinearizeTuple using NodeBuilder
        tensorflow::NodeBuilder node_builder("prelinearize_tuple", "PrelinearizeTuple");
        
        // Add inputs
        for (const auto& input : inputs) {
            node_builder.Input(input.node());
        }
        
        // Add shapes attribute
        std::vector<tensorflow::PartialTensorShape> partial_shapes;
        for (const auto& shape : shapes) {
            partial_shapes.push_back(tensorflow::PartialTensorShape(shape));
        }
        node_builder.Attr("shapes", partial_shapes);
        
        // Add layouts attribute if not empty
        if (!layouts.empty()) {
            node_builder.Attr("layouts", layouts);
        }
        
        // Build the node
        tensorflow::Node* node;
        tensorflow::Status status = node_builder.Finalize(root.graph(), &node);
        
        if (!status.ok()) {
            return 0;
        }
        
        // Create an output
        tensorflow::Output output(node, 0);
        
        // Run the session
        status = session.Run({output}, &outputs);
        
        if (!status.ok()) {
            return 0;
        }

    } catch (const std::exception& e) {
        return 0;
    } 

    return 0;
}