#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/framework/fake_input.h>
#include <tensorflow/core/kernels/ops_testutil.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/version.h>
#include <tensorflow/core/framework/resource_mgr.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
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
    case tensorflow::DT_STRING:
      {
        auto flat = tensor.flat<tensorflow::tstring>();
        const size_t num_elements = flat.size();
        for (size_t i = 0; i < num_elements; ++i) {
          if (offset < total_size) {
            uint8_t str_len = data[offset] % 10;
            offset++;
            std::string str;
            for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
              str += static_cast<char>(data[offset]);
              offset++;
            }
            flat(i) = str;
          } else {
            flat(i) = "";
          }
        }
      }
      break;
    default:
      break;
  }
}

tensorflow::DataType parseDataType(uint8_t selector) {
  tensorflow::DataType dtype; 
  switch (selector % 21) {  
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
      dtype = tensorflow::DT_STRING;
      break;
    case 7:
      dtype = tensorflow::DT_COMPLEX64;
      break;
    case 8:
      dtype = tensorflow::DT_INT64;
      break;
    case 9:
      dtype = tensorflow::DT_BOOL;
      break;
    case 10:
      dtype = tensorflow::DT_QINT8;
      break;
    case 11:
      dtype = tensorflow::DT_QUINT8;
      break;
    case 12:
      dtype = tensorflow::DT_QINT32;
      break;
    case 13:
      dtype = tensorflow::DT_BFLOAT16;
      break;
    case 14:
      dtype = tensorflow::DT_QINT16;
      break;
    case 15:
      dtype = tensorflow::DT_QUINT16;
      break;
    case 16:
      dtype = tensorflow::DT_UINT16;
      break;
    case 17:
      dtype = tensorflow::DT_COMPLEX128;
      break;
    case 18:
      dtype = tensorflow::DT_HALF;
      break;
    case 19:
      dtype = tensorflow::DT_UINT32;
      break;
    case 20:
      dtype = tensorflow::DT_UINT64;
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
        
        if (size < 10) {
            return 0;
        }

        uint8_t handle_rank = parseRank(data[offset++]);
        std::vector<int64_t> handle_shape = parseShape(data, offset, size, handle_rank);
        
        tensorflow::TensorShape handle_tensor_shape;
        for (int64_t dim : handle_shape) {
            handle_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor handle_tensor(tensorflow::DT_STRING, handle_tensor_shape);
        fillTensorWithDataByType(handle_tensor, tensorflow::DT_STRING, data, offset, size);
        
        std::cout << "Handle tensor shape: ";
        for (int i = 0; i < handle_tensor_shape.dims(); ++i) {
            std::cout << handle_tensor_shape.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Handle tensor data: ";
        auto handle_flat = handle_tensor.flat<tensorflow::tstring>();
        for (int i = 0; i < std::min(5, static_cast<int>(handle_flat.size())); ++i) {
            std::cout << handle_flat(i) << " ";
        }
        std::cout << std::endl;

        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        tensorflow::Node* handle_node = tensorflow::ops::Const(handle_tensor, builder.opts());
        
        tensorflow::Node* accumulator_num_accumulated_node = tensorflow::ops::UnaryOp(
            "AccumulatorNumAccumulated", handle_node, builder.opts());
        
        tensorflow::GraphDef graph;
        tensorflow::Status status = builder.ToGraphDef(&graph);
        if (!status.ok()) {
            std::cout << "Failed to build graph: " << status.ToString() << std::endl;
            return 0;
        }
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        status = session->Create(graph);
        if (!status.ok()) {
            std::cout << "Failed to create session: " << status.ToString() << std::endl;
            return 0;
        }
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {accumulator_num_accumulated_node->name() + ":0"}, {}, &outputs);
        if (!status.ok()) {
            std::cout << "Failed to run session: " << status.ToString() << std::endl;
            return 0;
        }
        
        if (!outputs.empty()) {
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].shape().dims(); ++i) {
                std::cout << outputs[0].shape().dim_size(i) << " ";
            }
            std::cout << std::endl;
            
            if (outputs[0].dtype() == tensorflow::DT_INT32) {
                auto output_flat = outputs[0].flat<int32_t>();
                std::cout << "Output value: " << output_flat(0) << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}