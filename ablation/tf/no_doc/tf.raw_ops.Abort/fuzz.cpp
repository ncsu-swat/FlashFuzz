#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/str_util.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/common_runtime/direct_session.h>
#include <tensorflow/core/framework/op_def_builder.h>
#include <tensorflow/core/framework/op_def.pb.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/lib/core/errors.h>
#include <tensorflow/core/lib/core/status.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/graph.h>
#include <tensorflow/core/graph/node_builder.h>
#include <tensorflow/core/common_runtime/device_factory.h>
#include <tensorflow/core/common_runtime/device_mgr.h>
#include <tensorflow/core/framework/device_base.h>
#include <tensorflow/core/framework/allocator.h>
#include <tensorflow/core/platform/env.h>

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
        
        if (size < 3) {
            return 0;
        }

        tensorflow::DataType error_msg_dtype = parseDataType(data[offset++]);
        uint8_t error_msg_rank = parseRank(data[offset++]);
        std::vector<int64_t> error_msg_shape = parseShape(data, offset, size, error_msg_rank);
        
        tensorflow::TensorShape error_msg_tensor_shape;
        for (int64_t dim : error_msg_shape) {
            error_msg_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor error_msg_tensor(error_msg_dtype, error_msg_tensor_shape);
        fillTensorWithDataByType(error_msg_tensor, error_msg_dtype, data, offset, size);
        
        std::cout << "Error message tensor dtype: " << tensorflow::DataTypeString(error_msg_dtype) << std::endl;
        std::cout << "Error message tensor shape: " << error_msg_tensor_shape.DebugString() << std::endl;
        
        tensorflow::DataType exit_without_error_dtype = parseDataType(data[offset % size]);
        offset++;
        uint8_t exit_without_error_rank = parseRank(data[offset % size]);
        offset++;
        std::vector<int64_t> exit_without_error_shape = parseShape(data, offset, size, exit_without_error_rank);
        
        tensorflow::TensorShape exit_without_error_tensor_shape;
        for (int64_t dim : exit_without_error_shape) {
            exit_without_error_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor exit_without_error_tensor(exit_without_error_dtype, exit_without_error_tensor_shape);
        fillTensorWithDataByType(exit_without_error_tensor, exit_without_error_dtype, data, offset, size);
        
        std::cout << "Exit without error tensor dtype: " << tensorflow::DataTypeString(exit_without_error_dtype) << std::endl;
        std::cout << "Exit without error tensor shape: " << exit_without_error_tensor_shape.DebugString() << std::endl;

        tensorflow::Graph graph(tensorflow::OpRegistry::Global());
        
        tensorflow::Node* error_msg_node;
        tensorflow::Status status = tensorflow::NodeBuilder("error_msg", "Placeholder")
            .Attr("dtype", error_msg_dtype)
            .Attr("shape", error_msg_tensor_shape)
            .Finalize(&graph, &error_msg_node);
        
        if (!status.ok()) {
            std::cout << "Failed to create error_msg placeholder: " << status.ToString() << std::endl;
            return 0;
        }
        
        tensorflow::Node* exit_without_error_node;
        status = tensorflow::NodeBuilder("exit_without_error", "Placeholder")
            .Attr("dtype", exit_without_error_dtype)
            .Attr("shape", exit_without_error_tensor_shape)
            .Finalize(&graph, &exit_without_error_node);
        
        if (!status.ok()) {
            std::cout << "Failed to create exit_without_error placeholder: " << status.ToString() << std::endl;
            return 0;
        }
        
        tensorflow::Node* abort_node;
        status = tensorflow::NodeBuilder("abort_op", "Abort")
            .Input(error_msg_node)
            .Input(exit_without_error_node)
            .Finalize(&graph, &abort_node);
        
        if (!status.ok()) {
            std::cout << "Failed to create Abort node: " << status.ToString() << std::endl;
            return 0;
        }
        
        std::cout << "Successfully created Abort operation with inputs" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}