#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/framework/node_def_util.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>

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
            size_t str_len = std::min(static_cast<size_t>(data[offset] % 32), total_size - offset - 1);
            offset++;
            if (offset + str_len <= total_size) {
              std::string str(reinterpret_cast<const char*>(data + offset), str_len);
              flat(i) = str;
              offset += str_len;
            } else {
              flat(i) = "";
            }
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

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        std::string handle_str;
        if (offset < size) {
            size_t str_len = std::min(static_cast<size_t>(data[offset] % 32), size - offset - 1);
            offset++;
            if (offset + str_len <= size) {
                handle_str = std::string(reinterpret_cast<const char*>(data + offset), str_len);
                offset += str_len;
            } else {
                handle_str = "accumulator_handle";
            }
        } else {
            handle_str = "accumulator_handle";
        }

        tensorflow::Tensor handle_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        handle_tensor.scalar<tensorflow::tstring>()() = handle_str;

        std::cout << "Handle string: " << handle_str << std::endl;
        std::cout << "Handle tensor shape: " << handle_tensor.shape().DebugString() << std::endl;
        std::cout << "Handle tensor dtype: " << tensorflow::DataTypeString(handle_tensor.dtype()) << std::endl;

        auto handle_op = tensorflow::ops::Const(root, handle_tensor);

        tensorflow::ClientSession session(root);
        
        tensorflow::NodeDef node_def;
        node_def.set_name("accumulator_num_accumulated");
        node_def.set_op("AccumulatorNumAccumulated");
        
        tensorflow::NodeDefBuilder builder("accumulator_num_accumulated", "AccumulatorNumAccumulated");
        builder.Input(handle_op.node()->name(), 0, tensorflow::DT_STRING);
        
        tensorflow::Status status = builder.Finalize(&node_def);
        if (!status.ok()) {
            std::cout << "NodeDefBuilder failed: " << status.ToString() << std::endl;
            return 0;
        }

        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({handle_op}, {"accumulator_num_accumulated:0"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "AccumulatorNumAccumulated output shape: " << outputs[0].shape().DebugString() << std::endl;
            std::cout << "AccumulatorNumAccumulated output dtype: " << tensorflow::DataTypeString(outputs[0].dtype()) << std::endl;
            if (outputs[0].dtype() == tensorflow::DT_INT32) {
                std::cout << "AccumulatorNumAccumulated result: " << outputs[0].scalar<int32_t>()() << std::endl;
            }
        } else {
            std::cout << "AccumulatorNumAccumulated failed: " << status.ToString() << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}