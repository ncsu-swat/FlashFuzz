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
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>

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
            size_t str_len = std::min(static_cast<size_t>(data[offset] % 10 + 1), total_size - offset - 1);
            offset++;
            if (offset + str_len <= total_size) {
              flat(i) = tensorflow::tstring(reinterpret_cast<const char*>(data + offset), str_len);
              offset += str_len;
            } else {
              flat(i) = tensorflow::tstring("");
            }
          } else {
            flat(i) = tensorflow::tstring("");
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

        std::string container = "";
        if (offset < size) {
            size_t container_len = std::min(static_cast<size_t>(data[offset] % 10), size - offset - 1);
            offset++;
            if (offset + container_len <= size) {
                container = std::string(reinterpret_cast<const char*>(data + offset), container_len);
                offset += container_len;
            }
        }

        std::string shared_name = "";
        if (offset < size) {
            size_t shared_name_len = std::min(static_cast<size_t>(data[offset] % 10), size - offset - 1);
            offset++;
            if (offset + shared_name_len <= size) {
                shared_name = std::string(reinterpret_cast<const char*>(data + offset), shared_name_len);
                offset += shared_name_len;
            }
        }

        std::cout << "Container: " << container << std::endl;
        std::cout << "Shared name: " << shared_name << std::endl;

        tensorflow::SessionOptions session_options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(session_options));
        
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        node_def->set_name("identity_reader");
        node_def->set_op("IdentityReader");
        
        tensorflow::AttrValue container_attr;
        container_attr.set_s(container);
        (*node_def->mutable_attr())["container"] = container_attr;
        
        tensorflow::AttrValue shared_name_attr;
        shared_name_attr.set_s(shared_name);
        (*node_def->mutable_attr())["shared_name"] = shared_name_attr;

        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            std::cout << "Failed to create session: " << status.ToString() << std::endl;
            return 0;
        }

        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {"identity_reader:0"}, {}, &outputs);
        
        if (status.ok()) {
            std::cout << "IdentityReader operation executed successfully" << std::endl;
            if (!outputs.empty()) {
                std::cout << "Output tensor shape: " << outputs[0].shape().DebugString() << std::endl;
                std::cout << "Output tensor dtype: " << tensorflow::DataTypeString(outputs[0].dtype()) << std::endl;
            }
        } else {
            std::cout << "IdentityReader operation failed: " << status.ToString() << std::endl;
        }

        session->Close();

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}