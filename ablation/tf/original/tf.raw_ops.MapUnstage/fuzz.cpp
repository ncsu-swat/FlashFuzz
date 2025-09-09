#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/cc/framework/scope.h>
#include <vector>
#include <memory>

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
    default:
      break;
  }
}

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
      dtype = tensorflow::DT_COMPLEX64;
      break;
    case 11:
      dtype = tensorflow::DT_COMPLEX128;
      break;
    case 12:
      dtype = tensorflow::DT_HALF;
      break;
    case 13:
      dtype = tensorflow::DT_UINT32;
      break;
    case 14:
      dtype = tensorflow::DT_UINT64;
      break;
    default:
      dtype = tensorflow::DT_FLOAT;
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

        uint8_t key_rank = parseRank(data[offset++]);
        std::vector<int64_t> key_shape = parseShape(data, offset, size, key_rank);
        tensorflow::TensorShape key_tensor_shape(key_shape);
        tensorflow::Tensor key_tensor(tensorflow::DT_INT64, key_tensor_shape);
        fillTensorWithDataByType(key_tensor, tensorflow::DT_INT64, data, offset, size);

        uint8_t indices_rank = parseRank(data[offset++]);
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        tensorflow::TensorShape indices_tensor_shape(indices_shape);
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, indices_tensor_shape);
        fillTensorWithDataByType(indices_tensor, tensorflow::DT_INT32, data, offset, size);

        if (offset >= size) {
            return 0;
        }

        uint8_t num_dtypes = (data[offset++] % 5) + 1;
        std::vector<tensorflow::DataType> dtypes;
        for (uint8_t i = 0; i < num_dtypes; ++i) {
            if (offset >= size) {
                dtypes.push_back(tensorflow::DT_FLOAT);
            } else {
                dtypes.push_back(parseDataType(data[offset++]));
            }
        }

        int capacity = 0;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&capacity, data + offset, sizeof(int));
            offset += sizeof(int);
            capacity = std::abs(capacity) % 1000;
        }

        int memory_limit = 0;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&memory_limit, data + offset, sizeof(int));
            offset += sizeof(int);
            memory_limit = std::abs(memory_limit) % 1000000;
        }

        std::string container = "";
        std::string shared_name = "";

        std::cout << "Key tensor shape: ";
        for (auto dim : key_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        std::cout << "Indices tensor shape: ";
        for (auto dim : indices_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        std::cout << "Number of dtypes: " << static_cast<int>(num_dtypes) << std::endl;
        std::cout << "Capacity: " << capacity << std::endl;
        std::cout << "Memory limit: " << memory_limit << std::endl;

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto key_placeholder = tensorflow::ops::Placeholder(root.WithOpName("key"), tensorflow::DT_INT64);
        auto indices_placeholder = tensorflow::ops::Placeholder(root.WithOpName("indices"), tensorflow::DT_INT32);

        tensorflow::Node* map_unstage_node;
        tensorflow::NodeBuilder builder("map_unstage", "MapUnstage");
        builder.Input(key_placeholder.node())
               .Input(indices_placeholder.node())
               .Attr("dtypes", dtypes)
               .Attr("capacity", capacity)
               .Attr("memory_limit", memory_limit)
               .Attr("container", container)
               .Attr("shared_name", shared_name);
        
        auto status = builder.Finalize(root.graph(), &map_unstage_node);
        if (!status.ok()) {
            std::cout << "Failed to create MapUnstage node: " << status.ToString() << std::endl;
            return 0;
        }

        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        status = session->Create(root.graph()->ToGraphDef());
        if (!status.ok()) {
            std::cout << "Failed to create session: " << status.ToString() << std::endl;
            return 0;
        }

        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"key", key_tensor},
            {"indices", indices_tensor}
        };

        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names;
        for (size_t i = 0; i < dtypes.size(); ++i) {
            output_names.push_back("map_unstage:" + std::to_string(i));
        }

        status = session->Run(inputs, output_names, {}, &outputs);
        if (!status.ok()) {
            std::cout << "MapUnstage operation failed: " << status.ToString() << std::endl;
        } else {
            std::cout << "MapUnstage operation succeeded with " << outputs.size() << " outputs" << std::endl;
        }

        session->Close();

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}