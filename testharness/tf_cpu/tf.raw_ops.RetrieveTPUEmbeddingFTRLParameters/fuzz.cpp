#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include <iostream>
#include <cstring>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
    void logError(const std::string& message, const uint8_t* data, size_t size) {
        std::cerr << "Error: " << message << std::endl;
    }
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
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        int32_t num_shards;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&num_shards, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            num_shards = std::abs(num_shards) % 100 + 1;
        } else {
            num_shards = 1;
        }

        int32_t shard_id;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&shard_id, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            shard_id = std::abs(shard_id) % num_shards;
        } else {
            shard_id = 0;
        }

        int32_t table_id = -1;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&table_id, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            table_id = std::abs(table_id) % 10;
        }

        std::string table_name = "";
        if (offset + 1 <= size) {
            uint8_t name_len = data[offset] % 20;
            offset++;
            if (offset + name_len <= size) {
                table_name = std::string(reinterpret_cast<const char*>(data + offset), name_len);
                offset += name_len;
            }
        }

        std::string config = "";
        if (offset + 1 <= size) {
            uint8_t config_len = data[offset] % 20;
            offset++;
            if (offset + config_len <= size) {
                config = std::string(reinterpret_cast<const char*>(data + offset), config_len);
                offset += config_len;
            }
        }

        std::cout << "num_shards: " << num_shards << std::endl;
        std::cout << "shard_id: " << shard_id << std::endl;
        std::cout << "table_id: " << table_id << std::endl;
        std::cout << "table_name: " << table_name << std::endl;
        std::cout << "config: " << config << std::endl;

        // Create operation using raw_ops
        auto num_shards_tensor = tensorflow::ops::Const(root, num_shards);
        auto shard_id_tensor = tensorflow::ops::Const(root, shard_id);
        
        std::vector<tensorflow::Output> outputs;
        tensorflow::NodeBuilder builder = tensorflow::NodeBuilder("RetrieveTPUEmbeddingFTRLParameters", "RetrieveTPUEmbeddingFTRLParameters")
            .Input(num_shards_tensor.node())
            .Input(shard_id_tensor.node())
            .Attr("table_id", table_id)
            .Attr("table_name", table_name)
            .Attr("config", config);
        
        tensorflow::Node* node;
        tensorflow::Status status = root.graph()->AddNode(builder, &node);
        if (!status.ok()) {
            std::cout << "Error creating node: " << status.ToString() << std::endl;
            return -1;
        }
        
        // Add outputs
        for (int i = 0; i < 3; i++) {
            outputs.push_back(tensorflow::Output(node, i));
        }

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> output_tensors;
        
        status = session.Run({outputs[0], outputs[1], outputs[2]}, &output_tensors);
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }

        if (output_tensors.size() >= 3) {
            std::cout << "Parameters shape: ";
            for (int i = 0; i < output_tensors[0].shape().dims(); ++i) {
                std::cout << output_tensors[0].shape().dim_size(i) << " ";
            }
            std::cout << std::endl;
            
            std::cout << "Accumulators shape: ";
            for (int i = 0; i < output_tensors[1].shape().dims(); ++i) {
                std::cout << output_tensors[1].shape().dim_size(i) << " ";
            }
            std::cout << std::endl;
            
            std::cout << "Linears shape: ";
            for (int i = 0; i < output_tensors[2].shape().dims(); ++i) {
                std::cout << output_tensors[2].shape().dim_size(i) << " ";
            }
            std::cout << std::endl;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}