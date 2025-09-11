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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        int num_shards = 1;
        int shard_id = 0;
        int table_id = -1;
        std::string table_name = "";
        std::string config = "";

        if (offset + sizeof(int) <= size) {
            std::memcpy(&num_shards, data + offset, sizeof(int));
            offset += sizeof(int);
            num_shards = std::abs(num_shards) % 10 + 1;
        }

        if (offset + sizeof(int) <= size) {
            std::memcpy(&shard_id, data + offset, sizeof(int));
            offset += sizeof(int);
            shard_id = std::abs(shard_id) % num_shards;
        }

        if (offset + sizeof(int) <= size) {
            std::memcpy(&table_id, data + offset, sizeof(int));
            offset += sizeof(int);
            table_id = table_id % 100;
        }

        size_t table_name_len = 0;
        if (offset + sizeof(size_t) <= size) {
            std::memcpy(&table_name_len, data + offset, sizeof(size_t));
            offset += sizeof(size_t);
            table_name_len = table_name_len % 50;
            
            if (offset + table_name_len <= size) {
                table_name = std::string(reinterpret_cast<const char*>(data + offset), table_name_len);
                offset += table_name_len;
            }
        }

        size_t config_len = 0;
        if (offset + sizeof(size_t) <= size) {
            std::memcpy(&config_len, data + offset, sizeof(size_t));
            offset += sizeof(size_t);
            config_len = config_len % 100;
            
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

        // Use raw_ops API instead of ops::RetrieveTPUEmbeddingAdadeltaParameters
        auto op_attrs = tensorflow::ops::Attrs()
            .WithAttr("table_id", table_id)
            .WithAttr("table_name", table_name)
            .WithAttr("config", config);

        auto scope = root.WithOpName("RetrieveTPUEmbeddingAdadeltaParameters");
        auto num_shards_tensor = tensorflow::ops::Const(scope, num_shards);
        auto shard_id_tensor = tensorflow::ops::Const(scope, shard_id);
        
        tensorflow::NodeDef node_def;
        node_def.set_name(scope.GetUniqueNameForOp("RetrieveTPUEmbeddingAdadeltaParameters"));
        node_def.set_op("RetrieveTPUEmbeddingAdadeltaParameters");
        
        // Add inputs
        tensorflow::NodeDefBuilder node_def_builder(node_def.name(), "RetrieveTPUEmbeddingAdadeltaParameters");
        node_def_builder.Input(tensorflow::ops::NodeOut(num_shards_tensor.node()));
        node_def_builder.Input(tensorflow::ops::NodeOut(shard_id_tensor.node()));
        
        // Add attributes
        node_def_builder.Attr("table_id", table_id);
        node_def_builder.Attr("table_name", table_name);
        node_def_builder.Attr("config", config);
        
        TF_CHECK_OK(node_def_builder.Finalize(&node_def));
        
        tensorflow::Status status;
        auto node = scope.graph()->AddNode(node_def, &status);
        TF_CHECK_OK(status);
        
        scope.UpdateStatus(scope.graph()->UpdateEdge(
            num_shards_tensor.node(), 0, node, 0));
        scope.UpdateStatus(scope.graph()->UpdateEdge(
            shard_id_tensor.node(), 0, node, 1));
        
        std::vector<tensorflow::Output> outputs;
        for (int i = 0; i < 3; ++i) {
            outputs.emplace_back(tensorflow::Output(node, i));
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
            
            std::cout << "Updates shape: ";
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
