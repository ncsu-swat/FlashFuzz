#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/op.h"
#include <cstring>
#include <vector>
#include <iostream>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
    void logError(const std::string& message, const uint8_t* data, size_t size) {
        std::cerr << "Error: " << message << std::endl;
    }
}

tensorflow::DataType parseDataType(uint8_t selector) {
    return tensorflow::DT_FLOAT;
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
    default:
      fillTensorWithData<float>(tensor, data, offset, total_size);
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t parameters_rank = parseRank(data[offset++]);
        std::vector<int64_t> parameters_shape = parseShape(data, offset, size, parameters_rank);
        tensorflow::TensorShape parameters_tensor_shape(parameters_shape);
        tensorflow::Tensor parameters_tensor(tensorflow::DT_FLOAT, parameters_tensor_shape);
        fillTensorWithDataByType(parameters_tensor, tensorflow::DT_FLOAT, data, offset, size);

        uint8_t accumulators_rank = parseRank(data[offset++]);
        std::vector<int64_t> accumulators_shape = parseShape(data, offset, size, accumulators_rank);
        tensorflow::TensorShape accumulators_tensor_shape(accumulators_shape);
        tensorflow::Tensor accumulators_tensor(tensorflow::DT_FLOAT, accumulators_tensor_shape);
        fillTensorWithDataByType(accumulators_tensor, tensorflow::DT_FLOAT, data, offset, size);

        int num_shards = 1;
        if (offset < size) {
            num_shards = static_cast<int>(data[offset++]) % 10 + 1;
        }

        int shard_id = 0;
        if (offset < size) {
            shard_id = static_cast<int>(data[offset++]) % num_shards;
        }

        int table_id = -1;
        if (offset < size) {
            table_id = static_cast<int>(data[offset++]) % 100;
        }

        std::string table_name = "";
        std::string config = "";

        auto parameters_input = tensorflow::ops::Const(root, parameters_tensor);
        auto accumulators_input = tensorflow::ops::Const(root, accumulators_tensor);

        // Use raw_ops directly since tpu_ops.h is not available
        auto load_op = tensorflow::ops::Operation(
            root.WithOpName("LoadTPUEmbeddingProximalAdagradParameters"),
            "LoadTPUEmbeddingProximalAdagradParameters",
            {parameters_input.output, accumulators_input.output},
            {
                {"num_shards", num_shards},
                {"shard_id", shard_id},
                {"table_id", table_id},
                {"table_name", table_name},
                {"config", config}
            }
        );

        tensorflow::ClientSession session(root);
        tensorflow::Status status = session.Run({load_op}, nullptr);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}