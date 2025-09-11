#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <cstring>
#include <vector>
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
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t rank1 = parseRank(data[offset++]);
        std::vector<int64_t> shape1 = parseShape(data, offset, size, rank1);
        tensorflow::TensorShape tensor_shape1(shape1);
        tensorflow::Tensor parameters_tensor(tensorflow::DT_FLOAT, tensor_shape1);
        fillTensorWithDataByType(parameters_tensor, tensorflow::DT_FLOAT, data, offset, size);

        uint8_t rank2 = parseRank(data[offset++]);
        std::vector<int64_t> shape2 = parseShape(data, offset, size, rank2);
        tensorflow::TensorShape tensor_shape2(shape2);
        tensorflow::Tensor ms_tensor(tensorflow::DT_FLOAT, tensor_shape2);
        fillTensorWithDataByType(ms_tensor, tensorflow::DT_FLOAT, data, offset, size);

        uint8_t rank3 = parseRank(data[offset++]);
        std::vector<int64_t> shape3 = parseShape(data, offset, size, rank3);
        tensorflow::TensorShape tensor_shape3(shape3);
        tensorflow::Tensor mom_tensor(tensorflow::DT_FLOAT, tensor_shape3);
        fillTensorWithDataByType(mom_tensor, tensorflow::DT_FLOAT, data, offset, size);

        int num_shards = 1;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&num_shards, data + offset, sizeof(int));
            offset += sizeof(int);
            num_shards = std::abs(num_shards) % 10 + 1;
        }

        int shard_id = 0;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&shard_id, data + offset, sizeof(int));
            offset += sizeof(int);
            shard_id = std::abs(shard_id) % num_shards;
        }

        int table_id = -1;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&table_id, data + offset, sizeof(int));
            offset += sizeof(int);
            table_id = table_id % 100;
        }

        std::string table_name = "";
        std::string config = "";

        auto parameters_input = tensorflow::ops::Const(root, parameters_tensor);
        auto ms_input = tensorflow::ops::Const(root, ms_tensor);
        auto mom_input = tensorflow::ops::Const(root, mom_tensor);

        // Use raw_ops directly since tpu_ops.h is not available
        tensorflow::NodeBuilder builder("LoadTPUEmbeddingRMSPropParameters", "LoadTPUEmbeddingRMSPropParameters");
        builder.Input(parameters_input.node());
        builder.Input(ms_input.node());
        builder.Input(mom_input.node());
        builder.Attr("num_shards", num_shards);
        builder.Attr("shard_id", shard_id);
        builder.Attr("table_id", table_id);
        builder.Attr("table_name", table_name);
        builder.Attr("config", config);

        tensorflow::Node* node;
        TF_CHECK_OK(builder.Finalize(root.graph(), &node));
        
        tensorflow::ClientSession session(root);
        TF_CHECK_OK(session.Run({}, {}, {node->name()}, nullptr));

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
