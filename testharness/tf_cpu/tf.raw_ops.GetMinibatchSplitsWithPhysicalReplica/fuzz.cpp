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

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << message << std::endl;
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
    case tensorflow::DT_STRING: {
      auto flat = tensor.flat<tensorflow::tstring>();
      const size_t num_elements = flat.size();
      for (size_t i = 0; i < num_elements; ++i) {
        if (offset < total_size) {
          uint8_t str_len = data[offset] % 10 + 1;
          offset++;
          std::string str;
          for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
            str += static_cast<char>(data[offset] % 26 + 'a');
            offset++;
          }
          flat(i) = str;
        } else {
          flat(i) = "default";
        }
      }
      break;
    }
    default:
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t program_key_rank = parseRank(data[offset++]);
        std::vector<int64_t> program_key_shape = parseShape(data, offset, size, program_key_rank);
        tensorflow::Tensor program_key_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(program_key_shape));
        fillTensorWithDataByType(program_key_tensor, tensorflow::DT_STRING, data, offset, size);
        auto program_key = tensorflow::ops::Const(root, program_key_tensor);

        uint8_t row_ids_rank = parseRank(data[offset++]);
        std::vector<int64_t> row_ids_shape = parseShape(data, offset, size, row_ids_rank);
        tensorflow::Tensor row_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(row_ids_shape));
        fillTensorWithDataByType(row_ids_tensor, tensorflow::DT_INT32, data, offset, size);
        auto row_ids = tensorflow::ops::Const(root, row_ids_tensor);

        uint8_t col_ids_rank = parseRank(data[offset++]);
        std::vector<int64_t> col_ids_shape = parseShape(data, offset, size, col_ids_rank);
        tensorflow::Tensor col_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(col_ids_shape));
        fillTensorWithDataByType(col_ids_tensor, tensorflow::DT_INT32, data, offset, size);
        auto col_ids = tensorflow::ops::Const(root, col_ids_tensor);

        uint8_t gains_rank = parseRank(data[offset++]);
        std::vector<int64_t> gains_shape = parseShape(data, offset, size, gains_rank);
        tensorflow::Tensor gains_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(gains_shape));
        fillTensorWithDataByType(gains_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto gains = tensorflow::ops::Const(root, gains_tensor);

        int sample_count = 1;
        if (offset < size) {
            sample_count = static_cast<int>(data[offset++] % 10) + 1;
        }

        int num_replica = 1;
        if (offset < size) {
            num_replica = static_cast<int>(data[offset++] % 10) + 1;
        }

        int table_vocab_size = 1;
        if (offset < size) {
            table_vocab_size = static_cast<int>(data[offset++] % 100) + 1;
        }

        int feature_width = 1;
        if (offset < size) {
            feature_width = static_cast<int>(data[offset++] % 10) + 1;
        }

        int num_sc_per_chip = 1;
        if (offset < size) {
            num_sc_per_chip = static_cast<int>(data[offset++] % 10) + 1;
        }

        std::string table_name = "test_table";
        if (offset < size) {
            uint8_t name_len = data[offset++] % 10 + 1;
            table_name = "";
            for (uint8_t i = 0; i < name_len && offset < size; ++i) {
                table_name += static_cast<char>(data[offset++] % 26 + 'a');
            }
        }

        std::string mini_batch_splits = "test_splits";
        if (offset < size) {
            uint8_t splits_len = data[offset++] % 10 + 1;
            mini_batch_splits = "";
            for (uint8_t i = 0; i < splits_len && offset < size; ++i) {
                mini_batch_splits += static_cast<char>(data[offset++] % 26 + 'a');
            }
        }

        // Use raw_ops API instead of ops namespace
        tensorflow::NodeDef node_def;
        node_def.set_op("GetMinibatchSplitsWithPhysicalReplica");
        node_def.set_name("get_minibatch_splits_with_physical_replica");
        
        // Set attributes
        auto* attrs = node_def.mutable_attr();
        (*attrs)["sample_count"].set_i(sample_count);
        (*attrs)["num_replica"].set_i(num_replica);
        (*attrs)["table_vocab_size"].set_i(table_vocab_size);
        (*attrs)["feature_width"].set_i(feature_width);
        (*attrs)["num_sc_per_chip"].set_i(num_sc_per_chip);
        (*attrs)["table_name"].set_s(table_name);
        (*attrs)["mini_batch_splits"].set_s(mini_batch_splits);

        // Create operation
        tensorflow::Status status;
        auto op = root.AddNode(node_def, &status);
        if (!status.ok()) {
            tf_fuzzer_utils::logError("Failed to create op: " + status.ToString(), data, size);
            return -1;
        }

        // Add inputs
        status = root.UpdateEdge(program_key.node(), 0, op, 0);
        if (!status.ok()) {
            tf_fuzzer_utils::logError("Failed to add input 0: " + status.ToString(), data, size);
            return -1;
        }
        
        status = root.UpdateEdge(row_ids.node(), 0, op, 1);
        if (!status.ok()) {
            tf_fuzzer_utils::logError("Failed to add input 1: " + status.ToString(), data, size);
            return -1;
        }
        
        status = root.UpdateEdge(col_ids.node(), 0, op, 2);
        if (!status.ok()) {
            tf_fuzzer_utils::logError("Failed to add input 2: " + status.ToString(), data, size);
            return -1;
        }
        
        status = root.UpdateEdge(gains.node(), 0, op, 3);
        if (!status.ok()) {
            tf_fuzzer_utils::logError("Failed to add input 3: " + status.ToString(), data, size);
            return -1;
        }

        // Create output operations
        auto sorted_row_ids = tensorflow::Output(op, 0);
        auto sorted_col_ids = tensorflow::Output(op, 1);
        auto sorted_gains = tensorflow::Output(op, 2);
        auto splits = tensorflow::Output(op, 3);
        auto id_counts = tensorflow::Output(op, 4);
        auto max_ids = tensorflow::Output(op, 5);
        auto max_uniques = tensorflow::Output(op, 6);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({sorted_row_ids, sorted_col_ids, sorted_gains, 
                             splits, id_counts, max_ids, max_uniques}, &outputs);
        
        if (!status.ok()) {
            tf_fuzzer_utils::logError("Session run failed: " + status.ToString(), data, size);
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
