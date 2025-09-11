#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
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

void fillStringTensor(tensorflow::Tensor& tensor, const uint8_t* data,
                      size_t& offset, size_t total_size) {
  auto flat = tensor.flat<tensorflow::tstring>();
  const size_t num_elements = flat.size();
  
  for (size_t i = 0; i < num_elements; ++i) {
    if (offset < total_size) {
      size_t str_len = std::min(static_cast<size_t>(10), total_size - offset);
      std::string str(reinterpret_cast<const char*>(data + offset), str_len);
      offset += str_len;
      flat(i) = str;
    } else {
      flat(i) = "";
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
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_COMPLEX64:
      fillTensorWithData<tensorflow::complex64>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_COMPLEX128:
      fillTensorWithData<tensorflow::complex128>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_STRING:
      fillStringTensor(tensor, data, offset, total_size);
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
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t program_key_rank = parseRank(data[offset++]);
        std::vector<int64_t> program_key_shape = parseShape(data, offset, size, program_key_rank);
        tensorflow::Tensor program_key_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(program_key_shape));
        fillStringTensor(program_key_tensor, data, offset, size);
        auto program_key = tensorflow::ops::Const(root, program_key_tensor);

        uint8_t row_ids_rank = parseRank(data[offset++]);
        std::vector<int64_t> row_ids_shape = parseShape(data, offset, size, row_ids_rank);
        tensorflow::Tensor row_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(row_ids_shape));
        fillTensorWithData<int32_t>(row_ids_tensor, data, offset, size);
        auto row_ids = tensorflow::ops::Const(root, row_ids_tensor);

        uint8_t col_ids_rank = parseRank(data[offset++]);
        std::vector<int64_t> col_ids_shape = parseShape(data, offset, size, col_ids_rank);
        tensorflow::Tensor col_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(col_ids_shape));
        fillTensorWithData<int32_t>(col_ids_tensor, data, offset, size);
        auto col_ids = tensorflow::ops::Const(root, col_ids_tensor);

        uint8_t gains_rank = parseRank(data[offset++]);
        std::vector<int64_t> gains_shape = parseShape(data, offset, size, gains_rank);
        tensorflow::Tensor gains_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(gains_shape));
        fillTensorWithData<float>(gains_tensor, data, offset, size);
        auto gains = tensorflow::ops::Const(root, gains_tensor);

        uint8_t splits_rank = parseRank(data[offset++]);
        std::vector<int64_t> splits_shape = parseShape(data, offset, size, splits_rank);
        tensorflow::Tensor splits_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(splits_shape));
        fillTensorWithData<int64_t>(splits_tensor, data, offset, size);
        auto splits = tensorflow::ops::Const(root, splits_tensor);

        uint8_t id_counts_rank = parseRank(data[offset++]);
        std::vector<int64_t> id_counts_shape = parseShape(data, offset, size, id_counts_rank);
        tensorflow::Tensor id_counts_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(id_counts_shape));
        fillTensorWithData<int32_t>(id_counts_tensor, data, offset, size);
        auto id_counts = tensorflow::ops::Const(root, id_counts_tensor);

        int sample_count = 1;
        int num_replica = 1;
        int max_minibatches_per_sc = 1;
        int max_ids_per_chip_per_sample = 1;
        int table_vocab_size = 1;
        int feature_width = 1;
        int num_sc_per_chip = 1;
        
        if (offset + sizeof(int) <= size) {
            std::memcpy(&sample_count, data + offset, sizeof(int));
            offset += sizeof(int);
            sample_count = std::max(1, std::abs(sample_count) % 100 + 1);
        }
        
        if (offset + sizeof(int) <= size) {
            std::memcpy(&num_replica, data + offset, sizeof(int));
            offset += sizeof(int);
            num_replica = std::max(1, std::abs(num_replica) % 10 + 1);
        }
        
        if (offset + sizeof(int) <= size) {
            std::memcpy(&max_minibatches_per_sc, data + offset, sizeof(int));
            offset += sizeof(int);
            max_minibatches_per_sc = std::max(1, std::abs(max_minibatches_per_sc) % 10 + 1);
        }
        
        if (offset + sizeof(int) <= size) {
            std::memcpy(&max_ids_per_chip_per_sample, data + offset, sizeof(int));
            offset += sizeof(int);
            max_ids_per_chip_per_sample = std::max(1, std::abs(max_ids_per_chip_per_sample) % 100 + 1);
        }
        
        if (offset + sizeof(int) <= size) {
            std::memcpy(&table_vocab_size, data + offset, sizeof(int));
            offset += sizeof(int);
            table_vocab_size = std::max(1, std::abs(table_vocab_size) % 1000 + 1);
        }
        
        if (offset + sizeof(int) <= size) {
            std::memcpy(&feature_width, data + offset, sizeof(int));
            offset += sizeof(int);
            feature_width = std::max(1, std::abs(feature_width) % 100 + 1);
        }
        
        if (offset + sizeof(int) <= size) {
            std::memcpy(&num_sc_per_chip, data + offset, sizeof(int));
            offset += sizeof(int);
            num_sc_per_chip = std::max(1, std::abs(num_sc_per_chip) % 10 + 1);
        }

        std::string table_name = "test_table";
        std::string mini_batch_in_csr = "test_csr";
        
        if (offset + 10 <= size) {
            table_name = std::string(reinterpret_cast<const char*>(data + offset), 10);
            offset += 10;
        }
        
        if (offset + 10 <= size) {
            mini_batch_in_csr = std::string(reinterpret_cast<const char*>(data + offset), 10);
            offset += 10;
        }

        // Create the operation using raw_ops
        tensorflow::OutputList outputs;
        tensorflow::NodeBuilder node_builder("GetMinibatchesInCsrWithPhysicalReplica", "GetMinibatchesInCsrWithPhysicalReplica");
        
        node_builder.Input(program_key.node())
                   .Input(row_ids.node())
                   .Input(col_ids.node())
                   .Input(gains.node())
                   .Input(splits.node())
                   .Input(id_counts.node())
                   .Attr("sample_count", sample_count)
                   .Attr("num_replica", num_replica)
                   .Attr("max_minibatches_per_sc", max_minibatches_per_sc)
                   .Attr("max_ids_per_chip_per_sample", max_ids_per_chip_per_sample)
                   .Attr("table_vocab_size", table_vocab_size)
                   .Attr("feature_width", feature_width)
                   .Attr("num_sc_per_chip", num_sc_per_chip)
                   .Attr("table_name", table_name)
                   .Attr("mini_batch_in_csr", mini_batch_in_csr);
        
        tensorflow::Node* node;
        tensorflow::Status status = node_builder.Finalize(root.graph(), &node);
        
        if (!status.ok()) {
            return -1;
        }
        
        for (int i = 0; i < 7; ++i) {
            outputs.push_back(tensorflow::Output(node, i));
        }
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> output_tensors;
        status = session.Run({outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6]}, &output_tensors);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
