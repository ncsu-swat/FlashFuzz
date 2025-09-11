#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/sparse_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <vector>
#include <cstring>
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
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t list_size_byte = data[offset++];
        int list_size = (list_size_byte % 3) + 1;
        
        std::vector<tensorflow::Input> sorted_row_ids_list;
        std::vector<tensorflow::Input> sorted_col_ids_list;
        std::vector<tensorflow::Input> sorted_gains_list;
        std::vector<tensorflow::Input> id_counts_list;
        
        for (int i = 0; i < list_size; ++i) {
            if (offset >= size) break;
            
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::Tensor row_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(shape));
            fillTensorWithDataByType(row_ids_tensor, tensorflow::DT_INT32, data, offset, size);
            auto row_ids_const = tensorflow::ops::Const(root, row_ids_tensor);
            sorted_row_ids_list.push_back(row_ids_const);
            
            tensorflow::Tensor col_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(shape));
            fillTensorWithDataByType(col_ids_tensor, tensorflow::DT_INT32, data, offset, size);
            auto col_ids_const = tensorflow::ops::Const(root, col_ids_tensor);
            sorted_col_ids_list.push_back(col_ids_const);
            
            tensorflow::Tensor gains_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(shape));
            fillTensorWithDataByType(gains_tensor, tensorflow::DT_FLOAT, data, offset, size);
            auto gains_const = tensorflow::ops::Const(root, gains_tensor);
            sorted_gains_list.push_back(gains_const);
            
            tensorflow::Tensor id_counts_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(shape));
            fillTensorWithDataByType(id_counts_tensor, tensorflow::DT_INT32, data, offset, size);
            auto id_counts_const = tensorflow::ops::Const(root, id_counts_tensor);
            id_counts_list.push_back(id_counts_const);
        }
        
        if (offset >= size) return 0;
        
        uint8_t splits_rank = parseRank(data[offset++]);
        std::vector<int64_t> splits_shape = parseShape(data, offset, size, splits_rank);
        tensorflow::Tensor splits_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(splits_shape));
        fillTensorWithDataByType(splits_tensor, tensorflow::DT_INT64, data, offset, size);
        auto splits_const = tensorflow::ops::Const(root, splits_tensor);
        
        if (offset + 6 * sizeof(int32_t) + 1 + 1 > size) return 0;
        
        int32_t sample_count_per_sc;
        std::memcpy(&sample_count_per_sc, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        sample_count_per_sc = std::abs(sample_count_per_sc) % 100 + 1;
        
        int32_t num_replica;
        std::memcpy(&num_replica, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        num_replica = std::abs(num_replica) % 10 + 1;
        
        int32_t max_minibatches_per_sc;
        std::memcpy(&max_minibatches_per_sc, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        max_minibatches_per_sc = std::abs(max_minibatches_per_sc) % 100 + 1;
        
        int32_t max_ids_per_chip_per_sample;
        std::memcpy(&max_ids_per_chip_per_sample, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        max_ids_per_chip_per_sample = std::abs(max_ids_per_chip_per_sample) % 1000 + 1;
        
        int32_t table_vocab_size;
        std::memcpy(&table_vocab_size, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        table_vocab_size = std::abs(table_vocab_size) % 10000 + 1;
        
        int32_t feature_width;
        std::memcpy(&feature_width, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        feature_width = std::abs(feature_width) % 100 + 1;
        
        std::string table_name = "test_table";
        
        bool allow_id_dropping = (data[offset++] % 2) == 1;
        
        // Use raw_ops approach instead of ops namespace
        auto op_attrs = tensorflow::AttrValueMap();
        op_attrs["sample_count_per_sc"] = tensorflow::AttrValue();
        op_attrs["sample_count_per_sc"].set_i(sample_count_per_sc);
        
        op_attrs["num_replica"] = tensorflow::AttrValue();
        op_attrs["num_replica"].set_i(num_replica);
        
        op_attrs["max_minibatches_per_sc"] = tensorflow::AttrValue();
        op_attrs["max_minibatches_per_sc"].set_i(max_minibatches_per_sc);
        
        op_attrs["max_ids_per_chip_per_sample"] = tensorflow::AttrValue();
        op_attrs["max_ids_per_chip_per_sample"].set_i(max_ids_per_chip_per_sample);
        
        op_attrs["table_vocab_size"] = tensorflow::AttrValue();
        op_attrs["table_vocab_size"].set_i(table_vocab_size);
        
        op_attrs["feature_width"] = tensorflow::AttrValue();
        op_attrs["feature_width"].set_i(feature_width);
        
        op_attrs["table_name"] = tensorflow::AttrValue();
        op_attrs["table_name"].set_s(table_name);
        
        op_attrs["allow_id_dropping"] = tensorflow::AttrValue();
        op_attrs["allow_id_dropping"].set_b(allow_id_dropping);
        
        std::vector<tensorflow::Output> inputs;
        for (const auto& input : sorted_row_ids_list) {
            inputs.push_back(input.node()->output(input.index()));
        }
        for (const auto& input : sorted_col_ids_list) {
            inputs.push_back(input.node()->output(input.index()));
        }
        for (const auto& input : sorted_gains_list) {
            inputs.push_back(input.node()->output(input.index()));
        }
        for (const auto& input : id_counts_list) {
            inputs.push_back(input.node()->output(input.index()));
        }
        inputs.push_back(splits_const.node()->output(splits_const.index()));
        
        auto op = root.AddOperation("ConvertToSparseCoreCsrWrappedCooTensor", 
                                   inputs, 
                                   op_attrs, 
                                   7);  // 7 outputs
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({
            tensorflow::Output(op.node(), 0),  // row_pointers
            tensorflow::Output(op.node(), 1),  // sorted_sample_ids
            tensorflow::Output(op.node(), 2),  // sorted_token_ids
            tensorflow::Output(op.node(), 3),  // sorted_gains
            tensorflow::Output(op.node(), 4),  // row_pointers_unpadded_size
            tensorflow::Output(op.node(), 5),  // ids_unpadded_size
            tensorflow::Output(op.node(), 6)   // num_minibatches_per_sc
        }, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
