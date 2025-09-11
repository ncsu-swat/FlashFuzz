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
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10
#define MAX_LIST_SIZE 5

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

int32_t parseInt32(const uint8_t* data, size_t& offset, size_t total_size) {
    if (offset + sizeof(int32_t) <= total_size) {
        int32_t value;
        std::memcpy(&value, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        return std::abs(value) % 1000 + 1;
    }
    return 1;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t list_size_byte = data[offset++];
        size_t list_size = (list_size_byte % MAX_LIST_SIZE) + 1;

        std::vector<tensorflow::Input> row_ids_list;
        std::vector<tensorflow::Input> col_ids_list;
        std::vector<tensorflow::Input> gains_list;
        std::vector<int64_t> sample_count_list;
        std::vector<int64_t> col_offset_list;

        for (size_t i = 0; i < list_size; ++i) {
            if (offset >= size) break;

            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);

            tensorflow::Tensor row_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(shape));
            fillTensorWithDataByType(row_ids_tensor, tensorflow::DT_INT32, data, offset, size);
            auto row_ids_const = tensorflow::ops::Const(root, row_ids_tensor);
            row_ids_list.push_back(row_ids_const);

            tensorflow::Tensor col_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(shape));
            fillTensorWithDataByType(col_ids_tensor, tensorflow::DT_INT32, data, offset, size);
            auto col_ids_const = tensorflow::ops::Const(root, col_ids_tensor);
            col_ids_list.push_back(col_ids_const);

            tensorflow::Tensor gains_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(shape));
            fillTensorWithDataByType(gains_tensor, tensorflow::DT_FLOAT, data, offset, size);
            auto gains_const = tensorflow::ops::Const(root, gains_tensor);
            gains_list.push_back(gains_const);

            int64_t sample_count = parseInt32(data, offset, size);
            sample_count_list.push_back(sample_count);

            int64_t col_offset = parseInt32(data, offset, size);
            col_offset_list.push_back(col_offset);
        }

        int64_t num_replica = parseInt32(data, offset, size);
        int64_t table_vocab_size = parseInt32(data, offset, size);
        int64_t feature_width = parseInt32(data, offset, size);
        int64_t num_sc_per_chip = parseInt32(data, offset, size);
        int64_t max_ids_per_sparse_core = parseInt32(data, offset, size);
        int64_t max_unique_ids_per_sparse_core = parseInt32(data, offset, size);

        std::string table_name = "test_table";

        // Create a vector of tensors for row_ids, col_ids, and gains
        std::vector<tensorflow::Output> row_ids_outputs;
        std::vector<tensorflow::Output> col_ids_outputs;
        std::vector<tensorflow::Output> gains_outputs;

        for (size_t i = 0; i < row_ids_list.size(); i++) {
            row_ids_outputs.push_back(row_ids_list[i].node()->output(0));
            col_ids_outputs.push_back(col_ids_list[i].node()->output(0));
            gains_outputs.push_back(gains_list[i].node()->output(0));
        }

        // Create a stack of tensors
        auto row_ids_stack = tensorflow::ops::Stack(root, row_ids_outputs);
        auto col_ids_stack = tensorflow::ops::Stack(root, col_ids_outputs);
        auto gains_stack = tensorflow::ops::Stack(root, gains_outputs);

        // Create tensors for the attributes
        tensorflow::Tensor sample_count_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({static_cast<int64_t>(sample_count_list.size())}));
        auto sample_count_flat = sample_count_tensor.flat<int64_t>();
        for (size_t i = 0; i < sample_count_list.size(); i++) {
            sample_count_flat(i) = sample_count_list[i];
        }
        auto sample_count_const = tensorflow::ops::Const(root, sample_count_tensor);

        tensorflow::Tensor col_offset_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({static_cast<int64_t>(col_offset_list.size())}));
        auto col_offset_flat = col_offset_tensor.flat<int64_t>();
        for (size_t i = 0; i < col_offset_list.size(); i++) {
            col_offset_flat(i) = col_offset_list[i];
        }
        auto col_offset_const = tensorflow::ops::Const(root, col_offset_tensor);

        // Create scalar tensors for the other attributes
        auto num_replica_const = tensorflow::ops::Const(root, num_replica);
        auto table_vocab_size_const = tensorflow::ops::Const(root, table_vocab_size);
        auto feature_width_const = tensorflow::ops::Const(root, feature_width);
        auto num_sc_per_chip_const = tensorflow::ops::Const(root, num_sc_per_chip);
        auto max_ids_per_sparse_core_const = tensorflow::ops::Const(root, max_ids_per_sparse_core);
        auto max_unique_ids_per_sparse_core_const = tensorflow::ops::Const(root, max_unique_ids_per_sparse_core);
        auto table_name_const = tensorflow::ops::Const(root, table_name);

        // Use raw_ops to call SortListOfSparseCoreCooTensors
        auto sort_op = tensorflow::ops::Raw(
            root.WithOpName("SortListOfSparseCoreCooTensors"),
            "SortListOfSparseCoreCooTensors",
            {row_ids_stack.output, col_ids_stack.output, gains_stack.output, 
             sample_count_const, col_offset_const, num_replica_const, 
             table_vocab_size_const, feature_width_const, num_sc_per_chip_const,
             max_ids_per_sparse_core_const, max_unique_ids_per_sparse_core_const,
             table_name_const},
            {tensorflow::DT_INT32, tensorflow::DT_INT32, tensorflow::DT_FLOAT, tensorflow::DT_INT32},
            {}
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({sort_op}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
