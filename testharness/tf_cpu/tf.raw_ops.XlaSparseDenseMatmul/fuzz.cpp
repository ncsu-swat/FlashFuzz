#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
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
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t row_ids_rank = parseRank(data[offset++]);
        std::vector<int64_t> row_ids_shape = parseShape(data, offset, size, row_ids_rank);
        tensorflow::Tensor row_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(row_ids_shape));
        fillTensorWithDataByType(row_ids_tensor, tensorflow::DT_INT32, data, offset, size);
        auto row_ids = tensorflow::ops::Const(root, row_ids_tensor);

        uint8_t col_ids_rank = parseRank(data[offset++]);
        std::vector<int64_t> col_ids_shape = parseShape(data, offset, size, col_ids_rank);
        tensorflow::Tensor col_ids_tensor(tensorflow::DT_UINT32, tensorflow::TensorShape(col_ids_shape));
        fillTensorWithDataByType(col_ids_tensor, tensorflow::DT_UINT32, data, offset, size);
        auto col_ids = tensorflow::ops::Const(root, col_ids_tensor);

        uint8_t values_rank = parseRank(data[offset++]);
        std::vector<int64_t> values_shape = parseShape(data, offset, size, values_rank);
        tensorflow::Tensor values_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(values_shape));
        fillTensorWithDataByType(values_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto values = tensorflow::ops::Const(root, values_tensor);

        uint8_t offsets_rank = parseRank(data[offset++]);
        std::vector<int64_t> offsets_shape = parseShape(data, offset, size, offsets_rank);
        tensorflow::Tensor offsets_tensor(tensorflow::DT_UINT32, tensorflow::TensorShape(offsets_shape));
        fillTensorWithDataByType(offsets_tensor, tensorflow::DT_UINT32, data, offset, size);
        auto offsets = tensorflow::ops::Const(root, offsets_tensor);

        uint8_t embedding_table_rank = parseRank(data[offset++]);
        std::vector<int64_t> embedding_table_shape = parseShape(data, offset, size, embedding_table_rank);
        tensorflow::Tensor embedding_table_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(embedding_table_shape));
        fillTensorWithDataByType(embedding_table_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto embedding_table = tensorflow::ops::Const(root, embedding_table_tensor);

        int max_ids_per_partition = 1;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&max_ids_per_partition, data + offset, sizeof(int));
            offset += sizeof(int);
            max_ids_per_partition = std::abs(max_ids_per_partition) % 100 + 1;
        }

        int max_unique_ids_per_partition = 1;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&max_unique_ids_per_partition, data + offset, sizeof(int));
            offset += sizeof(int);
            max_unique_ids_per_partition = std::abs(max_unique_ids_per_partition) % 100 + 1;
        }

        int input_size = 1;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&input_size, data + offset, sizeof(int));
            offset += sizeof(int);
            input_size = std::abs(input_size) % 100 + 1;
        }

        std::cout << "row_ids shape: ";
        for (auto dim : row_ids_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "col_ids shape: ";
        for (auto dim : col_ids_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "values shape: ";
        for (auto dim : values_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "offsets shape: ";
        for (auto dim : offsets_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "embedding_table shape: ";
        for (auto dim : embedding_table_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "max_ids_per_partition: " << max_ids_per_partition << std::endl;
        std::cout << "max_unique_ids_per_partition: " << max_unique_ids_per_partition << std::endl;
        std::cout << "input_size: " << input_size << std::endl;

        tensorflow::Node* xla_sparse_dense_matmul_node;
        tensorflow::Status status = tensorflow::NodeBuilder("XlaSparseDenseMatmul", "XlaSparseDenseMatmul")
            .Input(row_ids.node())
            .Input(col_ids.node())
            .Input(values.node())
            .Input(offsets.node())
            .Input(embedding_table.node())
            .Attr("max_ids_per_partition", max_ids_per_partition)
            .Attr("max_unique_ids_per_partition", max_unique_ids_per_partition)
            .Attr("input_size", input_size)
            .Finalize(root.graph(), &xla_sparse_dense_matmul_node);

        if (!status.ok()) {
            std::cout << "Error creating XlaSparseDenseMatmul node: " << status.ToString() << std::endl;
            return -1;
        }

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({tensorflow::Output(xla_sparse_dense_matmul_node, 0),
                             tensorflow::Output(xla_sparse_dense_matmul_node, 1),
                             tensorflow::Output(xla_sparse_dense_matmul_node, 2),
                             tensorflow::Output(xla_sparse_dense_matmul_node, 3),
                             tensorflow::Output(xla_sparse_dense_matmul_node, 4)}, &outputs);
        
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}