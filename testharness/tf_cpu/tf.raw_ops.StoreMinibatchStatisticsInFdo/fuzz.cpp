#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
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

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 3) {
        case 0:
            dtype = tensorflow::DT_STRING;
            break;
        case 1:
            dtype = tensorflow::DT_INT32;
            break;
        case 2:
            dtype = tensorflow::DT_INT32;
            break;
        default:
            dtype = tensorflow::DT_INT32;
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
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_STRING:
            fillStringTensor(tensor, data, offset, total_size);
            break;
        default:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
    }
}

// Register the StoreMinibatchStatisticsInFdo operation
REGISTER_OP("StoreMinibatchStatisticsInFdo")
    .Input("program_key: string")
    .Input("max_ids: int32")
    .Input("max_uniques: int32")
    .Attr("sample_count: int = 1")
    .Attr("num_replica: int = 1")
    .Attr("feature_width: int = 1")
    .Attr("num_sc_per_chip: int = 1")
    .Attr("table_name: string = ''")
    .Attr("mini_batch_splits: string = ''");

// Define a wrapper for the StoreMinibatchStatisticsInFdo operation
namespace tensorflow {
namespace ops {

class StoreMinibatchStatisticsInFdo {
 public:
  StoreMinibatchStatisticsInFdo(const ::tensorflow::Scope& scope,
                               ::tensorflow::Input program_key,
                               ::tensorflow::Input max_ids,
                               ::tensorflow::Input max_uniques) {
    if (!scope.ok()) return;
    auto _program_key = ::tensorflow::ops::AsNodeOut(scope, program_key);
    if (!scope.ok()) return;
    auto _max_ids = ::tensorflow::ops::AsNodeOut(scope, max_ids);
    if (!scope.ok()) return;
    auto _max_uniques = ::tensorflow::ops::AsNodeOut(scope, max_uniques);
    if (!scope.ok()) return;
    
    ::tensorflow::Node* ret;
    const auto unique_name = scope.GetUniqueNameForOp("StoreMinibatchStatisticsInFdo");
    auto builder = ::tensorflow::NodeBuilder(unique_name, "StoreMinibatchStatisticsInFdo")
                       .Input(_program_key)
                       .Input(_max_ids)
                       .Input(_max_uniques);
    
    if (sample_count_ != 1) builder.Attr("sample_count", sample_count_);
    if (num_replica_ != 1) builder.Attr("num_replica", num_replica_);
    if (feature_width_ != 1) builder.Attr("feature_width", feature_width_);
    if (num_sc_per_chip_ != 1) builder.Attr("num_sc_per_chip", num_sc_per_chip_);
    if (table_name_ != "") builder.Attr("table_name", table_name_);
    if (mini_batch_splits_ != "") builder.Attr("mini_batch_splits", mini_batch_splits_);
    
    scope.UpdateBuilder(&builder);
    scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
    if (!scope.ok()) return;
    scope.UpdateStatus(scope.DoShapeInference(ret));
    this->operation = ::tensorflow::Operation(ret);
  }
  
  StoreMinibatchStatisticsInFdo& SampleCount(int64_t sample_count) {
    sample_count_ = sample_count;
    return *this;
  }
  
  StoreMinibatchStatisticsInFdo& NumReplica(int64_t num_replica) {
    num_replica_ = num_replica;
    return *this;
  }
  
  StoreMinibatchStatisticsInFdo& FeatureWidth(int64_t feature_width) {
    feature_width_ = feature_width;
    return *this;
  }
  
  StoreMinibatchStatisticsInFdo& NumScPerChip(int64_t num_sc_per_chip) {
    num_sc_per_chip_ = num_sc_per_chip;
    return *this;
  }
  
  StoreMinibatchStatisticsInFdo& TableName(const std::string& table_name) {
    table_name_ = table_name;
    return *this;
  }
  
  StoreMinibatchStatisticsInFdo& MiniBatchSplits(const std::string& mini_batch_splits) {
    mini_batch_splits_ = mini_batch_splits;
    return *this;
  }
  
  operator ::tensorflow::Operation() const { return operation; }
  
 private:
  ::tensorflow::Operation operation;
  int64_t sample_count_ = 1;
  int64_t num_replica_ = 1;
  int64_t feature_width_ = 1;
  int64_t num_sc_per_chip_ = 1;
  std::string table_name_ = "";
  std::string mini_batch_splits_ = "";
};

}  // namespace ops
}  // namespace tensorflow

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType program_key_dtype = tensorflow::DT_STRING;
        uint8_t program_key_rank = parseRank(data[offset++]);
        std::vector<int64_t> program_key_shape = parseShape(data, offset, size, program_key_rank);
        tensorflow::Tensor program_key_tensor(program_key_dtype, tensorflow::TensorShape(program_key_shape));
        fillTensorWithDataByType(program_key_tensor, program_key_dtype, data, offset, size);
        auto program_key = tensorflow::ops::Const(root, program_key_tensor);

        tensorflow::DataType max_ids_dtype = tensorflow::DT_INT32;
        uint8_t max_ids_rank = parseRank(data[offset++]);
        std::vector<int64_t> max_ids_shape = parseShape(data, offset, size, max_ids_rank);
        tensorflow::Tensor max_ids_tensor(max_ids_dtype, tensorflow::TensorShape(max_ids_shape));
        fillTensorWithDataByType(max_ids_tensor, max_ids_dtype, data, offset, size);
        auto max_ids = tensorflow::ops::Const(root, max_ids_tensor);

        tensorflow::DataType max_uniques_dtype = tensorflow::DT_INT32;
        uint8_t max_uniques_rank = parseRank(data[offset++]);
        std::vector<int64_t> max_uniques_shape = parseShape(data, offset, size, max_uniques_rank);
        tensorflow::Tensor max_uniques_tensor(max_uniques_dtype, tensorflow::TensorShape(max_uniques_shape));
        fillTensorWithDataByType(max_uniques_tensor, max_uniques_dtype, data, offset, size);
        auto max_uniques = tensorflow::ops::Const(root, max_uniques_tensor);

        int sample_count = 1;
        if (offset < size) {
            sample_count = std::max(1, static_cast<int>(data[offset++] % 10 + 1));
        }

        int num_replica = 1;
        if (offset < size) {
            num_replica = std::max(1, static_cast<int>(data[offset++] % 10 + 1));
        }

        int feature_width = 1;
        if (offset < size) {
            feature_width = std::max(1, static_cast<int>(data[offset++] % 10 + 1));
        }

        int num_sc_per_chip = 1;
        if (offset < size) {
            num_sc_per_chip = std::max(1, static_cast<int>(data[offset++] % 10 + 1));
        }

        std::string table_name = "test_table";
        if (offset < size) {
            size_t name_len = std::min(static_cast<size_t>(10), size - offset);
            table_name = std::string(reinterpret_cast<const char*>(data + offset), name_len);
            offset += name_len;
        }

        std::string mini_batch_splits = "test_splits";
        if (offset < size) {
            size_t splits_len = std::min(static_cast<size_t>(10), size - offset);
            mini_batch_splits = std::string(reinterpret_cast<const char*>(data + offset), splits_len);
            offset += splits_len;
        }

        std::cout << "program_key shape: ";
        for (auto dim : program_key_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "max_ids shape: ";
        for (auto dim : max_ids_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "max_uniques shape: ";
        for (auto dim : max_uniques_shape) std::cout << dim << " ";
        std::cout << std::endl;

        std::cout << "sample_count: " << sample_count << std::endl;
        std::cout << "num_replica: " << num_replica << std::endl;
        std::cout << "feature_width: " << feature_width << std::endl;
        std::cout << "num_sc_per_chip: " << num_sc_per_chip << std::endl;
        std::cout << "table_name: " << table_name << std::endl;
        std::cout << "mini_batch_splits: " << mini_batch_splits << std::endl;

        auto store_op = tensorflow::ops::StoreMinibatchStatisticsInFdo(
            root,
            program_key,
            max_ids,
            max_uniques)
            .SampleCount(sample_count)
            .NumReplica(num_replica)
            .FeatureWidth(feature_width)
            .NumScPerChip(num_sc_per_chip)
            .TableName(table_name)
            .MiniBatchSplits(mini_batch_splits);

        tensorflow::ClientSession session(root);
        tensorflow::Status status = session.Run({store_op}, nullptr);
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
