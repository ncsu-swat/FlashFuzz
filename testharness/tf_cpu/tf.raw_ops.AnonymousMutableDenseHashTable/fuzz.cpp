#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/lookup_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/graph/node_builder.h"
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
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType selectKeyType(uint8_t selector) {
    switch (selector % 3) {
        case 0:
            return tensorflow::DT_INT32;
        case 1:
            return tensorflow::DT_INT64;
        default:
            return tensorflow::DT_STRING;
    }
}

tensorflow::DataType selectValueType(uint8_t selector, tensorflow::DataType key_dtype) {
    if (key_dtype == tensorflow::DT_INT32) {
        switch (selector % 3) {
            case 0:
                return tensorflow::DT_FLOAT;
            case 1:
                return tensorflow::DT_DOUBLE;
            default:
                return tensorflow::DT_INT32;
        }
    }

    if (key_dtype == tensorflow::DT_INT64) {
        switch (selector % 5) {
            case 0:
                return tensorflow::DT_BOOL;
            case 1:
                return tensorflow::DT_FLOAT;
            case 2:
                return tensorflow::DT_DOUBLE;
            case 3:
                return tensorflow::DT_INT32;
            default:
                return tensorflow::DT_INT64;
        }
    }

    switch (selector % 5) {
        case 0:
            return tensorflow::DT_BOOL;
        case 1:
            return tensorflow::DT_FLOAT;
        case 2:
            return tensorflow::DT_DOUBLE;
        case 3:
            return tensorflow::DT_INT32;
        default:
            return tensorflow::DT_INT64;
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

void fillStringTensor(tensorflow::Tensor& tensor, const uint8_t* data,
                      size_t& offset, size_t total_size) {
    auto flat = tensor.flat<tensorflow::tstring>();
    const size_t num_elements = flat.size();

    for (size_t i = 0; i < num_elements; ++i) {
        if (offset < total_size) {
            uint8_t str_len = data[offset] % 10 + 1;
            offset++;
            
            std::string str;
            for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                str += static_cast<char>(data[offset] % 128);
                offset++;
            }
            flat(i) = tensorflow::tstring(str);
        } else {
            flat(i) = tensorflow::tstring("default");
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
        case tensorflow::DT_STRING:
            fillStringTensor(tensor, data, offset, total_size);
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
        case tensorflow::DT_QINT8:
            fillTensorWithData<tensorflow::qint8>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QUINT8:
            fillTensorWithData<tensorflow::quint8>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT32:
            fillTensorWithData<tensorflow::qint32>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT16:
            fillTensorWithData<tensorflow::qint16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QUINT16:
            fillTensorWithData<tensorflow::quint16>(tensor, data, offset, total_size);
            break;
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType key_dtype = selectKeyType(data[offset++]);
        tensorflow::DataType value_dtype = selectValueType(data[offset++], key_dtype);

        uint8_t value_shape_rank = parseRank(data[offset++]);
        std::vector<int64_t> value_shape_dims = parseShape(data, offset, size, value_shape_rank);
        
        int64_t initial_num_buckets = 131072;
        if (offset + sizeof(int32_t) <= size) {
            int32_t bucket_bytes;
            std::memcpy(&bucket_bytes, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            bucket_bytes = std::abs(bucket_bytes) % 1048576 + 1024;
            int64_t power = 1;
            while (power < bucket_bytes) {
                power *= 2;
            }
            initial_num_buckets = power;
        }
        
        float max_load_factor = 0.8f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&max_load_factor, data + offset, sizeof(float));
            offset += sizeof(float);
            max_load_factor = std::abs(max_load_factor);
            if (max_load_factor > 1.0f || max_load_factor <= 0.0f) {
                max_load_factor = 0.8f;
            }
        }

        tensorflow::Tensor empty_key_tensor(key_dtype, tensorflow::TensorShape({}));
        tensorflow::Tensor deleted_key_tensor(key_dtype, tensorflow::TensorShape({}));
        fillTensorWithDataByType(empty_key_tensor, key_dtype, data, offset, size);
        fillTensorWithDataByType(deleted_key_tensor, key_dtype, data, offset, size);

        if (key_dtype == tensorflow::DT_INT32) {
            auto empty_val = empty_key_tensor.scalar<int32_t>()();
            auto del_val = deleted_key_tensor.scalar<int32_t>()();
            if (del_val == empty_val) {
                deleted_key_tensor.scalar<int32_t>()() = empty_val + 1;
            }
        } else if (key_dtype == tensorflow::DT_INT64) {
            auto empty_val = empty_key_tensor.scalar<int64_t>()();
            auto del_val = deleted_key_tensor.scalar<int64_t>()();
            if (del_val == empty_val) {
                deleted_key_tensor.scalar<int64_t>()() = empty_val + 1;
            }
        } else if (key_dtype == tensorflow::DT_STRING) {
            auto empty_val = empty_key_tensor.scalar<tensorflow::tstring>()();
            auto del_val = deleted_key_tensor.scalar<tensorflow::tstring>()();
            if (del_val == empty_val) {
                deleted_key_tensor.scalar<tensorflow::tstring>()() = empty_val + "_del";
            }
        }

        auto empty_key_op = tensorflow::ops::Const(root, empty_key_tensor);
        auto deleted_key_op = tensorflow::ops::Const(root, deleted_key_tensor);

        tensorflow::Node* table_node = nullptr;
        auto builder = tensorflow::NodeBuilder(
                           root.GetUniqueNameForOp("AnonymousMutableDenseHashTable"),
                           "AnonymousMutableDenseHashTable")
                           .Input(tensorflow::ops::AsNodeOut(root, empty_key_op))
                           .Input(tensorflow::ops::AsNodeOut(root, deleted_key_op))
                           .Attr("key_dtype", key_dtype)
                           .Attr("value_dtype", value_dtype)
                           .Attr("value_shape", tensorflow::PartialTensorShape(value_shape_dims))
                           .Attr("initial_num_buckets", initial_num_buckets)
                           .Attr("max_load_factor", max_load_factor);
        root.UpdateStatus(builder.Finalize(root.graph(), &table_node));
        if (!root.ok() || table_node == nullptr) {
            return -1;
        }
        tensorflow::Output hash_table(table_node, 0);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({hash_table}, &outputs);
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
