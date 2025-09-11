#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/random_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
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

tensorflow::DataType parseIndexDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
        case 0:
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_UINT32;
            break;
        case 2:
            dtype = tensorflow::DT_INT64;
            break;
        case 3:
            dtype = tensorflow::DT_UINT64;
            break;
    }
    return dtype;
}

tensorflow::DataType parseSeedDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
        case 0:
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_UINT32;
            break;
        case 2:
            dtype = tensorflow::DT_INT64;
            break;
        case 3:
            dtype = tensorflow::DT_UINT64;
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

void fillTensorWithDataByType(tensorflow::Tensor& tensor,
                              tensorflow::DataType dtype, const uint8_t* data,
                              size_t& offset, size_t total_size) {
    switch (dtype) {
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT32:
            fillTensorWithData<uint32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT64:
            fillTensorWithData<uint64_t>(tensor, data, offset, total_size);
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
        tensorflow::DataType index_dtype = parseIndexDataType(data[offset++]);
        tensorflow::DataType seed_dtype = parseSeedDataType(data[offset++]);
        
        uint8_t index_rank = parseRank(data[offset++]);
        std::vector<int64_t> index_shape = parseShape(data, offset, size, index_rank);
        
        uint8_t seed_rank = data[offset++] % 2 == 0 ? 1 : 2;
        std::vector<int64_t> seed_shape;
        if (seed_rank == 1) {
            seed_shape = {3};
        } else {
            if (offset < size) {
                int64_t n = 1 + (data[offset++] % 5);
                seed_shape = {n, 3};
            } else {
                seed_shape = {2, 3};
            }
        }
        
        uint8_t max_index_rank = parseRank(data[offset++]);
        std::vector<int64_t> max_index_shape = parseShape(data, offset, size, max_index_rank);
        
        int rounds = 4;
        if (offset < size) {
            rounds = 1 + (data[offset++] % 8);
        }

        tensorflow::TensorShape index_tensor_shape(index_shape);
        tensorflow::Tensor index_tensor(index_dtype, index_tensor_shape);
        fillTensorWithDataByType(index_tensor, index_dtype, data, offset, size);

        tensorflow::TensorShape seed_tensor_shape(seed_shape);
        tensorflow::Tensor seed_tensor(seed_dtype, seed_tensor_shape);
        fillTensorWithDataByType(seed_tensor, seed_dtype, data, offset, size);

        tensorflow::TensorShape max_index_tensor_shape(max_index_shape);
        tensorflow::Tensor max_index_tensor(index_dtype, max_index_tensor_shape);
        fillTensorWithDataByType(max_index_tensor, index_dtype, data, offset, size);

        auto index_placeholder = tensorflow::ops::Placeholder(root, index_dtype);
        auto seed_placeholder = tensorflow::ops::Placeholder(root, seed_dtype);
        auto max_index_placeholder = tensorflow::ops::Placeholder(root, index_dtype);

        // Use raw_ops namespace for RandomIndexShuffle
        auto random_index_shuffle = tensorflow::ops::_RandomIndexShuffle(
            root, index_placeholder, seed_placeholder, max_index_placeholder,
            tensorflow::ops::_RandomIndexShuffle::Attrs().Rounds(rounds));

        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(
            {{index_placeholder, index_tensor},
             {seed_placeholder, seed_tensor},
             {max_index_placeholder, max_index_tensor}},
            {random_index_shuffle}, &outputs);
            
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
