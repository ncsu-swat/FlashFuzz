#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/dataset_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype; 
    switch (selector % 21) {  
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 2:
            dtype = tensorflow::DT_INT32;
            break;
        case 3:
            dtype = tensorflow::DT_UINT8;
            break;
        case 4:
            dtype = tensorflow::DT_INT16;
            break;
        case 5:
            dtype = tensorflow::DT_INT8;
            break;
        case 6:
            dtype = tensorflow::DT_STRING;
            break;
        case 7:
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 8:
            dtype = tensorflow::DT_INT64;
            break;
        case 9:
            dtype = tensorflow::DT_BOOL;
            break;
        case 10:
            dtype = tensorflow::DT_QINT8;
            break;
        case 11:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 12:
            dtype = tensorflow::DT_QINT32;
            break;
        case 13:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 14:
            dtype = tensorflow::DT_QINT16;
            break;
        case 15:
            dtype = tensorflow::DT_QUINT16;
            break;
        case 16:
            dtype = tensorflow::DT_UINT16;
            break;
        case 17:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 18:
            dtype = tensorflow::DT_HALF;
            break;
        case 19:
            dtype = tensorflow::DT_UINT32;
            break;
        case 20:
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

void fillStringTensor(tensorflow::Tensor& tensor, const uint8_t* data,
                      size_t& offset, size_t total_size) {
    auto flat = tensor.flat<tensorflow::tstring>();
    const size_t num_elements = flat.size();

    for (size_t i = 0; i < num_elements; ++i) {
        if (offset < total_size) {
            uint8_t str_len = data[offset] % 32;
            offset++;
            
            std::string str;
            for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                str += static_cast<char>(data[offset]);
                offset++;
            }
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
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::Tensor dataset_id_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        fillStringTensor(dataset_id_tensor, data, offset, size);

        tensorflow::Tensor processing_mode_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        fillStringTensor(processing_mode_tensor, data, offset, size);

        tensorflow::Tensor address_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        fillStringTensor(address_tensor, data, offset, size);

        tensorflow::Tensor protocol_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        fillStringTensor(protocol_tensor, data, offset, size);

        tensorflow::Tensor job_name_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        fillStringTensor(job_name_tensor, data, offset, size);

        tensorflow::Tensor consumer_index_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        fillTensorWithDataByType(consumer_index_tensor, tensorflow::DT_INT64, data, offset, size);

        tensorflow::Tensor num_consumers_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        fillTensorWithDataByType(num_consumers_tensor, tensorflow::DT_INT64, data, offset, size);

        tensorflow::Tensor max_outstanding_requests_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        fillTensorWithDataByType(max_outstanding_requests_tensor, tensorflow::DT_INT64, data, offset, size);

        // Create a variable for iteration counter
        auto iteration_counter = tensorflow::ops::Variable(root.WithOpName("iteration_counter"), 
                                                          tensorflow::TensorShape({}), 
                                                          tensorflow::DT_INT64);
        auto iteration_counter_init = tensorflow::ops::Assign(root, iteration_counter, 
                                                             tensorflow::ops::Const(root, int64_t(0)));

        std::vector<tensorflow::DataType> output_types;
        if (offset < size) {
            uint8_t num_types = (data[offset] % 3) + 1;
            offset++;
            for (uint8_t i = 0; i < num_types && offset < size; ++i) {
                output_types.push_back(parseDataType(data[offset]));
                offset++;
            }
        } else {
            output_types.push_back(tensorflow::DT_FLOAT);
        }

        std::vector<tensorflow::PartialTensorShape> output_shapes;
        for (size_t i = 0; i < output_types.size() && offset < size; ++i) {
            uint8_t rank = parseRank(data[offset]);
            offset++;
            auto shape = parseShape(data, offset, size, rank);
            output_shapes.push_back(tensorflow::PartialTensorShape(shape));
        }
        if (output_shapes.empty()) {
            output_shapes.push_back(tensorflow::PartialTensorShape({1}));
        }

        auto uncompress_fn = tensorflow::ops::NoOp(root);

        int64_t task_refresh_interval_hint_ms = -1;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&task_refresh_interval_hint_ms, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }

        std::string data_transfer_protocol = "";
        std::string target_workers = "AUTO";
        bool uncompress = false;
        std::string cross_trainer_cache_options = "";

        if (offset < size) {
            uncompress = (data[offset] % 2) == 1;
            offset++;
        }

        auto dataset_id_const = tensorflow::ops::Const(root, dataset_id_tensor);
        auto processing_mode_const = tensorflow::ops::Const(root, processing_mode_tensor);
        auto address_const = tensorflow::ops::Const(root, address_tensor);
        auto protocol_const = tensorflow::ops::Const(root, protocol_tensor);
        auto job_name_const = tensorflow::ops::Const(root, job_name_tensor);
        auto consumer_index_const = tensorflow::ops::Const(root, consumer_index_tensor);
        auto num_consumers_const = tensorflow::ops::Const(root, num_consumers_tensor);
        auto max_outstanding_requests_const = tensorflow::ops::Const(root, max_outstanding_requests_tensor);

        // Use raw_ops namespace for DataServiceDatasetV4
        auto dataset = tensorflow::ops::DataServiceDataset(
            root,
            dataset_id_const,
            processing_mode_const,
            address_const,
            protocol_const,
            job_name_const,
            consumer_index_const,
            num_consumers_const,
            max_outstanding_requests_const,
            iteration_counter,
            output_types,
            output_shapes,
            uncompress_fn,
            tensorflow::ops::DataServiceDataset::Attrs()
                .TaskRefreshIntervalHintMs(task_refresh_interval_hint_ms)
                .DataTransferProtocol(data_transfer_protocol)
                .TargetWorkers(target_workers)
                .Uncompress(uncompress)
        );

        tensorflow::ClientSession session(root);
        TF_CHECK_OK(session.Run({iteration_counter_init}, nullptr));

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}