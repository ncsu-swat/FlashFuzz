#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/dataset_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include <cstring>
#include <vector>
#include <iostream>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& msg, const uint8_t* data, size_t size) {
    std::cerr << msg << std::endl;
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
            {
                auto flat = tensor.flat<tensorflow::tstring>();
                const size_t num_elements = flat.size();
                for (size_t i = 0; i < num_elements; ++i) {
                    if (offset < total_size) {
                        uint8_t str_len = data[offset] % 10 + 1;
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
        tensorflow::Tensor dataset_id_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        if (offset + sizeof(int64_t) <= size) {
            int64_t dataset_id_val;
            std::memcpy(&dataset_id_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dataset_id_tensor.scalar<int64_t>()() = std::abs(dataset_id_val) % 1000;
        } else {
            dataset_id_tensor.scalar<int64_t>()() = 1;
        }

        tensorflow::Tensor processing_mode_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        if (offset < size) {
            uint8_t mode_selector = data[offset++];
            std::string mode = (mode_selector % 2 == 0) ? "parallel_epochs" : "distributed_epoch";
            processing_mode_tensor.scalar<tensorflow::tstring>()() = mode;
        } else {
            processing_mode_tensor.scalar<tensorflow::tstring>()() = "parallel_epochs";
        }

        tensorflow::Tensor address_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        if (offset < size) {
            uint8_t addr_len = data[offset] % 20 + 10;
            offset++;
            std::string address = "localhost:";
            for (uint8_t i = 0; i < 4 && offset < size; ++i) {
                address += std::to_string(data[offset] % 10);
                offset++;
            }
            address_tensor.scalar<tensorflow::tstring>()() = address;
        } else {
            address_tensor.scalar<tensorflow::tstring>()() = "localhost:5000";
        }

        tensorflow::Tensor protocol_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        if (offset < size) {
            uint8_t proto_selector = data[offset++];
            std::string protocol = (proto_selector % 2 == 0) ? "grpc" : "grpc+loas";
            protocol_tensor.scalar<tensorflow::tstring>()() = protocol;
        } else {
            protocol_tensor.scalar<tensorflow::tstring>()() = "grpc";
        }

        tensorflow::Tensor job_name_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        if (offset < size) {
            uint8_t job_len = data[offset] % 10 + 5;
            offset++;
            std::string job_name = "job_";
            for (uint8_t i = 0; i < job_len && offset < size; ++i) {
                job_name += static_cast<char>('a' + (data[offset] % 26));
                offset++;
            }
            job_name_tensor.scalar<tensorflow::tstring>()() = job_name;
        } else {
            job_name_tensor.scalar<tensorflow::tstring>()() = "test_job";
        }

        tensorflow::Tensor max_outstanding_requests_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        if (offset + sizeof(int64_t) <= size) {
            int64_t max_requests;
            std::memcpy(&max_requests, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            max_outstanding_requests_tensor.scalar<int64_t>()() = std::abs(max_requests) % 100 + 1;
        } else {
            max_outstanding_requests_tensor.scalar<int64_t>()() = 10;
        }

        // Create a variable handle for iteration counter
        auto iteration_counter = tensorflow::ops::Variable(root, {}, tensorflow::DT_INT64);

        uint8_t num_output_types = 1;
        if (offset < size) {
            num_output_types = (data[offset++] % 3) + 1;
        }

        std::vector<tensorflow::DataType> output_types;
        std::vector<tensorflow::PartialTensorShape> output_shapes;
        
        for (uint8_t i = 0; i < num_output_types; ++i) {
            tensorflow::DataType dtype = tensorflow::DT_FLOAT;
            if (offset < size) {
                dtype = parseDataType(data[offset++]);
            }
            output_types.push_back(dtype);

            uint8_t rank = 1;
            if (offset < size) {
                rank = parseRank(data[offset++]);
            }
            
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            tensorflow::PartialTensorShape tensor_shape;
            tensorflow::PartialTensorShape::MakePartialShape(shape.data(), shape.size(), &tensor_shape);
            output_shapes.push_back(tensor_shape);
        }

        int64_t task_refresh_interval = -1;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&task_refresh_interval, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            task_refresh_interval = task_refresh_interval % 10000;
        }

        std::string data_transfer_protocol = "";
        if (offset < size) {
            uint8_t proto_selector = data[offset++];
            if (proto_selector % 3 == 0) {
                data_transfer_protocol = "grpc";
            } else if (proto_selector % 3 == 1) {
                data_transfer_protocol = "grpc+loas";
            }
        }

        std::string target_workers = "AUTO";
        if (offset < size) {
            uint8_t worker_selector = data[offset++];
            if (worker_selector % 3 == 0) {
                target_workers = "AUTO";
            } else if (worker_selector % 3 == 1) {
                target_workers = "ANY";
            } else {
                target_workers = "LOCAL";
            }
        }

        std::string cross_trainer_cache_options = "";

        auto dataset_id_op = tensorflow::ops::Const(root, dataset_id_tensor);
        auto processing_mode_op = tensorflow::ops::Const(root, processing_mode_tensor);
        auto address_op = tensorflow::ops::Const(root, address_tensor);
        auto protocol_op = tensorflow::ops::Const(root, protocol_tensor);
        auto job_name_op = tensorflow::ops::Const(root, job_name_tensor);
        auto max_outstanding_requests_op = tensorflow::ops::Const(root, max_outstanding_requests_tensor);

        // Create DataServiceDataset operation using raw_ops
        tensorflow::ops::DataServiceDatasetV3 data_service_dataset(
            root,
            dataset_id_op,
            processing_mode_op,
            address_op,
            protocol_op,
            job_name_op,
            max_outstanding_requests_op,
            iteration_counter,
            output_types,
            output_shapes,
            task_refresh_interval,
            data_transfer_protocol,
            target_workers,
            cross_trainer_cache_options
        );

        tensorflow::ClientSession session(root);

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}