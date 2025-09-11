#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/dataset_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
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
                            str += static_cast<char>(data[offset] % 128);
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
    std::cout << "Start Fuzzing" << std::endl;
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
            std::string processing_mode;
            switch (mode_selector % 3) {
                case 0: processing_mode = "parallel_epochs"; break;
                case 1: processing_mode = "distributed_epoch"; break;
                case 2: processing_mode = "parallel_epochs"; break;
            }
            processing_mode_tensor.scalar<tensorflow::tstring>()() = processing_mode;
        } else {
            processing_mode_tensor.scalar<tensorflow::tstring>()() = "parallel_epochs";
        }

        tensorflow::Tensor address_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        if (offset < size) {
            uint8_t addr_len = data[offset++] % 20 + 5;
            std::string address = "localhost:";
            for (uint8_t i = 0; i < 4 && offset < size; ++i) {
                address += std::to_string(data[offset++] % 10);
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
            uint8_t job_len = data[offset++] % 10 + 3;
            std::string job_name = "job_";
            for (uint8_t i = 0; i < job_len && offset < size; ++i) {
                job_name += static_cast<char>('a' + (data[offset++] % 26));
            }
            job_name_tensor.scalar<tensorflow::tstring>()() = job_name;
        } else {
            job_name_tensor.scalar<tensorflow::tstring>()() = "job_test";
        }

        tensorflow::Tensor consumer_index_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        if (offset + sizeof(int64_t) <= size) {
            int64_t consumer_index_val;
            std::memcpy(&consumer_index_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            consumer_index_tensor.scalar<int64_t>()() = std::abs(consumer_index_val) % 10;
        } else {
            consumer_index_tensor.scalar<int64_t>()() = 0;
        }

        tensorflow::Tensor num_consumers_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        if (offset + sizeof(int64_t) <= size) {
            int64_t num_consumers_val;
            std::memcpy(&num_consumers_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            num_consumers_tensor.scalar<int64_t>()() = std::abs(num_consumers_val) % 10 + 1;
        } else {
            num_consumers_tensor.scalar<int64_t>()() = 1;
        }

        tensorflow::Tensor max_outstanding_requests_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        if (offset + sizeof(int64_t) <= size) {
            int64_t max_outstanding_val;
            std::memcpy(&max_outstanding_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            max_outstanding_requests_tensor.scalar<int64_t>()() = std::abs(max_outstanding_val) % 100 + 1;
        } else {
            max_outstanding_requests_tensor.scalar<int64_t>()() = 10;
        }

        auto iteration_counter = tensorflow::ops::Variable(root.WithOpName("iteration_counter"), {}, tensorflow::DT_INT64);
        auto iteration_counter_init = tensorflow::ops::Assign(root.WithOpName("iteration_counter_init"), iteration_counter, tensorflow::ops::Const(root, 0));

        uint8_t num_output_types = 1;
        if (offset < size) {
            num_output_types = data[offset++] % 3 + 1;
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
            
            std::vector<int64_t> shape_dims = parseShape(data, offset, size, rank);
            tensorflow::PartialTensorShape shape(shape_dims);
            output_shapes.push_back(shape);
        }

        int64_t task_refresh_interval_hint_ms = -1;
        if (offset + sizeof(int64_t) <= size) {
            int64_t refresh_val;
            std::memcpy(&refresh_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            task_refresh_interval_hint_ms = std::abs(refresh_val) % 10000;
        }

        std::string data_transfer_protocol = "";
        if (offset < size) {
            uint8_t proto_selector = data[offset++];
            if (proto_selector % 2 == 1) {
                data_transfer_protocol = "grpc";
            }
        }

        std::string target_workers = "AUTO";
        if (offset < size) {
            uint8_t target_selector = data[offset++];
            switch (target_selector % 3) {
                case 0: target_workers = "AUTO"; break;
                case 1: target_workers = "ANY"; break;
                case 2: target_workers = "LOCAL"; break;
            }
        }

        std::string cross_trainer_cache_options = "";

        // Create the DataServiceDatasetV2 operation using raw_ops
        auto dataset_id = tensorflow::ops::Const(root, dataset_id_tensor);
        auto processing_mode = tensorflow::ops::Const(root, processing_mode_tensor);
        auto address = tensorflow::ops::Const(root, address_tensor);
        auto protocol = tensorflow::ops::Const(root, protocol_tensor);
        auto job_name = tensorflow::ops::Const(root, job_name_tensor);
        auto consumer_index = tensorflow::ops::Const(root, consumer_index_tensor);
        auto num_consumers = tensorflow::ops::Const(root, num_consumers_tensor);
        auto max_outstanding_requests = tensorflow::ops::Const(root, max_outstanding_requests_tensor);

        tensorflow::NodeDef node_def;
        node_def.set_name("DataServiceDatasetV2");
        node_def.set_op("DataServiceDatasetV2");
        
        // Add inputs
        tensorflow::AddNodeInput("dataset_id", &node_def);
        tensorflow::AddNodeInput("processing_mode", &node_def);
        tensorflow::AddNodeInput("address", &node_def);
        tensorflow::AddNodeInput("protocol", &node_def);
        tensorflow::AddNodeInput("job_name", &node_def);
        tensorflow::AddNodeInput("consumer_index", &node_def);
        tensorflow::AddNodeInput("num_consumers", &node_def);
        tensorflow::AddNodeInput("max_outstanding_requests", &node_def);
        tensorflow::AddNodeInput("iteration_counter", &node_def);
        
        // Add attributes
        tensorflow::AddNodeAttr("output_types", output_types, &node_def);
        tensorflow::AddNodeAttr("output_shapes", output_shapes, &node_def);
        tensorflow::AddNodeAttr("task_refresh_interval_hint_ms", task_refresh_interval_hint_ms, &node_def);
        tensorflow::AddNodeAttr("data_transfer_protocol", data_transfer_protocol, &node_def);
        tensorflow::AddNodeAttr("target_workers", target_workers, &node_def);
        tensorflow::AddNodeAttr("cross_trainer_cache_options", cross_trainer_cache_options, &node_def);

        tensorflow::ClientSession session(root);
        TF_CHECK_OK(session.Run({iteration_counter_init}, nullptr));
        
    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
