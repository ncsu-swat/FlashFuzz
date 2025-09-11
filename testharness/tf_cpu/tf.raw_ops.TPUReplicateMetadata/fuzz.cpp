#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <cstring>
#include <vector>
#include <iostream>

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
                for (int i = 0; i < flat.size(); ++i) {
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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        int32_t num_replicas = 1;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&num_replicas, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            num_replicas = std::abs(num_replicas) % 8 + 1;
        }

        int32_t num_cores_per_replica = 1;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&num_cores_per_replica, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            num_cores_per_replica = std::abs(num_cores_per_replica) % 4 + 1;
        }

        std::string topology = "";
        if (offset < size) {
            uint8_t topology_len = data[offset] % 10;
            offset++;
            for (uint8_t i = 0; i < topology_len && offset < size; ++i) {
                topology += static_cast<char>(data[offset]);
                offset++;
            }
        }

        bool use_tpu = true;
        if (offset < size) {
            use_tpu = (data[offset] % 2) == 1;
            offset++;
        }

        std::vector<int32_t> device_assignment;
        if (offset < size) {
            uint8_t assignment_len = data[offset] % 5;
            offset++;
            for (uint8_t i = 0; i < assignment_len && offset + sizeof(int32_t) <= size; ++i) {
                int32_t device_id;
                std::memcpy(&device_id, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                device_assignment.push_back(std::abs(device_id) % 16);
            }
        }

        std::vector<int32_t> computation_shape;
        if (offset < size) {
            uint8_t shape_len = data[offset] % 4;
            offset++;
            for (uint8_t i = 0; i < shape_len && offset + sizeof(int32_t) <= size; ++i) {
                int32_t shape_dim;
                std::memcpy(&shape_dim, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                computation_shape.push_back(std::abs(shape_dim) % 8 + 1);
            }
        }

        std::vector<std::string> host_compute_core;
        if (offset < size) {
            uint8_t core_len = data[offset] % 3;
            offset++;
            for (uint8_t i = 0; i < core_len; ++i) {
                std::string core_name = "core_" + std::to_string(i);
                host_compute_core.push_back(core_name);
            }
        }

        std::vector<std::string> padding_map;
        if (offset < size) {
            uint8_t padding_len = data[offset] % 3;
            offset++;
            for (uint8_t i = 0; i < padding_len; ++i) {
                std::string padding_entry = "padding_" + std::to_string(i);
                padding_map.push_back(padding_entry);
            }
        }

        std::string step_marker_location = "STEP_MARK_AT_ENTRY";
        if (offset < size) {
            uint8_t marker_choice = data[offset] % 3;
            offset++;
            switch (marker_choice) {
                case 0:
                    step_marker_location = "STEP_MARK_AT_ENTRY";
                    break;
                case 1:
                    step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP";
                    break;
                case 2:
                    step_marker_location = "STEP_MARK_AT_SECOND_LEVEL_WHILE_LOOP";
                    break;
            }
        }

        bool allow_soft_placement = false;
        if (offset < size) {
            allow_soft_placement = (data[offset] % 2) == 1;
            offset++;
        }

        bool use_spmd_for_xla_partitioning = false;
        if (offset < size) {
            use_spmd_for_xla_partitioning = (data[offset] % 2) == 1;
            offset++;
        }

        std::string tpu_compile_options_proto = "";
        if (offset < size) {
            uint8_t proto_len = data[offset] % 10;
            offset++;
            for (uint8_t i = 0; i < proto_len && offset < size; ++i) {
                tpu_compile_options_proto += static_cast<char>(data[offset]);
                offset++;
            }
        }

        std::cout << "num_replicas: " << num_replicas << std::endl;
        std::cout << "num_cores_per_replica: " << num_cores_per_replica << std::endl;
        std::cout << "topology: " << topology << std::endl;
        std::cout << "use_tpu: " << use_tpu << std::endl;
        std::cout << "device_assignment size: " << device_assignment.size() << std::endl;
        std::cout << "computation_shape size: " << computation_shape.size() << std::endl;
        std::cout << "host_compute_core size: " << host_compute_core.size() << std::endl;
        std::cout << "padding_map size: " << padding_map.size() << std::endl;
        std::cout << "step_marker_location: " << step_marker_location << std::endl;
        std::cout << "allow_soft_placement: " << allow_soft_placement << std::endl;
        std::cout << "use_spmd_for_xla_partitioning: " << use_spmd_for_xla_partitioning << std::endl;
        std::cout << "tpu_compile_options_proto: " << tpu_compile_options_proto << std::endl;

        // Use raw_ops API instead of the missing TPU ops header
        auto op_attrs = tensorflow::ops::Attrs()
            .Set("num_replicas", num_replicas)
            .Set("num_cores_per_replica", num_cores_per_replica)
            .Set("topology", topology)
            .Set("use_tpu", use_tpu)
            .Set("device_assignment", device_assignment)
            .Set("computation_shape", computation_shape)
            .Set("host_compute_core", host_compute_core)
            .Set("padding_map", padding_map)
            .Set("step_marker_location", step_marker_location)
            .Set("allow_soft_placement", allow_soft_placement)
            .Set("use_spmd_for_xla_partitioning", use_spmd_for_xla_partitioning)
            .Set("tpu_compile_options_proto", tpu_compile_options_proto);

        auto op = tensorflow::Operation(root.WithOpName("TPUReplicateMetadata").WithControlDependencies({}).node());
        
        tensorflow::ClientSession session(root);
        tensorflow::Status status = session.Run({}, {}, {op}, nullptr);
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
