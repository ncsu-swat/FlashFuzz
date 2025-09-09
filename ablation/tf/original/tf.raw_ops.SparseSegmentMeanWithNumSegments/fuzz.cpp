#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/sparse_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/framework/types.pb.h>
#include <tensorflow/core/platform/types.h>

constexpr uint8_t MIN_RANK = 1;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
        case 0:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 1:
            dtype = tensorflow::DT_HALF;
            break;
        case 2:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 3:
            dtype = tensorflow::DT_DOUBLE;
            break;
    }
    return dtype;
}

tensorflow::DataType parseIndexType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
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
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_BFLOAT16:
            fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
            break;
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType data_dtype = parseDataType(data[offset++]);
        tensorflow::DataType indices_dtype = parseIndexType(data[offset++]);
        tensorflow::DataType segment_ids_dtype = parseIndexType(data[offset++]);
        tensorflow::DataType num_segments_dtype = parseIndexType(data[offset++]);
        
        uint8_t data_rank = parseRank(data[offset++]);
        
        std::vector<int64_t> data_shape = parseShape(data, offset, size, data_rank);
        
        if (offset >= size) {
            return 0;
        }
        
        int64_t num_indices = 1 + (data[offset++] % 5);
        int64_t num_segments_val = 1 + (data[offset++] % 3);
        
        tensorflow::TensorShape data_tensor_shape;
        for (int64_t dim : data_shape) {
            data_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor data_tensor(data_dtype, data_tensor_shape);
        fillTensorWithDataByType(data_tensor, data_dtype, data, offset, size);
        
        tensorflow::TensorShape indices_shape({num_indices});
        tensorflow::Tensor indices_tensor(indices_dtype, indices_shape);
        fillTensorWithDataByType(indices_tensor, indices_dtype, data, offset, size);
        
        tensorflow::TensorShape segment_ids_shape({num_indices});
        tensorflow::Tensor segment_ids_tensor(segment_ids_dtype, segment_ids_shape);
        fillTensorWithDataByType(segment_ids_tensor, segment_ids_dtype, data, offset, size);
        
        if (segment_ids_dtype == tensorflow::DT_INT32) {
            auto flat = segment_ids_tensor.flat<int32_t>();
            for (int i = 0; i < flat.size(); ++i) {
                flat(i) = std::abs(flat(i)) % num_segments_val;
            }
        } else {
            auto flat = segment_ids_tensor.flat<int64_t>();
            for (int i = 0; i < flat.size(); ++i) {
                flat(i) = std::abs(flat(i)) % num_segments_val;
            }
        }
        
        if (indices_dtype == tensorflow::DT_INT32) {
            auto flat = indices_tensor.flat<int32_t>();
            int64_t max_index = data_tensor_shape.dim_size(0);
            for (int i = 0; i < flat.size(); ++i) {
                flat(i) = std::abs(flat(i)) % max_index;
            }
        } else {
            auto flat = indices_tensor.flat<int64_t>();
            int64_t max_index = data_tensor_shape.dim_size(0);
            for (int i = 0; i < flat.size(); ++i) {
                flat(i) = std::abs(flat(i)) % max_index;
            }
        }
        
        tensorflow::TensorShape num_segments_shape({});
        tensorflow::Tensor num_segments_tensor(num_segments_dtype, num_segments_shape);
        if (num_segments_dtype == tensorflow::DT_INT32) {
            num_segments_tensor.scalar<int32_t>()() = static_cast<int32_t>(num_segments_val);
        } else {
            num_segments_tensor.scalar<int64_t>()() = num_segments_val;
        }
        
        bool sparse_gradient = (data[offset % size] % 2) == 1;
        
        std::cout << "Data tensor shape: ";
        for (int i = 0; i < data_tensor_shape.dims(); ++i) {
            std::cout << data_tensor_shape.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Indices shape: " << num_indices << std::endl;
        std::cout << "Segment IDs shape: " << num_indices << std::endl;
        std::cout << "Num segments: " << num_segments_val << std::endl;
        std::cout << "Sparse gradient: " << sparse_gradient << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto data_placeholder = tensorflow::ops::Placeholder(root, data_dtype);
        auto indices_placeholder = tensorflow::ops::Placeholder(root, indices_dtype);
        auto segment_ids_placeholder = tensorflow::ops::Placeholder(root, segment_ids_dtype);
        auto num_segments_placeholder = tensorflow::ops::Placeholder(root, num_segments_dtype);
        
        auto sparse_segment_mean = tensorflow::ops::SparseSegmentMeanWithNumSegments(
            root, data_placeholder, indices_placeholder, segment_ids_placeholder, 
            num_segments_placeholder, 
            tensorflow::ops::SparseSegmentMeanWithNumSegments::Attrs().SparseGradient(sparse_gradient));
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({
            {data_placeholder, data_tensor},
            {indices_placeholder, indices_tensor},
            {segment_ids_placeholder, segment_ids_tensor},
            {num_segments_placeholder, num_segments_tensor}
        }, {sparse_segment_mean}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "Operation completed successfully" << std::endl;
            std::cout << "Output shape: ";
            for (int i = 0; i < outputs[0].shape().dims(); ++i) {
                std::cout << outputs[0].shape().dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}