#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/common_runtime/direct_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/core/framework/node_def_util.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/util/sparse/sparse_tensor.h>

constexpr uint8_t MIN_RANK = 1;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 17) {
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
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 7:
            dtype = tensorflow::DT_INT64;
            break;
        case 8:
            dtype = tensorflow::DT_QINT8;
            break;
        case 9:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 10:
            dtype = tensorflow::DT_QINT32;
            break;
        case 11:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 12:
            dtype = tensorflow::DT_QINT16;
            break;
        case 13:
            dtype = tensorflow::DT_QUINT16;
            break;
        case 14:
            dtype = tensorflow::DT_UINT16;
            break;
        case 15:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 16:
            dtype = tensorflow::DT_HALF;
            break;
        default:
            dtype = tensorflow::DT_FLOAT;
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
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType values_dtype = parseDataType(data[offset++]);
        uint8_t sparse_rank = parseRank(data[offset++]);
        
        std::vector<int64_t> sparse_shape = parseShape(data, offset, size, sparse_rank);
        
        if (offset >= size) return 0;
        
        uint8_t num_sparse_elements_byte = data[offset++];
        int64_t num_sparse_elements = 1 + (num_sparse_elements_byte % 10);
        
        tensorflow::TensorShape indices_shape({num_sparse_elements, sparse_rank});
        tensorflow::Tensor input_indices(tensorflow::DT_INT64, indices_shape);
        fillTensorWithData<int64_t>(input_indices, data, offset, size);
        
        auto indices_matrix = input_indices.matrix<int64_t>();
        for (int i = 0; i < num_sparse_elements; ++i) {
            for (int j = 0; j < sparse_rank; ++j) {
                int64_t idx = indices_matrix(i, j);
                if (j < sparse_shape.size()) {
                    indices_matrix(i, j) = std::abs(idx) % sparse_shape[j];
                }
            }
        }
        
        tensorflow::TensorShape values_shape({num_sparse_elements});
        tensorflow::Tensor input_values(values_dtype, values_shape);
        fillTensorWithDataByType(input_values, values_dtype, data, offset, size);
        
        tensorflow::TensorShape shape_tensor_shape({sparse_rank});
        tensorflow::Tensor input_shape(tensorflow::DT_INT64, shape_tensor_shape);
        auto shape_flat = input_shape.flat<int64_t>();
        for (int i = 0; i < sparse_rank; ++i) {
            shape_flat(i) = sparse_shape[i];
        }
        
        if (offset >= size) return 0;
        uint8_t num_reduction_axes_byte = data[offset++];
        int32_t num_reduction_axes = 1 + (num_reduction_axes_byte % sparse_rank);
        
        tensorflow::TensorShape reduction_axes_shape({num_reduction_axes});
        tensorflow::Tensor reduction_axes(tensorflow::DT_INT32, reduction_axes_shape);
        auto axes_flat = reduction_axes.flat<int32_t>();
        for (int i = 0; i < num_reduction_axes; ++i) {
            if (offset < size) {
                int32_t axis = static_cast<int32_t>(data[offset++] % sparse_rank);
                axes_flat(i) = axis;
            } else {
                axes_flat(i) = i % sparse_rank;
            }
        }
        
        bool keep_dims = (offset < size) ? (data[offset++] % 2 == 1) : false;
        
        std::cout << "Input indices shape: " << input_indices.shape().DebugString() << std::endl;
        std::cout << "Input values shape: " << input_values.shape().DebugString() << std::endl;
        std::cout << "Input shape: " << input_shape.shape().DebugString() << std::endl;
        std::cout << "Reduction axes shape: " << reduction_axes.shape().DebugString() << std::endl;
        std::cout << "Keep dims: " << keep_dims << std::endl;
        
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        node_def->set_name("sparse_reduce_sum_sparse");
        node_def->set_op("SparseReduceSumSparse");
        
        node_def->add_input("input_indices");
        node_def->add_input("input_values");
        node_def->add_input("input_shape");
        node_def->add_input("reduction_axes");
        
        tensorflow::AttrValue keep_dims_attr;
        keep_dims_attr.set_b(keep_dims);
        (*node_def->mutable_attr())["keep_dims"] = keep_dims_attr;
        
        tensorflow::AttrValue dtype_attr;
        dtype_attr.set_type(values_dtype);
        (*node_def->mutable_attr())["T"] = dtype_attr;
        
        tensorflow::NodeDef* indices_node = graph_def.add_node();
        indices_node->set_name("input_indices");
        indices_node->set_op("Placeholder");
        tensorflow::AttrValue indices_dtype_attr;
        indices_dtype_attr.set_type(tensorflow::DT_INT64);
        (*indices_node->mutable_attr())["dtype"] = indices_dtype_attr;
        
        tensorflow::NodeDef* values_node = graph_def.add_node();
        values_node->set_name("input_values");
        values_node->set_op("Placeholder");
        tensorflow::AttrValue values_dtype_attr;
        values_dtype_attr.set_type(values_dtype);
        (*values_node->mutable_attr())["dtype"] = values_dtype_attr;
        
        tensorflow::NodeDef* shape_node = graph_def.add_node();
        shape_node->set_name("input_shape");
        shape_node->set_op("Placeholder");
        tensorflow::AttrValue shape_dtype_attr;
        shape_dtype_attr.set_type(tensorflow::DT_INT64);
        (*shape_node->mutable_attr())["dtype"] = shape_dtype_attr;
        
        tensorflow::NodeDef* axes_node = graph_def.add_node();
        axes_node->set_name("reduction_axes");
        axes_node->set_op("Placeholder");
        tensorflow::AttrValue axes_dtype_attr;
        axes_dtype_attr.set_type(tensorflow::DT_INT32);
        (*axes_node->mutable_attr())["dtype"] = axes_dtype_attr;
        
        tensorflow::SessionOptions session_options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(session_options));
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            std::cout << "Failed to create session: " << status.ToString() << std::endl;
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"input_indices", input_indices},
            {"input_values", input_values},
            {"input_shape", input_shape},
            {"reduction_axes", reduction_axes}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {
            "sparse_reduce_sum_sparse:0",
            "sparse_reduce_sum_sparse:1", 
            "sparse_reduce_sum_sparse:2"
        };
        
        status = session->Run(inputs, output_names, {}, &outputs);
        if (!status.ok()) {
            std::cout << "Failed to run session: " << status.ToString() << std::endl;
            return 0;
        }
        
        if (outputs.size() == 3) {
            std::cout << "Output indices shape: " << outputs[0].shape().DebugString() << std::endl;
            std::cout << "Output values shape: " << outputs[1].shape().DebugString() << std::endl;
            std::cout << "Output shape: " << outputs[2].shape().DebugString() << std::endl;
        }
        
        session->Close();
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}