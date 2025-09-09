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
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/common_runtime/direct_session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/framework/node_def_util.h>
#include <tensorflow/core/kernels/ops_util.h>

constexpr uint8_t MIN_RANK = 1;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 10) {
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
            dtype = tensorflow::DT_INT64;
            break;
        case 4:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 5:
            dtype = tensorflow::DT_HALF;
            break;
        case 6:
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 7:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 8:
            dtype = tensorflow::DT_INT16;
            break;
        case 9:
            dtype = tensorflow::DT_INT8;
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

        tensorflow::DataType indices_dtype = (data[offset] % 2 == 0) ? tensorflow::DT_INT32 : tensorflow::DT_INT64;
        offset++;

        tensorflow::DataType values_dtype = parseDataType(data[offset]);
        offset++;

        bool adjoint_a = (data[offset] % 2 == 1);
        offset++;

        bool adjoint_b = (data[offset] % 2 == 1);
        offset++;

        if (offset + 8 > size) return 0;
        
        int64_t nnz_raw;
        std::memcpy(&nnz_raw, data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        int64_t nnz = 1 + (std::abs(nnz_raw) % 10);

        if (offset + 16 > size) return 0;
        
        int64_t sparse_rows_raw, sparse_cols_raw;
        std::memcpy(&sparse_rows_raw, data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        std::memcpy(&sparse_cols_raw, data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        
        int64_t sparse_rows = 1 + (std::abs(sparse_rows_raw) % 10);
        int64_t sparse_cols = 1 + (std::abs(sparse_cols_raw) % 10);

        if (offset + 16 > size) return 0;
        
        int64_t dense_rows_raw, dense_cols_raw;
        std::memcpy(&dense_rows_raw, data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        std::memcpy(&dense_cols_raw, data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        
        int64_t dense_rows = 1 + (std::abs(dense_rows_raw) % 10);
        int64_t dense_cols = 1 + (std::abs(dense_cols_raw) % 10);

        if (!adjoint_a && sparse_cols != dense_rows) {
            sparse_cols = dense_rows;
        }
        if (adjoint_a && sparse_rows != dense_rows) {
            sparse_rows = dense_rows;
        }
        if (adjoint_b && dense_rows != dense_cols) {
            dense_rows = dense_cols;
        }

        tensorflow::Tensor a_indices(indices_dtype, tensorflow::TensorShape({nnz, 2}));
        tensorflow::Tensor a_values(values_dtype, tensorflow::TensorShape({nnz}));
        tensorflow::Tensor a_shape(tensorflow::DT_INT64, tensorflow::TensorShape({2}));
        tensorflow::Tensor b(values_dtype, tensorflow::TensorShape({dense_rows, dense_cols}));

        if (indices_dtype == tensorflow::DT_INT32) {
            auto indices_flat = a_indices.flat<int32_t>();
            for (int64_t i = 0; i < nnz; ++i) {
                if (offset + sizeof(int32_t) <= size) {
                    int32_t row;
                    std::memcpy(&row, data + offset, sizeof(int32_t));
                    offset += sizeof(int32_t);
                    indices_flat(i * 2) = std::abs(row) % sparse_rows;
                } else {
                    indices_flat(i * 2) = 0;
                }
                
                if (offset + sizeof(int32_t) <= size) {
                    int32_t col;
                    std::memcpy(&col, data + offset, sizeof(int32_t));
                    offset += sizeof(int32_t);
                    indices_flat(i * 2 + 1) = std::abs(col) % sparse_cols;
                } else {
                    indices_flat(i * 2 + 1) = 0;
                }
            }
        } else {
            auto indices_flat = a_indices.flat<int64_t>();
            for (int64_t i = 0; i < nnz; ++i) {
                if (offset + sizeof(int64_t) <= size) {
                    int64_t row;
                    std::memcpy(&row, data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    indices_flat(i * 2) = std::abs(row) % sparse_rows;
                } else {
                    indices_flat(i * 2) = 0;
                }
                
                if (offset + sizeof(int64_t) <= size) {
                    int64_t col;
                    std::memcpy(&col, data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    indices_flat(i * 2 + 1) = std::abs(col) % sparse_cols;
                } else {
                    indices_flat(i * 2 + 1) = 0;
                }
            }
        }

        fillTensorWithDataByType(a_values, values_dtype, data, offset, size);

        auto shape_flat = a_shape.flat<int64_t>();
        shape_flat(0) = sparse_rows;
        shape_flat(1) = sparse_cols;

        fillTensorWithDataByType(b, values_dtype, data, offset, size);

        std::cout << "a_indices shape: [" << a_indices.shape().dim_size(0) << ", " << a_indices.shape().dim_size(1) << "]" << std::endl;
        std::cout << "a_values shape: [" << a_values.shape().dim_size(0) << "]" << std::endl;
        std::cout << "a_shape: [" << shape_flat(0) << ", " << shape_flat(1) << "]" << std::endl;
        std::cout << "b shape: [" << b.shape().dim_size(0) << ", " << b.shape().dim_size(1) << "]" << std::endl;
        std::cout << "adjoint_a: " << adjoint_a << ", adjoint_b: " << adjoint_b << std::endl;

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto a_indices_placeholder = tensorflow::ops::Placeholder(root.WithOpName("a_indices"), indices_dtype);
        auto a_values_placeholder = tensorflow::ops::Placeholder(root.WithOpName("a_values"), values_dtype);
        auto a_shape_placeholder = tensorflow::ops::Placeholder(root.WithOpName("a_shape"), tensorflow::DT_INT64);
        auto b_placeholder = tensorflow::ops::Placeholder(root.WithOpName("b"), values_dtype);

        auto sparse_matmul = tensorflow::ops::SparseTensorDenseMatMul(
            root.WithOpName("sparse_matmul"),
            a_indices_placeholder,
            a_values_placeholder,
            a_shape_placeholder,
            b_placeholder,
            tensorflow::ops::SparseTensorDenseMatMul::Attrs().AdjointA(adjoint_a).AdjointB(adjoint_b)
        );

        tensorflow::GraphDef graph;
        TF_CHECK_OK(root.ToGraphDef(&graph));

        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        TF_CHECK_OK(session->Create(graph));

        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"a_indices", a_indices},
            {"a_values", a_values},
            {"a_shape", a_shape},
            {"b", b}
        };

        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session->Run(inputs, {"sparse_matmul"}, {}, &outputs);

        if (status.ok() && !outputs.empty()) {
            std::cout << "Output shape: [" << outputs[0].shape().dim_size(0) << ", " << outputs[0].shape().dim_size(1) << "]" << std::endl;
        } else {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}