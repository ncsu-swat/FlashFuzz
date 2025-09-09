#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/framework/fake_input.h>
#include <tensorflow/core/kernels/ops_testutil.h>
#include <tensorflow/core/platform/test.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions and parameters
        int64_t sparse_rows = (data[offset] % 10) + 1;
        offset++;
        int64_t sparse_cols = (data[offset] % 10) + 1;
        offset++;
        int64_t dense_cols = (data[offset] % 10) + 1;
        offset++;
        int64_t nnz = std::min((int64_t)(data[offset] % 20) + 1, sparse_rows * sparse_cols);
        offset++;
        
        bool adjoint_a = data[offset] % 2;
        offset++;
        bool adjoint_b = data[offset] % 2;
        offset++;
        
        if (offset + nnz * 16 + sparse_cols * dense_cols * 4 > size) return 0;
        
        // Create sparse tensor indices
        tensorflow::Tensor sparse_indices(tensorflow::DT_INT64, tensorflow::TensorShape({nnz, 2}));
        auto indices_matrix = sparse_indices.matrix<int64_t>();
        
        for (int64_t i = 0; i < nnz; i++) {
            indices_matrix(i, 0) = (data[offset] % sparse_rows);
            offset++;
            indices_matrix(i, 1) = (data[offset] % sparse_cols);
            offset++;
        }
        
        // Create sparse tensor values
        tensorflow::Tensor sparse_values(tensorflow::DT_FLOAT, tensorflow::TensorShape({nnz}));
        auto values_vec = sparse_values.vec<float>();
        
        for (int64_t i = 0; i < nnz; i++) {
            float val;
            if (offset + 4 <= size) {
                memcpy(&val, data + offset, sizeof(float));
                offset += 4;
            } else {
                val = 1.0f;
            }
            values_vec(i) = val;
        }
        
        // Create sparse tensor shape
        tensorflow::Tensor sparse_shape(tensorflow::DT_INT64, tensorflow::TensorShape({2}));
        auto shape_vec = sparse_shape.vec<int64_t>();
        shape_vec(0) = sparse_rows;
        shape_vec(1) = sparse_cols;
        
        // Create dense tensor
        tensorflow::Tensor dense_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({sparse_cols, dense_cols}));
        auto dense_matrix = dense_tensor.matrix<float>();
        
        for (int64_t i = 0; i < sparse_cols; i++) {
            for (int64_t j = 0; j < dense_cols; j++) {
                float val;
                if (offset + 4 <= size) {
                    memcpy(&val, data + offset, sizeof(float));
                    offset += 4;
                } else {
                    val = 1.0f;
                }
                dense_matrix(i, j) = val;
            }
        }
        
        // Create OpKernelContext for testing
        tensorflow::test::OpsTestBase test_base;
        
        // Set up the operation
        tensorflow::NodeDef node_def;
        node_def.set_name("sparse_tensor_dense_matmul");
        node_def.set_op("SparseTensorDenseMatMul");
        node_def.add_input("sparse_indices");
        node_def.add_input("sparse_values");
        node_def.add_input("sparse_shape");
        node_def.add_input("dense");
        
        auto attr = node_def.mutable_attr();
        (*attr)["T"].set_type(tensorflow::DT_FLOAT);
        (*attr)["Tindices"].set_type(tensorflow::DT_INT64);
        (*attr)["adjoint_a"].set_b(adjoint_a);
        (*attr)["adjoint_b"].set_b(adjoint_b);
        
        tensorflow::Status status = test_base.InitOp(node_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Add inputs
        test_base.AddInputFromArray<int64_t>(sparse_indices.shape(), sparse_indices.flat<int64_t>());
        test_base.AddInputFromArray<float>(sparse_values.shape(), sparse_values.flat<float>());
        test_base.AddInputFromArray<int64_t>(sparse_shape.shape(), sparse_shape.flat<int64_t>());
        test_base.AddInputFromArray<float>(dense_tensor.shape(), dense_tensor.flat<float>());
        
        // Run the operation
        status = test_base.RunOpKernel();
        if (!status.ok()) {
            return 0;
        }
        
        // Get output
        tensorflow::Tensor* output = test_base.GetOutput(0);
        if (output != nullptr) {
            // Verify output shape is reasonable
            auto output_shape = output->shape();
            if (output_shape.dims() == 2) {
                int64_t expected_rows = adjoint_a ? sparse_cols : sparse_rows;
                int64_t expected_cols = adjoint_b ? sparse_cols : dense_cols;
                
                if (output_shape.dim_size(0) == expected_rows && 
                    output_shape.dim_size(1) == expected_cols) {
                    // Access output data to ensure it's computed
                    auto output_matrix = output->matrix<float>();
                    volatile float sum = 0.0f;
                    for (int i = 0; i < output_shape.dim_size(0); i++) {
                        for (int j = 0; j < output_shape.dim_size(1); j++) {
                            sum += output_matrix(i, j);
                        }
                    }
                }
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}