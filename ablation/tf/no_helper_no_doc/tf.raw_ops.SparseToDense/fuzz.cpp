#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/framework/fake_input.h>
#include <tensorflow/core/kernels/ops_testutil.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions for output shape
        int32_t output_shape_dim = (data[offset] % 4) + 1;
        offset++;
        
        if (size < offset + output_shape_dim * 4 + 8) return 0;
        
        // Create output shape tensor
        tensorflow::Tensor output_shape(tensorflow::DT_INT32, tensorflow::TensorShape({output_shape_dim}));
        auto output_shape_flat = output_shape.flat<int32_t>();
        int64_t total_elements = 1;
        for (int i = 0; i < output_shape_dim; i++) {
            int32_t dim_size = std::max(1, static_cast<int32_t>(data[offset] % 10) + 1);
            output_shape_flat(i) = dim_size;
            total_elements *= dim_size;
            offset++;
        }
        
        // Limit total elements to prevent excessive memory usage
        if (total_elements > 10000) {
            for (int i = 0; i < output_shape_dim; i++) {
                output_shape_flat(i) = std::min(output_shape_flat(i), 10);
            }
        }
        
        // Extract number of sparse indices
        int32_t num_indices = std::min(static_cast<int32_t>(data[offset % size] % 20) + 1, 100);
        offset = (offset + 1) % size;
        
        // Create sparse indices tensor
        tensorflow::Tensor sparse_indices(tensorflow::DT_INT64, tensorflow::TensorShape({num_indices, output_shape_dim}));
        auto indices_matrix = sparse_indices.matrix<int64_t>();
        
        for (int i = 0; i < num_indices; i++) {
            for (int j = 0; j < output_shape_dim; j++) {
                int64_t idx = data[(offset + i * output_shape_dim + j) % size] % output_shape_flat(j);
                indices_matrix(i, j) = idx;
            }
        }
        offset = (offset + num_indices * output_shape_dim) % size;
        
        // Create sparse values tensor
        tensorflow::Tensor sparse_values(tensorflow::DT_FLOAT, tensorflow::TensorShape({num_indices}));
        auto values_flat = sparse_values.flat<float>();
        
        for (int i = 0; i < num_indices; i++) {
            float val = static_cast<float>(data[(offset + i) % size]) / 255.0f;
            values_flat(i) = val;
        }
        offset = (offset + num_indices) % size;
        
        // Create default value tensor
        tensorflow::Tensor default_value(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        float default_val = static_cast<float>(data[offset % size]) / 255.0f;
        default_value.scalar<float>()() = default_val;
        
        // Create a simple test using OpsTestBase
        class SparseToDenseTest : public tensorflow::OpsTestBase {};
        SparseToDenseTest test;
        
        tensorflow::NodeDef node_def;
        tensorflow::Status status = tensorflow::NodeDefBuilder("sparse_to_dense", "SparseToDense")
            .Input(tensorflow::FakeInput(tensorflow::DT_INT64))
            .Input(tensorflow::FakeInput(tensorflow::DT_INT32))
            .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))
            .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))
            .Attr("T", tensorflow::DT_FLOAT)
            .Attr("Tindices", tensorflow::DT_INT64)
            .Attr("validate_indices", true)
            .Finalize(&node_def);
            
        if (!status.ok()) {
            return 0;
        }
        
        status = test.InitOp(node_def);
        if (!status.ok()) {
            return 0;
        }
        
        test.AddInputFromArray<int64_t>(sparse_indices.shape(), sparse_indices.flat<int64_t>());
        test.AddInputFromArray<int32_t>(output_shape.shape(), output_shape.flat<int32_t>());
        test.AddInputFromArray<float>(sparse_values.shape(), sparse_values.flat<float>());
        test.AddInputFromArray<float>(default_value.shape(), default_value.flat<float>());
        
        status = test.RunOpKernel();
        if (!status.ok()) {
            return 0;
        }
        
        // Get output tensor
        tensorflow::Tensor* output = test.GetOutput(0);
        if (output != nullptr) {
            // Basic validation - check that output has expected shape
            if (output->dims() == output_shape_dim) {
                for (int i = 0; i < output_shape_dim; i++) {
                    if (output->dim_size(i) != output_shape_flat(i)) {
                        return 0;
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