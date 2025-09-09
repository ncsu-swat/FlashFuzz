#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/fractional_avg_pool_op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/framework/fake_input.h>
#include <tensorflow/core/kernels/ops_testutil.h>
#include <tensorflow/core/platform/test.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions for input tensor
        int batch_size = (data[offset] % 4) + 1;
        offset++;
        int height = (data[offset] % 32) + 8;
        offset++;
        int width = (data[offset] % 32) + 8;
        offset++;
        int channels = (data[offset] % 8) + 1;
        offset++;
        
        // Extract pooling ratios
        float pooling_ratio_h = 1.0f + (data[offset] % 50) / 100.0f;
        offset++;
        float pooling_ratio_w = 1.0f + (data[offset] % 50) / 100.0f;
        offset++;
        
        // Extract boolean flags
        bool pseudo_random = (data[offset] % 2) == 1;
        offset++;
        bool overlapping = (data[offset] % 2) == 1;
        offset++;
        bool deterministic = (data[offset] % 2) == 1;
        offset++;
        
        // Extract seed values
        int64_t seed = 0;
        int64_t seed2 = 0;
        if (offset + 16 <= size) {
            memcpy(&seed, data + offset, 8);
            offset += 8;
            memcpy(&seed2, data + offset, 8);
            offset += 8;
        }
        
        // Create input tensor
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, 
            tensorflow::TensorShape({batch_size, height, width, channels}));
        
        auto input_flat = input_tensor.flat<float>();
        for (int i = 0; i < input_flat.size() && offset < size; ++i) {
            input_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        
        // Create test context
        tensorflow::test::OpsTestBase test_base;
        
        // Build node definition
        tensorflow::NodeDef node_def;
        tensorflow::Status status = tensorflow::NodeDefBuilder("fractional_avg_pool", "FractionalAvgPool")
            .Input(tensorflow::test::FakeInput(tensorflow::DT_FLOAT))
            .Attr("pooling_ratio", std::vector<float>{1.0f, pooling_ratio_h, pooling_ratio_w, 1.0f})
            .Attr("pseudo_random", pseudo_random)
            .Attr("overlapping", overlapping)
            .Attr("deterministic", deterministic)
            .Attr("seed", seed)
            .Attr("seed2", seed2)
            .Finalize(&node_def);
            
        if (!status.ok()) {
            return 0;
        }
        
        status = test_base.InitOp(node_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Add input
        test_base.AddInputFromArray<float>(input_tensor.shape(), input_flat.data());
        
        // Run the operation
        status = test_base.RunOpKernel();
        if (!status.ok()) {
            return 0;
        }
        
        // Get outputs
        tensorflow::Tensor* output = test_base.GetOutput(0);
        tensorflow::Tensor* row_pooling_sequence = test_base.GetOutput(1);
        tensorflow::Tensor* col_pooling_sequence = test_base.GetOutput(2);
        
        // Basic validation
        if (output && output->NumElements() > 0) {
            auto output_flat = output->flat<float>();
            for (int i = 0; i < std::min(10, static_cast<int>(output_flat.size())); ++i) {
                float val = output_flat(i);
                if (std::isnan(val) || std::isinf(val)) {
                    return 0;
                }
            }
        }
        
        if (row_pooling_sequence && row_pooling_sequence->NumElements() > 0) {
            auto row_seq = row_pooling_sequence->flat<int64_t>();
            for (int i = 0; i < std::min(5, static_cast<int>(row_seq.size())); ++i) {
                if (row_seq(i) < 0) {
                    return 0;
                }
            }
        }
        
        if (col_pooling_sequence && col_pooling_sequence->NumElements() > 0) {
            auto col_seq = col_pooling_sequence->flat<int64_t>();
            for (int i = 0; i < std::min(5, static_cast<int>(col_seq.size())); ++i) {
                if (col_seq(i) < 0) {
                    return 0;
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