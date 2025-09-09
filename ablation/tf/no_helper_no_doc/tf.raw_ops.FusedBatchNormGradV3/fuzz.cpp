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
        
        if (size < 32) return 0;
        
        // Extract dimensions from fuzz data
        int batch_size = (data[offset] % 8) + 1;
        offset++;
        int height = (data[offset] % 16) + 1;
        offset++;
        int width = (data[offset] % 16) + 1;
        offset++;
        int channels = (data[offset] % 16) + 1;
        offset++;
        
        // Extract epsilon
        float epsilon = 1e-5f;
        if (offset + 4 <= size) {
            memcpy(&epsilon, data + offset, sizeof(float));
            epsilon = std::abs(epsilon);
            if (epsilon < 1e-10f) epsilon = 1e-5f;
            offset += 4;
        }
        
        // Extract is_training flag
        bool is_training = (data[offset % size] % 2) == 1;
        
        using namespace tensorflow;
        
        // Create input tensors
        TensorShape input_shape({batch_size, height, width, channels});
        TensorShape scale_shape({channels});
        
        Tensor y_backprop(DT_FLOAT, input_shape);
        Tensor x(DT_FLOAT, input_shape);
        Tensor scale(DT_FLOAT, scale_shape);
        Tensor reserve_space_1(DT_FLOAT, scale_shape);
        Tensor reserve_space_2(DT_FLOAT, scale_shape);
        Tensor reserve_space_3(DT_FLOAT, scale_shape);
        
        // Fill tensors with fuzz data
        auto y_backprop_flat = y_backprop.flat<float>();
        auto x_flat = x.flat<float>();
        auto scale_flat = scale.flat<float>();
        auto reserve_1_flat = reserve_space_1.flat<float>();
        auto reserve_2_flat = reserve_space_2.flat<float>();
        auto reserve_3_flat = reserve_space_3.flat<float>();
        
        size_t data_idx = offset;
        
        // Fill y_backprop
        for (int i = 0; i < y_backprop_flat.size() && data_idx < size; ++i) {
            float val = static_cast<float>(data[data_idx % size]) / 255.0f - 0.5f;
            y_backprop_flat(i) = val;
            data_idx++;
        }
        
        // Fill x
        for (int i = 0; i < x_flat.size() && data_idx < size; ++i) {
            float val = static_cast<float>(data[data_idx % size]) / 255.0f - 0.5f;
            x_flat(i) = val;
            data_idx++;
        }
        
        // Fill scale
        for (int i = 0; i < scale_flat.size() && data_idx < size; ++i) {
            float val = static_cast<float>(data[data_idx % size]) / 255.0f + 0.1f;
            scale_flat(i) = val;
            data_idx++;
        }
        
        // Fill reserve spaces (mean and variance from forward pass)
        for (int i = 0; i < reserve_1_flat.size() && data_idx < size; ++i) {
            float val = static_cast<float>(data[data_idx % size]) / 255.0f - 0.5f;
            reserve_1_flat(i) = val; // mean
            data_idx++;
        }
        
        for (int i = 0; i < reserve_2_flat.size() && data_idx < size; ++i) {
            float val = static_cast<float>(data[data_idx % size]) / 255.0f + 0.1f;
            reserve_2_flat(i) = val; // variance
            data_idx++;
        }
        
        for (int i = 0; i < reserve_3_flat.size() && data_idx < size; ++i) {
            float val = static_cast<float>(data[data_idx % size]) / 255.0f;
            reserve_3_flat(i) = val;
            data_idx++;
        }
        
        // Create test context
        class FusedBatchNormGradV3Test : public tensorflow::OpsTestBase {
        public:
            void RunTest(const Tensor& y_backprop, const Tensor& x, const Tensor& scale,
                        const Tensor& reserve_space_1, const Tensor& reserve_space_2,
                        const Tensor& reserve_space_3, float epsilon, bool is_training) {
                
                TF_ASSERT_OK(NodeDefBuilder("fused_batch_norm_grad_v3", "FusedBatchNormGradV3")
                    .Input(FakeInput(DT_FLOAT))  // y_backprop
                    .Input(FakeInput(DT_FLOAT))  // x
                    .Input(FakeInput(DT_FLOAT))  // scale
                    .Input(FakeInput(DT_FLOAT))  // reserve_space_1
                    .Input(FakeInput(DT_FLOAT))  // reserve_space_2
                    .Input(FakeInput(DT_FLOAT))  // reserve_space_3
                    .Attr("epsilon", epsilon)
                    .Attr("data_format", "NHWC")
                    .Attr("is_training", is_training)
                    .Finalize(node_def()));
                
                TF_ASSERT_OK(InitOp());
                
                AddInputFromArray<float>(y_backprop.shape(), y_backprop.flat<float>());
                AddInputFromArray<float>(x.shape(), x.flat<float>());
                AddInputFromArray<float>(scale.shape(), scale.flat<float>());
                AddInputFromArray<float>(reserve_space_1.shape(), reserve_space_1.flat<float>());
                AddInputFromArray<float>(reserve_space_2.shape(), reserve_space_2.flat<float>());
                AddInputFromArray<float>(reserve_space_3.shape(), reserve_space_3.flat<float>());
                
                TF_ASSERT_OK(RunOpKernel());
                
                // Check output shapes
                const Tensor& x_backprop = *GetOutput(0);
                const Tensor& scale_backprop = *GetOutput(1);
                const Tensor& offset_backprop = *GetOutput(2);
                
                EXPECT_EQ(x_backprop.shape(), x.shape());
                EXPECT_EQ(scale_backprop.shape(), scale.shape());
                EXPECT_EQ(offset_backprop.shape(), scale.shape());
            }
        };
        
        FusedBatchNormGradV3Test test;
        test.RunTest(y_backprop, x, scale, reserve_space_1, reserve_space_2, 
                    reserve_space_3, epsilon, is_training);
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}