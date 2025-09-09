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
        
        // Extract dimensions
        int32_t dim1 = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        int32_t dim2 = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        // Clamp dimensions to reasonable values
        dim1 = std::max(1, std::min(100, abs(dim1)));
        dim2 = std::max(1, std::min(100, abs(dim2)));
        
        tensorflow::TensorShape shape({dim1, dim2});
        size_t tensor_size = dim1 * dim2;
        size_t float_bytes = tensor_size * sizeof(float);
        
        // Check if we have enough data
        if (size < offset + 7 * float_bytes + 3 * sizeof(float)) return 0;
        
        // Create input tensors
        tensorflow::Tensor var(tensorflow::DT_FLOAT, shape);
        tensorflow::Tensor accum(tensorflow::DT_FLOAT, shape);
        tensorflow::Tensor linear(tensorflow::DT_FLOAT, shape);
        tensorflow::Tensor grad(tensorflow::DT_FLOAT, shape);
        tensorflow::Tensor lr(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor l1(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor l2(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor l2_shrinkage(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor lr_power(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        
        // Fill tensors with fuzz data
        auto var_flat = var.flat<float>();
        auto accum_flat = accum.flat<float>();
        auto linear_flat = linear.flat<float>();
        auto grad_flat = grad.flat<float>();
        
        std::memcpy(var_flat.data(), data + offset, float_bytes);
        offset += float_bytes;
        std::memcpy(accum_flat.data(), data + offset, float_bytes);
        offset += float_bytes;
        std::memcpy(linear_flat.data(), data + offset, float_bytes);
        offset += float_bytes;
        std::memcpy(grad_flat.data(), data + offset, float_bytes);
        offset += float_bytes;
        
        // Fill scalar tensors
        lr.scalar<float>()() = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        l1.scalar<float>()() = std::abs(*reinterpret_cast<const float*>(data + offset));
        offset += sizeof(float);
        l2.scalar<float>()() = std::abs(*reinterpret_cast<const float*>(data + offset));
        offset += sizeof(float);
        l2_shrinkage.scalar<float>()() = std::abs(*reinterpret_cast<const float*>(data + offset));
        offset += sizeof(float);
        lr_power.scalar<float>()() = *reinterpret_cast<const float*>(data + offset);
        
        // Ensure accum values are positive (required for FTRL)
        for (int i = 0; i < tensor_size; ++i) {
            accum_flat(i) = std::abs(accum_flat(i)) + 1e-8f;
        }
        
        // Create operation using OpsTestBase
        class ApplyFtrlV2Test : public tensorflow::OpsTestBase {
        public:
            void RunTest(const tensorflow::Tensor& var, const tensorflow::Tensor& accum,
                        const tensorflow::Tensor& linear, const tensorflow::Tensor& grad,
                        const tensorflow::Tensor& lr, const tensorflow::Tensor& l1,
                        const tensorflow::Tensor& l2, const tensorflow::Tensor& l2_shrinkage,
                        const tensorflow::Tensor& lr_power) {
                tensorflow::NodeDef node_def;
                tensorflow::Status status = tensorflow::NodeDefBuilder("apply_ftrl_v2", "ApplyFtrlV2")
                    .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))
                    .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))
                    .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))
                    .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))
                    .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))
                    .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))
                    .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))
                    .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))
                    .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))
                    .Attr("use_locking", false)
                    .Finalize(&node_def);
                
                if (!status.ok()) return;
                
                status = InitOp(node_def);
                if (!status.ok()) return;
                
                AddInputFromArray<float>(var.shape(), var.flat<float>());
                AddInputFromArray<float>(accum.shape(), accum.flat<float>());
                AddInputFromArray<float>(linear.shape(), linear.flat<float>());
                AddInputFromArray<float>(grad.shape(), grad.flat<float>());
                AddInputFromArray<float>(lr.shape(), lr.flat<float>());
                AddInputFromArray<float>(l1.shape(), l1.flat<float>());
                AddInputFromArray<float>(l2.shape(), l2.flat<float>());
                AddInputFromArray<float>(l2_shrinkage.shape(), l2_shrinkage.flat<float>());
                AddInputFromArray<float>(lr_power.shape(), lr_power.flat<float>());
                
                status = RunOpKernel();
                // Ignore status - we're fuzzing, errors are expected
            }
        };
        
        ApplyFtrlV2Test test;
        test.RunTest(var, accum, linear, grad, lr, l1, l2, l2_shrinkage, lr_power);
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}