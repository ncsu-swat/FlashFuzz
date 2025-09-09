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
        
        // Extract basic parameters from fuzz input
        int32_t tensor_size = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        bool pred_value = *reinterpret_cast<const bool*>(data + offset);
        offset += sizeof(bool);
        
        tensorflow::DataType dtype = static_cast<tensorflow::DataType>(*reinterpret_cast<const int32_t*>(data + offset) % 19 + 1);
        offset += sizeof(int32_t);
        
        // Clamp tensor size to reasonable bounds
        tensor_size = std::abs(tensor_size) % 1000 + 1;
        
        // Create input tensor based on remaining data
        tensorflow::TensorShape shape({tensor_size});
        tensorflow::Tensor input_tensor(dtype, shape);
        
        // Fill tensor with fuzz data
        size_t remaining_data = size - offset;
        if (remaining_data > 0) {
            size_t tensor_bytes = input_tensor.TotalBytes();
            size_t copy_bytes = std::min(remaining_data, tensor_bytes);
            if (copy_bytes > 0) {
                std::memcpy(input_tensor.tensor_data().data(), data + offset, copy_bytes);
            }
        }
        
        // Create predicate tensor
        tensorflow::Tensor pred_tensor(tensorflow::DT_BOOL, tensorflow::TensorShape({}));
        pred_tensor.scalar<bool>()() = pred_value;
        
        // Create a simple test using OpsTestBase
        class RefSwitchTest : public tensorflow::OpsTestBase {
        public:
            void RunTest(const tensorflow::Tensor& input, const tensorflow::Tensor& pred) {
                tensorflow::NodeDefBuilder builder("ref_switch", "RefSwitch");
                builder.Input(tensorflow::FakeInput(input.dtype()));
                builder.Input(tensorflow::FakeInput(tensorflow::DT_BOOL));
                
                tensorflow::NodeDef node_def;
                tensorflow::Status status = builder.Finalize(&node_def);
                if (!status.ok()) return;
                
                status = InitOp(node_def);
                if (!status.ok()) return;
                
                AddInputFromArray<tensorflow::int32>(input.shape(), 
                    reinterpret_cast<const tensorflow::int32*>(input.tensor_data().data()));
                AddInputFromArray<bool>(pred.shape(), {pred.scalar<bool>()()});
                
                status = RunOpKernel();
                // Don't check status as we're fuzzing - just let it run
            }
        };
        
        RefSwitchTest test;
        test.RunTest(input_tensor, pred_tensor);
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}