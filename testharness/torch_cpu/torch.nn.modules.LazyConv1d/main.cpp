#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 4 more bytes for parameters
        if (Size - offset < 4) {
            return 0;
        }
        
        // Extract parameters for LazyConv1d
        uint8_t out_channels_byte = Data[offset++];
        uint8_t kernel_size_byte = Data[offset++];
        uint8_t stride_byte = Data[offset++];
        uint8_t padding_byte = Data[offset++];
        
        // Convert to reasonable values
        int64_t out_channels = (out_channels_byte % 16) + 1;  // 1-16 output channels
        int64_t kernel_size = (kernel_size_byte % 7) + 1;     // 1-7 kernel size
        int64_t stride = (stride_byte % 3) + 1;               // 1-3 stride
        int64_t padding = padding_byte % 4;                   // 0-3 padding
        
        // Create LazyConv1d module using Conv1d as LazyConv1d is not available
        torch::nn::Conv1d conv1d(torch::nn::Conv1dOptions(1, out_channels, kernel_size)
                                 .stride(stride)
                                 .padding(padding));
        
        // Reshape input tensor if needed to make it compatible with Conv1d
        // Conv1d expects input of shape [batch_size, in_channels, sequence_length]
        if (input_tensor.dim() == 0) {
            // Scalar tensor - reshape to [1, 1, 1]
            input_tensor = input_tensor.reshape({1, 1, 1});
        } else if (input_tensor.dim() == 1) {
            // 1D tensor - reshape to [1, 1, length]
            input_tensor = input_tensor.reshape({1, 1, input_tensor.size(0)});
        } else if (input_tensor.dim() == 2) {
            // 2D tensor - reshape to [batch_size, 1, sequence_length]
            input_tensor = input_tensor.reshape({input_tensor.size(0), 1, input_tensor.size(1)});
        } else if (input_tensor.dim() > 3) {
            // Higher dim tensor - reshape to 3D by flattening extra dimensions
            int64_t batch_size = input_tensor.size(0);
            int64_t in_channels = 1;
            for (int i = 1; i < input_tensor.dim() - 1; i++) {
                in_channels *= input_tensor.size(i);
            }
            int64_t seq_length = input_tensor.size(input_tensor.dim() - 1);
            input_tensor = input_tensor.reshape({batch_size, in_channels, seq_length});
        }
        
        // Convert to float if not already a floating point type
        if (!torch::isFloatingType(input_tensor.scalar_type())) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Apply the Conv1d operation
        torch::Tensor output = conv1d->forward(input_tensor);
        
        // Perform some operation on the output to ensure it's used
        auto sum = output.sum();
        
        // Access the value to ensure computation is not optimized away
        float sum_val = sum.item<float>();
        (void)sum_val;  // Suppress unused variable warning
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
