#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple model to test with parallel
        torch::nn::Linear model(input1.size(-1), 10);
        
        // Get number of GPUs to use (if available)
        int num_gpus = 0;
        if (offset + 1 < Size) {
            num_gpus = Data[offset++] % 4;  // Use 0-3 GPUs
        }
        
        // Get parallel mode
        int mode = 0;
        if (offset + 1 < Size) {
            mode = Data[offset++] % 3;  // 0: data parallel, 1: distributed data parallel, 2: model parallel
        }
        
        // Test the deprecated parallel functionality
        try {
            if (mode == 0) {
                // Simple forward pass without parallel (since torch::nn::parallel is not available)
                auto output = model->forward(input1);
            }
            else if (mode == 1) {
                // Test with CUDA if available
                if (torch::cuda::is_available() && num_gpus > 0) {
                    auto device = torch::Device(torch::kCUDA, 0);
                    model->to(device);
                    input1 = input1.to(device);
                    auto output = model->forward(input1);
                }
            }
            else {
                // Test other functionality
                if (torch::cuda::is_available() && num_gpus > 0) {
                    auto device = torch::Device(torch::kCUDA, 0);
                    model->to(device);
                    input1 = input1.to(device);
                    auto output = model->forward(input1);
                }
            }
        }
        catch (const c10::Error &e) {
            // Expected exceptions from PyTorch operations
        }
        
        // Try to test with different batch sizes
        if (offset + 1 < Size && input1.dim() > 0) {
            try {
                int batch_size = (Data[offset++] % 8) + 1;  // 1-8 batch size
                
                // Create a new input with the specified batch size
                std::vector<int64_t> new_shape;
                new_shape.push_back(batch_size);
                for (int i = 1; i < input1.dim(); i++) {
                    new_shape.push_back(input1.size(i));
                }
                
                torch::Tensor batched_input;
                try {
                    batched_input = torch::ones(new_shape, input1.options());
                    
                    // Test forward pass with batched input
                    auto output = model->forward(batched_input);
                }
                catch (const c10::Error &e) {
                    // Expected exceptions from tensor creation or forward operation
                }
            }
            catch (const std::exception &e) {
                // Handle other exceptions
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
