#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }
    
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Extract parameters for the Bilinear module from fuzzer data
        int64_t in1_features = (Data[offset++] % 16) + 1;  // 1-16
        int64_t in2_features = (Data[offset++] % 16) + 1;  // 1-16
        int64_t out_features = (Data[offset++] % 16) + 1;  // 1-16
        bool bias = Data[offset++] & 0x1;
        int64_t batch_size = (Data[offset++] % 8) + 1;     // 1-8
        
        // Create the Bilinear module
        torch::nn::BilinearOptions options(in1_features, in2_features, out_features);
        options.bias(bias);
        torch::nn::Bilinear bilinear(options);
        
        // Create input tensors with correct shapes for bilinear operation
        // Bilinear expects: input1 (N, in1_features), input2 (N, in2_features)
        torch::Tensor input1;
        torch::Tensor input2;
        
        // Try to create tensors from remaining fuzzer data
        if (offset < Size) {
            input1 = fuzzer_utils::createTensor(Data, Size, offset);
        }
        if (offset < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Ensure input1 has the correct shape (batch_size, in1_features)
        try {
            if (!input1.defined() || input1.numel() == 0) {
                input1 = torch::randn({batch_size, in1_features});
            } else {
                // Flatten and take what we need, or pad with random values
                input1 = input1.flatten().to(torch::kFloat);
                int64_t needed = batch_size * in1_features;
                if (input1.numel() >= needed) {
                    input1 = input1.slice(0, 0, needed).reshape({batch_size, in1_features});
                } else {
                    // Pad with random values
                    torch::Tensor padding = torch::randn({needed - input1.numel()});
                    input1 = torch::cat({input1, padding}).reshape({batch_size, in1_features});
                }
            }
        } catch (...) {
            input1 = torch::randn({batch_size, in1_features});
        }
        
        // Ensure input2 has the correct shape (batch_size, in2_features)
        try {
            if (!input2.defined() || input2.numel() == 0) {
                input2 = torch::randn({batch_size, in2_features});
            } else {
                // Flatten and take what we need, or pad with random values
                input2 = input2.flatten().to(torch::kFloat);
                int64_t needed = batch_size * in2_features;
                if (input2.numel() >= needed) {
                    input2 = input2.slice(0, 0, needed).reshape({batch_size, in2_features});
                } else {
                    // Pad with random values
                    torch::Tensor padding = torch::randn({needed - input2.numel()});
                    input2 = torch::cat({input2, padding}).reshape({batch_size, in2_features});
                }
            }
        } catch (...) {
            input2 = torch::randn({batch_size, in2_features});
        }
        
        // Ensure both tensors are float type (required by Bilinear)
        input1 = input1.to(torch::kFloat);
        input2 = input2.to(torch::kFloat);
        
        // Apply the bilinear module
        torch::Tensor output = bilinear->forward(input1, input2);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        auto mean = output.mean();
        
        // Test with different batch sizes if we have enough data
        if (Size > 20) {
            int64_t batch2 = (Data[Size - 1] % 4) + 1;
            torch::Tensor input1_v2 = torch::randn({batch2, in1_features});
            torch::Tensor input2_v2 = torch::randn({batch2, in2_features});
            torch::Tensor output2 = bilinear->forward(input1_v2, input2_v2);
            (void)output2.sum();
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}