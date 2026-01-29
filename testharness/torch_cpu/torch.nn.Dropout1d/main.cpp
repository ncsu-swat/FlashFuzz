#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }
    
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Extract parameters for dropout1d from the data
        double p = 0.5; // Default dropout probability
        bool inplace = false;
        bool training = true;
        bool use_3d_input = false;
        
        // Parse p value if we have enough data
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure p is in valid range [0, 1]
            if (std::isnan(p) || std::isinf(p)) {
                p = 0.5;
            } else {
                p = std::abs(p);
                p = p - std::floor(p); // Get fractional part to ensure 0 <= p <= 1
            }
        }
        
        // Parse inplace flag if we have enough data
        if (offset < Size) {
            inplace = Data[offset++] & 0x1;
        }
        
        // Parse training mode if we have enough data
        if (offset < Size) {
            training = Data[offset++] & 0x1;
        }
        
        // Decide whether to use 2D or 3D input
        if (offset < Size) {
            use_3d_input = Data[offset++] & 0x1;
        }
        
        // Create input tensor with appropriate shape for dropout1d
        // dropout1d expects (N, C) or (N, C, L) shaped input
        torch::Tensor input;
        
        if (use_3d_input) {
            // 3D input: (N, C, L)
            int64_t N = 1 + (offset < Size ? Data[offset++] % 8 : 1);
            int64_t C = 1 + (offset < Size ? Data[offset++] % 16 : 4);
            int64_t L = 1 + (offset < Size ? Data[offset++] % 32 : 8);
            input = torch::randn({N, C, L});
        } else {
            // 2D input: (N, C)
            int64_t N = 1 + (offset < Size ? Data[offset++] % 8 : 1);
            int64_t C = 1 + (offset < Size ? Data[offset++] % 16 : 4);
            input = torch::randn({N, C});
        }
        
        // If inplace, we need a tensor that allows modification
        if (inplace) {
            input = input.clone();
        }
        
        // Use functional dropout1d API
        // torch::nn::functional::dropout1d(input, options)
        auto options = torch::nn::functional::Dropout1dFuncOptions()
                           .p(p)
                           .training(training)
                           .inplace(inplace);
        
        torch::Tensor output = torch::nn::functional::dropout1d(input, options);
        
        // Force computation to ensure any potential errors are triggered
        output.sum().item<float>();
        
        // Additional coverage: also test with training=false if we tested with training=true
        if (training) {
            auto eval_options = torch::nn::functional::Dropout1dFuncOptions()
                                    .p(p)
                                    .training(false)
                                    .inplace(false);
            torch::Tensor eval_output = torch::nn::functional::dropout1d(input.clone(), eval_options);
            eval_output.sum().item<float>();
        }
        
        // Test with different p values for more coverage
        if (offset < Size) {
            double p2 = static_cast<double>(Data[offset++]) / 255.0;
            auto options2 = torch::nn::functional::Dropout1dFuncOptions()
                                .p(p2)
                                .training(training)
                                .inplace(false);
            torch::Tensor output2 = torch::nn::functional::dropout1d(input.clone(), options2);
            output2.sum().item<float>();
        }
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
}