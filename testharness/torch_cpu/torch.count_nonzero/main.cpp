#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get tensor dimensions for valid dim selection
        int64_t ndim = input_tensor.dim();
        
        // Decide which variant to call based on remaining data
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 3;
            
            switch (variant) {
                case 0: {
                    // Basic count_nonzero without additional parameters
                    torch::Tensor result = torch::count_nonzero(input_tensor);
                    // Result is a scalar tensor
                    if (result.defined() && result.numel() == 1) {
                        volatile auto val = result.item<int64_t>();
                        (void)val;
                    }
                    break;
                }
                    
                case 1: {
                    // count_nonzero with single dim parameter (using optional<int64_t>)
                    if (ndim > 0 && offset < Size) {
                        // Select a valid dimension
                        int64_t dim_val = static_cast<int64_t>(Data[offset++] % ndim);
                        
                        try {
                            // Explicitly use std::optional<int64_t> to disambiguate the call
                            torch::Tensor result = torch::count_nonzero(input_tensor, std::optional<int64_t>(dim_val));
                            // Access result to ensure computation
                            if (result.defined() && result.numel() > 0) {
                                volatile auto sum = result.sum().item<int64_t>();
                                (void)sum;
                            }
                        } catch (const c10::Error&) {
                            // Expected for some tensor configurations
                        }
                    } else {
                        // Fallback to no-dim version
                        torch::Tensor result = torch::count_nonzero(input_tensor);
                        if (result.defined()) {
                            volatile auto val = result.item<int64_t>();
                            (void)val;
                        }
                    }
                    break;
                }
                
                case 2: {
                    // count_nonzero with multiple dims (IntArrayRef)
                    if (ndim > 1 && offset + 1 < Size) {
                        // Create a vector of dims
                        uint8_t num_dims = (Data[offset++] % std::min(static_cast<int64_t>(3), ndim)) + 1;
                        std::vector<int64_t> dims;
                        
                        for (uint8_t i = 0; i < num_dims && offset < Size; i++) {
                            int64_t d = static_cast<int64_t>(Data[offset++] % ndim);
                            // Avoid duplicate dims
                            bool duplicate = false;
                            for (auto existing_dim : dims) {
                                if (existing_dim == d) {
                                    duplicate = true;
                                    break;
                                }
                            }
                            if (!duplicate) {
                                dims.push_back(d);
                            }
                        }
                        
                        if (!dims.empty()) {
                            try {
                                // Use IntArrayRef overload for multiple dimensions
                                torch::Tensor result = torch::count_nonzero(input_tensor, at::IntArrayRef(dims));
                                if (result.defined() && result.numel() > 0) {
                                    volatile auto sum = result.sum().item<int64_t>();
                                    (void)sum;
                                }
                            } catch (const c10::Error&) {
                                // Expected for some configurations
                            }
                        }
                    } else {
                        // Fallback to no-dim version
                        torch::Tensor result = torch::count_nonzero(input_tensor);
                        if (result.defined()) {
                            volatile auto val = result.item<int64_t>();
                            (void)val;
                        }
                    }
                    break;
                }
            }
        } else {
            // Default to basic count_nonzero if no more data
            torch::Tensor result = torch::count_nonzero(input_tensor);
            if (result.defined()) {
                volatile auto val = result.item<int64_t>();
                (void)val;
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