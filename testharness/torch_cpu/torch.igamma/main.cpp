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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create the first tensor (a)
        torch::Tensor a = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create the second tensor (x)
        if (offset < Size) {
            torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Make sure both tensors have compatible types for igamma
            // Convert to floating point if needed
            if (!a.is_floating_point()) {
                a = a.to(torch::kFloat);
            }
            
            if (!x.is_floating_point()) {
                x = x.to(torch::kFloat);
            }
            
            // Apply torch.igamma operation
            // igamma(a, x) is the lower incomplete gamma function
            torch::Tensor result;
            
            // Try broadcasting if shapes don't match
            try {
                result = torch::igamma(a, x);
            }
            catch (const std::exception& e) {
                // Try with scalar values if tensor operation fails
                if (a.numel() > 0 && x.numel() > 0) {
                    try {
                        // Try with first elements as scalars converted to tensors
                        auto a_scalar_tensor = torch::tensor(a.item<double>());
                        auto x_scalar_tensor = torch::tensor(x.item<double>());
                        result = torch::igamma(a_scalar_tensor, x_scalar_tensor);
                    }
                    catch (const std::exception& e) {
                        // Silently ignore this specific failure
                    }
                }
            }
            
            // Try the variant with scalar inputs converted to tensors
            if (offset + 8 <= Size) {
                double a_scalar, x_scalar;
                std::memcpy(&a_scalar, Data + offset, sizeof(double));
                offset += sizeof(double);
                std::memcpy(&x_scalar, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                try {
                    auto a_scalar_tensor = torch::tensor(a_scalar);
                    result = torch::igamma(a_scalar_tensor, x);
                }
                catch (const std::exception& e) {
                    // Silently ignore this specific failure
                }
                
                try {
                    auto x_scalar_tensor = torch::tensor(x_scalar);
                    result = torch::igamma(a, x_scalar_tensor);
                }
                catch (const std::exception& e) {
                    // Silently ignore this specific failure
                }
                
                try {
                    auto a_scalar_tensor = torch::tensor(a_scalar);
                    auto x_scalar_tensor = torch::tensor(x_scalar);
                    result = torch::igamma(a_scalar_tensor, x_scalar_tensor);
                }
                catch (const std::exception& e) {
                    // Silently ignore this specific failure
                }
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
