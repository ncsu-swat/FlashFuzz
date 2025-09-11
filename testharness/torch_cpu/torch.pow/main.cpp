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
        
        // Create base tensor
        torch::Tensor base = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create exponent tensor or scalar
        if (offset < Size && Data[offset] % 2 == 0) {
            // Use tensor exponent
            torch::Tensor exponent = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply torch.pow operation
            torch::Tensor result = torch::pow(base, exponent);
        } else {
            // Use scalar exponent
            double exponent = 0.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&exponent, Data + offset, sizeof(double));
                offset += sizeof(double);
            } else if (offset < Size) {
                // Not enough data for a double, use the remaining bytes
                exponent = static_cast<double>(Data[offset]);
                offset++;
            }
            
            // Apply torch.pow operation with scalar exponent
            torch::Tensor result = torch::pow(base, exponent);
        }
        
        // Test self-version of pow
        if (offset < Size) {
            // Create another exponent
            if (Data[offset] % 2 == 0) {
                // Use tensor exponent for self version
                torch::Tensor exponent = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Clone base to avoid modifying the original
                torch::Tensor base_clone = base.clone();
                
                // Apply in-place pow
                base_clone.pow_(exponent);
            } else {
                // Use scalar exponent for self version
                double exponent = 0.0;
                if (offset + sizeof(double) <= Size) {
                    std::memcpy(&exponent, Data + offset, sizeof(double));
                } else if (offset < Size) {
                    exponent = static_cast<double>(Data[offset]);
                }
                
                // Clone base to avoid modifying the original
                torch::Tensor base_clone = base.clone();
                
                // Apply in-place pow
                base_clone.pow_(exponent);
            }
        }
        
        // Test the static version of pow
        if (offset < Size) {
            // Create another base tensor
            torch::Tensor another_base = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Create another exponent
            if (offset < Size && Data[offset] % 2 == 0) {
                // Use tensor exponent
                torch::Tensor exponent = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Apply static pow
                torch::Tensor result = at::pow(another_base, exponent);
            } else {
                // Use scalar exponent
                double exponent = 0.0;
                if (offset + sizeof(double) <= Size) {
                    std::memcpy(&exponent, Data + offset, sizeof(double));
                } else if (offset < Size) {
                    exponent = static_cast<double>(Data[offset]);
                }
                
                // Apply static pow
                torch::Tensor result = at::pow(another_base, exponent);
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
