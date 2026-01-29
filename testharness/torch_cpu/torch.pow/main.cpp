#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstring>        // For std::memcpy

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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create base tensor
        torch::Tensor base = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create exponent tensor or scalar
        if (offset < Size && Data[offset] % 2 == 0) {
            offset++; // Consume the decision byte
            // Use tensor exponent
            torch::Tensor exponent = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply torch.pow operation
            torch::Tensor result = torch::pow(base, exponent);
        } else {
            if (offset < Size) offset++; // Consume the decision byte
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
                offset++; // Consume the decision byte
                // Use tensor exponent for self version
                torch::Tensor exponent = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Clone base to avoid modifying the original
                torch::Tensor base_clone = base.clone();
                
                // Apply in-place pow
                try {
                    base_clone.pow_(exponent);
                } catch (...) {
                    // Shape mismatch or other expected failures
                }
            } else {
                offset++; // Consume the decision byte
                // Use scalar exponent for self version
                double exponent = 0.0;
                if (offset + sizeof(double) <= Size) {
                    std::memcpy(&exponent, Data + offset, sizeof(double));
                    offset += sizeof(double);
                } else if (offset < Size) {
                    exponent = static_cast<double>(Data[offset]);
                    offset++;
                }
                
                // Clone base to avoid modifying the original
                torch::Tensor base_clone = base.clone();
                
                // Apply in-place pow
                try {
                    base_clone.pow_(exponent);
                } catch (...) {
                    // Expected failures for certain exponent values
                }
            }
        }
        
        // Test scalar base with tensor exponent (torch.pow(scalar, tensor))
        if (offset < Size) {
            double scalar_base = 0.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&scalar_base, Data + offset, sizeof(double));
                offset += sizeof(double);
            } else {
                scalar_base = static_cast<double>(Data[offset]);
                offset++;
            }
            
            torch::Tensor exponent = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply pow with scalar base
            try {
                torch::Tensor result = torch::pow(scalar_base, exponent);
            } catch (...) {
                // Expected failures for certain inputs
            }
        }
        
        // Test with output tensor (out parameter version)
        if (offset < Size) {
            torch::Tensor another_base = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor out = torch::empty_like(another_base);
            
            double exponent = 2.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&exponent, Data + offset, sizeof(double));
                offset += sizeof(double);
            }
            
            try {
                torch::pow_out(out, another_base, exponent);
            } catch (...) {
                // Expected failures
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