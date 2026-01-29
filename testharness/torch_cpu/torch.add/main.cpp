#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout
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
        
        // Create first tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor or scalar
        bool use_scalar = false;
        if (offset < Size) {
            use_scalar = Data[offset++] % 2 == 0;
        }
        
        if (use_scalar) {
            // Use a scalar value for the second argument
            double scalar_value = 0.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&scalar_value, Data + offset, sizeof(double));
                offset += sizeof(double);
            }
            
            // Test torch::add with scalar
            torch::Tensor result = torch::add(tensor1, scalar_value);
            
            // Try with alpha parameter if we have more data
            if (offset + sizeof(double) <= Size) {
                double alpha;
                std::memcpy(&alpha, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                torch::Tensor result_with_alpha = torch::add(tensor1, scalar_value, alpha);
            }
        } else {
            // Create a second tensor for tensor-tensor addition
            torch::Tensor tensor2;
            if (offset < Size) {
                tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Try tensor-tensor addition
                try {
                    torch::Tensor result = torch::add(tensor1, tensor2);
                } catch (const std::exception&) {
                    // Shapes might be incompatible, that's expected in some cases
                }
                
                // Try with alpha parameter if we have more data
                if (offset + sizeof(double) <= Size) {
                    double alpha;
                    std::memcpy(&alpha, Data + offset, sizeof(double));
                    offset += sizeof(double);
                    
                    try {
                        torch::Tensor result_with_alpha = torch::add(tensor1, tensor2, alpha);
                    } catch (const std::exception&) {
                        // Shapes might be incompatible, that's expected in some cases
                    }
                }
                
                // Try in-place addition if we have more data
                if (offset < Size && Data[offset++] % 2 == 0) {
                    try {
                        tensor1.add_(tensor2);
                    } catch (const std::exception&) {
                        // Shapes might be incompatible, that's expected in some cases
                    }
                    
                    // Try with alpha
                    if (offset + sizeof(double) <= Size) {
                        double alpha;
                        std::memcpy(&alpha, Data + offset, sizeof(double));
                        offset += sizeof(double);
                        
                        try {
                            tensor1.add_(tensor2, alpha);
                        } catch (const std::exception&) {
                            // Shapes might be incompatible, that's expected in some cases
                        }
                    }
                }
            }
        }
        
        // Try in-place addition with scalar if we have more data
        if (offset + sizeof(double) <= Size) {
            double scalar_value;
            std::memcpy(&scalar_value, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            tensor1.add_(scalar_value);
            
            // Try with alpha
            if (offset + sizeof(double) <= Size) {
                double alpha;
                std::memcpy(&alpha, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                tensor1.add_(scalar_value, alpha);
            }
        }
        
        // Try out_variant if we have more data
        if (offset < Size && Data[offset++] % 2 == 0) {
            torch::Tensor out = torch::empty_like(tensor1);
            
            if (use_scalar && offset + sizeof(double) <= Size) {
                double scalar_value;
                std::memcpy(&scalar_value, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                torch::add_out(out, tensor1, scalar_value);
            } else if (!use_scalar && offset < Size) {
                torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                try {
                    torch::add_out(out, tensor1, tensor2);
                } catch (const std::exception&) {
                    // Shapes might be incompatible, that's expected in some cases
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