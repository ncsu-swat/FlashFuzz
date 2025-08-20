#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor x1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data for a second tensor
        if (offset < Size) {
            torch::Tensor x2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Get operation type from the next byte if available
            uint8_t op_type = 0;
            if (offset < Size) {
                op_type = Data[offset++];
            }
            
            // Apply different operations based on the op_type
            switch (op_type % 6) {
                case 0: {
                    // Test add operation
                    torch::Tensor result = torch::add(x1, x2);
                    break;
                }
                case 1: {
                    // Test add_scalar operation
                    double scalar = 1.0;
                    if (offset + sizeof(double) <= Size) {
                        std::memcpy(&scalar, Data + offset, sizeof(double));
                        offset += sizeof(double);
                    }
                    torch::Tensor result = torch::add(x1, scalar);
                    break;
                }
                case 2: {
                    // Test mul operation
                    torch::Tensor result = torch::mul(x1, x2);
                    break;
                }
                case 3: {
                    // Test mul_scalar operation
                    double scalar = 1.0;
                    if (offset + sizeof(double) <= Size) {
                        std::memcpy(&scalar, Data + offset, sizeof(double));
                        offset += sizeof(double);
                    }
                    torch::Tensor result = torch::mul(x1, scalar);
                    break;
                }
                case 4: {
                    // Test cat operation
                    std::vector<torch::Tensor> tensors = {x1, x2};
                    int64_t dim = 0;
                    if (offset + sizeof(int64_t) <= Size) {
                        std::memcpy(&dim, Data + offset, sizeof(int64_t));
                        offset += sizeof(int64_t);
                    }
                    torch::Tensor result = torch::cat(tensors, dim);
                    break;
                }
                case 5: {
                    // Test add_relu operation (add followed by relu)
                    torch::Tensor result = torch::relu(torch::add(x1, x2));
                    break;
                }
            }
        } else {
            // If we don't have enough data for a second tensor, test operations that work with a single tensor
            double scalar = 1.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&scalar, Data + offset, sizeof(double));
                offset += sizeof(double);
            }
            
            // Test scalar operations
            torch::Tensor result1 = torch::add(x1, scalar);
            torch::Tensor result2 = torch::mul(x1, scalar);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}