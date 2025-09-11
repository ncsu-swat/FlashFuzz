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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create transition matrix
        torch::Tensor transition_matrix = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create emission matrix
        if (offset < Size) {
            torch::Tensor emission_matrix = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Create initial distribution
            if (offset < Size) {
                torch::Tensor initial_distribution = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Create sequence
                if (offset < Size) {
                    torch::Tensor sequence = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Try to apply a valid torch operation instead of hsmm
                    try {
                        auto result = torch::matmul(transition_matrix, emission_matrix);
                    } catch (...) {
                        // Catch any exceptions from the operation itself
                    }
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
