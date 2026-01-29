#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create Softsign module
        torch::nn::Softsign softsign;
        
        // Apply Softsign to the input tensor
        torch::Tensor output = softsign->forward(input);
        
        // Alternative direct function call
        torch::Tensor output2 = torch::nn::functional::softsign(input);
        
        // Try with different tensor options
        if (offset + 1 < Size) {
            // Create a tensor with different options
            torch::Tensor input2 = input.clone();
            
            // Try different tensor properties - wrap in silent try-catch
            // as dtype conversions can fail for some tensor types
            try {
                if (Data[offset] % 4 == 0) {
                    input2 = input2.to(torch::kFloat32);
                } else if (Data[offset] % 4 == 1) {
                    input2 = input2.to(torch::kDouble);
                } else if (Data[offset] % 4 == 2) {
                    input2 = input2.contiguous();
                } else {
                    input2 = input2.to(torch::kFloat64);
                }
                
                // Apply Softsign to the modified tensor
                torch::Tensor output3 = softsign->forward(input2);
            } catch (...) {
                // Silently ignore expected failures from dtype conversion
            }
            
            offset++;
        }
        
        // Try with non-contiguous tensor if possible
        if (input.dim() > 1 && input.size(0) > 1) {
            try {
                torch::Tensor non_contiguous = input.transpose(0, input.dim() - 1);
                if (!non_contiguous.is_contiguous()) {
                    torch::Tensor output4 = softsign->forward(non_contiguous);
                }
            } catch (...) {
                // Silently ignore
            }
        }
        
        // Try with empty tensor
        if (offset + 1 < Size && Data[offset] % 2 == 0) {
            try {
                std::vector<int64_t> empty_shape = {0};
                torch::Tensor empty_tensor = torch::empty(empty_shape, torch::kFloat32);
                torch::Tensor empty_output = softsign->forward(empty_tensor);
            } catch (...) {
                // Silently ignore
            }
        }
        
        // Try with scalar tensor
        if (offset + 1 < Size) {
            try {
                torch::Tensor scalar_tensor;
                if (Data[offset] % 3 == 0) {
                    scalar_tensor = torch::tensor(3.14f);
                } else if (Data[offset] % 3 == 1) {
                    scalar_tensor = torch::tensor(-100.0f);
                } else {
                    scalar_tensor = torch::tensor(0.0f);
                }
                
                torch::Tensor scalar_output = softsign->forward(scalar_tensor);
            } catch (...) {
                // Silently ignore
            }
        }
        
        // Test with special float values
        if (offset < Size) {
            try {
                torch::Tensor special_tensor;
                uint8_t choice = Data[offset] % 4;
                if (choice == 0) {
                    special_tensor = torch::tensor({std::numeric_limits<float>::infinity()});
                } else if (choice == 1) {
                    special_tensor = torch::tensor({-std::numeric_limits<float>::infinity()});
                } else if (choice == 2) {
                    special_tensor = torch::tensor({std::numeric_limits<float>::quiet_NaN()});
                } else {
                    special_tensor = torch::tensor({std::numeric_limits<float>::max(), std::numeric_limits<float>::min()});
                }
                torch::Tensor special_output = softsign->forward(special_tensor);
            } catch (...) {
                // Silently ignore
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}