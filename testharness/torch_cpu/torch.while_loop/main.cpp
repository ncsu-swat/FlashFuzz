#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic operation
        if (Size < 4) {
            return 0;
        }
        
        // Create initial tensor
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create condition function
        auto cond_fn = [](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            const torch::Tensor& x = inputs[0];
            // Create a condition based on sum of tensor elements
            // Return true if sum is positive, false otherwise
            torch::Tensor sum = torch::sum(x);
            return sum > 0;
        };
        
        // Create body function
        auto body_fn = [](const std::vector<torch::Tensor>& inputs) -> std::vector<torch::Tensor> {
            const torch::Tensor& x = inputs[0];
            // Perform some operation on the tensor
            // For example, subtract 1 from each element
            torch::Tensor result = x - 1;
            return {result};
        };
        
        // Get max iterations from the input data
        uint32_t max_iterations = 10;
        if (offset + sizeof(uint32_t) <= Size) {
            std::memcpy(&max_iterations, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            // Limit max iterations to avoid excessive computation
            max_iterations = max_iterations % 100;
        }
        
        // Apply torch::jit::while_loop
        std::vector<torch::Tensor> outputs = torch::jit::while_loop(
            cond_fn,
            body_fn,
            {x},
            max_iterations
        );
        
        // Try another variant with different tensors if we have more data
        if (offset + 4 < Size) {
            torch::Tensor y = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Create a more complex condition function
            auto cond_fn2 = [](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
                const torch::Tensor& x = inputs[0];
                const torch::Tensor& y = inputs[1];
                // Check if any element in x is greater than the corresponding element in y
                return torch::any(x > y);
            };
            
            // Create a more complex body function
            auto body_fn2 = [](const std::vector<torch::Tensor>& inputs) -> std::vector<torch::Tensor> {
                const torch::Tensor& x = inputs[0];
                const torch::Tensor& y = inputs[1];
                // Perform operations on the tensors
                torch::Tensor new_x = x * 0.9;
                torch::Tensor new_y = y + 0.1;
                return {new_x, new_y};
            };
            
            // Apply torch::jit::while_loop with multiple tensors
            std::vector<torch::Tensor> outputs2 = torch::jit::while_loop(
                cond_fn2,
                body_fn2,
                {x, y},
                max_iterations
            );
        }
        
        // Try a variant that might hit edge cases
        if (offset + 4 < Size) {
            // Create a tensor that might be empty or have zeros
            torch::Tensor z = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Create a condition that might never be true
            auto cond_fn3 = [](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
                const torch::Tensor& z = inputs[0];
                // This condition might never be true if z has all negative values
                return torch::all(z > 0);
            };
            
            // Create a body function that modifies the tensor
            auto body_fn3 = [](const std::vector<torch::Tensor>& inputs) -> std::vector<torch::Tensor> {
                const torch::Tensor& z = inputs[0];
                // This operation might lead to NaN or Inf values
                torch::Tensor result = torch::sqrt(z);
                return {result};
            };
            
            // Apply torch::jit::while_loop with potentially problematic inputs
            std::vector<torch::Tensor> outputs3 = torch::jit::while_loop(
                cond_fn3,
                body_fn3,
                {z},
                max_iterations
            );
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}