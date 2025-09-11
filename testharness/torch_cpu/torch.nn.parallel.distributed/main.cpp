#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse parameters for distributed operations
        uint8_t world_size = 0;
        uint8_t rank = 0;
        
        if (offset + 2 <= Size) {
            world_size = (Data[offset] % 8) + 1; // 1-8 processes
            offset++;
            rank = Data[offset] % world_size; // 0 to world_size-1
            offset++;
        }
        
        // Test basic tensor operations that would be used in distributed context
        try {
            // Test tensor operations that simulate distributed behavior
            if (offset < Size) {
                uint8_t op_type = Data[offset++] % 5;
                
                switch (op_type) {
                    case 0: {
                        // Simulate all_reduce with sum
                        auto result = input.sum();
                        break;
                    }
                    case 1: {
                        // Simulate broadcast by cloning
                        auto result = input.clone();
                        break;
                    }
                    case 2: {
                        // Simulate reduce with mean
                        auto result = input.mean();
                        break;
                    }
                    case 3: {
                        // Simulate all_gather by expanding
                        if (input.dim() > 0) {
                            auto result = input.expand({world_size, -1});
                        }
                        break;
                    }
                    case 4: {
                        // Simulate gather by selecting
                        if (input.dim() > 0 && input.size(0) > rank) {
                            auto result = input.select(0, rank);
                        }
                        break;
                    }
                }
            }
            
            // Test model operations that would be used with distributed training
            if (input.dim() > 0 && input.size(-1) > 0) {
                auto model = torch::nn::Linear(input.size(-1), input.size(-1));
                
                // Forward pass
                auto output = model->forward(input);
                
                // Backward pass
                if (output.requires_grad()) {
                    auto loss = output.mean();
                    loss.backward();
                }
            }
        }
        catch (const c10::Error& e) {
            // Expected exceptions from operations with invalid parameters
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
