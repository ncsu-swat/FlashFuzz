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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a simple BatchNorm2d module for testing freeze_bn_stats
        int64_t num_features = (Data[0] % 8) + 1;
        
        // Create a BatchNorm2d module
        auto bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_features));
        
        // Create input tensor for the module
        torch::Tensor input;
        if (Size > 4) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create a default tensor if not enough data
            int64_t batch_size = 2;
            int64_t height = 10;
            int64_t width = 10;
            input = torch::rand({batch_size, num_features, height, width});
        }
        
        // Reshape input if necessary to match expected dimensions
        if (input.dim() < 4) {
            int64_t batch_size = 2;
            int64_t height = 10;
            int64_t width = 10;
            input = input.reshape({batch_size, num_features, height, width});
        } else if (input.size(1) != num_features) {
            input = input.reshape({input.size(0), num_features, -1, input.size(3)});
        }
        
        // Test the module before freezing
        torch::Tensor output_before;
        try {
            output_before = bn->forward(input);
        } catch (const std::exception& e) {
            // If forward fails, try with a more standard input
            int64_t batch_size = 2;
            int64_t height = 10;
            int64_t width = 10;
            input = torch::rand({batch_size, num_features, height, width});
            output_before = bn->forward(input);
        }
        
        // Set module to training mode first
        bn->train();
        
        // Run forward pass to accumulate statistics
        torch::Tensor train_output = bn->forward(input);
        
        // Now freeze the BatchNorm statistics by setting to eval mode
        bn->eval();
        
        // Test the module after freezing
        torch::Tensor output_after = bn->forward(input);
        
        // Try to switch back to training and test again
        try {
            bn->train();
            torch::Tensor output_unfrozen = bn->forward(input);
        } catch (const std::exception& e) {
            // Ignore exceptions from this usage
        }
        
        // Test with different input sizes if we have enough data
        if (Size > 10) {
            try {
                int64_t new_batch = (Data[3] % 4) + 1;
                int64_t new_height = (Data[4] % 8) + 4;
                int64_t new_width = (Data[5] % 8) + 4;
                
                torch::Tensor new_input = torch::rand({new_batch, num_features, new_height, new_width});
                
                // Test in training mode
                bn->train();
                torch::Tensor train_out = bn->forward(new_input);
                
                // Test in eval mode (frozen stats)
                bn->eval();
                torch::Tensor eval_out = bn->forward(new_input);
                
            } catch (const std::exception& e) {
                // Ignore exceptions
            }
        }
        
        // Test with BatchNorm1d if we have enough data
        if (Size > 15) {
            try {
                int64_t features_1d = (Data[6] % 10) + 1;
                auto bn1d = torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(features_1d));
                
                // Create appropriate input
                torch::Tensor input_1d = torch::rand({2, features_1d});
                
                // Test training mode
                bn1d->train();
                torch::Tensor out_1d_train = bn1d->forward(input_1d);
                
                // Test eval mode (frozen stats)
                bn1d->eval();
                torch::Tensor out_1d_eval = bn1d->forward(input_1d);
                
            } catch (const std::exception& e) {
                // Ignore exceptions
            }
        }
        
        // Test with BatchNorm3d if we have enough data
        if (Size > 20) {
            try {
                int64_t features_3d = (Data[7] % 6) + 1;
                auto bn3d = torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(features_3d));
                
                // Create appropriate input
                torch::Tensor input_3d = torch::rand({1, features_3d, 4, 4, 4});
                
                // Test training mode
                bn3d->train();
                torch::Tensor out_3d_train = bn3d->forward(input_3d);
                
                // Test eval mode (frozen stats)
                bn3d->eval();
                torch::Tensor out_3d_eval = bn3d->forward(input_3d);
                
            } catch (const std::exception& e) {
                // Ignore exceptions
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
