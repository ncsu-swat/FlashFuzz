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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for BatchNorm2d + ReLU
        uint8_t num_features = 0;
        if (offset < Size) {
            num_features = Data[offset++] % 64 + 1; // Ensure at least 1 feature
        } else {
            num_features = 3; // Default value
        }
        
        // Create parameters for BatchNorm2d
        double eps = 1e-5;
        double momentum = 0.1;
        bool affine = true;
        bool track_running_stats = true;
        
        // Modify parameters if we have more data
        if (offset + 3 < Size) {
            // Extract eps (small positive value)
            uint32_t eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(eps_raw));
            offset += sizeof(eps_raw);
            eps = std::abs(static_cast<double>(eps_raw) / std::numeric_limits<uint32_t>::max()) + 1e-10;
            
            // Extract momentum (between 0 and 1)
            if (offset < Size) {
                momentum = static_cast<double>(Data[offset++]) / 255.0;
            }
            
            // Extract boolean parameters
            if (offset < Size) {
                affine = Data[offset++] & 1;
            }
            
            if (offset < Size) {
                track_running_stats = Data[offset++] & 1;
            }
        }
        
        // Create BatchNorm2d module
        torch::nn::BatchNorm2d bn(
            torch::nn::BatchNorm2dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Create ReLU module
        torch::nn::ReLU relu;
        
        // Ensure input has correct shape for BatchNorm2d (N, C, H, W)
        // If not, reshape it to a valid shape
        if (input.dim() != 4) {
            // Create a valid 4D tensor shape
            std::vector<int64_t> new_shape;
            
            // Batch size
            int64_t batch_size = 1;
            
            // Channels should match num_features
            int64_t channels = num_features;
            
            // Height and width can be any positive value
            int64_t height = 2;
            int64_t width = 2;
            
            // If we have more data, use it to determine dimensions
            if (offset + 3 < Size) {
                batch_size = (Data[offset++] % 4) + 1;
                height = (Data[offset++] % 8) + 1;
                width = (Data[offset++] % 8) + 1;
            }
            
            new_shape = {batch_size, channels, height, width};
            
            // Create a new tensor with the right shape
            input = torch::ones(new_shape, input.options());
        } else {
            // If already 4D, ensure the channel dimension matches num_features
            auto input_sizes = input.sizes();
            if (input_sizes[1] != num_features) {
                std::vector<int64_t> new_shape = {
                    input_sizes[0],
                    num_features,
                    input_sizes[2],
                    input_sizes[3]
                };
                input = torch::ones(new_shape, input.options());
            }
        }
        
        // Apply BatchNorm2d followed by ReLU to simulate BNReLU2d
        torch::Tensor bn_output = bn->forward(input);
        torch::Tensor output = relu->forward(bn_output);
        
        // Try training mode as well
        bn->train();
        relu->train();
        torch::Tensor bn_train_output = bn->forward(input);
        torch::Tensor train_output = relu->forward(bn_train_output);
        
        // Try eval mode
        bn->eval();
        relu->eval();
        torch::Tensor bn_eval_output = bn->forward(input);
        torch::Tensor eval_output = relu->forward(bn_eval_output);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
