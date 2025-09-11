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
        
        // Ensure input has at least 3 dimensions (N, C, H, W) for UpsamplingNearest2d
        if (input.dim() < 3) {
            // Expand dimensions if needed
            while (input.dim() < 3) {
                input = input.unsqueeze(0);
            }
            // Add one more dimension if needed to make it 4D (N, C, H, W)
            if (input.dim() == 3) {
                input = input.unsqueeze(0);
            }
        }
        
        // Extract scale factors or output size parameters from the remaining data
        double scale_h = 1.0;
        double scale_w = 1.0;
        int64_t output_h = 0;
        int64_t output_w = 0;
        bool use_size = false;
        
        if (offset + sizeof(uint32_t) <= Size) {
            uint32_t scale_h_bits;
            std::memcpy(&scale_h_bits, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            scale_h = static_cast<double>(scale_h_bits) / 1000.0 + 0.1; // Ensure positive scale
        }
        
        if (offset + sizeof(uint32_t) <= Size) {
            uint32_t scale_w_bits;
            std::memcpy(&scale_w_bits, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            scale_w = static_cast<double>(scale_w_bits) / 1000.0 + 0.1; // Ensure positive scale
        }
        
        // Decide whether to use size or scale factor
        if (offset < Size) {
            use_size = Data[offset++] % 2 == 0;
        }
        
        if (use_size && offset + sizeof(int64_t) <= Size) {
            std::memcpy(&output_h, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            output_h = std::abs(output_h) % 100 + 1; // Ensure positive size
        }
        
        if (use_size && offset + sizeof(int64_t) <= Size) {
            std::memcpy(&output_w, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            output_w = std::abs(output_w) % 100 + 1; // Ensure positive size
        }
        
        // Create UpsamplingNearest2d module
        torch::nn::Upsample upsampler = nullptr;
        
        if (use_size) {
            upsampler = torch::nn::Upsample(
                torch::nn::UpsampleOptions().size(std::vector<int64_t>{output_h, output_w}).mode(torch::kNearest)
            );
        } else {
            upsampler = torch::nn::Upsample(
                torch::nn::UpsampleOptions().scale_factor(std::vector<double>{scale_h, scale_w}).mode(torch::kNearest)
            );
        }
        
        // Apply upsampling
        torch::Tensor output = upsampler->forward(input);
        
        // Try to access output properties to ensure computation is complete
        auto output_sizes = output.sizes();
        auto output_dtype = output.dtype();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
