#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has 4 dimensions (N, C, H, W) for UpsamplingNearest2d
        if (input.dim() < 4) {
            while (input.dim() < 4) {
                input = input.unsqueeze(0);
            }
        } else if (input.dim() > 4) {
            // Flatten extra dimensions into batch
            auto sizes = input.sizes();
            int64_t batch = 1;
            for (int i = 0; i < input.dim() - 3; i++) {
                batch *= sizes[i];
            }
            input = input.reshape({batch, sizes[input.dim()-3], sizes[input.dim()-2], sizes[input.dim()-1]});
        }
        
        // Ensure input has positive spatial dimensions
        if (input.size(2) <= 0 || input.size(3) <= 0) {
            return 0;
        }
        
        // Extract scale factors or output size parameters from the remaining data
        double scale_h = 1.0;
        double scale_w = 1.0;
        int64_t output_h = 0;
        int64_t output_w = 0;
        bool use_size = false;
        
        if (offset + sizeof(uint16_t) <= Size) {
            uint16_t scale_h_bits;
            std::memcpy(&scale_h_bits, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            // Scale factor between 0.5 and 4.0 to avoid memory issues
            scale_h = 0.5 + (static_cast<double>(scale_h_bits % 350) / 100.0);
        }
        
        if (offset + sizeof(uint16_t) <= Size) {
            uint16_t scale_w_bits;
            std::memcpy(&scale_w_bits, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            // Scale factor between 0.5 and 4.0 to avoid memory issues
            scale_w = 0.5 + (static_cast<double>(scale_w_bits % 350) / 100.0);
        }
        
        // Decide whether to use size or scale factor
        if (offset < Size) {
            use_size = Data[offset++] % 2 == 0;
        }
        
        if (use_size) {
            if (offset + sizeof(uint16_t) <= Size) {
                uint16_t h_bits;
                std::memcpy(&h_bits, Data + offset, sizeof(uint16_t));
                offset += sizeof(uint16_t);
                output_h = (h_bits % 64) + 1; // Size between 1 and 64
            } else {
                output_h = input.size(2); // Default to same size
            }
            
            if (offset + sizeof(uint16_t) <= Size) {
                uint16_t w_bits;
                std::memcpy(&w_bits, Data + offset, sizeof(uint16_t));
                offset += sizeof(uint16_t);
                output_w = (w_bits % 64) + 1; // Size between 1 and 64
            } else {
                output_w = input.size(3); // Default to same size
            }
        }
        
        // Create UpsamplingNearest2d module and apply
        torch::Tensor output;
        
        if (use_size) {
            torch::nn::Upsample upsampler(
                torch::nn::UpsampleOptions()
                    .size(std::vector<int64_t>{output_h, output_w})
                    .mode(torch::kNearest)
            );
            output = upsampler->forward(input);
        } else {
            torch::nn::Upsample upsampler(
                torch::nn::UpsampleOptions()
                    .scale_factor(std::vector<double>{scale_h, scale_w})
                    .mode(torch::kNearest)
            );
            output = upsampler->forward(input);
        }
        
        // Try to access output properties to ensure computation is complete
        auto output_sizes = output.sizes();
        auto output_dtype = output.dtype();
        (void)output_sizes;
        (void)output_dtype;
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}