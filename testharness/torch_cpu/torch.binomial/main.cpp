#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        
        // Create the count tensor (number of trials)
        torch::Tensor count_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have some data left for the probability
        if (offset >= Size) {
            return 0;
        }
        
        // Create the probability tensor
        torch::Tensor prob_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Binomial requires non-negative integer counts
        // Convert to float first (binomial actually expects float tensors for count)
        torch::Tensor count;
        if (count_tensor.scalar_type() == torch::kBool) {
            count = count_tensor.to(torch::kFloat);
        } else {
            count = torch::abs(count_tensor.to(torch::kFloat));
        }
        
        // Clamp count to reasonable values to avoid excessive computation
        count = torch::clamp(count, 0, 100);
        
        // Binomial requires probabilities in [0, 1] as float
        torch::Tensor prob;
        if (prob_tensor.scalar_type() == torch::kBool) {
            prob = prob_tensor.to(torch::kFloat);
        } else {
            prob = prob_tensor.to(torch::kFloat);
            // Clamp to [0, 1] range
            prob = torch::clamp(torch::abs(prob), 0.0, 1.0);
            // If any values exceed 1 after abs, normalize them
            auto max_val = torch::max(prob).item<float>();
            if (max_val > 1.0) {
                prob = prob / max_val;
            }
        }
        
        // Ensure shapes are broadcastable by using same shape
        // Get shapes and broadcast to common shape
        auto count_sizes = count.sizes().vec();
        auto prob_sizes = prob.sizes().vec();
        
        // If shapes don't match, reshape prob to match count's shape
        if (count_sizes != prob_sizes) {
            try {
                // Try to broadcast - this may fail silently if incompatible
                auto expanded = torch::broadcast_tensors({count, prob});
                count = expanded[0].contiguous();
                prob = expanded[1].contiguous();
            } catch (...) {
                // If broadcast fails, reshape prob to match count
                int64_t count_numel = count.numel();
                if (count_numel > 0) {
                    prob = prob.flatten().slice(0, 0, std::min(prob.numel(), count_numel));
                    if (prob.numel() < count_numel) {
                        // Repeat to fill
                        int64_t repeats = (count_numel + prob.numel() - 1) / prob.numel();
                        prob = prob.repeat({repeats}).slice(0, 0, count_numel);
                    }
                    prob = prob.reshape(count.sizes());
                }
            }
        }
        
        // Apply the binomial operation
        torch::Tensor result = torch::binomial(count, prob);
        
        // Try different variants if we have more data
        if (offset + 1 < Size) {
            uint8_t variant = Data[offset++];
            
            try {
                if (variant % 3 == 0) {
                    // Test with a generator
                    auto gen = torch::make_generator<torch::CPUGeneratorImpl>();
                    result = torch::binomial(count, prob, gen);
                } else if (variant % 3 == 1) {
                    // Test with out tensor - binomial returns float type
                    torch::Tensor out = torch::empty_like(count);
                    torch::binomial_outf(count, prob, c10::nullopt, out);
                    result = out;
                } else {
                    // Test with generator and out tensor
                    auto gen = torch::make_generator<torch::CPUGeneratorImpl>();
                    torch::Tensor out = torch::empty_like(count);
                    torch::binomial_outf(count, prob, gen, out);
                    result = out;
                }
            } catch (...) {
                // Silently ignore variant-specific failures
            }
        }
        
        // Verify the result is valid
        if (result.numel() > 0) {
            // Binomial should return non-negative values
            auto min_val = torch::min(result).item<float>();
            
            if (min_val < 0) {
                throw std::runtime_error("Binomial result out of expected range");
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}