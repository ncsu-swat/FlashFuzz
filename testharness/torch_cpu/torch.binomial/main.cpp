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
        
        // Create the count tensor (number of trials)
        torch::Tensor count_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have some data left for the probability
        if (offset >= Size) {
            return 0;
        }
        
        // Create the probability tensor
        torch::Tensor prob_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try to make the count tensor valid for binomial
        // Binomial requires non-negative integer counts
        torch::Tensor count;
        if (count_tensor.dtype() == torch::kBool) {
            // Convert boolean to long
            count = count_tensor.to(torch::kLong);
        } else if (torch::isIntegralType(count_tensor.scalar_type(), false)) {
            // For integer types, take absolute value and convert to long
            count = torch::abs(count_tensor).to(torch::kLong);
        } else {
            // For floating point types, convert to long and take absolute value
            count = torch::abs(count_tensor).to(torch::kLong);
        }
        
        // Clamp count to reasonable values to avoid excessive computation
        count = torch::clamp(count, 0, 100);
        
        // Try to make the probability tensor valid for binomial
        // Binomial requires probabilities in [0, 1]
        torch::Tensor prob;
        if (prob_tensor.dtype() == torch::kBool) {
            // Convert boolean to float
            prob = prob_tensor.to(torch::kFloat);
        } else if (torch::isFloatingType(prob_tensor.scalar_type())) {
            // For floating point types, clamp to [0, 1]
            prob = torch::clamp(prob_tensor, 0.0, 1.0);
        } else {
            // For integer types, convert to float and normalize to [0, 1]
            prob = prob_tensor.to(torch::kFloat);
            if (prob.numel() > 0) {
                auto max_val = torch::max(prob).item<float>();
                if (max_val > 0) {
                    prob = prob / max_val;
                } else {
                    // If all values are negative or zero, just use abs and normalize
                    prob = torch::abs(prob);
                    auto new_max = torch::max(prob).item<float>();
                    if (new_max > 0) {
                        prob = prob / new_max;
                    } else {
                        // If still all zeros, just use a tensor of 0.5s
                        prob = torch::ones_like(prob) * 0.5;
                    }
                }
            }
        }
        
        // Apply the binomial operation
        torch::Tensor result = torch::binomial(count, prob);
        
        // Try different variants if we have more data
        if (offset + 1 < Size) {
            uint8_t variant = Data[offset++];
            
            if (variant % 3 == 0 && prob.dim() > 0) {
                // Test with a generator
                torch::Generator gen;
                result = torch::binomial(count, prob, gen);
            } else if (variant % 3 == 1) {
                // Test with out tensor
                torch::Tensor out = torch::empty_like(count, torch::kLong);
                torch::binomial_out(out, count, prob);
                result = out;
            } else {
                // Test with both generator and out tensor
                torch::Generator gen;
                torch::Tensor out = torch::empty_like(count, torch::kLong);
                torch::binomial_out(out, count, prob, gen);
                result = out;
            }
        }
        
        // Verify the result is valid
        if (result.numel() > 0) {
            // Binomial should return non-negative integers
            auto min_val = torch::min(result).item<int64_t>();
            
            if (min_val < 0) {
                throw std::runtime_error("Binomial result out of expected range");
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
