

#     TorchScript implementation of Shapley Value Sampling.
#     See https://captum.ai/api/shapley_value_sampling.html for
#     reference. From that source:
#     A perturbation based approach to compute attribution, based on the concept
#     of Shapley Values from cooperative game theory. This method involves taking
#     a random permutation of the input features and adding them one-by-one to the
#     given baseline. The output difference after adding each feature corresponds
#     to its attribution, and these difference are averaged when repeating this
#     process n_samples times, each time choosing a new random permutation of
#     the input features.


# Script inputs

# Tensors:
#     tensors[0] - x : Input tensor to the model
#     tensors[1] - baselines : Optional - reference values which replace each feature when
#         ablated; if no baselines are provided, baselines are set
#         to all zeros

# Keys:
#     keys[0] - model_key: Redis key name where the model is stored as RedisAI model.
        
# Args:
#     args[0] - n_samples: number of random feature permutations performed
#     args[1] - number_of_outputs - number of model outputs
#     args[2] - output_tensor_index - index of the tested output tensor
#     args[3] - Optional - target: output indices for which Shapley Value Sampling is
#             computed; if model returns a single scalar, target can be
#             None



def generate_permutations(x, n_samples:int):
    n_features = torch.numel(x[0])
    return [torch.randperm(n_features) for _ in range(n_samples)]

def index_with_target(x, target:int):
    # Transpose is used in order to avoid iterating batches
    x_t = torch.transpose(x, 0, -1)
    x_target_t = x_t[target]
    return torch.transpose(x_target_t, 0, -1)

# binary classification - no need for target (output size is 1)
# multiple output (output vector - target specifies the output index to explain.
def shapely_sample(tensors: List[Tensor], keys: List[str], args: List[str]):
    model_key = keys[0]
    x = tensors[0]
    n_samples = int(args[0])
    number_of_outputs = int(args[1])
    output_tensor_index = int(args[2])
    if(len(args) == 4):
        target = int(args[3])
    else:
        target = None


    attrib = torch.zeros_like(x)

    if len(tensors) == 2:
        baselines = tensors[1]
    else:
        baselines = torch.zeros_like(x)

    permutations = generate_permutations(x, n_samples)

    n_features = torch.numel(x[0])

    for permutation in permutations:
        current = x.clone()
        for batch_i in range(current.shape[0]):
            current[batch_i] = baselines[
                int(torch.randint(low=0, high=baselines.shape[0], size=(1,)))
            ]
        prev_out = redisAI.model_execute(model_key, [current], number_of_outputs)
        prev_out_target = index_with_target(prev_out[output_tensor_index], target) if target is not None else prev_out[output_tensor_index]


        # Check how current output target differs from the prev output target
        for feature_i in range(n_features):
            permuted_feature_i = int(permutation[feature_i])
            current[:, permuted_feature_i] = x[:, permuted_feature_i]
            out = redisAI.model_execute(model_key, [current], number_of_outputs)
            out_target = index_with_target(out[output_tensor_index], target) if target is not None else out[output_tensor_index]
            # Add the contribution of the feature added in current iteration
            attrib[:, permuted_feature_i] += out_target - prev_out_target
            prev_out_target = out_target

    attrib /= n_samples

    return attrib
