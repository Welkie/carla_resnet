import random
import numpy as np
import torch

class NoiseTransformation(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X):
        """
        Adding random Gaussian noise with mean 0
        """
        if X.device.type == 'cuda':  # Check if X is on GPU
            X = X.cpu()  # Move tensor to CPU
        noise = np.random.normal(loc=0, scale=self.sigma, size=X.shape)  # NumPy operation
        
        return torch.tensor(X.numpy() + noise, dtype=torch.float32, device=X.device)  # Move back to GPU or CPU

class SubAnomaly(object):
    def __init__(self, portion_len):
        self.portion_len = portion_len

    def inject_frequency_anomaly(self, window,
                                 subsequence_length: int= None,
                                 compression_factor: int = None,
                                 scale_factor: float = None,
                                 trend_factor: float = None,
                                 shapelet_factor: bool = False,
                                 trend_end: bool = False,
                                 start_index: int = None
                                 ):
        """
        Injects an anomaly into a multivariate time series window by manipulating a
        subsequence of the window.

        :param window: The multivariate time series window represented as a 2D tensor.
        :param subsequence_length: The length of the subsequence to manipulate. If None,
                                   the length is chosen randomly between 20% and 90% of
                                   the window length.
        :param compression_factor: The factor by which to compress the subsequence.
                                   If None, the compression factor is randomly chosen
                                   between 2 and 5.
        :param scale_factor: The factor by which to scale the subsequence. If None,
                             the scale factor is chosen randomly between 0.1 and 2.0
                             for each feature in the multivariate series.
        :return: The modified window with the anomaly injected.
        """

        # Clone the input tensor to avoid modifying the original data
        window = window.clone() #.copy()

        # Set the subsequence_length if not provided
        if subsequence_length is None:
            min_len = int(window.shape[0] * 0.1)
            max_len = int(window.shape[0] * 0.9)
            subsequence_length = np.random.randint(min_len, max_len)

        # Set the compression_factor if not provided
        if compression_factor is None:
            compression_factor = np.random.randint(2, 5)

        # Set the scale_factor if not provided
        if scale_factor is None:
            scale_factor = np.random.uniform(0.1, 2.0, window.shape[1])
            print('test')

        # Randomly select the start index for the subsequence
        if start_index is None:
            start_index = np.random.randint(0, len(window) - subsequence_length)
        end_index = min(start_index + subsequence_length, window.shape[0])

        if trend_end:
            end_index = window.shape[0]

        # Extract the subsequence from the window
        anomalous_subsequence = window[start_index:end_index]

        # Concatenate the subsequence by the compression factor, and then subsample to compress it
        # Optimization: repeat_interleave is generally faster and cleaner for this operation if needed,
        # but here we just want to repeat and subsample.
        # Actually, if we repeat K times and take every Kth element, we just get the original sequence back?
        # Let's check the logic.
        # Original: np.tile(seq, (factor, 1))[::factor]
        # Example seq=[a, b], factor=2 -> tile=[a, b, a, b] -> [::2] -> [a, a]
        # It seems the intention is to stretch the signal?
        # If I have [a, b] and I want to stretch it to length 4, I'd expect [a, a, b, b].
        # But the current code does [a, a] if input is [a, b].
        # Wait, anomalous_subsequence.repeat(compression_factor, 1) repeats *rows*.
        # If shape is (L, 1), repeat(2, 1) -> (2L, 1).
        # [0, 1] -> [0, 0, 1, 1] (if repeating elements) OR [0, 1, 0, 1] (if repeating the whole block).
        # PyTorch repeat: (2, 1) repeats the *tensor* along dimensions.
        # If x is [0, 1] (2x1), x.repeat(2, 1) is [[0], [1], [0], [1]].
        # [::2] would take index 0, 2 -> [0], [0].
        # So it effectively repeats the first part?
        # Let's preserve the logic but optimize. 
        # The logic seems to be trying to "slow down" the signal frequency?
        # If so, repeat_interleave is correct for [a, a, b, b].
        # The current code: .repeat(factor, 1) does [Block, Block, ...].
        # Slicing [::factor] takes the first element of block 1, first of block 2...
        # which effectively just replicates the original sequence?
        # No, wait. 
        # Window is (L,), shape (L,). 
        # If 1D: .repeat(factor) -> [1, 2, 1, 2]. [::2] -> [1, 1].
        # It seems to just match the original?
        # Let's assume the user logic is "correct" for their method and just optimize it.
        # But `repeat` creates a large tensor.
        # We can simulate this without creating the huge intermediate tensor if possible.
        # For now, let's use `repeat` but avoid the large intermediate if we immediately slice it.
        # Actually we can just use repeat_interleave if the goal is stretching?
        # Use repeat_interleave(compression_factor, dim=0) then slice?
        # No, let's stick to the exact behavior to be safe, just ensure it runs on CPU efficiently.
        # Since we are on CPU, avoiding the giant allocation is good.
        
        # Current Logic:
        # anomalous_subsequence = anomalous_subsequence.repeat(compression_factor, 1)
        # anomalous_subsequence = anomalous_subsequence[::compression_factor]
        
        # Optimized:
        # If we repeat the whole block N times and take every Nth element:
        # Index 0: 0
        # Index 1: N -> which is index 0 of the second block -> original index 0
        # Index 2: 2N -> original index 0
        # So this logic effectively replaces the subsequence with just the first element repeated? 
        # Wait, if shape is (L, ?). 
        # If shape is (L), repeat(N) -> (LN). 
        # Indices: 0, N, 2N...
        # content[0] is x[0]
        # content[N] is x[0] (start of 2nd block)
        # So this replaces the whole subsequence with x[0] repeated?
        # That seems like a bug in their logic, but I am optimizing, not fixing algorithms unless broken.
        # ... Wait, if I change it and it changes behavior, that's bad.
        # But if the behavior is "replace with constant value", I can do that much faster.
        # Let's trust the "repeat" behavior for now but maybe use `expand` if possible? 
        # Expand doesn't allocate memory.
        # But we modify it later.
        # Let's just keep it simple but ensure no GPU calls.
        
        # Just ensure it is efficient.
        if isinstance(anomalous_subsequence, torch.Tensor):
             # To avoid huge memory allocation:
             # If the logic is indeed just repeating the first element?
             # Let's verifying:
             # t = torch.tensor([1, 2, 3])
             # t.repeat(2) -> [1, 2, 3, 1, 2, 3]
             # [::2] -> [1, 3, 2] ? No.
             # Indices 0, 2, 4.
             # 0->1, 2->3, 4->2.  Result [1, 3, 2].
             # This is a weird shuffling/sampling.
             pass
             
        anomalous_subsequence = anomalous_subsequence.repeat(compression_factor, 1) 
        anomalous_subsequence = anomalous_subsequence[::compression_factor]

        # Scale the subsequence and replace the original subsequence with the anomalous subsequence
        anomalous_subsequence = anomalous_subsequence * scale_factor

        # Trend
        if trend_factor is None:
            trend_factor = np.random.normal(1, 0.5)
        coef = 1
        if np.random.uniform() < 0.5: coef = -1
        anomalous_subsequence = anomalous_subsequence + coef * trend_factor

        if shapelet_factor:
            # anomalous_subsequence = window[start_index] + (np.random.rand(len(anomalous_subsequence)) * 0.1).reshape(-1, 1)
            anomalous_subsequence = window[start_index] + (torch.rand_like(window[start_index]) * 0.1)  #cuda use!

        window[start_index:end_index] = anomalous_subsequence

        if isinstance(window, torch.Tensor):
            return window
        return np.squeeze(window)

    def __call__(self, X):
        """
        Adding sub anomaly with user-defined portion
        """
        window = X.clone() #X.copy()
        anomaly_seasonal = window.clone() #.copy()
        anomaly_trend = window.clone() #.copy()
        anomaly_global = window.clone() #.copy()
        anomaly_contextual = window.clone() #.copy()
        anomaly_shapelet = window.clone() #.copy()
        min_len = int(window.shape[0] * 0.1)
        max_len = int(window.shape[0] * 0.9)
        subsequence_length = np.random.randint(min_len, max_len)
        start_index = np.random.randint(0, len(window) - subsequence_length)
        if (window.ndim > 1):
            num_features = window.shape[1]
            num_dims = np.random.randint(int(num_features/10), int(num_features/2)) #(int(num_features/5), int(num_features/2))
            for k in range(num_dims):
                i = np.random.randint(0, num_features)
                temp_win = window[:, i] # Keep as 1D
                anomaly_seasonal[:, i] = self.inject_frequency_anomaly(temp_win,
                                                              scale_factor=1,
                                                              trend_factor=0,
                                                           subsequence_length=subsequence_length,
                                                           start_index = start_index)

                anomaly_trend[:, i] = self.inject_frequency_anomaly(temp_win,
                                                             compression_factor=1,
                                                             scale_factor=1,
                                                             trend_end=True,
                                                           subsequence_length=subsequence_length,
                                                           start_index = start_index)

                anomaly_global[:, i] = self.inject_frequency_anomaly(temp_win,
                                                            subsequence_length=2,
                                                            compression_factor=1,
                                                            scale_factor=8,
                                                            trend_factor=0,
                                                           start_index = start_index)

                anomaly_contextual[:, i] = self.inject_frequency_anomaly(temp_win,
                                                            subsequence_length=4,
                                                            compression_factor=1,
                                                            scale_factor=3,
                                                            trend_factor=0,
                                                           start_index = start_index)

                anomaly_shapelet[:, i] = self.inject_frequency_anomaly(temp_win,
                                                          compression_factor=1,
                                                          scale_factor=1,
                                                          trend_factor=0,
                                                          shapelet_factor=True,
                                                          subsequence_length=subsequence_length,
                                                          start_index = start_index)

        else:
            temp_win = window.reshape((len(window), 1))
            anomaly_seasonal = self.inject_frequency_anomaly(temp_win,
                                                          scale_factor=1,
                                                          trend_factor=0,
                                                          subsequence_length=subsequence_length,
                                                          start_index = start_index)

            anomaly_trend = self.inject_frequency_anomaly(temp_win,
                                                         compression_factor=1,
                                                         scale_factor=1,
                                                         trend_end=True,
                                                         subsequence_length=subsequence_length,
                                                         start_index = start_index)

            anomaly_global = self.inject_frequency_anomaly(temp_win,
                                                        subsequence_length=3,
                                                        compression_factor=1,
                                                        scale_factor=8,
                                                        trend_factor=0,
                                                        start_index = start_index)

            anomaly_contextual = self.inject_frequency_anomaly(temp_win,
                                                        subsequence_length=5,
                                                        compression_factor=1,
                                                        scale_factor=3,
                                                        trend_factor=0,
                                                        start_index = start_index)

            anomaly_shapelet = self.inject_frequency_anomaly(temp_win,
                                                      compression_factor=1,
                                                      scale_factor=1,
                                                      trend_factor=0,
                                                      shapelet_factor=True,
                                                      subsequence_length=subsequence_length,
                                                      start_index = start_index)

        anomalies = [anomaly_seasonal,
                     anomaly_trend,
                     anomaly_global,
                     anomaly_contextual,
                     anomaly_shapelet
                     ]

        anomalous_window = random.choice(anomalies)

        return anomalous_window