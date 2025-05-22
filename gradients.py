import torch

class KernelISEstimator:
    def __init__(self, data, target_policy, logging_policy, kernel, tau, marginal_density_model, num_epochs=10):
        """
        data: list of tuples (x_i, y_i, r_i)
        target_policy: evaluation policy π (supports sample_latent, sample_action, log_grad)
        logging_policy: logging policy π₀ (supports sample_action(x))
        kernel: function K(y, y', x, tau)
        tau: temperature for kernel
        marginal_density_model: model that approximates logging marginal density π₀(y | x)
        num_epochs: training iterations for h_model
        """
        self.data = data
        self.target_policy = target_policy
        self.logging_policy = logging_policy
        self.kernel = kernel
        self.tau = tau
        self.marginal_density_model = marginal_density_model
        self.num_epochs = num_epochs

    def estimateLMD(self):
        """Train h_model to estimate logging marginal density."""
        for _ in range(self.num_epochs):
            for x_i, y_i, _ in self.data:
                sampled_action = self.logging_policy.sample_action(x_i)
                k_val = self.kernel(y_i, sampled_action, x_i, self.tau)
                predicted_density = self.marginal_density_model.predict(x_i, y_i)
                loss = (predicted_density - k_val) ** 2
                self.marginal_density_model.update(x_i, y_i, loss)

    def _estimate_policy_gradient(self) -> torch.Tensor:
        grad = []
        for x_i, y_i, r_i in self.data:
            W_k, A_k = self.target_policy.sample_latent(x_i) 
            y = self.target_policy.sample_action(x_i, A_k, W_k)

            k_val = self.kernel(y, y_i, x_i, self.tau)
            density_estimate = self.marginal_density_model.predict(x_i, y_i)
            is_weight = k_val / density_estimate

            log_grad = self.target_policy.log_grad(W_k, A_k, x_i)
            grad.append(-is_weight.detach() * r_i.detach() * log_grad)

        return torch.stack(grad).mean()

    def estimate_policy_value(self) -> float:
        return self.estimate_policy_value_tensor().item()
    
    def estimate_policy_value_tensor(self) -> torch.Tensor:
        self.estimateLMD()
        return self._estimate_policy_gradient()