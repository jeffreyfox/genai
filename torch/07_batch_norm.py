import torch

# ✏️ YOUR IMPLEMENTATION HERE

def my_batch_norm(
    x,
    gamma,
    beta,
    running_mean,
    running_var,
    eps=1e-5,
    momentum=0.1,
    training=True,
):
    if training:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, correction=0)
        x = gamma * (x - batch_mean) * torch.rsqrt(batch_var + eps) + beta
        with torch.no_grad():
          running_mean.mul_(1 - momentum).add_(momentum * batch_mean)
          running_var.mul_(1 - momentum).add_(momentum * batch_var)
    else:
        x = gamma * (x - running_mean) * torch.rsqrt(running_var + eps) + beta
    return x


# 🧪 Debug
x = torch.randn(8, 4)
gamma = torch.ones(4)
beta = torch.zeros(4)

# Running stats typically live on the same device and shape as features
running_mean = torch.zeros(4)
running_var = torch.ones(4)

# Training mode: uses batch stats and updates running_mean / running_var
out_train = my_batch_norm(x, gamma, beta, running_mean, running_var, training=True)
print("[Train] Output shape:", out_train.shape)
print("[Train] Column means:", out_train.mean(dim=0))   # should be ~0
print("[Train] Column stds: ", out_train.std(dim=0))    # should be ~1
print("Updated running_mean:", running_mean)
print("Updated running_var:", running_var)

# Inference mode: uses running_mean / running_var only
out_eval = my_batch_norm(x, gamma, beta, running_mean, running_var, training=False)
print("[Eval] Output shape:", out_eval.shape)
