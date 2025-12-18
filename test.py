import torch
print(torch.__version__)
print(torch.cuda.get_device_name(0))
# 這裡應該要顯示 "NVIDIA GeForce RTX 5080"
print(torch.rand(5, 3).cuda()) 
# 這行如果沒報錯，並且印出 Tensor device='cuda:0'，就是成功了！