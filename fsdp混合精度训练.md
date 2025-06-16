# pytorch fsdp 混合精度训练
混合精度基本上是大模型训练的标配，不单能够节省很多显存，还可以提高训练速度，且精度几乎不受损失。
来看下 pytroch fsdp 的混合精度训练如何使用，以及遇到的一些坑。
## HF模型加载
使用 AutoModel.from_pretrained 加载 HF 模型，直接调用接口，这其中有和精度相关的参数。
如果是纯文本模型，那么在config.json中的 text_config 下会有一个 "torch_dtype" 字段。
如果是多模态模型，那么还会有一个 vision_config 下有一个类似的 "torch_dtype" 字段，这两个一般情况下是一样的。
```
# 首先 load 一下 config
model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
# 如果在这里指定 torch_dtype，transformers 会无视模型 config.json 中的 torch_dtype，转而使用显示指定的dtype。
AutoModel.from_pretrained(local_path, torch_dtype=torch.bfloat16, config=model_config, trust_remote_code=trust_remote_code)
```
## fsdp 封装
加载完模型后，就需要把模型转换为 fsdp module，
## 设置混合精度参数
```
param_dtype = torch.bfloat16
reduce_dtype = torch.float32
buffer_dtype = torch.float32

mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)
```
## ckpt 保存和加载
在存ckpt的时候发现一些fsdp和HF的坑
```
cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
    state_dict = model.state_dict()
```
首先用一个 context 拿到完整的参数字典，
然后
```
model.save_pretrained(local_path, state_dict=state_dict)
```
这样保存下来的是 fp32 的模型，这里比较奇怪，因为混合精度设置的时候，我们设置的是 param_dtype = torch.bfloadt16，按理说这里会保存成 bf16 才对
于是我尝试把 state_dict 转换成 bf16 再save
```
bf16_state_dict = {
    k: v.to(torch.bfloat16) if torch.is_floating_point(v) else v
    for k, v in state_dict.items()
}
model.save_pretrained(local_path, state_dict=bf16_state_dict)
```
但是这样存的依旧是 fp32 模型，究其原因，model 是一个 fp32 的模型，所以使用 .save_pretrained 接口仍然会保存 fp32 的模型，并且会把 bf16 转为 fp32 再保存，很不合理。
所以这里只能再创建一个新的相同结构的 bf16 模型，再用这个模型的 .save_pretrained 方法保存（感觉huggingface这里实现的不太合理，应该能通过参数修改保存什么类型的模型最好）
