# 记一次vllm bug排查
```
(main_task pid=11534)   File "/usr/local/lib/python3.10/dist-packages/vllm/entrypoints/llm.py", line 464, in generate
(main_task pid=11534)     outputs = self._run_engine(use_tqdm=use_tqdm)
(main_task pid=11534)   File "/usr/local/lib/python3.10/dist-packages/vllm/entrypoints/llm.py", line 1371, in _run_engine
(main_task pid=11534)     step_outputs = self.llm_engine.step()
(main_task pid=11534)   File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/llm_engine.py", line 209, in step
(main_task pid=11534)     outputs = self.engine_core.get_output()
(main_task pid=11534)   File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/core_client.py", line 167, in get_output
(main_task pid=11534)     return self.engine_core.step()
(main_task pid=11534)   File "/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/core.py", line 192, in step
(main_task pid=11534)     output = self.model_executor.execute_model(scheduler_output)
(main_task pid=11534)   File "/usr/local/lib/python3.10/dist-packages/vllm/v1/executor/abstract.py", line 80, in execute_model
(main_task pid=11534)     output = self.collective_rpc("execute_model",
(main_task pid=11534)   File "/usr/local/lib/python3.10/dist-packages/vllm/executor/uniproc_executor.py", line 56, in collective_rpc
(main_task pid=11534)     answer = run_method(self.driver_worker, method, args, kwargs)
(main_task pid=11534)   File "/usr/local/lib/python3.10/dist-packages/vllm/utils.py", line 2216, in run_method
(main_task pid=11534)     return func(*args, **kwargs)
(main_task pid=11534)   File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
(main_task pid=11534)     return func(*args, **kwargs)
(main_task pid=11534)   File "/usr/local/lib/python3.10/dist-packages/vllm/v1/worker/gpu_worker.py", line 242, in execute_model
(main_task pid=11534)     output = self.model_runner.execute_model(scheduler_output)
(main_task pid=11534)   File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
(main_task pid=11534)     return func(*args, **kwargs)
(main_task pid=11534)   File "/usr/local/lib/python3.10/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 996, in execute_model
(main_task pid=11534)     inputs_embeds = self.model.get_input_embeddings(
(main_task pid=11534)   File "/data/juicefs_sharing_data/11171634/code/verl_mm/vivo-verl/verl/v_custom_models/vllm/bluelmv/bluelmv.py", line 554, in get_input_embeddings
(main_task pid=11534)     inputs_embeds = embed_multimodal(
(main_task pid=11534)   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/utils.py", line 408, in embed_multimodal
(main_task pid=11534)     return _merge_multimodal_embeddings(
(main_task pid=11534)   File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/utils.py", line 371, in _merge_multimodal_embeddings
(main_task pid=11534)     raise ValueError(
(main_task pid=11534) ValueError: Attempted to assign 1 x 1963 + 1 x 1963 + 1 x 1963 + 1 x 1963 + 1 x 1963 + 1 x 1963 + 1 x 1963 = 13741 multimodal tokens to 13722 placeholders
```
现象就是设置max_num_batched_tokens比较小的时候，如果丢给它很多数据，就会出现这个错误，一开始是定位到 schedule 的时候一部分text被截断了，由于chunked_prefill，但是encoder部分的embedding还是生成了那么多，最终导致对不上。但我一直觉得这个结论不成立，
vllm官网明确说了multimodal支持chunked_prefill，其次，按理说应该会判断这一条的image_embedding能不能被调度。所以想再debug一下看一下这个问题怎么解决。
经过详细的debug，定位到，vllm 0.8.1 版本确实会把带有image的request截断
vllm随后很快修复了这个问题
在_try_schedule_encoder_inputs中加入了如下的判断，详细见 https://github.com/vllm-project/vllm/blob/v0.8.5.post1/vllm/v1/core/sched/scheduler.py#L596
```
if (self.scheduler_config.disable_chunked_mm_input
                    and num_computed_tokens < start_pos
                    and (num_computed_tokens + num_new_tokens)
                    < (start_pos + num_encoder_tokens)):
                num_new_tokens = start_pos - num_computed_tokens
                break
```
当开启disable_chunked_mm_input并且经过text schedule得到的num_new_tokens数量没有cover掉num_encoder_tokens时，会把num_new_tokens回滚到start_pos，也就是说，整个encoder_tokens的部分不会被调度，只调度start_pos之前的文本部分。
