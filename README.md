# Practice-for-LLM

- **笔记**

  - [哈工大《ChatGPT调研报告》笔记](大模型梳理/哈工大《ChatGPT调研报告》笔记.md)

  - [**GPT3**](大模型梳理/GPT3.md)

  - [InstructGPT](大模型梳理/InstructGPT.md)


- **论文**
- [ ]  **A Survey of Large Language Models** 💥
- [x]  **GPT研究报告**
- [x]  GPT1, 2, 3, 4原文
    - [x]  **Improving language understanding by generative pre-training.**
    - [x]  **Language models are unsupervised multitask learners.**
    - [x]  **Language Models are Few-Shot Learners**
    - [ ]  **GPT-4 Technical Report**
- [x]  InstructGPT ****Training language models to follow instructions with human feedback****
- [x]  Self-Instruct ****Self-Instruct: Aligning Language Model with Self Generated Instructions****
- [x]  LLaMA ****LLaMA: Open and Efficient Foundation Language Models****
- [x]  Alpaca ****Alpaca: A Strong, Replicable Instruction-Following Model****
- [x]  BLOOM 和 BLOOMZ ****BLOOM: A 176B-Parameter Open-Access Multilingual Language Model 写的比较详细，可以作为指南****
- [ ]  OPT 和 OPT-IML  **Scaling Language Model Instruction Meta Learning through the Lens of Generalization**
- [ ]  Lora **LoRA: Low-Rank Adaptation of Large Language Models**
- [ ]  GPT4早期实验报告 ****GPTs are GPTs: An Early Look at the Labor Market Impact Potential of Large Language Models****


**参数规模、数据集、模型能力的探究**

- [ ]  **Scaling laws for neural language models**
- [ ]  **Training compute-optimal large language models**
- [ ]  LLaMA ****LLaMA: Open and Efficient Foundation Language Models****

### 工程方面


- **框架类**
- [ ]  **DP，DDP，ZeRO，FSDP**
- [ ]  **Megatron**
- [ ]  [**microsoft/DeepSpeed**](https://github.com/microsoft/DeepSpeed) 💥
- [ ]  [**hwchase17/langchain**](https://github.com/hwchase17/langchain) 基于LLM开发应用的SDK 💥
- [ ]  [**facebookresearch / fairscale**](https://github.com/facebookresearch/fairscale) 和deepspeed类似，[Fit More and Train Faster With ZeRO via DeepSpeed and FairScale](https://huggingface.co/blog/zero-deepspeed-fairscale)
- [ ]  [**hpcaitech/ColossalAI**](https://github.com/hpcaitech/ColossalAI) 同上，中文友好
- [ ]  [**huggingface / peft**](https://github.com/huggingface/peft) [PEFT: 在低资源硬件上对十亿规模模型进行参数高效微调](https://zhuanlan.zhihu.com/p/610503561) 实现了lora、prefix tuning、prompt tuning、p-tuning
- [ ]  [**lvwerra / trl**](https://github.com/lvwerra/trl)  TRL - Transformer Reinforcement Learning，实现了PPO Trainer [在一张 24 GB 的消费级显卡上用 RLHF 微调 20B LLMs](https://zhuanlan.zhihu.com/p/616346543) 💥


- **实现类**
- [ ]  [**karpathy/nanoGPT**](https://github.com/karpathy/nanoGPT) 💥
- [ ]  [**yizhongw / self-instruct**](https://github.com/yizhongw/self-instruct) Self-Instruct的代码
- [ ]  [**tatsu-lab/stanford_alpaca**](https://github.com/tatsu-lab/stanford_alpaca) 基于self-instrction的LLaMa指令精调 ****[standford-alpaca微调记录](https://zhuanlan.zhihu.com/p/616119919)**** 💥


- **实践类**
- [ ]  [**tloen / alpaca-lora**](https://github.com/tloen/alpaca-lora) 基于Lora的LLaMa指令精调 ****[训练个中文版ChatGPT没那么难：不用A100，开源Alpaca-LoRA+RTX 4090就能搞定](https://zhuanlan.zhihu.com/p/617221484)****
- [ ]  [**LC1332 / Chinese-alpaca-lora**](https://github.com/LC1332/Chinese-alpaca-lora) 基于Lora的中文LLaMa指令精调 ****[Alpaca-Lora 轻量级 ChatGPT 的开源实现](https://zhuanlan.zhihu.com/p/615646636)****
- [ ]  [**LianjiaTech / BELLE**](https://github.com/LianjiaTech/BELLE) 中文LLaMa指令精调
- [ ]  [**databrickslabs / dolly**](https://github.com/databrickslabs/dolly) [0门槛克隆ChatGPT！30分钟训完，60亿参数性能堪比GPT-3.5](https://zhuanlan.zhihu.com/p/617345561)
- [ ]  [**LLM省内存方法**](https://zhuanlan.zhihu.com/p/616858352)
- [ ]  ****[总结开源可用的Instruct/Prompt Tuning数据](https://zhuanlan.zhihu.com/p/615277009)****
- [ ]  ****[总结当下可用的大模型LLMs](https://zhuanlan.zhihu.com/p/611403556)****
- [ ]  ****[使用 DeepSpeed 和 Accelerate 进行超快 BLOOM 模型推理](https://zhuanlan.zhihu.com/p/602142554?utm_medium=social&utm_oi=46337705902080&utm_psn=1622986060279119872&utm_source=wechat_session)****
- [ ]  [**dair-ai/Prompt-Engineering-Guide**](https://github.com/dair-ai/Prompt-Engineering-Guide) [中文版](https://github.com/wangxuqi/Prompt-Engineering-Guide-Chinese)  prompt工程师指南 
- [ ]  ****[复现ChatGPT的难点与平替](https://zhuanlan.zhihu.com/p/607847588)****
- [ ]  ****[平替chatGPT的开源方案](https://zhuanlan.zhihu.com/p/618926239?utm_medium=social&utm_oi=46337705902080&utm_psn=1625959298282090496&utm_source=wechat_session)****
- [ ]  **[ChatGPT的低成本“平替”当下实现路线](https://mp.weixin.qq.com/s/5SNJLLs9Hw0uvjkcLQflvA)**
- [ ]  ****[Stealing Large Language Models: 关于对ChatGPT进行模型窃取的一些工作](https://zhuanlan.zhihu.com/p/621179159?utm_medium=social&utm_oi=46337705902080&utm_psn=1629976091313029120&utm_source=wechat_session)****
- [ ]  **[微软开源Deep Speed Chat，人人拥有ChatGPT！](https://mp.weixin.qq.com/s/6y5e9MvSXXLCj-q7FI08Kw)** 💥