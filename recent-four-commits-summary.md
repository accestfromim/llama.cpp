# 最近四次提交改动说明

本轮工作的：将上级目录 `D:\Speculation Decoding\llama.cpp` 中的官方 llama.cpp 投机解码相关能力，按当前 Fairy2i 分支的兼容需求进行**最小范围移植**。官方参考副本当前 HEAD 为 `c264f65ff9`；本轮围绕 `speculative-simple` 完成了从底层算法、通用框架到示例接入的迁移。

1. **新增 n-gram 投机解码基础能力**（`3702a4f7`）  
   增加 n-gram 简单匹配、映射表匹配和哈希匹配等草稿生成实现，并补充对应的命令行参数、配置结构及构建配置。该能力可基于已生成的 token 历史预测后续 token，减少对独立草稿模型的依赖。

2. **引入按类型组合的投机解码框架**（`f8bd5768`）  
   将原有单一草稿模型逻辑抽象为统一框架，支持 `draft-simple`、多种 n-gram 策略等类型按顺序配置和回退；同时统一了初始化、草稿生成、接受结果反馈等生命周期接口。

3. **示例程序接入新框架**（`d59f0fa9`）  
   `speculative-simple` 改为使用新框架：有草稿模型时默认采用模型草稿，无草稿模型时可直接采用 n-gram 策略；完善生成长度控制、接受率统计和无草稿模型场景的处理。

4. **补齐聊天模板支持**（`77fd72bc`）  
   `speculative-simple` 支持与主程序一致的聊天模板、会话模式和推理相关参数，使聊天模型的 prompt 能按模板构造后再进行投机解码；同步开放相关命令行选项给该示例使用。



## 本阶段移植边界

- **已移植：** `draft-simple` 与 `ngram-simple`、`ngram-map-k`、`ngram-map-k4v`、`ngram-mod` 的统一选择和调用框架，以及 `speculative-simple` 的单序列使用路径。
- **保留兼容：** 原有 draft-model 接口和逻辑继续保留；`draft-simple` 在新框架内复用旧的草稿生成实现，以降低迁移风险。
- **暂不移植：** `llama-server`、server slot/多序列/并行解码、KV cache checkpoint 与上下文移位、EAGLE3、DFlash、MTP/NextN、hidden state/embedding 处理、ngram-cache 封装，以及 Fairy2i kernel 或计算图改动。
- **原因：** 上述能力依赖更完整的 `process`、embedding、状态保存与 server 生命周期接口。本阶段先以 `speculative-simple` 验证公共框架和 n-gram 路径，后续再分阶段向官方能力对齐。
