# Qwen-SigLIP Multimodal Learning
尝试使用 Qwen2.5-0.5B 和 SigLIP 搭建的简单多模态模型。分享训练和 SFT 相关代码,记录一下探索和学习的过程。欢迎一起交流讨论~

# 使用说明
**模型：**
- Qwen/Qwen2.5-0.5B-Instruct：http://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- google/siglip-base-patch16-224：https://huggingface.co/google/siglip-base-patch16-224

**数据：**
- 预训练数据：https://huggingface.co/SYSUMage/LLaVA-CC3M-Pretrain-595K
- 中文文本数据：https://huggingface.co/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions

**SFT数据:**
- SFT数据：https://huggingface.co/datasets/jingyaogong/minimind-v_dataset

## 训练配置
- 显卡 & 显存：vGPU-32GB(32GB) * 2 （通过 Accelertae 库简单实现单机多卡训练）
- CPU：24 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz
- 内存：180GB
- 硬盘：50GB

# 在线演示
Link：

# 心得
在实现过程中的一些心得和踩坑记录:
- 1、数据处理环节使用 chat template 和特殊的 image_pad token 来处理多模态输入,可以让文本和图像的拼接更加自然。同时做了异常处理,防止读取图片失败导致训练中断
- 2、一开始出现了显存 OOM 报错，通过将图像 token 从 196 压缩到 49 个,大大减少了显存占用
- 3、使用两个线性层来对齐视觉和语言模型的特征维度,结构简单但效果不错；冻结基础模型的参数可以让训练更加稳定,也能节省显存
    PS：开始只冻结 SigLIP，训练效果比较差；后来把 Qwen 参数也冻结之后，效果明显变好

# TODO
- 进一步优化模型效果
- 尝试其他模态的多模态大模型，比如语言视频等

# 参考资料
- Qwen：https://github.com/QwenLM/Qwen
- SigLIP：https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/image_text/siglip

## 以探索为目标的项目,代码实现可能存在不足之处。如果您发现任何问题,欢迎提出issue或PR，提前感谢您的指正