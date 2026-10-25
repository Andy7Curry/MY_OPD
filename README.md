# gemma OPD 31B~12B


## 项目简介

学生模型为gemma的12B模型，教师模型为31B，任务类型为GUI agent的能力提升
训练基于ms-swift的框架
采用grpo的sampled token模式的OPD对能力提升4个点左右（相比初始的学生模型）
采用top k64 reverseKL的gkd模式对效果无提升，且出现了5次点击同一个无意义位置没有纠偏的bad case出现
采用top k64 forwardKL的gkd模式在训练的loss中发现中间的一个尖峰，并且最终的loss和初始无多大区别，效果直接崩盘
对于top k64 forwardKL的gkd模式监控学生的top 64个token占学生在整个vocab上的原始概率比重Pst，观察到随着训练进行，Pst下降，可能为训练效果不佳的原因，因为理论上蒸馏过程不应该影响整体top 64的原始概率比重，至少应该保持不变，下降只能说明模型学得更分散了
对于top k64 forwardKL的gkd模式监控学生的top 64个token占学生在整个vocab上的原始概率比重Pst————待测
## 一些说明

output部分的forward的监控模式由于数据迁移不完全，还没有训完就迁移过来了


## 环境依赖

ms-swift
对于监控训练中的top K个token占学生在整个vocab上的原始概率比重Pst需要用gkd_trainer.py,gkd_trainer.py.bkd为原始的copy




4. 推送到远端分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request (PR)

## 许可证 (License)

本项目采用 [MIT 许可证](LICENSE) 开源。请查看 LICENSE 文件了解更多信息。
