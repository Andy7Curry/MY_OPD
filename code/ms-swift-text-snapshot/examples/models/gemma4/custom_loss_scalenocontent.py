import re
from typing import List, Tuple

from swift.loss_scale import LossScale, loss_scale_map


class ThinkAnswerLossScaleNoContent(LossScale):
    """
    自定义损失权重（无 content 版本，think:answer = 1:2）：
    - <think>...</think> 整段，包括标签：0.5
    - <answer>...</answer> 整段，包括标签：1.0
    - 其他 assistant 文本（如无 <answer> 的收尾发言、无 <think> 的残留样本）：1.0
    """

    # 因为存在 0.5，不是纯 0/1 loss scale
    is_binary = False

    tag_pattern = re.compile(
        r'<think>.*?</think>|<answer>.*?</answer>',
        re.DOTALL
    )

    def get_loss_scale(
        self,
        context: str,
        **kwargs
    ) -> Tuple[List[str], List[float]]:

        # 非字符串情况直接保持默认权重
        if not isinstance(context, str):
            return [context], [1.0]

        contexts = []
        loss_scales = []

        last_end = 0

        for match in self.tag_pattern.finditer(context):
            start, end = match.span()

            # 标签前面的普通文本
            if start > last_end:
                contexts.append(context[last_end:start])
                loss_scales.append(1.0)

            # <think>...</think> 或 <answer>...</answer>，包括标签本身
            tag_text = match.group()
            scale = 0.5 if tag_text.startswith('<think>') else 1.0
            contexts.append(tag_text)
            loss_scales.append(scale)

            last_end = end

        # 最后一段普通文本
        if last_end < len(context):
            contexts.append(context[last_end:])
            loss_scales.append(1.0)

        # 没有匹配到 think/answer
        if not contexts:
            return [context], [1.0]

        return contexts, loss_scales


# 注册
loss_scale_map['think_answer_weight_nocontent'] = ThinkAnswerLossScaleNoContent
