import re
from typing import List, Tuple

from swift.loss_scale import LossScale, loss_scale_map


class ThinkAnswerLossScale(LossScale):
    """
    自定义损失权重：
    - <think>...</think> 整段，包括标签：0.5
    - <answer>...</answer>：1.0
    - 其他 assistant 文本：1.0
    """

    # 因为存在 0.5，不是纯 0/1 loss scale
    is_binary = False

    think_pattern = re.compile(
        r'<think>.*?</think>',
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

        for match in self.think_pattern.finditer(context):
            start, end = match.span()

            # <think> 前面的普通文本
            if start > last_end:
                contexts.append(context[last_end:start])
                loss_scales.append(1.0)

            # <think>...</think>，包括标签本身
            contexts.append(context[start:end])
            loss_scales.append(0.5)

            last_end = end

        # 最后一段普通文本
        if last_end < len(context):
            contexts.append(context[last_end:])
            loss_scales.append(1.0)

        # 没有匹配到 think
        if not contexts:
            return [context], [1.0]

        return contexts, loss_scales


# 注册
loss_scale_map['think_answer_weight'] = ThinkAnswerLossScale