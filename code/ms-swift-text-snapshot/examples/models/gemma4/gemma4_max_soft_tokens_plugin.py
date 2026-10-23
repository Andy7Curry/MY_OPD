import os

from swift.template.templates.gemma import Gemma4Template


_original_encode = Gemma4Template._encode


def _patched_encode(self, inputs):
    max_soft_tokens = int(
        os.environ.get('GEMMA4_MAX_SOFT_TOKENS', '280')
    )

    if max_soft_tokens not in {70, 140, 280, 560, 1120}:
        raise ValueError(
            f'Invalid GEMMA4_MAX_SOFT_TOKENS={max_soft_tokens}'
        )

    processor = self.processor
    image_processor = processor.image_processor

    old_value = image_processor.max_soft_tokens
    image_processor.max_soft_tokens = max_soft_tokens

    try:
        return _original_encode(self, inputs)
    finally:
        image_processor.max_soft_tokens = old_value


Gemma4Template._encode = _patched_encode

print(
    "[Gemma4 plugin] Gemma4Template patched, "
    f"max_soft_tokens={os.environ.get('GEMMA4_MAX_SOFT_TOKENS', '280')}",
    flush=True,
)