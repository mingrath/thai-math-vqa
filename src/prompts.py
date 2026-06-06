"""Prompt templates for Thai Math VQA.

Strategy (from research): force step-by-step reasoning, then a single boxed
final answer in the SAME surface form the problem expects, so the exact-match
metric can match it after normalization.
"""

SYSTEM_PROMPT = (
    "You are an expert at solving Thai school mathematics problems shown as images "
    "(arithmetic, algebra, geometry, combinatorics, word problems). "
    "Read the Thai text, diagram, and math symbols carefully and reason precisely."
)

# Thai-language instruction. Asks for transcription -> reasoning -> one boxed answer.
USER_INSTRUCTION = (
    "จงแก้โจทย์คณิตศาสตร์ในภาพนี้\n"
    "1. อ่านและถอดความโจทย์ (ข้อความไทย ตัวเลข และไดอะแกรม) ให้ครบถ้วน\n"
    "2. คิดและแสดงวิธีทำทีละขั้นตอน\n"
    "3. ให้คำตอบสุดท้ายในรูปแบบที่โจทย์ต้องการ (จำนวนเต็ม ทศนิยม LaTeX หรือวลีไทยพร้อมหน่วย)\n\n"
    "บรรทัดสุดท้ายต้องเป็นคำตอบเดียวในรูปแบบ: \\boxed{คำตอบ}"
)

# Short/greedy variant: answer only, no reasoning (faster baseline).
USER_INSTRUCTION_SHORT = (
    "จงแก้โจทย์คณิตศาสตร์ในภาพนี้ แล้วตอบเฉพาะคำตอบสุดท้ายเท่านั้น "
    "ในรูปแบบ \\boxed{คำตอบ} ห้ามอธิบาย"
)


def build_messages(image_path: str, cot: bool = True, min_pixels=None, max_pixels=None):
    """Build a chat-message list for Qwen-VL-style processors / qwen_vl_utils."""
    image_item = {"type": "image", "image": image_path}
    if min_pixels is not None:
        image_item["min_pixels"] = min_pixels
    if max_pixels is not None:
        image_item["max_pixels"] = max_pixels
    instruction = USER_INSTRUCTION if cot else USER_INSTRUCTION_SHORT
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [image_item, {"type": "text", "text": instruction}]},
    ]
