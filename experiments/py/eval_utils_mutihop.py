import typing
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer


from dsets import AttributeSnippets
def compute_multihop_quality_mquake(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
    use_cot: bool = True,
    max_new_tokens: int = 64,
) -> dict:
    """
    生成式多跳/CoT 评测（单条 record）
    :param model: 已编辑模型
    :param tok: 分词器
    :param record: MQuAKE 数据集单条样本，需包含 paraphrase_prompts, new_answer, new_answer_alias
    :param use_cot: 是否使用 CoT 提示
    :param max_new_tokens: 生成最大长度
    :return: 字典，包含每个 paraphrase_prompt 的命中布尔值列表
    """
    correct_flags = []
    golds = [record["new_answer"].strip()] + [a.strip() for a in record.get("new_answer_alias", [])]

    for q in record["paraphrase_prompts"]:
        prompt = q + ("\nLet's think step by step.\nAnswer:" if use_cot else "\nAnswer:")
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tok.eos_token_id,
            )
        output_text = tok.decode(outputs[0], skip_special_tokens=True)
        # 提取答案片段
        if "Answer:" in output_text:
            ans = output_text.split("Answer:")[-1].strip().split("\n")[0]
        else:
            ans = output_text.strip().split("\n")[0]
        correct_flags.append(any(ans == g for g in golds))

    key = "multihop_cot_correct" if use_cot else "multihop_correct"
    return {key: correct_flags}