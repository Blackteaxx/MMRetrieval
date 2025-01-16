from typing import List

import vllm
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM


def get_system_prompt():
    system_prompt = """
        You are a product expert. You will be given two product images and their descriptions.
        You should consider the following rules carefully when comparing the products:
        - **Brand** and **measurements**
        - **Color**, **shape**, and visual features
        - Any additional **details** from the descriptions

        Decide if they represent the exact same-style product. 
        Your answer must be a single word: "yes" for the same style, "no" for different styles.
    """
    return system_prompt


class Reranker:
    def __init__(
        self,
        model_path: str,
        quantization: str = "awq",
        tensor_parallel_size: int = 1,
        max_model_len: int = 5000,
        enforce_eager: bool = True,
        limit_mm_per_prompt: int = 2,
        choices: List[str] = ["yes", "no"],
    ):
        """初始化reranker模型

        参数:
            model_path (str): 模型路径
            quantization (str, optional): 量化方法. 默认为 "awq".
            tensor_parallel_size (int, optional): 张量并行大小. 默认为 1.
            max_model_len (int, optional): 模型最大长度. 默认为 5000.
            enforce_eager (bool, optional): 是否强制使用eager模式. 默认为 True.
            limit_mm_per_prompt (int, optional): 每个提示的多模态限制. 默认为 2.
            choices (List[str], optional): 用于logits限制的两个token，第一个元素代表是，第二个元素代表否. 默认为 ["yes", "no"].
        """
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.choices = choices

        logits_processor = MultipleChoiceLogitsProcessor(
            tokenizer=self.tokenizer,
            choices=choices,
        )

        self.sampling_params = vllm.SamplingParams(
            n=1,
            temperature=0,
            max_tokens=1,
            logits_processors=[logits_processor],
        )

        self.model = LLM(
            model_path,
            quantization=quantization,
            dtype="half",
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt={
                "image": limit_mm_per_prompt,
            },
        )

    def rerank(
        self,
        query_text: str,
        pred_texts: List[str],
        query_image_path: str,
        pred_image_paths: List[str],
        pred_ids: List[int],
        use_tqdm=False,
    ):
        """Use multi-modal inputs to generate reranked results.
        Reference: https://docs.vllm.ai/en/latest/serving/multimodal_inputs.html

        Args:
            query_text (_type_): _description_
            preds_text (_type_): _description_
            query_image_path (_type_): _description_
            preds_image_path (_type_): _description_
            preds_ids (_type_): _description
            use_tqdm (_type_): _description

        Returns:
            _type_: _description_
        """

        # text
        text_pairs = [(query_text, pred) for pred in pred_texts]
        messages = [self._genearate_pair_message(pair) for pair in text_pairs]
        prompts = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # image
        query_image = self._open_image(query_image_path)
        image_pairs = [
            (query_image, self._open_image(pred)) for pred in pred_image_paths
        ]

        # construct mm inputs
        mm_inputs = []
        for prompt, image in zip(prompts, image_pairs):
            mm_inputs.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": image,
                    },
                }
            )

        # generate reranked results
        reranked_request_results = self.model.generate(
            mm_inputs, self.sampling_params, use_tqdm=use_tqdm
        )

        reranked_results = [res.outputs[0].text for res in reranked_request_results]

        reranked_ids = [
            pred_ids[i]
            for i in range(len(reranked_results))
            if reranked_results[i] == self.choices[0]
        ]

        return reranked_ids

    def _genearate_pair_message(query_text, pred_text):
        # query_text, pred_text = pair
        system_prompt = get_system_prompt()

        message = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Product 1: {query_text}"},
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": f"Product 2: {pred_text}"},
                    {
                        "type": "image",
                    },
                ],
            },
        ]

        return message

    def _open_image(self, image_path):
        image = Image.open(image_path)
        return image
