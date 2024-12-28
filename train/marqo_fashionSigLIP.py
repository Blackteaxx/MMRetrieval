import torch
from open_clip import create_model
from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.siglip.modeling_siglip import SiglipOutput
from typing import Optional, Tuple, Union, List
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType
import string
import ftfy
import html

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def canonicalize_text(
    text,
    *,
    keep_punctuation_exact_string=None,
    trans_punctuation: dict = str.maketrans("", "", string.punctuation),
):
    """Returns canonicalized `text` (lowercase and punctuation removed).

    From: https://github.com/google-research/big_vision/blob/53f18caf27a9419231bbf08d3388b07671616d3d/big_vision/evaluators/proj/image_text/prompt_engineering.py#L94

    Args:
      text: string to be canonicalized.
      keep_punctuation_exact_string: If provided, then this exact string kept.
        For example providing '{}' will keep any occurrences of '{}' (but will
        still remove '{' and '}' that appear separately).
    """
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(trans_punctuation)
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(trans_punctuation)
    text = text.lower()
    text = " ".join(text.split())
    return text.strip()

def _clean_canonicalize(x):
    # basic, remove whitespace, remove punctuation, lower case
    return canonicalize_text(basic_clean(x))

class MarqoFashionSigLIPConfig(PretrainedConfig):
    def __init__(
        self,
        open_clip_model_name: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.open_clip_model_name = open_clip_model_name
        
class MarqoFashionSigLIPProcessor(ProcessorMixin):
    r"""
    Constructs a Siglip processor which wraps a Siglip image processor and a Siglip tokenizer into a single processor.

    [`SiglipProcessor`] offers all the functionalities of [`SiglipImageProcessor`] and [`SiglipTokenizer`]. See the
    [`~SiglipProcessor.__call__`] and [`~SiglipProcessor.decode`] for more information.

    Args:
        image_processor ([`SiglipImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`T5TokenizerFast`]):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = "T5TokenizerFast"

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: int = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to SiglipTokenizer's [`~SiglipTokenizer.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` argument to
        SiglipImageProcessor's [`~SiglipImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """

        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            if isinstance(text, str):
                text = [text]
            text = [_clean_canonicalize(raw_text) for raw_text in text]
            encoding = self.tokenizer(
                text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
            )

        if images is not None:
            try:
                images = [image.convert('RGB') for image in images] if isinstance(images, list) else images.convert('RGB')
            except:
                images = images
            image_features = self.image_processor(images, return_tensors=return_tensors)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features.pixel_values
            return encoding
        elif text is not None:
            return encoding
        else:
            return BatchFeature(data=dict(**image_features), tensor_type=return_tensors)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SiglipTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SiglipTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names with CLIP->Siglip, T5->Siglip
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
        
class MarqoFashionSigLIP(PreTrainedModel):
    config_class = MarqoFashionSigLIPConfig

    def __init__(self, config: MarqoFashionSigLIPConfig):
        super().__init__(config)
        self.config = config
        self.model = create_model(config.open_clip_model_name, output_dict=True)
        self.model.eval()
        self.model.to(self.device)
        
    def get_image_features(
        self, 
        pixel_values: torch.FloatTensor, 
        normalize: bool = False,
        **kwargs
        ) -> torch.FloatTensor:
        
        with torch.inference_mode():
            image_features = self.model.encode_image(pixel_values, normalize=normalize)
        return image_features
    
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        normalize: bool = False,
        **kwargs
    ) -> torch.FloatTensor:
        
        with torch.inference_mode():
            text_features = self.model.encode_text(input_ids, normalize=normalize)
        return text_features
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SiglipOutput]:

        vision_outputs = self.get_image_features(pixel_values=pixel_values, normalize=True)
        text_outputs = self.get_text_features(input_ids=input_ids, normalize=True)

        logits_per_text = text_outputs @ vision_outputs.T
        logits_per_image = logits_per_text.T

        if not return_dict:
            return logits_per_image, logits_per_text, text_outputs, vision_outputs

        return SiglipOutput(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_outputs,
            image_embeds=vision_outputs
        )
