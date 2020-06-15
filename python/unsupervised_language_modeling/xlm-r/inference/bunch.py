from typing import Any, Dict, List, NewType, Tuple, Optional


class FakeTokenizer:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

        self._pad_token = '<pad>'
        self.pad_token_id = 1
        self.mask_token = '<mask>'
        self.mask_token_id = 250001
        
        self.max_len = 100

    def convert_tokens_to_ids(self, tokens):
        if tokens == '<mask>':
            return 250001

    
    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))


    def __len__(self):
        return 250005
