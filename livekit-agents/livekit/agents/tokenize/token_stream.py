from __future__ import annotations

import typing
from typing import Callable, Union

from ..utils import aio, shortuuid
from .tokenizer import SentenceStream, TokenData, WordStream

# Tokenizers can either provide us with a list of tokens or a list of tokens along with their start and end indices.
# If the start and end indices are not available, we attempt to locate the token within the text using str.find.
TokenizeCallable = Callable[[str], Union[list[str], list[tuple[str, int, int]]]]


class BufferedTokenStream:
    def __init__(
        self,
        *,
        tokenize_fnc: TokenizeCallable,
        min_token_len: int,
        min_ctx_len: int,
    ) -> None:
        self._event_ch = aio.Chan[TokenData]()
        self._tokenize_fnc = tokenize_fnc
        self._min_ctx_len = min_ctx_len
        self._min_token_len = min_token_len
        self._current_segment_id = shortuuid()

        self._buf_tokens: list[str] = []  # <= min_token_len
        self._in_buf = ""
        self._out_buf = ""

    @typing.no_type_check
    def push_text(self, text: str) -> None:
        print(f"Inside push_text, text: {text}")
        self._check_not_closed()

        # Special handling for periods
        if "." in text:
            for i, char in enumerate(text):
                if char == ".":
                    # Add everything up to the period to the buffer
                    self._in_buf += text[: i + 1]

                    # Process the buffer immediately regardless of size
                    self._process_buffer(force_process=True)

                    # Continue with the rest of the text
                    if i + 1 < len(text):
                        self.push_text(text[i + 1 :])
                    return

        # Normal processing for text without periods
        self._in_buf += text

        if len(self._in_buf) < self._min_ctx_len:
            print(f"Inside push_text, len(self._in_buf) < self._min_ctx_len, returning")
            return

        self._process_buffer()

    @typing.no_type_check
    def _process_buffer(self, force_process=False) -> None:
        """Process the input buffer, optionally forcing processing regardless of buffer size."""
        if not force_process and len(self._in_buf) < self._min_ctx_len:
            return

        while True:
            tokens = self._tokenize_fnc(self._in_buf)
            print(f"Inside _process_buffer, tokens: {tokens}")

            if len(tokens) <= 1 and not force_process:
                print(
                    f"Inside _process_buffer, tokens: {tokens}, len(tokens) <= 1, breaking"
                )
                break

            # If we're forcing processing with one token, we need to process it
            if len(tokens) == 1 and force_process:
                if self._out_buf:
                    self._out_buf += " "

                tok = tokens[0]
                tok_text = tok
                if isinstance(tok, tuple):
                    tok_text = tok[0]

                self._out_buf += tok_text

                # Send the token immediately if it contains a period
                # or if it meets the minimum token length
                if "." in self._out_buf or len(self._out_buf) >= self._min_token_len:
                    self._event_ch.send_nowait(
                        TokenData(
                            token=self._out_buf, segment_id=self._current_segment_id
                        )
                    )
                    self._out_buf = ""

                # Clear the input buffer
                self._in_buf = ""
                break

            # Regular processing for multiple tokens
            if len(tokens) > 1:
                if self._out_buf:
                    self._out_buf += " "

                tok = tokens.pop(0)
                tok_text = tok
                if isinstance(tok, tuple):
                    tok_text = tok[0]

                self._out_buf += tok_text

                # Send immediately if contains period or meets minimum length
                if "." in self._out_buf or len(self._out_buf) >= self._min_token_len:
                    self._event_ch.send_nowait(
                        TokenData(
                            token=self._out_buf, segment_id=self._current_segment_id
                        )
                    )
                    self._out_buf = ""

                if isinstance(tok, tuple):
                    self._in_buf = self._in_buf[tok[2] :]
                else:
                    tok_i = max(self._in_buf.find(tok), 0)
                    self._in_buf = self._in_buf[tok_i + len(tok) :].lstrip()
            else:
                # No tokens to process
                break

    @typing.no_type_check
    def flush(self) -> None:
        self._check_not_closed()

        if self._in_buf or self._out_buf:
            tokens = self._tokenize_fnc(self._in_buf)
            if tokens:
                if self._out_buf:
                    self._out_buf += " "

                if isinstance(tokens[0], tuple):
                    self._out_buf += " ".join([tok[0] for tok in tokens])
                else:
                    self._out_buf += " ".join(tokens)

            if self._out_buf:
                self._event_ch.send_nowait(
                    TokenData(token=self._out_buf, segment_id=self._current_segment_id)
                )

            self._current_segment_id = shortuuid()

        self._in_buf = ""
        self._out_buf = ""

    def end_input(self) -> None:
        self.flush()
        self._event_ch.close()

    async def aclose(self) -> None:
        self._event_ch.close()

    def _check_not_closed(self) -> None:
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    def __aiter__(self) -> "BufferedTokenStream":
        return self

    async def __anext__(self) -> TokenData:
        return await self._event_ch.__anext__()


class BufferedSentenceStream(BufferedTokenStream, SentenceStream):
    def __init__(
        self,
        *,
        tokenizer: TokenizeCallable,
        min_token_len: int,
        min_ctx_len: int,
    ) -> None:
        super().__init__(
            tokenize_fnc=tokenizer,
            min_token_len=min_token_len,
            min_ctx_len=min_ctx_len,
        )


class BufferedWordStream(BufferedTokenStream, WordStream):
    def __init__(
        self,
        *,
        tokenizer: TokenizeCallable,
        min_token_len: int,
        min_ctx_len: int,
    ) -> None:
        super().__init__(
            tokenize_fnc=tokenizer,
            min_token_len=min_token_len,
            min_ctx_len=min_ctx_len,
        )
