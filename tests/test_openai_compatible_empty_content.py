"""Tests for OpenAI-compatible provider empty-content handling."""

import unittest
from unittest.mock import Mock

from providers.openai_compatible import EmptyContentError, OpenAICompatibleProvider


def _response(content, finish_reason="stop", reasoning_tokens=None):
    """Build a minimal chat-completion response double."""
    message = Mock()
    message.content = content
    choice = Mock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = Mock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 0 if not content else 5
    if reasoning_tokens is None:
        del usage.completion_tokens_details
    else:
        usage.completion_tokens_details.reasoning_tokens = reasoning_tokens

    response = Mock()
    response.choices = [choice]
    response.usage = usage
    response.model = "test-model"
    response.id = "resp-1"
    response.created = 0
    return response


class _TestProvider(OpenAICompatibleProvider):
    FRIENDLY_NAME = "Test"
    MODEL_CAPABILITIES = {"test-model": {"context_window": 4096}}

    def get_capabilities(self, model_name):
        # use_openai_response_api must be explicitly False: a bare Mock returns
        # a truthy attribute for it and routes the call to the /v1/responses
        # endpoint instead of chat completions.
        return Mock(use_openai_response_api=False)

    def get_provider_type(self):
        return Mock()

    def validate_model_name(self, model_name):
        return True

    def list_models(self, **kwargs):
        return ["test-model"]


class TestEmptyContentDetection(unittest.TestCase):
    """A 200 OK with no content must not read as success."""

    def setUp(self):
        self.provider = _TestProvider("test-key")

    def _generate(self, response):
        self.provider._client = Mock()
        self.provider._client.chat.completions.create = Mock(return_value=response)
        return self.provider.generate_content(prompt="hi", model_name="test-model")

    def test_empty_string_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._generate(_response("", finish_reason="length", reasoning_tokens=2048))
        self.assertIn("empty content", str(ctx.exception))

    def test_none_content_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._generate(_response(None, finish_reason="length"))
        self.assertIn("empty content", str(ctx.exception))

    def test_whitespace_only_content_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._generate(_response("   \n\t  ", finish_reason="length"))
        self.assertIn("empty content", str(ctx.exception))

    def test_error_names_finish_reason_and_reasoning_tokens(self):
        """The two fields that explain the failure must reach the operator."""
        with self.assertRaises(RuntimeError) as ctx:
            self._generate(_response("", finish_reason="length", reasoning_tokens=8503))
        message = str(ctx.exception)
        self.assertIn("finish_reason=length", message)
        self.assertIn("reasoning_tokens=8503", message)

    def test_missing_usage_details_does_not_mask_the_error(self):
        """Providers that omit completion_tokens_details still get the error."""
        with self.assertRaises(RuntimeError) as ctx:
            self._generate(_response("", finish_reason="length", reasoning_tokens=None))
        self.assertIn("empty content", str(ctx.exception))

    def test_normal_content_unaffected(self):
        result = self._generate(_response("hello"))
        self.assertEqual(result.content, "hello")

    def test_content_that_is_only_zero_is_not_empty(self):
        """ "0" is falsy-looking but is a legitimate answer."""
        result = self._generate(_response("0"))
        self.assertEqual(result.content, "0")


class TestEmptyContentRetryClassification(unittest.TestCase):
    """Retryability must come from finish_reason, never from the message text.

    The message embeds a token count, and every classifier in
    ``_is_error_retryable`` below the typed check inspects the string. Without
    an explicit isinstance branch, a reasoning_tokens value of 1500 matches the
    "500" retry indicator and 429 matches the rate-limit heuristic, making
    retry behaviour a function of arbitrary digits.
    """

    def setUp(self):
        self.provider = _TestProvider("test-key")

    def _error(self, reasoning_tokens, finish_reason):
        return EmptyContentError(
            f"Test returned empty content for test-model "
            f"(finish_reason={finish_reason}, reasoning_tokens={reasoning_tokens}). "
            f"For reasoning models this usually means the output budget was spent "
            f"on reasoning; reduce the input size or raise max_tokens.",
            finish_reason=finish_reason,
            reasoning_tokens=reasoning_tokens,
        )

    def test_budget_exhaustion_is_not_retryable(self):
        """finish_reason=length cannot change on an identical retry."""
        self.assertFalse(self.provider._is_error_retryable(self._error(2048, "length")))

    def test_unexplained_empty_content_is_retryable(self):
        self.assertTrue(self.provider._is_error_retryable(self._error(2048, "stop")))

    def test_decision_ignores_digits_in_the_token_count(self):
        """Every one of these previously flipped the answer via string matching."""
        for tokens in (429, 1429, 1500, 8503, 65504, 502):
            with self.subTest(reasoning_tokens=tokens):
                self.assertFalse(
                    self.provider._is_error_retryable(self._error(tokens, "length")),
                    f"reasoning_tokens={tokens} leaked into the retry decision",
                )
                self.assertTrue(
                    self.provider._is_error_retryable(self._error(tokens, "stop")),
                    f"reasoning_tokens={tokens} leaked into the retry decision",
                )

    def test_missing_finish_reason_is_retryable(self):
        """No finish_reason means no evidence of budget exhaustion."""
        self.assertTrue(self.provider._is_error_retryable(self._error(2048, None)))

    def test_plain_runtime_error_still_uses_string_classification(self):
        """The typed branch must not swallow ordinary errors."""
        self.assertTrue(self.provider._is_error_retryable(RuntimeError("connection reset")))
        self.assertFalse(self.provider._is_error_retryable(RuntimeError("bad request")))


if __name__ == "__main__":
    unittest.main()
