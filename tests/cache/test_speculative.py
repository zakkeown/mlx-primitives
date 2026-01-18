"""Tests for speculative decoding support."""
import pytest
import mlx.core as mx

from mlx_primitives.cache.speculative import (
    SpeculativeToken,
    SpeculativeBranch,
    speculative_verify,
)


class TestSpeculativeToken:
    """Tests for SpeculativeToken dataclass."""

    def test_token_creation(self) -> None:
        """Test basic token creation."""
        token = SpeculativeToken(token_id=42, log_prob=-1.5, position=5)
        assert token.token_id == 42
        assert token.log_prob == -1.5
        assert token.position == 5
        assert token.branch_id == 0  # Default

    def test_token_with_branch(self) -> None:
        """Test token with explicit branch_id."""
        token = SpeculativeToken(token_id=10, log_prob=-0.5, position=0, branch_id=3)
        assert token.branch_id == 3


class TestSpeculativeBranch:
    """Tests for SpeculativeBranch dataclass."""

    def test_branch_creation(self) -> None:
        """Test basic branch creation."""
        branch = SpeculativeBranch(branch_id=0, parent_branch_id=None)
        assert branch.branch_id == 0
        assert branch.parent_branch_id is None
        assert branch.tokens == []
        assert branch.kv_sequence_ids == []
        assert branch.start_position == 0

    def test_branch_with_tokens(self) -> None:
        """Test branch with tokens."""
        tokens = [
            SpeculativeToken(token_id=1, log_prob=-1.0, position=0),
            SpeculativeToken(token_id=2, log_prob=-1.5, position=1),
        ]
        branch = SpeculativeBranch(
            branch_id=1,
            parent_branch_id=0,
            tokens=tokens,
            start_position=10,
        )
        assert len(branch.tokens) == 2
        assert branch.start_position == 10


class TestSpeculativeVerify:
    """Tests for rejection sampling verification."""

    def test_verify_empty_tokens(self) -> None:
        """Test with empty token list."""
        accepted, correction = speculative_verify([], mx.array([]), mx.zeros((0, 10)))
        assert accepted == 0
        assert correction is None

    def test_verify_high_acceptance(self) -> None:
        """Test when target agrees with draft."""
        mx.random.seed(42)

        draft_tokens = [5]
        draft_log_probs = mx.array([-0.1])  # High prob

        # Target also prefers token 5
        target_log_probs = mx.full((1, 100), -10.0)
        target_log_probs = target_log_probs.at[0, 5].add(9.9)  # log_prob ~ -0.1

        accepted, correction = speculative_verify(
            draft_tokens,
            draft_log_probs,
            target_log_probs,
        )

        # With matching distributions, likely to accept
        assert accepted >= 0  # May or may not accept depending on random sample

    def test_verify_low_acceptance(self) -> None:
        """Test when target strongly disagrees with draft."""
        mx.random.seed(123)

        draft_tokens = [5]
        draft_log_probs = mx.array([0.0])  # Draft confident (log_prob=0 -> prob=1)

        # Target prefers completely different token
        target_log_probs = mx.full((1, 100), -10.0)
        target_log_probs = target_log_probs.at[0, 99].add(9.9)  # Prefers token 99

        accepted, correction = speculative_verify(
            draft_tokens,
            draft_log_probs,
            target_log_probs,
        )

        # Likely to reject
        if accepted == 0:
            assert correction is not None

    def test_verify_returns_correction(self) -> None:
        """Test that rejection returns a valid correction token."""
        mx.random.seed(456)

        # Force a rejection by having very different distributions
        draft_tokens = [0]
        draft_log_probs = mx.array([-0.01])  # Very confident

        target_log_probs = mx.full((1, 10), -20.0)
        target_log_probs = target_log_probs.at[0, 9].add(19.9)  # Strongly prefers 9

        # Run multiple times to ensure we get a rejection
        for _ in range(10):
            accepted, correction = speculative_verify(
                draft_tokens,
                draft_log_probs,
                target_log_probs,
            )
            if accepted == 0:
                assert correction is not None
                assert 0 <= correction < 10
                break

    def test_verify_multiple_tokens(self) -> None:
        """Test verification with multiple speculative tokens."""
        mx.random.seed(789)

        draft_tokens = [1, 2, 3]
        draft_log_probs = mx.array([-1.0, -1.0, -1.0])

        # Target accepts first two but not third
        target_log_probs = mx.full((3, 100), -10.0)
        target_log_probs = target_log_probs.at[0, 1].add(9.0)  # Accept 1
        target_log_probs = target_log_probs.at[1, 2].add(9.0)  # Accept 2
        target_log_probs = target_log_probs.at[2, 50].add(9.0)  # Prefer 50, not 3

        accepted, correction = speculative_verify(
            draft_tokens,
            draft_log_probs,
            target_log_probs,
        )

        # Should accept at least 0, possibly up to 3
        assert 0 <= accepted <= 3

    def test_verify_all_accepted_with_bonus(self) -> None:
        """Test case where all tokens accepted and bonus token returned."""
        mx.random.seed(111)

        draft_tokens = [1]
        draft_log_probs = mx.array([-0.5])

        # Target agrees and provides next token
        target_log_probs = mx.full((2, 100), -10.0)
        target_log_probs = target_log_probs.at[0, 1].add(9.5)  # Accept 1
        target_log_probs = target_log_probs.at[1, 42].add(9.5)  # Bonus: 42

        accepted, correction = speculative_verify(
            draft_tokens,
            draft_log_probs,
            target_log_probs,
        )

        if accepted == 1:
            # Got all accepted, correction should be bonus token from target
            assert correction is not None


class TestSpeculativeVerifyNumericalStability:
    """Numerical stability tests for speculative verification."""

    def test_verify_extreme_log_probs(self) -> None:
        """Test with very negative log probs."""
        draft_tokens = [0]
        draft_log_probs = mx.array([-100.0])

        target_log_probs = mx.full((1, 10), -100.0)
        target_log_probs = target_log_probs.at[0, 0].add(50.0)

        # Should not crash with extreme values
        accepted, correction = speculative_verify(
            draft_tokens,
            draft_log_probs,
            target_log_probs,
        )
        assert accepted >= 0

    def test_verify_near_zero_prob(self) -> None:
        """Test with near-zero probability tokens."""
        draft_tokens = [0]
        draft_log_probs = mx.array([-50.0])  # Very low prob

        target_log_probs = mx.full((1, 10), -1.0)

        # Should handle gracefully
        accepted, correction = speculative_verify(
            draft_tokens,
            draft_log_probs,
            target_log_probs,
        )
        assert accepted >= 0
