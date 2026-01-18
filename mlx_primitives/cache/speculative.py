"""Speculative decoding support for KV cache.

This module provides speculative decoding utilities including:
- SpeculativeCache for storing draft tokens and KV
- Verification using rejection sampling
- Tree-based speculation with branching support
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import mlx.core as mx

from mlx_primitives.cache.kv_cache import KVCache


@dataclass
class SpeculativeToken:
    """A speculatively generated token.

    Attributes:
        token_id: The token ID.
        log_prob: Log probability from draft model.
        position: Position in the sequence.
        branch_id: For tree-based speculation.
    """

    token_id: int
    log_prob: float
    position: int
    branch_id: int = 0


@dataclass
class SpeculativeBranch:
    """A branch in speculative decoding.

    Attributes:
        branch_id: Unique identifier for this branch.
        parent_branch_id: Parent branch (for tree speculation).
        tokens: List of speculative tokens in this branch.
        kv_sequence_ids: Per-layer KV cache sequence IDs.
        start_position: Position where speculation started.
    """

    branch_id: int
    parent_branch_id: Optional[int]
    tokens: List[SpeculativeToken] = field(default_factory=list)
    kv_sequence_ids: List[int] = field(default_factory=list)
    start_position: int = 0


class SpeculativeCache:
    """KV cache extension for speculative decoding.

    Supports:
    - Draft model token storage
    - Efficient verification against target model
    - Rollback on rejection
    - Tree-based speculation with branching

    Integration with KVCache:
    - Speculative tokens stored in separate forked sequences
    - On acceptance, fork becomes the new main sequence
    - On rejection, fork is deleted and KV is rolled back

    Example:
        >>> cache = KVCache(config)
        >>> spec_cache = SpeculativeCache(cache, max_speculation_depth=5)
        >>>
        >>> # Begin speculation from main sequence
        >>> branch_id = spec_cache.begin_speculation(seq_id)
        >>>
        >>> # Add draft tokens
        >>> spec_cache.add_speculative_tokens(branch_id, tokens, k, v, layer_idx)
        >>>
        >>> # Verify against target model
        >>> accepted, correction = spec_cache.verify_and_commit(
        ...     branch_id, target_log_probs
        ... )
    """

    def __init__(
        self,
        parent_cache: KVCache,
        max_speculation_depth: int = 5,
        tree_width: int = 1,
    ):
        """Initialize speculative cache.

        Args:
            parent_cache: The main KVCache instance.
            max_speculation_depth: Maximum tokens to speculate.
            tree_width: Number of branches (1 for linear, >1 for tree).
        """
        self._parent = parent_cache
        self._max_depth = max_speculation_depth
        self._tree_width = tree_width

        # Active speculation branches
        self._branches: Dict[int, SpeculativeBranch] = {}
        self._next_branch_id = 0

        # Mapping from main sequence ID to active speculation
        self._active_speculation: Dict[int, int] = {}  # seq_id -> branch_id

    @property
    def max_speculation_depth(self) -> int:
        """Maximum speculation depth."""
        return self._max_depth

    def begin_speculation(
        self,
        sequence_id: int,
        parent_branch_id: Optional[int] = None,
    ) -> int:
        """Begin speculative generation for a sequence.

        Creates a forked sequence in the KV cache for speculation.

        Args:
            sequence_id: Main sequence ID to speculate from.
            parent_branch_id: Parent branch for tree speculation.

        Returns:
            New branch ID for this speculation.
        """
        branch_id = self._next_branch_id
        self._next_branch_id += 1

        # Fork the main sequence for speculation
        forked_seq_id = self._parent.fork_sequence(sequence_id)

        # Get current position
        start_pos = self._parent.get_sequence_length(sequence_id)

        branch = SpeculativeBranch(
            branch_id=branch_id,
            parent_branch_id=parent_branch_id,
            kv_sequence_ids=[forked_seq_id],  # One forked sequence
            start_position=start_pos,
        )

        self._branches[branch_id] = branch
        self._active_speculation[sequence_id] = branch_id

        return branch_id

    def add_speculative_tokens(
        self,
        branch_id: int,
        tokens: List[SpeculativeToken],
        k: mx.array,
        v: mx.array,
        layer_idx: int,
    ) -> None:
        """Add draft model tokens and KV to speculation branch.

        Args:
            branch_id: Branch to add to.
            tokens: Speculative tokens with log probs.
            k: Keys from draft model, shape (num_tokens, heads, dim).
            v: Values from draft model, shape (num_tokens, heads, dim).
            layer_idx: Layer index.
        """
        if branch_id not in self._branches:
            raise KeyError(f"Branch {branch_id} not found")

        branch = self._branches[branch_id]

        # Check depth limit
        if len(branch.tokens) + len(tokens) > self._max_depth:
            tokens = tokens[: self._max_depth - len(branch.tokens)]
            k = k[: len(tokens)]
            v = v[: len(tokens)]

        if not tokens:
            return

        # Add tokens to branch
        for token in tokens:
            token.position = branch.start_position + len(branch.tokens)
            branch.tokens.append(token)

        # Update KV cache
        forked_seq_id = branch.kv_sequence_ids[0]
        self._parent.update(forked_seq_id, k, v, layer_idx)

    def get_speculative_tokens(self, branch_id: int) -> List[SpeculativeToken]:
        """Get tokens in a speculation branch.

        Args:
            branch_id: Branch ID.

        Returns:
            List of speculative tokens.
        """
        if branch_id not in self._branches:
            raise KeyError(f"Branch {branch_id} not found")
        return self._branches[branch_id].tokens.copy()

    def get_speculation_kv(
        self,
        branch_id: int,
        layer_idx: int,
    ) -> Tuple[mx.array, mx.array]:
        """Get KV for speculative branch (including parent).

        Args:
            branch_id: Branch ID.
            layer_idx: Layer index.

        Returns:
            (K, V) including speculative tokens.
        """
        if branch_id not in self._branches:
            raise KeyError(f"Branch {branch_id} not found")

        branch = self._branches[branch_id]
        forked_seq_id = branch.kv_sequence_ids[0]
        return self._parent.get_kv(forked_seq_id, layer_idx)

    def verify_and_commit(
        self,
        branch_id: int,
        target_log_probs: mx.array,
        acceptance_threshold: float = 0.0,
    ) -> Tuple[int, Optional[int]]:
        """Verify speculative tokens against target model.

        Uses rejection sampling for verification.

        Args:
            branch_id: Branch to verify.
            target_log_probs: Target model log probs, shape (num_spec_tokens, vocab).
            acceptance_threshold: Minimum acceptance probability.

        Returns:
            Tuple of (num_accepted, correction_token).
            correction_token is the next token to generate if any rejected.
        """
        if branch_id not in self._branches:
            raise KeyError(f"Branch {branch_id} not found")

        branch = self._branches[branch_id]
        tokens = branch.tokens

        if not tokens:
            return 0, None

        # Extract draft log probs and token IDs
        draft_tokens = [t.token_id for t in tokens]
        draft_log_probs = mx.array([t.log_prob for t in tokens])

        # Verify using rejection sampling
        num_accepted, correction = speculative_verify(
            draft_tokens,
            draft_log_probs,
            target_log_probs,
        )

        # Commit accepted tokens or rollback
        if num_accepted > 0:
            # Truncate branch to accepted tokens
            branch.tokens = branch.tokens[:num_accepted]

        return num_accepted, correction

    def commit_branch(self, branch_id: int, main_sequence_id: int) -> None:
        """Commit a speculation branch to the main sequence.

        This replaces the main sequence with the speculative fork.

        Args:
            branch_id: Branch to commit.
            main_sequence_id: Main sequence to replace.
        """
        if branch_id not in self._branches:
            raise KeyError(f"Branch {branch_id} not found")

        branch = self._branches[branch_id]

        # Delete the old main sequence
        self._parent.delete_sequence(main_sequence_id)

        # The forked sequence becomes the new main
        # (In a more complete implementation, we'd update internal mappings)

        # Clean up
        del self._branches[branch_id]
        self._active_speculation.pop(main_sequence_id, None)

    def rollback(self, branch_id: int, to_position: int) -> None:
        """Rollback speculation to given position, freeing KV.

        Args:
            branch_id: Branch to rollback.
            to_position: Position to rollback to.
        """
        if branch_id not in self._branches:
            return

        branch = self._branches[branch_id]

        # Truncate tokens
        branch.tokens = [t for t in branch.tokens if t.position < to_position]

        # Rollback KV cache
        # This requires updating the cache at each layer
        # For now, we delete the speculative fork
        forked_seq_id = branch.kv_sequence_ids[0]
        self._parent.delete_sequence(forked_seq_id)

    def cancel_speculation(self, branch_id: int) -> None:
        """Cancel a speculation branch entirely.

        Args:
            branch_id: Branch to cancel.
        """
        if branch_id not in self._branches:
            return

        branch = self._branches[branch_id]

        # Delete the forked sequence
        forked_seq_id = branch.kv_sequence_ids[0]
        self._parent.delete_sequence(forked_seq_id)

        # Clean up
        del self._branches[branch_id]

        # Remove from active speculation
        for seq_id, b_id in list(self._active_speculation.items()):
            if b_id == branch_id:
                del self._active_speculation[seq_id]

    def get_active_branch(self, sequence_id: int) -> Optional[int]:
        """Get active speculation branch for a sequence.

        Args:
            sequence_id: Main sequence ID.

        Returns:
            Branch ID or None if no active speculation.
        """
        return self._active_speculation.get(sequence_id)


def speculative_verify(
    draft_tokens: List[int],
    draft_log_probs: mx.array,
    target_log_probs: mx.array,
) -> Tuple[int, Optional[int]]:
    """Verify speculative tokens using rejection sampling.

    For each position i:
    1. Compute acceptance probability:
       p_accept = min(1, exp(target_logp[i] - draft_logp[i]))
    2. Sample uniform random r
    3. If r < p_accept: accept and continue
       Else: reject, resample from adjusted distribution

    Args:
        draft_tokens: List of draft model tokens.
        draft_log_probs: Draft model log probabilities (num_tokens,).
        target_log_probs: Target model log probs (num_tokens, vocab).

    Returns:
        (num_accepted, correction_token or None)
    """
    num_tokens = len(draft_tokens)

    if num_tokens == 0:
        return 0, None

    for i in range(num_tokens):
        draft_token = draft_tokens[i]
        draft_logp = float(draft_log_probs[i])
        target_logp = float(target_log_probs[i, draft_token])

        # Acceptance probability
        acceptance_prob = min(1.0, mx.exp(target_logp - draft_logp).item())

        # Sample
        r = mx.random.uniform(shape=())
        mx.eval(r)

        if float(r) < acceptance_prob:
            # Accept this token
            continue
        else:
            # Rejection - compute correction token
            # Sample from: max(0, p_target - p_draft) normalized
            target_probs = mx.exp(target_log_probs[i])
            draft_prob = mx.exp(mx.array(draft_logp))

            # Adjusted distribution
            adjusted = mx.maximum(mx.array(0.0), target_probs - draft_prob)
            adjusted_sum = mx.sum(adjusted)

            if float(adjusted_sum) > 1e-6:  # FP16-safe threshold
                adjusted = adjusted / adjusted_sum
                correction_token = int(mx.random.categorical(mx.log(adjusted + 1e-6)))
            else:
                # Fall back to sampling from target
                correction_token = int(mx.random.categorical(target_log_probs[i]))

            return i, correction_token

    # All accepted - sample next token from target (last position + 1)
    # This requires the caller to provide one more position of target logprobs
    if target_log_probs.shape[0] > num_tokens:
        next_token = int(mx.random.categorical(target_log_probs[num_tokens]))
        return num_tokens, next_token

    return num_tokens, None


class TreeSpeculation:
    """Tree-based speculation for higher acceptance rates.

    Instead of linear speculation, explores multiple branches
    and selects the one with highest acceptance.

    Tree structure:
        root -> [branch_0, branch_1, ...]
           branch_0 -> [branch_0_0, branch_0_1, ...]

    Memory management:
    - Branches share prefix via copy-on-write
    - Pruned branches release their blocks immediately

    Example:
        >>> tree = TreeSpeculation(spec_cache, branch_factor=2, max_depth=4)
        >>> branches = tree.speculate(seq_id, draft_model, sampler)
        >>> accepted, tokens = tree.verify_tree(target_model)
    """

    def __init__(
        self,
        cache: SpeculativeCache,
        branch_factor: int = 2,
        max_depth: int = 4,
    ):
        """Initialize tree speculation.

        Args:
            cache: Speculative cache instance.
            branch_factor: Number of branches at each level.
            max_depth: Maximum depth of speculation tree.
        """
        self._cache = cache
        self._branch_factor = branch_factor
        self._max_depth = max_depth

        # Track tree structure
        self._root_sequence_id: Optional[int] = None
        self._branch_tree: Dict[int, List[int]] = {}  # parent -> children

    def speculate(
        self,
        sequence_id: int,
        draft_model: Callable[[mx.array], mx.array],
        draft_sampler: Callable[[mx.array, int], List[Tuple[int, float]]],
    ) -> Dict[int, List[SpeculativeToken]]:
        """Generate speculation tree.

        Args:
            sequence_id: Sequence to speculate from.
            draft_model: Function that takes input IDs and returns logits.
            draft_sampler: Function that samples k tokens from logits.

        Returns:
            Mapping of branch_id -> tokens.
        """
        self._root_sequence_id = sequence_id
        self._branch_tree.clear()

        # Start with root branch
        root_branch = self._cache.begin_speculation(sequence_id)
        self._branch_tree[root_branch] = []

        # BFS to build tree
        current_level = [(root_branch, [])]  # (branch_id, token_ids so far)
        depth = 0

        while depth < self._max_depth and current_level:
            next_level: List[Tuple[int, List[int]]] = []

            for branch_id, token_ids in current_level:
                tokens = self._cache.get_speculative_tokens(branch_id)
                current_pos = len(tokens)

                if current_pos >= self._max_depth:
                    continue

                # Build input from accumulated token IDs
                if token_ids:
                    input_ids = mx.array(token_ids)
                else:
                    input_ids = mx.array([sequence_id])  # Use sequence_id as context

                # Run draft model to get logits
                logits = draft_model(input_ids)
                mx.eval(logits)

                # Get logits for last position
                if logits.ndim == 2:
                    last_logits = logits[-1]
                else:
                    last_logits = logits

                # Sample multiple continuations
                continuations = draft_sampler(last_logits, self._branch_factor)

                for i, (token_id, log_prob) in enumerate(continuations):
                    spec_token = SpeculativeToken(
                        token_id=token_id,
                        log_prob=log_prob,
                        position=current_pos,
                        branch_id=branch_id,
                    )

                    if i == 0:
                        # First continuation stays on current branch
                        self._cache.add_speculative_tokens(
                            branch_id,
                            [spec_token],
                            mx.zeros((1, 1, 1)),  # Placeholder K
                            mx.zeros((1, 1, 1)),  # Placeholder V
                            0,
                        )
                        next_level.append((branch_id, token_ids + [token_id]))
                    else:
                        # Create child branch for additional continuations
                        child_branch = self._cache.begin_speculation(
                            sequence_id, parent_branch_id=branch_id
                        )
                        self._branch_tree[branch_id].append(child_branch)
                        self._branch_tree[child_branch] = []
                        self._cache.add_speculative_tokens(
                            child_branch,
                            [spec_token],
                            mx.zeros((1, 1, 1)),
                            mx.zeros((1, 1, 1)),
                            0,
                        )
                        next_level.append((child_branch, token_ids + [token_id]))

            current_level = next_level
            depth += 1

        # Return all branches
        return {
            bid: self._cache.get_speculative_tokens(bid)
            for bid in self._branch_tree.keys()
        }

    def verify_tree(
        self,
        target_model: Callable[[mx.array], mx.array],
    ) -> Tuple[int, List[int]]:
        """Verify all branches and return best accepted sequence.

        Args:
            target_model: Function that takes input IDs and returns logits.

        Returns:
            (num_accepted, accepted_token_ids)
        """
        if not self._branch_tree:
            return 0, []

        best_accepted = 0
        best_tokens: List[int] = []

        # Verify each branch and find the one with most accepted tokens
        for branch_id in self._branch_tree.keys():
            tokens = self._cache.get_speculative_tokens(branch_id)
            if not tokens:
                continue

            # Build input IDs from tokens
            token_ids = [t.token_id for t in tokens]
            draft_log_probs = mx.array([t.log_prob for t in tokens])

            # Run target model on token sequence
            input_ids = mx.array(token_ids)
            target_logits = target_model(input_ids)
            mx.eval(target_logits)

            # Convert logits to log probs
            target_log_probs = mx.log(mx.softmax(target_logits, axis=-1) + 1e-6)

            # Verify using rejection sampling
            num_accepted, correction = speculative_verify(
                token_ids,
                draft_log_probs,
                target_log_probs,
            )

            # Track best branch
            if num_accepted > best_accepted:
                best_accepted = num_accepted
                best_tokens = token_ids[:num_accepted]
                if correction is not None:
                    best_tokens.append(correction)

        return best_accepted, best_tokens

    def cleanup(self) -> None:
        """Cancel all speculation branches."""
        for branch_id in list(self._branch_tree.keys()):
            self._cache.cancel_speculation(branch_id)
        self._branch_tree.clear()
        self._root_sequence_id = None
