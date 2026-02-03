from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx


class MLP(nnx.Module):
    """A simple multi-layer perceptron."""

    def __init__(self, layer_sizes: Sequence[int], rngs: nnx.Rngs):
        """Initialize the network.

        Args:
            layer_sizes: Sizes of all layers, including input and output.
            rngs: Random number generators for initialization.
        """
        self.num_hidden = len(layer_sizes) - 2

        # TODO: use nnx.scan to scan over layers, reducing compile times
        for i, (input_size, output_size) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:], strict=False)
        ):
            setattr(
                self, f"l{i}", nnx.Linear(input_size, output_size, rngs=rngs)
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the network."""
        for i in range(self.num_hidden):
            x = getattr(self, f"l{i}")(x)
            x = nnx.swish(x)
        x = getattr(self, f"l{self.num_hidden}")(x)
        return x


class DenoisingMLP(nnx.Module):
    """A simple multi-layer perceptron for action sequence denoising.

    Computes U* = NNet(U, y, t), where U is the noisy action sequence, y is the
    initial observation, and t is the time step in the denoising process.
    """

    def __init__(
        self,
        action_size: int,
        observation_size: int,
        horizon: int,
        hidden_layers: Sequence[int],
        rngs: nnx.Rngs,
    ):
        """Initialize the network.

        Args:
            action_size: Dimension of the actions (u).
            observation_size: Dimension of the observations (y).
            horizon: Number of steps in the action sequence (U = [u0, u1, ...]).
            hidden_layers: Sizes of all hidden layers.
            rngs: Random number generators for initialization.
        """
        self.action_size = action_size
        self.observation_size = observation_size
        self.horizon = horizon
        self.hidden_layers = hidden_layers

        input_size = horizon * action_size + observation_size + 1
        output_size = horizon * action_size
        self.mlp = MLP(
            [input_size] + list(hidden_layers) + [output_size], rngs=rngs
        )

    def __call__(self, u: jax.Array, y: jax.Array, t: jax.Array, use_running_average: bool = True) -> jax.Array:
        """Forward pass through the network."""
        batches = u.shape[:-2]
        u_flat = u.reshape(batches + (self.horizon * self.action_size,))
        x = jnp.concatenate([u_flat, y, t], axis=-1)
        x = self.mlp(x)
        return x.reshape(batches + (self.horizon, self.action_size))


class PositionalEmbedding(nnx.Module):
    """A simple sinusoidal positional embedding layer (MP1 style)."""

    def __init__(self, dim: int):
        """Initialize the positional embedding.

        Args:
            dim: Dimension to lift the input to.
        """
        self.half_dim = dim // 2

    def __call__(self, t: jax.Array) -> jax.Array:
        """Compute the positional embedding.
        
        Args:
            t: Time values, shape (...,) - will be squeezed and expanded
            
        Returns:
            Embedding of shape (..., dim)
        """
        # Handle both scalar and array inputs
        t_squeezed = jnp.squeeze(t)
        if t_squeezed.ndim == 0:
            t_squeezed = t_squeezed[None]
        
        freqs = jnp.arange(1, self.half_dim + 1) * jnp.pi
        emb = freqs[None, :] * t_squeezed[..., None]  # (..., half_dim)
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class TimeEmbedding(nnx.Module):
    """MP1-style time embedding with MLP (matches MP1 diffusion_step_encoder, lines 174-179)."""

    def __init__(self, dim: int, rngs: nnx.Rngs):
        """Initialize the time embedding.
        
        Args:
            dim: Output dimension
            rngs: Random number generators
        """
        self.pos_emb = PositionalEmbedding(dim)
        # MP1 uses: Linear(dim, dim*4) -> Mish -> Linear(dim*4, dim)
        self.linear1 = nnx.Linear(dim, dim * 4, rngs=rngs)
        self.linear2 = nnx.Linear(dim * 4, dim, rngs=rngs)

    def __call__(self, t: jax.Array) -> jax.Array:
        """Compute time embedding (MP1 style).
        
        Args:
            t: Time values, shape (batch,) or (batch, 1)
            
        Returns:
            Embedding of shape (batch, dim)
        """
        # Ensure t is 1D
        if t.ndim > 1:
            t = jnp.squeeze(t, axis=-1)
        assert t.ndim == 1, f"t should be 1D, got shape {t.shape}"
        
        # Positional embedding (MP1 line 175: SinusoidalPosEmb)
        pos_emb = self.pos_emb(t)  # (batch, dim)
        
        # MLP: Linear -> Mish -> Linear (MP1 lines 176-178)
        x = self.linear1(pos_emb)
        x = nnx.swish(x)  # Mish activation (swish approximates Mish)
        x = self.linear2(x)
        
        return x


class Conv1DBlock(nnx.Module):
    """A simple temporal convolutional block (BatchNorm removed for consistency).

         ----------     --------------------
    ---> | Conv1d | --> | Swish Activation | --->
         ----------     --------------------

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        rngs: nnx.Rngs,
    ):
        """Initialize the block.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            kernel_size: Size of the convolutional kernel.
            rngs: Random number generators for initialization.
        """
        self.c = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the block."""
        x = self.c(x)
        x = nnx.swish(x)
        return x


class ConditionalResidualBlock(nnx.Module):
    """A temporal convolutional block with FiLM conditional information.

        -------------------------------------------------------------
        |                                                           |
        |  -----------             -----------     -----------      |
    x ---> | Encoder | --> (+) --> | Dropout | --> | Decoder | --> (+) -->
           -----------      |      -----------     -----------
                            |
                       ----------
    y -----------------| Linear |
                       ----------

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        cond_features: int,
        kernel_size: int,
        rngs: nnx.Rngs,
    ):
        """Initialize the block.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            cond_features: Number of conditioning features.
            kernel_size: Size of the convolutional kernel.
            rngs: Random number generators for initialization.
        """
        self.encoder = Conv1DBlock(in_features, out_features, kernel_size, rngs)
        self.decoder = Conv1DBlock(
            out_features, out_features, kernel_size, rngs
        )
        self.linear = nnx.LinearGeneral(
            cond_features, (1, out_features), rngs=rngs
        )
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
        self.residual = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=1,
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Forward pass through the block."""
        z = self.encoder(x)
        z += self.linear(y)
        z = self.dropout(z)  # Dropout always applied (can be disabled via rate=0)
        z = self.decoder(z)
        return z + self.residual(x)


class DenoisingCNN(nnx.Module):
    class ConditionalUNet1D(nnx.Module):
        """Conditional 1D UNet for MeanFlow (MP1 style), with FiLM conditioning and dual time encoders.
        
        Matches MP1's ConditionalUnet1D architecture exactly:
        - Dual time encoders for t and r (MP1 lines 300, 310)
        - Time embeddings are added: timestep_embed + rs_embed (MP1 line 312)
        - FiLM conditioning with observation projection
        """

        def __init__(
            self,
            action_size: int,
            observation_size: int,
            horizon: int,
            base_dim: int,
            depth: int,
            rngs: nnx.Rngs,
            time_emb_dim: int = 32,
        ):
            """Initialize the Conditional UNet1D (MP1 style).

            Args:
                action_size: Dimension of the actions (u).
                observation_size: Dimension of the observations (y).
                horizon: Number of steps in the action sequence (U = [u0, u1, ...]).
                base_dim: Base feature dimension for UNet.
                depth: Number of down/up blocks.
                rngs: Random number generators for initialization.
                time_emb_dim: Dimension of the time embedding.
            """
            self.action_size = action_size
            self.observation_size = observation_size
            self.horizon = horizon
            self.depth = depth
            self.base_dim = base_dim
            self.time_emb_dim = time_emb_dim

            # Dual time encoders (MP1 lines 174-179, 264-269)
            # t encoder (timestep)
            self.time_emb_t = TimeEmbedding(time_emb_dim, rngs)
            # r encoder (rs, MP1 line 310)
            self.time_emb_r = TimeEmbedding(time_emb_dim, rngs)
            
            self.obs_proj = nnx.Linear(observation_size, base_dim, rngs=rngs)

            # Down path
            for i in range(depth):
                setattr(
                    self,
                    f"down{i}",
                    ConditionalResidualBlock(
                        in_features=base_dim if i > 0 else action_size,
                        out_features=base_dim,
                        cond_features=base_dim + time_emb_dim,
                        kernel_size=3,
                        rngs=rngs,
                    ),
                )
            # Up path
            for i in range(depth):
                setattr(
                    self,
                    f"up{i}",
                    ConditionalResidualBlock(
                        in_features=base_dim * 2,
                        out_features=base_dim,
                        cond_features=base_dim + time_emb_dim,
                        kernel_size=3,
                        rngs=rngs,
                    ),
                )
            self.final = nnx.Linear(base_dim, action_size, rngs=rngs)

        def __call__(
            self, 
            u: jax.Array, 
            y: jax.Array, 
            t: jax.Array, 
            r: jax.Array | None = None,
            use_running_average: bool = True
        ) -> jax.Array:
            """Forward pass through the Conditional UNet1D (MP1 style).
            
            Args:
                u: Action sequence, shape (..., horizon, action_size)
                y: Observation, shape (..., observation_size)
                t: Timestep, shape (...,) or (..., 1) - will be squeezed to 1D
                r: r parameter (MP1 style), shape (...,) or (..., 1). If None, uses zeros.
                
            Returns:
                Velocity prediction, shape (..., horizon, action_size)
            """
            # Input validation with assertions
            assert u.shape[-2:] == (self.horizon, self.action_size), \
                f"u must have shape (..., {self.horizon}, {self.action_size}), got {u.shape}"
            assert y.shape[-1] == self.observation_size, \
                f"y must have shape (..., {self.observation_size}), got {y.shape}"
            
            # Handle t: ensure it's 1D (batch,)
            if t.ndim > 1:
                t = jnp.squeeze(t, axis=-1)
            assert t.ndim == 1, f"t should be 1D after squeezing, got shape {t.shape}"
            batch_size = t.shape[0]
            
            # Handle r: if None, use zeros (MP1 uses r=0 in inference)
            if r is None:
                r = jnp.zeros_like(t)
            else:
                if r.ndim > 1:
                    r = jnp.squeeze(r, axis=-1)
                assert r.ndim == 1, f"r should be 1D after squeezing, got shape {r.shape}"
                assert r.shape[0] == batch_size, \
                    f"r batch size {r.shape[0]} must match t batch size {batch_size}"
            
            # Dual time embeddings (MP1 lines 300, 310, 312)
            t_emb = self.time_emb_t(t)  # (batch, time_emb_dim)
            r_emb = self.time_emb_r(r)  # (batch, time_emb_dim)
            time_emb = t_emb + r_emb  # (batch, time_emb_dim) - MP1 line 312
            
            # Observation projection
            obs_emb = self.obs_proj(y)  # (batch, base_dim)
            
            # Concatenate for conditioning (MP1 line 317)
            cond = jnp.concatenate([obs_emb, time_emb], axis=-1)  # (batch, base_dim + time_emb_dim)
            
            # Expand cond for broadcasting: (batch, 1, base_dim + time_emb_dim)
            cond = cond[:, None, :]  # (batch, 1, cond_dim)

            # Down path
            skips = []
            x = u
            actual_depth = 0
            for i in range(self.depth):
                # Broadcast cond to match x's spatial dimensions for FiLM
                # x: (batch, horizon, features), cond: (batch, 1, cond_dim)
                # FiLM expects (batch, cond_dim) so we squeeze the middle dimension
                cond_squeezed = jnp.squeeze(cond, axis=1)  # (batch, cond_dim)
                x = getattr(self, f"down{i}")(x, cond_squeezed)
                skips.append(x)
                actual_depth += 1
                # Downsample in time dimension only if we have enough length
                if x.shape[-2] > 1:
                    # Simple downsampling: take every other element
                    x = x[..., ::2, :]
                else:
                    break  # Can't downsample further

            # Bottleneck
            # (Optional: add more layers here if needed)

            # Up path
            for i in range(actual_depth):
                skip_idx = actual_depth - 1 - i
                # Upsample to match skip connection shape
                if x.shape[-2] != skips[skip_idx].shape[-2]:
                    # Use nearest neighbor upsampling
                    target_shape = skips[skip_idx].shape
                    x = jax.image.resize(x, target_shape, method="nearest")
                x = jnp.concatenate([x, skips[skip_idx]], axis=-1)
                # Broadcast cond for FiLM
                cond_squeezed = jnp.squeeze(cond, axis=1)  # (batch, cond_dim)
                x = getattr(self, f"up{i}")(x, cond_squeezed)

            out = self.final(x)
            assert out.shape[-2:] == (self.horizon, self.action_size), \
                f"Output shape {out.shape} must end with ({self.horizon}, {self.action_size})"
            return out
    """A denoising convolutional network with FiLM conditioning.

    Based on Diffusion Policy, https://arxiv.org/abs/2303.04137v5.
    """

    def __init__(
        self,
        action_size: int,
        observation_size: int,
        horizon: int,
        feature_dims: Sequence[int],
        rngs: nnx.Rngs,
        kernel_size: int = 3,
        timestep_embedding_dim: int = 32,
    ):
        """Initialize the network.

        Args:
            action_size: Dimension of the actions (u).
            observation_size: Dimension of the observations (y).
            horizon: Number of steps in the action sequence (U = [u0, u1, ...]).
            feature_dims: List of feature dimensions.
            rngs: Random number generators for initialization.
            kernel_size: Size of the convolutional kernel.
            timestep_embedding_dim: Dimension of the positional embedding.
        """
        self.action_size = action_size
        self.observation_size = observation_size
        self.horizon = horizon
        self.num_layers = len(feature_dims) + 1
        self.positional_embedding = PositionalEmbedding(timestep_embedding_dim)

        feature_sizes = [action_size] + list(feature_dims) + [action_size]
        for i, (input_size, output_size) in enumerate(
            zip(feature_sizes[:-1], feature_sizes[1:], strict=False)
        ):
            setattr(
                self,
                f"l{i}",
                ConditionalResidualBlock(
                    input_size,
                    output_size,
                    observation_size + timestep_embedding_dim,
                    kernel_size,
                    rngs,
                ),
            )

    def __call__(self, u: jax.Array, y: jax.Array, t: jax.Array, use_running_average: bool = True) -> jax.Array:
        """Forward pass through the network."""
        emb = self.positional_embedding(t)
        y = jnp.concatenate([y, emb], axis=-1)

        x = self.l0(u, y)
        for i in range(1, self.num_layers):
            x = getattr(self, f"l{i}")(x, y)

        return x + u
