import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence

import distrax


class ActorCriticConvSymbolicCraftax(nn.Module):
    action_dim: Sequence[int]
    map_obs_shape: Sequence[int]
    layer_width: int

    @nn.compact
    def __call__(self, obs):
        # Split into map and flat obs
        flat_map_obs_shape = (
            self.map_obs_shape[0] * self.map_obs_shape[1] * self.map_obs_shape[2]
        )
        image_obs = obs[:, :flat_map_obs_shape]
        image_dim = self.map_obs_shape
        image_obs = image_obs.reshape((image_obs.shape[0], *image_dim))

        flat_obs = obs[:, flat_map_obs_shape:]

        # Convolutions on map
        image_embedding = nn.Conv(features=32, kernel_size=(2, 2))(image_obs)
        image_embedding = nn.relu(image_embedding)
        image_embedding = nn.max_pool(
            image_embedding, window_shape=(2, 2), strides=(1, 1)
        )
        image_embedding = nn.Conv(features=32, kernel_size=(2, 2))(image_embedding)
        image_embedding = nn.relu(image_embedding)
        image_embedding = nn.max_pool(
            image_embedding, window_shape=(2, 2), strides=(1, 1)
        )
        image_embedding = image_embedding.reshape(image_embedding.shape[0], -1)
        # image_embedding = jnp.concatenate([image_embedding, obs[:, : CraftaxEnv.get_flat_map_obs_shape()]], axis=-1)

        # Combine embeddings
        embedding = jnp.concatenate([image_embedding, flat_obs], axis=-1)
        embedding = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        actor_mean = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)

from flax import linen as nn
from transformers import FlaxBertModel, BertTokenizer
import jax.numpy as jnp
import distrax

class ActorCriticConvWithIdxEmbedding(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    vocab_size: int = 4  # Размер словаря для индексов, по умолчанию 4
    embedding_dim: int = 4  # Размер эмбеддингов, по умолчанию 4
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs, text_indices):
        # obs - входное изображение
        # text_indices - индексы строк, которые будут преобразованы в эмбеддинги
      #  text_indices = text_indices[:,0]
        print("OBS...", obs.shape)
        print("text indices...", text_indices.shape)
        
        # Обработка изображения (не изменяется)
        x = nn.Conv(features=32, kernel_size=(5, 5))(obs)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

        image_embedding = x.reshape(x.shape[0], -1)

        # Эмбеддинг слой для текстовых индексов
        text_embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.embedding_dim)(text_indices)
        
        # Убираем лишнюю ось у text_embedding
        text_embedding = jnp.squeeze(text_embedding, axis=1)  # Теперь text_embedding имеет размер [batch_size, embedding_dim]

        # Объединение эмбеддингов изображения и текстовых эмбеддингов
        combined_embedding = jnp.concatenate([image_embedding, text_embedding], axis=-1)

        # Акторная сеть
        actor_mean = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(combined_embedding)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        # Критическая сеть
        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(combined_embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        print("Done")
        return pi, jnp.squeeze(critic, axis=-1)

import jax.numpy as jnp
import flax.linen as nn
import distrax
from flax.linen.initializers import orthogonal, constant


class ActorCriticConvWithBiFiLM(nn.Module):
    action_dim: int
    layer_width: int
    activation: str = "tanh"
    bert_model_name: str = "bert-base-uncased"

    def compute_film_params_from_text(self, text_embedding, num_channels):
        """
        Compute FiLM parameters (γ and β) from text embedding.
        """
        gamma = nn.Dense(num_channels)(text_embedding)   # W_γ * x_text + b_γ
        beta = nn.Dense(num_channels)(text_embedding)    # W_β * x_text + b_β
        return gamma, beta

    def compute_film_params_from_image(self, image_tensor, num_channels):
        """
        Compute FiLM parameters (Γ and B) from image features.
        image_tensor: [batch, height, width, channels]
        """
        # Convolution to extract spatial features
        gamma = nn.Conv(features=num_channels, kernel_size=(1, 1))(image_tensor)
        beta = nn.Conv(features=num_channels, kernel_size=(1, 1))(image_tensor)
        return gamma, beta

    @nn.compact
    def __call__(self, obs, text_embedding):
        # === First convolutional block with FiLM modulation from text ===
        x = nn.Conv(features=32, kernel_size=(5, 5))(obs)
        x = nn.relu(x)
        gamma, beta = self.compute_film_params_from_text(text_embedding, num_channels=32)
        gamma = gamma[:, None, None, :]  # Expand dimensions for broadcasting over spatial coordinates
        beta = beta[:, None, None, :]
        x = (1 + gamma) * x + beta
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

        # === Second convolutional block with FiLM modulation from text ===
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        gamma, beta = self.compute_film_params_from_text(text_embedding, num_channels=32)
        gamma = gamma[:, None, None, :]
        beta = beta[:, None, None, :]
        x = (1 + gamma) * x + beta
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

        # === Third convolutional block with FiLM modulation from text ===
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        gamma, beta = self.compute_film_params_from_text(text_embedding, num_channels=32)
        gamma = gamma[:, None, None, :]
        beta = beta[:, None, None, :]
        x = (1 + gamma) * x + beta
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

        # === Bidirectional FiLM Modulation ===
        # Compute Γ and B from the image
        gamma_text, beta_text = self.compute_film_params_from_image(
            obs, num_channels=text_embedding.shape[-1]
        )

        # Apply modulation to text embedding
        text_embedding = (1 + gamma_text.mean(axis=(1, 2))) * text_embedding + beta_text.mean(axis=(1, 2))

        # === Flatten feature map ===
        image_embedding = x.reshape(x.shape[0], -1)

        # === Combine image and text embeddings ===
        combined_embedding = jnp.concatenate([image_embedding, text_embedding], axis=-1)

        # === Actor network ===
        actor_hidden = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(combined_embedding)
        actor_hidden = nn.relu(actor_hidden)
        actor_hidden = nn.Dense(
            self.layer_width, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_hidden)
        actor_hidden = nn.relu(actor_hidden)
        actor_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_hidden)

        # Categorical distribution over actions
        pi = distrax.Categorical(logits=actor_logits)

        # === Critic network ===
        critic_hidden = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(combined_embedding)
        critic_hidden = nn.relu(critic_hidden)
        critic_value = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic_hidden)

        return pi, jnp.squeeze(critic_value, axis=-1)

class ActorCriticConvWithFiLM(nn.Module):
    action_dim: int
    layer_width: int

    def compute_film_params(self, text_embedding, num_channels):
        gamma = nn.Dense(num_channels)(text_embedding)
        beta = nn.Dense(num_channels)(text_embedding)
        gamma = gamma[:, None, None, :]
        beta = beta[:, None, None, :]
        return gamma, beta
    
    @nn.compact
    def __call__(self, obs, text_embedding):
        # Первый свёрточный FiLM-блок
        x_conv = nn.Conv(features=32, kernel_size=(5, 5))(obs)
        x_conv = nn.LayerNorm()(x_conv)
        x_conv = nn.relu(x_conv)

        gamma, beta = self.compute_film_params(text_embedding, num_channels=32)
        x_film = gamma * x_conv + beta
        x = nn.max_pool(x_film + x_conv, window_shape=(3, 3), strides=(3, 3))

        # Второй свёрточный FiLM-блок
        x_conv = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x_conv = nn.LayerNorm()(x_conv)
        x_conv = nn.relu(x_conv)

        gamma, beta = self.compute_film_params(text_embedding, num_channels=32)
        x_film = gamma * x_conv + beta
        x = nn.max_pool(x_film + x_conv, window_shape=(3, 3), strides=(3, 3))

        # Третий свёрточный FiLM-блок
        x_conv = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x_conv = nn.LayerNorm()(x_conv)
        x_conv = nn.relu(x_conv)

        gamma, beta = self.compute_film_params(text_embedding, num_channels=32)
        x_film = gamma * x_conv + beta
        x = nn.max_pool(x_film + x_conv, window_shape=(3, 3), strides=(3, 3))

        # Преобразование карты признаков в вектор
        image_embedding = x.reshape(x.shape[0], -1)

        # Residual соединение между изображением и текстовым эмбеддингом
        combined_embedding = jnp.concatenate([image_embedding, text_embedding], axis=-1)

        # Actor с residual
        actor_hidden = nn.Dense(self.layer_width)(combined_embedding)
        actor_hidden = nn.relu(actor_hidden)
        actor_residual = nn.Dense(self.layer_width)(actor_hidden)
        actor_hidden = nn.relu(actor_hidden + actor_residual)
        actor_logits = nn.Dense(self.action_dim)(actor_hidden)

        pi = distrax.Categorical(logits=actor_logits)

        # Critic с residual
        critic_hidden = nn.Dense(self.layer_width)(combined_embedding)
        critic_hidden = nn.relu(critic_hidden)
        critic_residual = nn.Dense(self.layer_width)(critic_hidden)
        critic_hidden = nn.relu(critic_hidden + critic_residual)
        critic_value = nn.Dense(1)(critic_hidden)

        return pi, jnp.squeeze(critic_value, axis=-1)

# class ActorCriticConvWithFiLM(nn.Module):
#     action_dim: int
#     layer_width: int
#     activation: str = "tanh"
#     # Если понадобится использовать BERT внутри модуля, можно добавить имя модели
#     bert_model_name: str = "bert-base-uncased"

#     def compute_film_params(self, text_embedding, num_channels):
#         """
#         Вычисляет FiLM-параметры (γ и β) для заданного количества каналов.
#         """
#         # МЛП, переводящий текстовый эмбеддинг в 2*num_channels
#         film_params = nn.Dense(2 * num_channels)(text_embedding)
#         gamma, beta = jnp.split(film_params, 2, axis=-1)
#         return gamma, beta

#     @nn.compact
#     def __call__(self, obs, text_embedding):
#         """
#         obs: изображение в виде тензора [batch, H, W, C]
#         text_embedding: текстовый эмбеддинг, например, pooler_output BERT [batch, emb_dim]
#         """
#         # Первый свёрточный блок с FiLM
#         x = nn.Conv(features=32, kernel_size=(5, 5))(obs)
#         x = nn.relu(x)
#         # Вычисление FiLM-параметров для 32 каналов
#         gamma, beta = self.compute_film_params(text_embedding, num_channels=32)
#         # Расширяем размерности, чтобы они применялись к каждому пространственному положению
#         gamma = gamma[:, None, None, :]
#         beta = beta[:, None, None, :]
#         # Применяем FiLM-модуляцию
#         x = gamma * x + beta
#         x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

#         # Второй свёрточный блок с FiLM
#         x = nn.Conv(features=32, kernel_size=(5, 5))(x)
#         x = nn.relu(x)
#         gamma, beta = self.compute_film_params(text_embedding, num_channels=32)
#         gamma = gamma[:, None, None, :]
#         beta = beta[:, None, None, :]
#         x = gamma * x + beta
#         x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

#         # Третий свёрточный блок с FiLM
#         x = nn.Conv(features=32, kernel_size=(5, 5))(x)
#         x = nn.relu(x)
#         gamma, beta = self.compute_film_params(text_embedding, num_channels=32)
#         gamma = gamma[:, None, None, :]
#         beta = beta[:, None, None, :]
#         x = gamma * x + beta
#         x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

#         # Преобразование карты признаков в вектор
#         image_embedding = x.reshape(x.shape[0], -1)

#         # Объединяем эмбеддинги изображения и текста
#         combined_embedding = jnp.concatenate([image_embedding, text_embedding], axis=-1)

#         # Сеть актёра
#         actor_hidden = nn.Dense(
#             self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
#         )(combined_embedding)
#         actor_hidden = nn.relu(actor_hidden)
#         actor_hidden = nn.Dense(
#             self.layer_width, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
#         )(actor_hidden)
#         actor_hidden = nn.relu(actor_hidden)
#         actor_logits = nn.Dense(
#             self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
#         )(actor_hidden)

#         # Формируем категориальное распределение действий
#         pi = distrax.Categorical(logits=actor_logits)

#         # Сеть критика
#         critic_hidden = nn.Dense(
#             self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
#         )(combined_embedding)
#         critic_hidden = nn.relu(critic_hidden)
#         critic_value = nn.Dense(
#             1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
#         )(critic_hidden)

#         return pi, jnp.squeeze(critic_value, axis=-1)

class ActorCriticConvWithBERT(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"
    bert_model_name: str = "bert-base-uncased"

    # def setup(self):
    #     # Load the BERT model and tokenizer
    #     self.bert_model = FlaxBertModel.from_pretrained(self.bert_model_name, cache_dir=".")
    #     self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name, cache_dir=".")

    @nn.compact
    def __call__(self, obs, encoded_input):

        print("OBS...", obs.shape)
        print("tokens...", encoded_input.shape)
        
       # obs, encoded_input = obs
        # Image processing (unchanged)
        x = nn.Conv(features=32, kernel_size=(5, 5))(obs)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

        image_embedding = x.reshape(x.shape[0], -1)

       
        #encoded_input = self.tokenizer(encoded_input, padding=True, truncation=True, return_tensors='np')
       # bert_output = self.bert_model(input_ids=encoded_input)
        
        text_embedding = encoded_input #[:,0] #bert_output.pooler_output  # Use the pooled output as the text embedding

        # if text_embedding.shape[0] == 1 and image_embedding.shape[0] > 1:
        #     text_embedding = jnp.repeat(text_embedding, image_embedding.shape[0], axis=0)
        # Combine image and text embeddings
        combined_embedding = jnp.concatenate([image_embedding, text_embedding], axis=-1)

        # Actor network
        actor_mean = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(combined_embedding)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        # Critic network
        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(combined_embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        print("Done")
        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticConv(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs):
        
        x = nn.Conv(features=32, kernel_size=(5, 5))(obs)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

        embedding = x.reshape(x.shape[0], -1)

        actor_mean = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticWithEmbedding(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_emb = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_emb = activation(actor_emb)

        actor_emb = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_emb)
        actor_emb = activation(actor_emb)

        actor_emb = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_emb)
        actor_emb = activation(actor_emb)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_emb)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1), actor_emb
