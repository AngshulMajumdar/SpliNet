import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
from transformers.models.fnet.modeling_fnet import (
    FNetEmbeddings,
    FNetIntermediate,
    FNetOnlyMLMHead,
    FNetOutput,
    FNetPreTrainedModel,
)

from .configuration_splinet import SpliNetConfig
from .mixer import MultiHeadSpliNetMixer


class SpliNetMixingBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mixer = MultiHeadSpliNetMixer(
            hidden_size=config.hidden_size,
            num_heads=config.splinet_num_heads,
            radius=config.splinet_radius,
        )
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

    def forward(self, hidden_states):
        mixed = self.mixer(hidden_states)
        return self.LayerNorm(hidden_states + mixed)


class SpliNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mixing = SpliNetMixingBlock(config)
        self.intermediate = FNetIntermediate(config)
        self.output = FNetOutput(config)

    def forward(self, hidden_states):
        mixed_output = self.mixing(hidden_states)
        intermediate_output = self.intermediate(mixed_output)
        return self.output(intermediate_output, mixed_output)


class SpliNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList(
            [SpliNetLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states, output_hidden_states=False):
        all_hidden_states = () if output_hidden_states else None
        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer_module(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class SpliNetModel(FNetPreTrainedModel):
    config_class = SpliNetConfig
    base_model_prefix = "splinet"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = FNetEmbeddings(config)
        self.encoder = SpliNetEncoder(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both.")

        if input_ids is not None:
            input_shape = input_ids.size()
            _, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            _, seq_length = input_shape
            device = inputs_embeds.device
        else:
            raise ValueError("input_ids or inputs_embeds must be provided")

        if seq_length > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_length} exceeds maximum "
                f"{self.config.max_position_embeddings}"
            )

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape,
                dtype=torch.long,
                device=device,
            )

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
        )


class SpliNetForMaskedLM(FNetPreTrainedModel):
    config_class = SpliNetConfig
    base_model_prefix = "splinet"

    _tied_weights_keys = {
        "cls.predictions.decoder.bias": "cls.predictions.bias",
        "cls.predictions.decoder.weight":
            "splinet.embeddings.word_embeddings.weight",
    }

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.splinet = SpliNetModel(config)
        self.cls = FNetOnlyMLMHead(config)
        self.post_init()

        if config.tie_word_embeddings:
            self.cls.predictions.decoder.weight = (
                self.splinet.embeddings.word_embeddings.weight
            )

    def get_input_embeddings(self):
        return self.splinet.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.splinet.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        outputs = self.splinet(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        prediction_scores = self.cls(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                prediction_scores.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
            )

        if not return_dict:
            result = (prediction_scores,)
            if output_hidden_states:
                result += (outputs.hidden_states,)
            if loss is not None:
                result = (loss,) + result
            return result

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
        )
