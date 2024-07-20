from typing import Iterable
from pathlib import Path
from datetime import datetime
import torch
import wandb

from miditok import MusicTokenizer
from symusic import Synthesizer, dump_wav
from tokenizer import get_tokenizer  # Import the tokenizer


def export_to_midi(tokenizer: MusicTokenizer, tokens: Iterable,
                   filepath: Path | str):
    score = tokenizer.decode(tokens)
    score.dump_midi(filepath)


def export_to_wav(tokenizer: MusicTokenizer,
                  tokens: Iterable,
                  filepath: Path | str,
                  sample_rate: int = 44100):
    score = tokenizer(tokens)
    synth = Synthesizer(sample_rate=sample_rate, )
    audio = synth.render(score, stereo=True)
    dump_wav(filepath, audio, sample_rate=sample_rate, use_int16=True)


def generate_music(input_ids, model, config, overtrained_song=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Input Token For Generation: {input_ids}")

    # Load the tokenizer
    tokenizer = get_tokenizer(config)
    eso_token = tokenizer['EOS_None']

    # prepare EOS token id
    print(f"eos_token_id: {eso_token}")

    model.to(device)
    input_ids = input_ids.to(device)

    # Print the current device for verification
    print("Current device:", device)

    # Print device information for the model and input tensor to confirm they're on the same device
    print("Model device:", next(model.parameters()).device)
    print("Input tensor device:", input_ids.device)

    if config.inference.custom:
        generated_sequence = model.generate(
            input_ids=input_ids,
            sequence_length=config.inference.max_length
        )
    else:
        # Use model.generate for sequence generation
        generated_sequence = model.model.generate(
            input_ids=input_ids,
            max_length=config.inference.max_length,
            temperature=config.inference.temperature,
            top_k=config.inference.top_k,
            eos_token_id=eso_token,
            repetition_penalty=config.inference.repetition_penalty)

    # Export the output to a .wav file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wav_filename = f'output_{timestamp}.wav'

    base_path = Path().cwd()
    test_path = base_path / "tests" / timestamp
    test_path.mkdir(exist_ok=True, parents=True)

    # Ensure the output is in the correct format for the tokenizer
    tokens = generated_sequence.cpu().numpy()
    export_to_wav(tokenizer, tokens, (test_path / wav_filename).as_posix())

    if overtrained_song is not None:
        overtrained_song = overtrained_song.unsqueeze(0).cpu()
        max_dim = min(generated_sequence.shape[-1], overtrained_song.shape[-1])
        export_to_wav(tokenizer, overtrained_song.numpy(),
                      (test_path / "input.wav").as_posix())
        overtrained_song = overtrained_song[:max_dim].unsqueeze(0)
        generated_sequence = generated_sequence[:, :max_dim].cpu()
        concatenated_song = torch.cat((overtrained_song, generated_sequence),
                                      dim=0)
        print("Concatenated songs:")
        print(concatenated_song)

        # Compare tensors element-wise
        matches = overtrained_song == generated_sequence

        # Calculate the percentage of elements that are the same
        percentage_same = torch.sum(matches).item() / matches.numel() * 100

        success_message = f"Percentage of elements that are the same: {percentage_same}%"
        print(success_message)
        try:
            wandb.log({"success_message": success_message, "concatenated_songs": concatenated_song,
                       "success": percentage_same})
        except:
            print("Cannot log to wnadb info about similarity of two sequences")


def compare_sequences(input_ids, model, config, base_sequence=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Input Token For Generation: {input_ids}")
    model.to(device)
    input_ids = input_ids.to(device)

    # Print the current device for verification
    print("Current device:", device)

    # Print device information for the model and input tensor to confirm they're on the same device
    print("Model device:", next(model.parameters()).device)
    print("Input tensor device:", input_ids.device)
    if config.inference.custom:
        generated_sequence = model.generate(
            input_ids=input_ids,
            sequence_length=config.inference.max_length
        )
    else:
        generated_sequence = model.model.generate(
            input_ids=input_ids,
            max_length=config.inference.max_length,
            temperature=config.inference.temperature,
            top_k=config.inference.top_k,
            repetition_penalty=config.inference.repetition_penalty)

    overtrained_song = base_sequence.unsqueeze(0).cpu()
    max_dim = min(generated_sequence.shape[-1], overtrained_song.shape[-1])

    overtrained_song = overtrained_song[:max_dim]
    generated_sequence = generated_sequence[:, :max_dim].cpu()
    print("ORIGIN SHAPE", overtrained_song.shape)
    print("GEN SHAPE", generated_sequence.shape)
    concatenated_song = torch.cat((overtrained_song, generated_sequence),
                                  dim=0)
    print("Concatenated songs:")
    print(concatenated_song)

    # Compare tensors element-wise
    matches = overtrained_song == generated_sequence

    # Calculate the percentage of elements that are the same
    percentage_same = torch.sum(matches).item() / matches.numel() * 100

    success_message = f"Percentage of elements that are the same: {percentage_same}%"
    print(success_message)
    try:
        wandb.log({"success_message": success_message, "concatenated_songs": concatenated_song, "success": percentage_same})
    except:
        print("Cannot log to wnadb info about similarity of two sequences")
