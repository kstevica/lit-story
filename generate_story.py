import json
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

# from generate import generate
# from scripts.prepare_alpaca import generate_prompt

# import pyperclip
import textwrap
from datetime import datetime
import readline
import asyncio
import websockets
import ssl

total_active_count = 0

def main(
    adapter_path: Path = Path("./"),
    checkpoint_dir: Path = Path(f"./"),
    quantize: Optional[str] = None,
    use_alpaca: bool = False,
    use_sockets: bool = False,
    use_port: int = 12350,
    use_llama: bool = False,
    tokenizer_path: str = None,
) -> None:
    """
    Args:
        adapter_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune_adapter.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained Parrot weights.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
        use_alpaca: Alpaca style prompt
        use_sockets: start as websockets server -> make sure you have SSL certificate in place
        use_port: when run with use_sockets, set a port
        use_llama: load llama model, make sure checkpoint_dir points to .pth file
        tokenizer_path: when llama base model is used, point to tokenizer.model
    """    
    max_new_tokens: int = 100
    top_k: int = 200
    temperature: float = 0.8

    fabric = L.Fabric(devices=1)
    dtype = torch.bfloat16 # torch.bfloat16 if fabric.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    print("Preparing to load model...", file=sys.stderr)
    t0 = time.time()

    model = None
    tokenizer = None

    if not use_llama:
        from lit_parrot import Tokenizer
        from lit_parrot.adapter import Parrot, Config
        from lit_parrot.utils import EmptyInitOnDevice, lazy_load, check_valid_checkpoint_dir
        check_valid_checkpoint_dir(checkpoint_dir)
        with open(checkpoint_dir / "lit_config.json") as fp:
            config = Config(**json.load(fp))
        with EmptyInitOnDevice(device=fabric.device, dtype=dtype, quantization_mode=quantize):
            model = Parrot(config)
        print("Loading model...", file=sys.stderr)
        with lazy_load(checkpoint_dir / "lit_model.pth") as pretrained_checkpoint:
            model.load_state_dict(pretrained_checkpoint, strict=False)
        tokenizer = Tokenizer(checkpoint_dir / "tokenizer.json", checkpoint_dir / "tokenizer_config.json")
        print("Loading adapter...", file=sys.stderr)
        adapter_checkpoint = torch.load(adapter_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(adapter_checkpoint, strict=False)

    
    if use_llama:
        from lit_llama import LLaMA, Tokenizer
        from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup        
        from lit_llama.model import LLaMA, LLaMAConfig
        with EmptyInitOnDevice(
                device=fabric.device, dtype=dtype, quantization_mode=quantize
        ):
            model = LLaMA.from_name("7B")
        model.config.block_size = 1024
        print('Model configured, GPU memory is allocated, loading to CPU memory...')
        checkpoint = torch.load(checkpoint_dir, map_location=lambda storage, loc: storage)        
        print('Model loaded, transfering to GPU memory...')
        model.load_state_dict(checkpoint, strict=False)
        tokenizer = Tokenizer(tokenizer_path)
        print("Loading adapter...", file=sys.stderr)
        adapter_checkpoint = torch.load(adapter_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(adapter_checkpoint, strict=False)

    print(f"Time to load: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)    

    first_run = True
    last_input = ''
    last_output = ''
    last_time = ''
    with open(f'save/story.txt', 'a') as fp:
        now = datetime.now() # current date and time
        date_time = now.strftime("%d.%m.%Y, %H:%M:%S")
        fp.write('\n\n-------------------------------------------------------------\n')
        fp.write(f'{date_time}\n')
        fp.write('-------------------------------------------------------------\n\n')

    if use_sockets:
        import asyncio
        import websockets
        import ssl
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_cert = "ssl/website.pem"
        ssl_key = "ssl/website.key"
        ssl_context.load_cert_chain(ssl_cert, keyfile=ssl_key)

        async def websocket_handler(websocket, path):
            while True:
                global total_active_count  
                message = await websocket.recv()          
                print(f'Got message: {message}')
                use_temperature = temperature
                use_max = max_new_tokens
                if message[0:1] == '[':
                    closed = message.find(']')
                    if closed == -1:
                        message = message[1:]
                    else:
                        commands = message[1:closed].split(';')
                        for cmd in commands:
                            one_split = cmd.split('=')
                            if one_split[0] == 'temp':
                                use_temperature = float(one_split[1])
                            if one_split[0] == 'max':
                                use_max = int(one_split[1])
                        message = message[(closed+1):]                        
                    print(f'temp={use_temperature}, max={use_max}')    
                    print(message)

                total_active_count += 1
                await asyncio.sleep(0)
                if total_active_count > 5:
                    total_active_count = total_active_count - 1
                    await websocket.send(f'[[;red;]Maximum concurrent users reached, try again in 10-20 seconds]')
                    await websocket.send(f'### END ###')  
                else:
                    print(f'Active users start: {total_active_count}')
                    output = await generate_output(model, tokenizer, message, fabric, use_max, use_temperature, top_k, use_alpaca, sock=websocket)
                    await websocket.send(output)
                    total_active_count = total_active_count - 1
                    await asyncio.sleep(0)
                    print(f'Active users end: {total_active_count}')
                    await websocket.send(f'### END ###')  

        start_server = websockets.serve(websocket_handler, "0.0.0.0", use_port, ssl=ssl_context)

        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever() 
        
        from threading import Event

        while True:
            Event().wait()

    if not use_sockets:
        while True:
            my_input = 'help'
            
            if not first_run:
                my_input = input(f"\ntemp: {temperature} | max: {max_new_tokens} | top_k: {top_k} > ")
            first_run = False

            should_do_prompt = True

            if len(my_input) < 4:
                should_do_prompt = False
                print('Input is too short or doesn\'t make sense! Try again!')

            if my_input == 'exit':
                print("Bye!")
                break

            if my_input == 'help':
                should_do_prompt = False
                print('\nAvailable commands are:')
                print('set temp=TEMPERATURE -> sets temperature, use 0.5 to 1.0 for best results')
                print('set max=MAX_TOKENS -> sets how many tokens will be in the answer, use 128 to 384 for best results')
                print('set top_k=TOP_K -> sets top_k, use whatever you want, this is black magic')
                print('save -> saves last result to save/story.txt')
                print('exit -> exit to system')
                print('help -> this help')

            if my_input == 'save':
                should_do_prompt = False
                with open(f'save/story.txt', 'a') as fp:
                    fp.write(f'{last_output}\n\n')
                print('Saved to save/story.txt')

            if my_input[0:6] == 'query ':
                use_instruction = my_input[6:]
                print('Not used')

            if my_input[0:4] == 'set ':            
                use_input = my_input[4:].strip()
                if use_input[0:5] == 'temp=':
                    t = use_input.split('=')
                    if len(t) == 2:
                        temperature = float(t[1])
                        should_do_prompt = False

                if use_input[0:4] == 'max=':
                    t = use_input.split('=')
                    if len(t) == 2:
                        max_new_tokens = int(t[1])
                        should_do_prompt = False

                if use_input[0:6] == 'top_k=':
                    t = use_input.split('=')
                    if len(t) == 2:
                        top_k = int(t[1])
                        should_do_prompt = False


            if should_do_prompt:
                last_input = my_input
                t0 = time.perf_counter()
                output = generate_output_console(model, tokenizer, my_input, fabric, max_new_tokens, temperature, top_k, use_alpaca)
                last_output = output
                # pyperclip.copy(output)
                t = time.perf_counter() - t0
                last_time = f'{t:.02f}s'
                tw = textwrap.wrap(output)
                print("\r                                                 \n" + '\n'.join(tw) + '\n')
                print(f"[Time for inference: {t:.02f}s total; Memory in use: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB; type 'help' for help]", file=sys.stderr)

@torch.no_grad()
def generate_console(
    model: torch.nn.Module,
    tokenizer,
    idx: torch.Tensor,
    max_new_tokens: int,
    max_seq_length: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    sock = None
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (B, T) with indices of the prompt sequence.
        max_new_tokens: The number of new tokens to generate.
        max_seq_length: The maximum sequence length allowed.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    B, T = idx.shape
    T_new = T + max_new_tokens
    empty = torch.empty(B, T_new, dtype=idx.dtype, device=idx.device)
    empty[:, :T] = idx
    idx = empty

    # generate max_new_tokens tokens
    internal_counter = 0
    tokens_to_decode = ''
    # s = spm.SentencePieceProcessor(model_file='spm.model')
    time_start = time.perf_counter()
    for t in range(T, T_new):        
        # print(t)
        # ignore the not-filled-yet tokens
        idx_cond = idx[:, :t]
        # if the sequence context is growing too long we must crop it at max_seq_length
        idx_cond = idx_cond if T <= max_seq_length else idx_cond[:, -max_seq_length:]

        # forward
        logits = model(idx_cond)
        logits = logits[:, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)        

        # concatenate the new column
        idx[:, t:] = idx_next        

        next_token = tokenizer.decode(idx_next[0].cpu())
        if next_token == '\n':
            next_token = ' '

        time_end = time.perf_counter() - time_start        
        print(f'\r{time_end:.02f}s {t}/{T_new}:      ', sep=' ', end="", flush=True)
        print(f'\r{time_end:.02f}s {t}/{T_new}: ' + next_token, sep=' ', end="", flush=True)

        if idx_next == tokenizer.eos_id or next_token[0:1] == '#':
            return idx

        internal_counter += 1        

    return idx


async def generate_output(model, tokenizer, prompt, fabric, max_new_tokens, temperature, top_k, use_alpaca, sock=None):    
    encoded =  tokenizer.encode(prompt, device=model.device)
    output = await generate_websocket(
        model,
        idx=encoded,
        max_seq_length=2048,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        tokenizer=tokenizer,
        sock=sock,
    )
    output = tokenizer.decode(output)
    if '#' in output:
        output = output[0:(output.index('#'))]  
    output = output.strip()
    last_char_01 = output.rfind('.')        
    last_char_02 = output.rfind('!')        
    last_char_03 = output.rfind('?')        
    last_max = max(last_char_01, last_char_02, last_char_03)
    output = output[0:last_max+1]
    return output


@torch.no_grad()
def generate_orig(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    max_seq_length: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
    tokenizer: Optional = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_new_tokens: The number of new tokens to generate.
        max_seq_length: The maximum sequence length allowed.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
        eos_id: If specified, stop generating any more token once the <eos> token is triggered
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = idx.size(0)
    T_new = min(T + max_new_tokens, max_seq_length)
    empty = torch.empty(T_new, dtype=idx.dtype, device=idx.device)
    empty[:T] = idx
    idx = empty

    time_start = time.perf_counter()
    # generate max_new_tokens tokens
    for t in range(T, T_new):
        # ignore the not-filled-yet tokens
        idx_cond = idx[:t]

        # forward
        logits = model(idx_cond.view(1, -1))
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[[-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # concatenate the new generation
        idx[t] = idx_next

        next_token = tokenizer.decode(idx_next[0].cpu())
        if next_token == '\n':
            next_token = ' '

        time_end = time.perf_counter() - time_start        
        print(f'\r{time_end:.02f}s {t}/{T_new}:      ', sep=' ', end="", flush=True)
        print(f'\r{time_end:.02f}s {t}/{T_new}: ' + next_token, sep=' ', end="", flush=True)

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id or next_token[0:1] == '#':
            return idx[: t + 1]  # include the EOS token

    return idx

@torch.no_grad()
async def generate_websocket(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    max_seq_length: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
    tokenizer: Optional = None,
    sock: Optional = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_new_tokens: The number of new tokens to generate.
        max_seq_length: The maximum sequence length allowed.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
        eos_id: If specified, stop generating any more token once the <eos> token is triggered
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = idx.size(0)
    T_new = min(T + max_new_tokens, max_seq_length)
    empty = torch.empty(T_new, dtype=idx.dtype, device=idx.device)
    empty[:T] = idx
    idx = empty

    time_start = time.perf_counter()
    # generate max_new_tokens tokens
    for t in range(T, T_new):
        # ignore the not-filled-yet tokens
        idx_cond = idx[:t]

        # forward
        logits = model(idx_cond.view(1, -1))
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[[-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # concatenate the new generation
        idx[t] = idx_next

        next_token = tokenizer.decode(idx_next[0].cpu())
        if next_token == '\n':
            next_token = ' '

        time_end = time.perf_counter() - time_start        
        print(f'\r{time_end:.02f}s {t}/{T_new}:      ', sep=' ', end="", flush=True)
        print(f'\r{time_end:.02f}s {t}/{T_new}: ' + next_token, sep=' ', end="", flush=True)

        if sock != None:
            await asyncio.sleep(0)
            await sock.send(f'{time_end:.02f}s {t}/{T_new}: {next_token}')
            await asyncio.sleep(0)


        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id or next_token[0:1] == '#':
            return idx[: t + 1]  # include the EOS token

    return idx


def generate_output_console(model, tokenizer, prompt, fabric, max_new_tokens, temperature, top_k, use_alpaca, sock=None):   
    encoded =  tokenizer.encode(prompt, device=model.device)
    output = generate_orig(
        model,
        idx=encoded,
        max_seq_length=2048,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        tokenizer=tokenizer,
        # sock=sock,
    )
    output = tokenizer.decode(output)
    if '#' in output:
        output = output[0:(output.index('#'))]  
    output = output.strip()
    last_char_01 = output.rfind('.')        
    last_char_02 = output.rfind('!')        
    last_char_03 = output.rfind('?')        
    last_max = max(last_char_01, last_char_02, last_char_03)
    output = output[0:last_max+1]
    return output

if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet",
    )
    CLI(main)
