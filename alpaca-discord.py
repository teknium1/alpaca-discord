import discord, torch
from queue import Queue
from transformers import LlamaTokenizer, LlamaForCausalLM

intents = discord.Intents.default()
intents.members = True

client = discord.Client(intents=intents)
tokenizer = LlamaTokenizer.from_pretrained("./alpaca/")

model = LlamaForCausalLM.from_pretrained(
    "alpaca",
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto"
)

queue = Queue()

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if client.user.mentioned_in(message):
        queue.put(message)
        if queue.qsize() == 1:
            await generate_response()

async def generate_response():
    message = queue.get()
    text = generate_prompt(message.content.replace("@Alpaca ", "").replace("<@1086483646946496573> ", ""))
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    generated_ids = model.generate(input_ids, max_new_tokens=250, do_sample=True, repetition_penalty=1.0, temperature=0.8, top_p=0.75, top_k=40)
    response = tokenizer.decode(generated_ids[0])

    await message.channel.send(response.replace(text, ""))
    if not queue.empty():
        await generate_response()

def generate_prompt(text):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Instruction:
    {text}
    ### Response:"""

#Load the API key from alpacakey.txt
with open("alpacakey.txt", "r") as f:
    key = f.read()

client.run(key)