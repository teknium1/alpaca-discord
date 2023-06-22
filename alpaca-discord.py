from concurrent.futures import ThreadPoolExecutor
import discord, torch
import asyncio
from transformers import LlamaTokenizer, LlamaForCausalLM

intents = discord.Intents.default()
intents.members = True

client = discord.Client(intents=intents)
tokenizer = LlamaTokenizer.from_pretrained("./alpaca/")

model = LlamaForCausalLM.from_pretrained(
    "alpaca",
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

queue = asyncio.Queue()

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    asyncio.get_running_loop().create_task(background_task())

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if isinstance(message.channel, discord.channel.DMChannel) or (client.user and client.user.mentioned_in(message)):
        if message.reference:
            pastMessage = await message.channel.fetch_message(message.reference.message_id)
        else:
            pastMessage = None
        await queue.put((message, pastMessage))

def sync_task(message):
    input_ids = tokenizer(message, return_tensors="pt").input_ids.to("cuda")
    generated_ids = model.generate(input_ids, max_new_tokens=250, do_sample=True, repetition_penalty=1.3, temperature=0.8, top_p=0.75, top_k=40)
    response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:])
    return response

async def background_task():
    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    print("Task Started. Waiting for inputs.")
    while True:
        msg_pair: tuple[discord.Message, discord.Message] = await queue.get()
        msg, past = msg_pair

        username = client.user.name
        user_id = client.user.id
        message_content = msg.content.replace(f"@{username} ", "").replace(f"<@{user_id}> ", "")
        past_content = None
        if past:
            past_content = past.content.replace(f"@{username} ", "").replace(f"<@{user_id}> ", "")
        text = generate_prompt(message_content, past_content)
        response = await loop.run_in_executor(executor, sync_task, text)
        print(f"Response: {text}\n{response}")
        await msg.reply(response, mention_author=False)

def generate_prompt(text, pastMessage):
    if pastMessage:
        return f"""### Instruction:
Your previous response to the prior instruction: {pastMessage}
        
Current instruction to respond to: {text}
### Response:"""
    else:
        return f"""### Instruction:
{text}
### Response:"""

#Load the API key from alpacakey.txt
with open("alpacakey.txt", "r") as f:
    key = f.read()

client.run(key)
