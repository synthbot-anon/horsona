import asyncio

from horsona.audiogen.gptsovits import GptSovitsTTS

GPT_SOVITS_ENDPOINT = "http://localhost:9880"
VOICE = "rarity"


async def main():
    gpt_sovits = GptSovitsTTS(GPT_SOVITS_ENDPOINT)
    i = 0
    while True:
        user_input = input("Text: ")
        speech = await gpt_sovits.generate_speech(VOICE, user_input)

        with open(f"{VOICE}_{i}.wav", "wb") as f:
            f.write(speech)

        print(f"Saved to {VOICE}_{i}.wav")
        print("-" * 60)
        i += 1


asyncio.run(main())
