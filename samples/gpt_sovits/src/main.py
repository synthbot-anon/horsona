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

        output_path = f"{VOICE}_{i}.wav"
        with open(output_path, "wb") as f:
            f.write(speech)
            i += 1

        print(f"Saved to {output_path}")
        print("-" * 60)


asyncio.run(main())
