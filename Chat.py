import os
import torch
from Training import (
    BigramLanguageModel,
    encode,
    decode,
    device,
    checkpoint_file
)

history_file = "chat_history.txt"
max_new_tokens = 200


model = BigramLanguageModel().to(device)

checkpoint = torch.load(checkpoint_file, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


conversation = ""

if os.path.exists(history_file):
    with open(history_file, "r", encoding="utf-8") as file:
        conversation = file.read()

    print("Poprzednia rozmowa:")
    print(conversation)


if __name__ == "__main__":
    while True:
        message = input("<user> ")

        if message.lower() == "exit":
            break

        user_line = f"<user> {message}\n"
        conversation += user_line

        context = torch.tensor(
            [encode(conversation)],
            dtype=torch.long,
            device=device
        )

        with torch.no_grad():
            output = model.generate(context, max_new_tokens)

        new_tokens = output[0, context.shape[1]:]
        response = decode(new_tokens.tolist())

        response = response.split("<user>")[0].strip()

        print(response)

        conversation += response + "\n"

        with open(history_file, "w", encoding="utf-8") as file:
            file.write(conversation)