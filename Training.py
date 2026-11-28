import torch
from torch import nn
from torch.nn import functional as F

# Ustawienia
batch_size = 64
sequence_length = 256
embedding_size = 384
heads = 6
layers = 6
dropout = 0.2
learning_rate = 0.0003
training_steps = 5000

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

with open("training_data.txt", encoding="utf-8") as file:
    text = file.read()

characters = sorted(set(text))
vocab_size = len(characters)

char_to_number = {}
for number, character in enumerate(characters):
    char_to_number[character] = number


def encode(text):
    return [char_to_number[character] for character in text]


def decode(numbers):
    return "".join(characters[number] for number in numbers)


data = torch.tensor(encode(text), dtype=torch.long)
split = int(len(data) * 0.9)

train_data = data[:split]
validation_data = data[split:]

if min(len(train_data), len(validation_data)) <= sequence_length:
    raise ValueError(
        "Tekst jest za krótki. Dodaj więcej danych lub zmniejsz sequence_length."
    )


def get_batch(data):
    starts = torch.randint(
        len(data) - sequence_length,
        (batch_size,)
    )

    inputs = []
    targets = []

    for start in starts:
        inputs.append(data[start:start + sequence_length])
        targets.append(data[start + 1:start + sequence_length + 1])

    inputs = torch.stack(inputs).to(device)
    targets = torch.stack(targets).to(device)

    return inputs, targets


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        self.attention_dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask):
        normalized = self.norm1(x)

        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            need_weights=False
        )

        x = x + self.attention_dropout(attention_output)
        x = x + self.feed_forward(self.norm2(x))

        return x


class TextModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.character_embedding = nn.Embedding(
            vocab_size, embedding_size
        )
        self.position_embedding = nn.Embedding(
            sequence_length, embedding_size
        )

        self.blocks = nn.ModuleList(
            [TransformerBlock() for _ in range(layers)]
        )

        self.norm = nn.LayerNorm(embedding_size)
        self.output = nn.Linear(embedding_size, vocab_size)

        mask = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool),
            diagonal=1
        )
        self.register_buffer("mask", mask)

    def forward(self, inputs, targets=None):
        length = inputs.shape[1]
        positions = torch.arange(length, device=inputs.device)

        x = (
            self.character_embedding(inputs)
            + self.position_embedding(positions)
        )

        mask = self.mask[:length, :length]

        for block in self.blocks:
            x = block(x, mask)

        logits = self.output(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, inputs, new_characters):
        for _ in range(new_characters):
            fragment = inputs[:, -sequence_length:]
            logits, _ = self(fragment)

            probabilities = F.softmax(logits[:, -1, :], dim=-1)
            next_character = torch.multinomial(probabilities, 1)

            inputs = torch.cat([inputs, next_character], dim=1)

        return inputs


@torch.no_grad()
def evaluate(model):
    model.eval()
    losses = {}

    for name, dataset in [
        ("train", train_data),
        ("validation", validation_data)
    ]:
        total_loss = 0

        for _ in range(20):
            inputs, targets = get_batch(dataset)
            _, loss = model(inputs, targets)
            total_loss += loss.item()

        losses[name] = total_loss / 20

    model.train()
    return losses


model = TextModel().to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

for step in range(training_steps):
    inputs, targets = get_batch(train_data)
    _, loss = model(inputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % 100 == 0:
        losses = evaluate(model)

        print(
            f"Krok {step + 1}: "
            f"trening = {losses['train']:.4f}, "
            f"walidacja = {losses['validation']:.4f}"
        )

model.eval()

start = torch.tensor(
    [encode(text[:1])],
    dtype=torch.long,
    device=device
)

result = model.generate(start, new_characters=500)
print(decode(result[0].tolist()))