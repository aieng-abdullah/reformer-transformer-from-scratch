import torch


def generate_text(model, start_seq, max_len, device):
    model.eval()
    generated = start_seq.to(device)

    with torch.no_grad():
        for _ in range(max_len - start_seq.size(1)):
            logits = model(generated)
            next_token_logits = logits[:, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

    return generated
