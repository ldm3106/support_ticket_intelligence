"""Record actual local predictions; examples are not an evaluation dataset."""
import torch
from support_ticket.config import REPORTS, save_json
from support_ticket.inference import TicketPredictor

torch.set_num_threads(4)
predictor = TicketPredictor()
texts = [
    "I was charged twice for the same card payment.",
    "My new card has not arrived yet.",
    "I forgot the passcode for my account.",
    "I lost my card. What should I do?",
    "How do I get a refund for a purchase?",
    "My printer is broken and I need technical help.",
]
results = [{"text": text, **predictor.predict_ticket(text)} for text in texts]
save_json(REPORTS / "example_predictions.json", results)
print(results)
